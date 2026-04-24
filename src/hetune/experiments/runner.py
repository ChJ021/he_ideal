from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from hetune.core.ids import OperatorKey
from hetune.core.serialization import load_yaml, save_yaml
from hetune.core.types import ExperimentPaths, SchedulePlan, SensitivityRecord
from hetune.cost.static import StaticCostModel
from hetune.execution.evaluator import PlaintextEvaluator
from hetune.experiments.artifacts import (
    write_artifacts_index,
    write_config_snapshots,
    write_manifest,
)
from hetune.experiments.config import LoadedExperimentConfig, load_experiment_config
from hetune.experiments.data import load_tokenized_dataset
from hetune.experiments.diagnostics import write_combination_diagnostics
from hetune.experiments.distillation import DistillationRunner, load_override_payload
from hetune.experiments.reporting import write_report
from hetune.experiments.visualization import write_sensitivity_heatmap
from hetune.models.hf_adapter import HFSequenceClassifierAdapter
from hetune.operators.registry import ApproximationRegistry, build_default_registry
from hetune.profiling.calibration import (
    calibration_coverage,
    collect_operator_calibration_stats,
    load_calibration_stats,
)
from hetune.profiling.profiler import SensitivityProfiler
from hetune.scheduling.policies import (
    BasePolicy,
    GreedyDowngradePolicy,
    UniformPolicy,
    ValidatedGreedyDowngradePolicy,
)
from hetune.security.validators import SecurityValidator


class ExperimentRunner:
    def __init__(
        self,
        config_path: str | Path,
        operator_scope: str | None = None,
        command_name: str = "run",
    ) -> None:
        self.config_path = Path(config_path)
        self.loaded = load_experiment_config(config_path)
        self.operator_scope = operator_scope or self.loaded.experiment.get(
            "operator_scope",
            "activation_norm",
        )
        configured_operator_types = (
            None if operator_scope is not None else self.loaded.experiment.get("operator_types")
        )
        self.operator_types = self._operator_types_for_scope(
            self.operator_scope,
            configured_operator_types,
        )
        self.command_name = command_name
        self.experiment_id = self.loaded.experiment["experiment_id"]
        self.paths = ExperimentPaths(
            experiment_id=self.experiment_id,
            root=self.loaded.root / self.loaded.experiment.get("output_root", "outputs"),
        )
        self.paths.ensure()
        write_config_snapshots(self.paths, self.loaded)
        self._write_run_manifest(command_name)

    def run(self) -> None:
        self.profile()
        self.tune()
        self._run_distillation_if_enabled()
        self.evaluate()
        self._write_run_manifest(self.command_name)
        write_artifacts_index(
            self.paths,
            self.experiment_id,
            self.operator_scope,
            self.operator_types,
        )

    def profile(self) -> Path:
        adapter, registry, evaluator = self._build_runtime(split_key="calibration_split")
        dataset = self._load_dataset(adapter, split_key="calibration_split")
        self._ensure_calibration_stats(adapter, dataset)
        profiler = SensitivityProfiler(evaluator, registry, self._metadata())
        records = profiler.profile_all(dataset)
        profile_path = self.paths.profile_dir() / "sensitivity_matrix.csv"
        profiler.save(records, profile_path)
        cost_model = self._build_cost_model(registry)
        cost_model.export_candidate_costs(
            self.paths.root
            / "cost_tables"
            / self.loaded.ckks.get("ckks_param_id", "static_ckks_128")
            / "candidate_costs.csv"
        )
        write_sensitivity_heatmap(
            profile_path,
            self.paths.figure_dir() / "sensitivity_heatmap.png",
        )
        return profile_path

    def tune(self) -> Path:
        adapter, registry, evaluator = self._build_runtime(split_key="calibration_split")
        dataset = self._load_dataset(adapter, split_key="calibration_split")
        self._ensure_calibration_stats(adapter, dataset)
        profile_path = self.paths.profile_dir() / "sensitivity_matrix.csv"
        if not profile_path.exists():
            self.profile()
            adapter.set_calibration_stats(
                load_calibration_stats(self._calibration_stats_path())
            )
        records = self._load_sensitivity_records(profile_path, adapter.operators)
        metadata = self._metadata()
        scheduler_cfg = self.loaded.experiment.get("scheduler", {})
        max_accuracy_drop = float(
            scheduler_cfg.get(
                "max_accuracy_drop",
                self.loaded.experiment.get("accuracy_tolerance", 0.01),
            )
        )
        constraints = {
            "max_accuracy_drop": max_accuracy_drop,
            "max_logit_kl": float(scheduler_cfg.get("max_logit_kl", 0.02)),
            "max_label_flip_rate": float(scheduler_cfg.get("max_label_flip_rate", 0.01)),
            "max_downgrades_per_layer": int(scheduler_cfg.get("max_downgrades_per_layer", 1)),
            "input_independent": True,
            "min_security_bits": self.loaded.ckks.get("security_bits", 128),
            "operator_scope": self.operator_scope,
            "operator_types": self.operator_types,
        }
        ckks_param_id = self.loaded.ckks.get("ckks_param_id", "static_ckks_128")
        base_schedule = BasePolicy(registry, ckks_param_id).generate(
            adapter.operators,
            metadata=metadata,
            constraints=constraints,
        )
        save_yaml(
            base_schedule.to_dict(),
            self.paths.schedule_dir() / "base_reference.yaml",
        )
        for quality in ("low", "mid", "high"):
            schedule = UniformPolicy(registry, ckks_param_id, quality).generate(
                adapter.operators,
                metadata=metadata,
                constraints=constraints,
            )
            save_yaml(
                schedule.to_dict(),
                self.paths.schedule_dir() / f"uniform_{quality}.yaml",
            )

        additive_greedy = GreedyDowngradePolicy(
            registry=registry,
            cost_model=self._build_cost_model(registry),
            max_accuracy_drop=max_accuracy_drop,
            ckks_param_id=ckks_param_id,
        ).generate(
            adapter.operators,
            sensitivity_records=records,
            metadata=metadata,
            constraints=constraints,
        )
        save_yaml(
            additive_greedy.to_dict(),
            self.paths.schedule_dir() / "hetune_additive_greedy.yaml",
        )

        def evaluate_schedule(schedule):
            return evaluator.run(dataset, schedule=schedule)

        if self.loaded.experiment.get("run_combination_diagnostics", True):
            write_combination_diagnostics(
                self.paths.profile_dir() / "combination_diagnostics.csv",
                adapter.operators,
                registry,
                evaluate_schedule,
                metadata,
                constraints,
                ckks_param_id,
                quality=self.loaded.experiment.get("combination_diagnostic_quality", "low"),
            )

        validated = ValidatedGreedyDowngradePolicy(
            registry=registry,
            cost_model=self._build_cost_model(registry),
            evaluate_schedule=evaluate_schedule,
            baseline_schedule=base_schedule,
            max_accuracy_drop=max_accuracy_drop,
            max_logit_kl=float(scheduler_cfg.get("max_logit_kl", 0.02)),
            max_label_flip_rate=float(scheduler_cfg.get("max_label_flip_rate", 0.01)),
            max_downgrades_per_layer=int(scheduler_cfg.get("max_downgrades_per_layer", 1)),
            protected_operator_types=tuple(
                scheduler_cfg.get("protected_operator_types", ["softmax"])
            ),
            min_quality_rank_by_operator_type=scheduler_cfg.get(
                "min_quality_rank_by_operator_type",
                {"layernorm": 45},
            ),
            ckks_param_id=ckks_param_id,
        ).generate(
            adapter.operators,
            sensitivity_records=records,
            metadata=metadata,
            constraints=constraints,
        )
        decisions = pd.DataFrame([decision.to_dict() for decision in validated.decisions])
        decisions.to_csv(
            self.paths.schedule_dir() / "validated_greedy_decisions.csv",
            index=False,
        )

        findings = SecurityValidator().validate(validated.schedule)
        if findings:
            raise ValueError(f"Generated schedule failed validation: {findings}")
        output = self.paths.schedule_dir() / "hetune_generated.yaml"
        save_yaml(validated.schedule.to_dict(), output)
        self._write_selection_summary(validated.schedule, registry)
        return output

    def evaluate(self) -> Path:
        adapter, registry, evaluator = self._build_runtime(split_key="validation_split")
        self._ensure_calibration_stats_for_adapter(adapter)
        dataset = self._load_dataset(adapter, split_key="validation_split")
        schedule_path = self.paths.schedule_dir() / "hetune_generated.yaml"
        if not schedule_path.exists():
            self.tune()
        schedule = SchedulePlan.from_dict(load_yaml(schedule_path))
        result = evaluator.run(dataset, schedule=schedule)
        cost_model = self._build_cost_model(registry)
        total_cost = cost_model.estimate_schedule(schedule)
        distilled_payload = self._load_distillation_payload()
        rows = []
        base_path = self.paths.schedule_dir() / "base_reference.yaml"
        if not base_path.exists():
            base_schedule = BasePolicy(
                registry,
                self.loaded.ckks.get("ckks_param_id", "static_ckks_128"),
            ).generate(
                adapter.operators,
                metadata=self._metadata(),
                constraints={"input_independent": True},
            )
            save_yaml(base_schedule.to_dict(), base_path)
        base_schedule = SchedulePlan.from_dict(load_yaml(base_path))
        base_result = evaluator.run(dataset, schedule=base_schedule)
        base_cost = cost_model.estimate_schedule(base_schedule)
        rows.append(
            {
                "schedule": "base",
                "accuracy": base_result.accuracy,
                **base_cost.to_dict(),
            }
        )
        rows.append(
            {
                "schedule": "hetune_generated",
                "accuracy": result.accuracy,
                **total_cost.to_dict(),
            }
        )
        if distilled_payload is not None:
            distilled_result = evaluator.run(
                dataset,
                schedule=schedule,
                parameter_overrides=list(distilled_payload.get("entries", [])),
            )
            rows.append(
                {
                    "schedule": "hetune_generated_distilled",
                    "accuracy": distilled_result.accuracy,
                    **total_cost.to_dict(),
                }
            )
        for quality in ("low", "mid", "high"):
            baseline_path = self.paths.schedule_dir() / f"uniform_{quality}.yaml"
            if not baseline_path.exists():
                continue
            baseline = SchedulePlan.from_dict(load_yaml(baseline_path))
            baseline_result = evaluator.run(dataset, schedule=baseline)
            baseline_cost = cost_model.estimate_schedule(baseline)
            rows.append(
                {
                    "schedule": f"uniform_{quality}",
                    "accuracy": baseline_result.accuracy,
                    **baseline_cost.to_dict(),
                }
            )
        metrics = pd.DataFrame(rows)
        metrics_path = self.paths.evaluation_dir() / "metrics.csv"
        metrics.to_csv(metrics_path, index=False)
        write_report(
            self.paths.report_dir() / "report.md",
            self.experiment_id,
            schedule,
            metrics,
            total_cost,
            decision_log_path=self.paths.schedule_dir() / "validated_greedy_decisions.csv",
            diagnostics_path=self.paths.profile_dir() / "combination_diagnostics.csv",
            calibration_stats_path=self._calibration_stats_path(),
            calibration_coverage=calibration_coverage(
                adapter.operators,
                adapter.calibration_stats,
            ),
            distillation_summary_path=self.paths.distillation_dir() / "summary.csv",
            distillation_report_path=self.paths.distillation_dir() / "report.md",
            distillation_overrides_path=self.paths.distillation_dir() / "overrides.pt",
            operator_scope=self.operator_scope,
            operator_types=self.operator_types,
        )
        write_artifacts_index(
            self.paths,
            self.experiment_id,
            self.operator_scope,
            self.operator_types,
        )
        return metrics_path

    def _build_runtime(
        self,
        split_key: str,
    ) -> tuple[HFSequenceClassifierAdapter, ApproximationRegistry, PlaintextEvaluator]:
        enabled_ids = self._enabled_candidates(self.loaded.approximations)
        registry = build_default_registry(enabled_ids)
        model_cfg = self.loaded.model
        adapter = HFSequenceClassifierAdapter(
            model_id=model_cfg["model_id"],
            model_name_or_path=model_cfg["model_name_or_path"],
            num_labels=int(model_cfg.get("num_labels", 2)),
            device=self.loaded.experiment.get("device", "cpu"),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        ).load()
        adapter.operators = [
            operator
            for operator in adapter.operators
            if operator.operator_type in set(self.operator_types)
        ]
        adapter.set_calibration_stats(load_calibration_stats(self._calibration_stats_path()))
        evaluator = PlaintextEvaluator(
            adapter,
            registry,
            batch_size=int(self.loaded.experiment.get("batch_size", 16)),
            max_batches=self.loaded.experiment.get("max_batches"),
        )
        return adapter, registry, evaluator

    def _ensure_calibration_stats(self, adapter: HFSequenceClassifierAdapter, dataset) -> Path:
        stats_path = self._calibration_stats_path()
        if stats_path.exists():
            adapter.set_calibration_stats(load_calibration_stats(stats_path))
            return stats_path
        collect_operator_calibration_stats(
            adapter=adapter,
            dataset=dataset,
            output_path=stats_path,
            batch_size=int(self.loaded.experiment.get("batch_size", 16)),
            max_batches=self.loaded.experiment.get("max_batches"),
        )
        return stats_path

    def _ensure_calibration_stats_for_adapter(
        self,
        adapter: HFSequenceClassifierAdapter,
    ) -> Path:
        stats_path = self._calibration_stats_path()
        if not stats_path.exists():
            calibration_adapter, _, _ = self._build_runtime(split_key="calibration_split")
            calibration_dataset = self._load_dataset(
                calibration_adapter,
                split_key="calibration_split",
            )
            self._ensure_calibration_stats(calibration_adapter, calibration_dataset)
        adapter.set_calibration_stats(load_calibration_stats(stats_path))
        return stats_path

    def _calibration_stats_path(self) -> Path:
        return self.paths.profile_dir() / "operator_calibration_stats.csv"

    def _load_dataset(self, adapter: HFSequenceClassifierAdapter, split_key: str):
        sample_key = "calibration_size" if split_key == "calibration_split" else "validation_size"
        return load_tokenized_dataset(
            self.loaded.dataset,
            adapter,
            split_key=split_key,
            sample_size=self.loaded.experiment.get(sample_key),
            max_length=int(self.loaded.experiment.get("sequence_length", 128)),
        )

    def _build_cost_model(self, registry: ApproximationRegistry) -> StaticCostModel:
        return StaticCostModel(
            registry=registry,
            ckks_param_id=self.loaded.ckks.get("ckks_param_id", "static_ckks_128"),
            weights=self.loaded.experiment.get("cost_weights", {}),
        )

    def _write_selection_summary(
        self,
        schedule: SchedulePlan,
        registry: ApproximationRegistry,
    ) -> None:
        cost_model = self._build_cost_model(registry)
        rows = []
        for entry in schedule.entries:
            cost = cost_model.estimate(entry.operator_key, entry.candidate_id)
            rows.append(
                {
                    "operator_id": entry.operator_key.id,
                    "layer_index": entry.operator_key.layer_index,
                    "operator_type": entry.operator_key.operator_type,
                    "operator_name": entry.operator_key.name,
                    "candidate_id": entry.candidate_id,
                    **cost.to_dict(),
                }
            )
        if not rows:
            return
        summary = pd.DataFrame(rows)
        summary.to_csv(self.paths.schedule_dir() / "selection_summary.csv", index=False)
        softmax = summary[summary["operator_type"] == "softmax"]
        if not softmax.empty:
            softmax.to_csv(self.paths.schedule_dir() / "softmax_selection.csv", index=False)

    def _metadata(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "model_id": self.loaded.model["model_id"],
            "dataset_id": self.loaded.dataset["dataset_id"],
            "sequence_length_bucket": self.loaded.experiment.get("sequence_length", 128),
            "backend_id": "plaintext_sim",
            "security_level": self.loaded.ckks.get("security_bits", 128),
            "operator_scope": self.operator_scope,
            "operator_types": self.operator_types,
        }

    def _run_distillation_if_enabled(self) -> Path | None:
        if "layernorm" not in set(self.operator_types):
            return None
        if not self.loaded.experiment.get("distillation", {}).get("enabled", False):
            return None
        runner = DistillationRunner(self.config_path, command_name="distill")
        return runner.run()

    def _load_distillation_payload(self) -> dict[str, Any] | None:
        overrides_path = self.paths.distillation_dir() / "overrides.pt"
        if not overrides_path.exists():
            return None
        return load_override_payload(overrides_path)

    def _write_run_manifest(self, command_name: str) -> None:
        write_manifest(
            self.paths,
            self.experiment_id,
            self.operator_scope,
            self.operator_types,
            command_name,
            {
                "profile": self.paths.profile_dir() / "sensitivity_matrix.csv",
                "calibration_stats": self._calibration_stats_path(),
                "schedule": self.paths.schedule_dir() / "hetune_generated.yaml",
                "decisions": self.paths.schedule_dir() / "validated_greedy_decisions.csv",
                "metrics": self.paths.evaluation_dir() / "metrics.csv",
                "distillation_summary": self.paths.distillation_dir() / "summary.csv",
                "distillation_overrides": self.paths.distillation_dir() / "overrides.pt",
                "distillation_report": self.paths.distillation_dir() / "report.md",
                "report": self.paths.report_dir() / "report.md",
            },
        )

    @staticmethod
    def _operator_types_for_scope(
        operator_scope: str,
        configured: list[str] | None,
    ) -> list[str]:
        if configured:
            return list(configured)
        scopes = {
            "activation_norm": ["gelu", "layernorm"],
            "softmax_only": ["softmax"],
            "all_nonlinear": ["gelu", "layernorm", "softmax"],
        }
        if operator_scope not in scopes:
            raise ValueError(f"Unknown operator_scope: {operator_scope}")
        return scopes[operator_scope]

    @staticmethod
    def _enabled_candidates(config: dict[str, Any]) -> set[str]:
        candidates = config.get("candidates", [])
        enabled: set[str] = set()
        for item in candidates:
            if item.get("enabled", True):
                enabled.add(item["candidate_id"])
        return enabled

    @staticmethod
    def _load_sensitivity_records(
        profile_path: Path,
        operators: list[OperatorKey],
    ) -> list[SensitivityRecord]:
        data = pd.read_csv(profile_path)
        by_id = {operator.id: operator for operator in operators}
        records: list[SensitivityRecord] = []
        for row in data.to_dict(orient="records"):
            operator = by_id.get(str(row["operator_id"]))
            if operator is None:
                continue
            records.append(
                SensitivityRecord(
                    operator_key=operator,
                    candidate_id=str(row["candidate_id"]),
                    baseline_accuracy=float(row["baseline_accuracy"]),
                    candidate_accuracy=float(row["candidate_accuracy"]),
                    accuracy_drop=float(row["accuracy_drop"]),
                    logit_kl=float(row["logit_kl"]),
                    label_flip_rate=float(row["label_flip_rate"]),
                    hidden_l2=float(row.get("hidden_l2", 0.0)),
                    attention_kl=float(row.get("attention_kl", 0.0)),
                )
            )
        return records
