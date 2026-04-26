from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from hetune.core.ids import OperatorKey
from hetune.core.serialization import load_yaml, save_yaml
from hetune.core.types import ExperimentPaths, SchedulePlan, SensitivityRecord
from hetune.cost.profiled import ProfiledHECostModel
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
    HEUniformPolicy,
    UniformPolicy,
    ValidatedGreedyDowngradePolicy,
)
from hetune.scheduling.he_planner import analyze_schedule_feasibility
from hetune.security.validators import SecurityValidator
from hetune.utils.paths import resolve_path


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
        cost_model = self._build_cost_model(registry)
        base_schedule = BasePolicy(registry, ckks_param_id).generate(
            adapter.operators,
            metadata=metadata,
            constraints=constraints,
        )
        save_yaml(
            base_schedule.to_dict(),
            self.paths.schedule_dir() / "base_reference.yaml",
        )
        if self._is_he_aware():
            constraints.update(
                {
                    "he_aware": True,
                    "available_levels": self.loaded.ckks.get("available_levels"),
                    "bootstrapping_supported": bool(
                        self.loaded.ckks.get("bootstrapping_supported", False)
                    ),
                    "max_bootstrap_count": scheduler_cfg.get("max_bootstrap_count"),
                }
            )
        initial_schedule = (
            HEUniformPolicy(
                registry=registry,
                cost_model=cost_model,
                ckks_config=self.loaded.ckks,
                ckks_param_id=ckks_param_id,
                quality="high",
            ).generate(
                adapter.operators,
                metadata=metadata,
                constraints=constraints,
            )
            if self._is_he_aware()
            else None
        )
        if initial_schedule is not None:
            initial_schedule = self._annotate_schedule_for_he(
                initial_schedule,
                cost_model,
            ).schedule
        for quality in ("low", "mid", "high"):
            if self._is_he_aware():
                schedule = HEUniformPolicy(
                    registry=registry,
                    cost_model=cost_model,
                    ckks_config=self.loaded.ckks,
                    ckks_param_id=ckks_param_id,
                    quality=quality,
                ).generate(
                    adapter.operators,
                    metadata=metadata,
                    constraints=constraints,
                )
                schedule = self._annotate_schedule_for_he(schedule, cost_model).schedule
            else:
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
            cost_model=cost_model,
            max_accuracy_drop=max_accuracy_drop,
            ckks_param_id=ckks_param_id,
            initial_schedule=initial_schedule,
        ).generate(
            adapter.operators,
            sensitivity_records=records,
            metadata=metadata,
            constraints=constraints,
        )
        if self._is_he_aware():
            additive_greedy = self._annotate_schedule_for_he(
                additive_greedy,
                cost_model,
            ).schedule
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
            cost_model=cost_model,
            evaluate_schedule=evaluate_schedule,
            baseline_schedule=base_schedule,
            initial_schedule=initial_schedule,
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
            schedule_constraint_checker=self._he_schedule_constraint_checker(cost_model)
            if self._is_he_aware()
            else None,
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
        final_schedule = validated.schedule
        if self._is_he_aware():
            final_schedule = self._annotate_schedule_for_he(
                final_schedule,
                cost_model,
            ).schedule
        output = self.paths.schedule_dir() / "hetune_generated.yaml"
        save_yaml(final_schedule.to_dict(), output)
        self._write_selection_summary(final_schedule, registry)
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
        he_result = self._annotate_schedule_for_he(schedule, cost_model) if self._is_he_aware() else None
        total_cost = he_result.total_cost if he_result is not None else cost_model.estimate_schedule(schedule)
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
        base_he_result = self._annotate_schedule_for_he(base_schedule, cost_model) if self._is_he_aware() else None
        base_cost = base_he_result.total_cost if base_he_result is not None else cost_model.estimate_schedule(base_schedule)
        rows.append(
            {
                "schedule": "base",
                "accuracy": base_result.accuracy,
                **self._schedule_profile_summary(cost_model, base_schedule),
                **base_cost.to_dict(),
            }
        )
        rows.append(
            {
                "schedule": "hetune_generated",
                "accuracy": result.accuracy,
                **self._schedule_profile_summary(cost_model, schedule),
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
                    **self._schedule_profile_summary(cost_model, schedule),
                    **total_cost.to_dict(),
                }
            )
        for quality in ("low", "mid", "high"):
            baseline_path = self.paths.schedule_dir() / f"uniform_{quality}.yaml"
            if not baseline_path.exists():
                continue
            baseline = SchedulePlan.from_dict(load_yaml(baseline_path))
            baseline_result = evaluator.run(dataset, schedule=baseline)
            baseline_he_result = self._annotate_schedule_for_he(baseline, cost_model) if self._is_he_aware() else None
            baseline_cost = baseline_he_result.total_cost if baseline_he_result is not None else cost_model.estimate_schedule(baseline)
            rows.append(
                {
                    "schedule": f"uniform_{quality}",
                    "accuracy": baseline_result.accuracy,
                    **self._schedule_profile_summary(cost_model, baseline),
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
            he_summary=self._he_summary_from_result(he_result, cost_model, schedule),
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
        registry = build_default_registry(enabled_ids, ckks_only=self._is_he_aware())
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

    def _build_cost_model(self, registry: ApproximationRegistry) -> Any:
        if self._is_he_aware():
            return ProfiledHECostModel(
                registry=registry,
                profile_path=self._profile_path(),
                ckks_param_id=self.loaded.ckks.get("ckks_param_id", "static_ckks_128"),
                backend_id=self.loaded.ckks.get("backend_id") or self.loaded.ckks.get("backend"),
                weights=self.loaded.experiment.get("cost_weights", {}),
                profile_required=self._profile_required(),
                profile_min_coverage=self._profile_min_coverage(),
            )
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
            cost_source = (
                cost_model.source_for(entry.candidate_id)
                if hasattr(cost_model, "source_for")
                else "static_fallback"
            )
            rows.append(
                {
                    "operator_id": entry.operator_key.id,
                    "layer_index": entry.operator_key.layer_index,
                    "operator_type": entry.operator_key.operator_type,
                    "operator_name": entry.operator_key.name,
                    "candidate_id": entry.candidate_id,
                    "ckks_param_id": entry.ckks_param_id,
                    "level_budget": entry.level_budget,
                    "bootstrap_policy": entry.bootstrap_policy,
                    "cost_source": cost_source,
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
            "backend_id": self.loaded.ckks.get("backend_id", "plaintext_sim")
            if self._is_he_aware()
            else "plaintext_sim",
            "security_level": self.loaded.ckks.get("security_bits", 128),
            "operator_scope": self.operator_scope,
            "operator_types": self.operator_types,
        }

    def _profile_required(self) -> bool:
        return bool(self.loaded.experiment.get("scheduler", {}).get("profile_required", False))

    def _profile_min_coverage(self) -> float:
        scheduler = self.loaded.experiment.get("scheduler", {})
        if "profile_min_coverage" in scheduler:
            return float(scheduler["profile_min_coverage"])
        return 1.0 if self._profile_required() else 0.0

    def _is_he_aware(self) -> bool:
        return bool(self.loaded.experiment.get("scheduler", {}).get("he_aware", False))

    def _profile_path(self) -> Path | None:
        if self.loaded.ckks.get("backend") == "static-only":
            return None
        profile = self.loaded.ckks.get("backend_profile_path")
        if not profile:
            return None
        return resolve_path(profile, self.loaded.root)

    def _annotate_schedule_for_he(
        self,
        schedule: SchedulePlan,
        cost_model: Any,
    ):
        return analyze_schedule_feasibility(
            schedule.metadata.get("policy", "schedule"),
            schedule,
            cost_model,
            self.loaded.ckks,
        )

    def _he_schedule_constraint_checker(self, cost_model: Any):
        def check(schedule: SchedulePlan) -> str | None:
            result = self._annotate_schedule_for_he(schedule, cost_model)
            max_bootstrap_count = self.loaded.experiment.get("scheduler", {}).get(
                "max_bootstrap_count"
            )
            if max_bootstrap_count is not None and result.estimated_bootstrap_count > int(
                max_bootstrap_count
            ):
                return "he_exceeds_max_bootstrap_count"
            if result.feasible:
                return None
            if result.first_violation_reason == "single_operator_exceeds_available_levels":
                return "he_infeasible_single_operator"
            if result.first_violation_reason == "level_budget_exceeded":
                return "he_infeasible_level_budget"
            return "he_infeasible"

        return check

    def _he_summary_from_result(
        self,
        result,
        cost_model: Any,
        schedule: SchedulePlan,
    ) -> dict[str, Any] | None:
        if result is None:
            return None
        summary = {
            "he_aware": True,
            "he_feasible": result.feasible,
            "estimated_bootstrap_count": result.estimated_bootstrap_count,
            "unsupported_rows": result.unsupported_count,
            "ckks_param_id": result.schedule.metadata.get("ckks_param_id"),
            "he_backend_id": result.schedule.metadata.get("he_backend_id"),
        }
        if result.first_violation_reason:
            summary["he_first_violation_reason"] = result.first_violation_reason
        summary.update(self._schedule_profile_summary(cost_model, schedule))
        return summary

    def _schedule_profile_summary(self, cost_model: Any, schedule: SchedulePlan) -> dict[str, Any]:
        if not hasattr(cost_model, "coverage_for_schedule") or not hasattr(cost_model, "load_summary"):
            return {}
        coverage = cost_model.coverage_for_schedule(schedule)
        missing_profile_ids = ",".join(coverage.used_candidate_ids_missing_profile)
        return {
            "profile_candidates_loaded": cost_model.load_summary.profile_candidates_loaded,
            "profile_entries": coverage.profile_entries,
            "static_fallback_entries": coverage.static_fallback_entries,
            "profile_coverage_rate": coverage.profile_coverage_rate,
            "used_candidates_with_profile": coverage.used_candidates_with_profile,
            "used_candidates_missing_profile": coverage.used_candidates_missing_profile,
            "used_candidate_ids_missing_profile": missing_profile_ids,
            "strict_profile_check_passed": coverage.strict_profile_check_passed,
            "strict_profile_check_reason": coverage.strict_profile_check_reason or "",
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
