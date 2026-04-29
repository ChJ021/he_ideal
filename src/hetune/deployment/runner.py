from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from hetune.core.serialization import load_yaml, save_json
from hetune.core.types import ExperimentPaths, SchedulePlan
from hetune.deployment.backend import (
    DeploymentCaseRequest,
    DeploymentCaseResult,
    HEBackend,
    HEBackendUnavailableError,
    HECaseExecutionError,
    OpenFHEExternalBackend,
)
from hetune.deployment.config import load_deployment_config
from hetune.deployment.forward_artifact import export_distilbert_forward_artifact
from hetune.experiments.distillation import load_override_payload
from hetune.utils.paths import resolve_path


@dataclass(slots=True)
class CaseSpec:
    name: str
    schedule_name: str
    overrides_path: Path | None = None
    requires_overrides: bool = False


class HEDeploymentRunner:
    def __init__(
        self,
        config_path: str | Path,
        *,
        backend: HEBackend | None = None,
        allow_unavailable_backend: bool | None = None,
    ) -> None:
        self.config = load_deployment_config(config_path)
        self.loaded = self.config.experiment
        experiment_id = self.loaded.experiment["experiment_id"]
        self.paths = ExperimentPaths(
            experiment_id=experiment_id,
            root=self.loaded.root / self.loaded.experiment.get("output_root", "outputs"),
        )
        self.output_dir = self.paths.he_deployment_dir()
        self.backend = backend or OpenFHEExternalBackend(
            self.config.backend_config,
            self.config.backend_config_path,
        )
        if allow_unavailable_backend is None:
            self.allow_unavailable_backend = not self.config.fail_on_unavailable_backend
        else:
            self.allow_unavailable_backend = allow_unavailable_backend

    def run(self) -> Path:
        self.paths.ensure()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        case_specs = [self._case_spec(item) for item in self.config.cases]
        availability = self.backend.availability()
        metadata = self._metadata(availability.to_dict())
        save_json(metadata, self.output_dir / "metadata.json")

        if not availability.available and not self.allow_unavailable_backend:
            raise HEBackendUnavailableError(availability.reason)

        results: list[DeploymentCaseResult] = []
        if not availability.available:
            results = [
                DeploymentCaseResult.infeasible(
                    case.name,
                    case.schedule_name,
                    availability.reason,
                    backend_metadata=availability.to_dict(),
                )
                for case in case_specs
            ]
        else:
            for case in case_specs:
                results.append(self._run_case(case, metadata))

        comparison_path = self.output_dir / "comparison.csv"
        pd.DataFrame([result.to_row() for result in results]).to_csv(
            comparison_path,
            index=False,
        )
        self._write_report(results, comparison_path, availability.to_dict())
        return comparison_path

    def _run_case(
        self,
        case: CaseSpec,
        metadata: dict[str, Any],
    ) -> DeploymentCaseResult:
        case_dir = self.output_dir / case.name
        case_dir.mkdir(parents=True, exist_ok=True)
        try:
            request = self._build_request(case, case_dir, metadata)
            result = self.backend.run_case(request)
            if self.config.fail_on_plaintext_accuracy_fallback:
                self._require_native_logits_accuracy(result)
            else:
                self._fill_accuracy_from_existing_metrics(case, result)
        except HECaseExecutionError as exc:
            if not self.config.continue_on_case_failure:
                raise
            result = DeploymentCaseResult.infeasible(
                case.name,
                case.schedule_name,
                str(exc),
                backend_metadata=getattr(self.backend, "metadata", lambda: {})(),
            )
        self._write_case_metrics(case_dir, result)
        return result

    def _fill_accuracy_from_existing_metrics(
        self,
        case: CaseSpec,
        result: DeploymentCaseResult,
    ) -> None:
        if result.accuracy is not None:
            return
        metrics_path = self.paths.evaluation_dir() / "metrics.csv"
        if not metrics_path.exists():
            return
        schedule_name = (
            "hetune_generated_distilled"
            if case.name == "post_distill"
            else case.schedule_name
        )
        frame = pd.read_csv(metrics_path)
        if "schedule" not in frame.columns or "accuracy" not in frame.columns:
            return
        matched = frame.loc[frame["schedule"] == schedule_name]
        if matched.empty:
            return
        result.accuracy = float(matched.iloc[0]["accuracy"])
        metadata = dict(result.backend_metadata or {})
        metadata["accuracy_source"] = "plaintext_evaluation_metrics"
        metadata["accuracy_schedule"] = schedule_name
        result.backend_metadata = metadata

    def _require_native_logits_accuracy(self, result: DeploymentCaseResult) -> None:
        metadata = result.backend_metadata or {}
        if metadata.get("accuracy_source") != "native_decrypted_logits":
            raise HECaseExecutionError(
                f"{result.case_name} did not return native decrypted-logits accuracy"
            )
        if result.accuracy is None:
            raise HECaseExecutionError(
                f"{result.case_name} returned no encrypted-forward accuracy"
            )
        if result.predictions_path is None:
            raise HECaseExecutionError(
                f"{result.case_name} returned no encrypted-forward predictions path"
            )

    def _build_request(
        self,
        case: CaseSpec,
        case_dir: Path,
        metadata: dict[str, Any],
    ) -> DeploymentCaseRequest:
        schedule_path = self.paths.schedule_dir() / f"{case.schedule_name}.yaml"
        if not schedule_path.exists():
            raise FileNotFoundError(f"Schedule not found for deployment case {case.name}: {schedule_path}")
        schedule = SchedulePlan.from_dict(load_yaml(schedule_path))
        if not schedule.entries:
            raise ValueError(f"Schedule has no entries for deployment case {case.name}: {schedule_path}")

        overrides_path = case.overrides_path
        overrides_hash = ""
        if case.requires_overrides:
            if overrides_path is None or not overrides_path.exists():
                raise FileNotFoundError(
                    f"Distilled deployment case {case.name} requires overrides: {overrides_path}"
                )
            payload = load_override_payload(overrides_path)
            if not payload.get("entries"):
                raise ValueError(f"Distillation overrides contain no entries: {overrides_path}")
            overrides_hash = _sha256(overrides_path)

        runner_mode = self.config.runner_mode
        sample_size = self.config.sample_size
        sequence_length = self.config.sequence_length
        forward_manifest_path: Path | None = None
        native_ckks_config = self._native_ckks_config()
        if runner_mode == "openfhe_distilbert_forward":
            sample_size = self.config.encrypted_sample_size
            sequence_length = self.config.encrypted_sequence_length
            artifact_root = self.config.forward_artifact_dir or self.output_dir / "forward_artifacts"
            artifact = export_distilbert_forward_artifact(
                output_dir=artifact_root / case.name,
                loaded_experiment=self.loaded,
                schedule=schedule,
                schedule_name=case.schedule_name,
                case_name=case.name,
                sample_size=sample_size,
                sequence_length=sequence_length,
                overrides_path=overrides_path,
                ckks_config=native_ckks_config,
            )
            forward_manifest_path = artifact.manifest_path

        return DeploymentCaseRequest(
            case_name=case.name,
            schedule_name=case.schedule_name,
            experiment_config_path=self.config.experiment_config_path,
            deployment_config_path=self.config.config_path,
            backend_config_path=self.config.backend_config_path,
            schedule_path=schedule_path,
            output_dir=case_dir,
            overrides_path=overrides_path,
            sample_size=sample_size,
            sequence_length=sequence_length,
            latency_repetitions=self.config.latency_repetitions,
            runner_mode=runner_mode,
            forward_manifest_path=forward_manifest_path,
            ckks_config=native_ckks_config,
            multiplicative_depth=_int_config(native_ckks_config, "multiplicative_depth", 60),
            scaling_mod_size=_int_config(native_ckks_config, "scaling_mod_size", 59),
            first_mod_size=_int_config(native_ckks_config, "first_mod_size", 60),
            poly_modulus_degree=_int_config(native_ckks_config, "poly_modulus_degree", 0),
            linear_kernel=self.config.linear_kernel,
            bsgs_baby_step=self.config.bsgs_baby_step,
            fuse_qkv=self.config.fuse_qkv,
            packing_strategy=self.config.packing_strategy,
            token_block_size=self.config.token_block_size,
            profile_native_stages=self.config.profile_native_stages,
            bootstrap_enabled=_bool_config(
                native_ckks_config,
                "bootstrap_enabled",
                bool(native_ckks_config.get("bootstrapping_supported", False)),
            ),
            bootstrap_level_budget=_int_pair_config(
                native_ckks_config,
                "bootstrap_level_budget",
                (4, 4),
            ),
            bootstrap_dim1=_int_pair_config(native_ckks_config, "bootstrap_dim1", (0, 0)),
            bootstrap_levels_after=_int_config(native_ckks_config, "bootstrap_levels_after", 10),
            bootstrap_num_iterations=_int_config(native_ckks_config, "bootstrap_num_iterations", 1),
            bootstrap_precision=_int_config(native_ckks_config, "bootstrap_precision", 0),
            bootstrap_auto_guard=_bool_config(native_ckks_config, "bootstrap_auto_guard", True),
            bootstrap_guard_min_levels=_int_config(native_ckks_config, "bootstrap_guard_min_levels", 2),
            metadata={
                **metadata,
                "schedule_hash": _sha256(schedule_path),
                "overrides_hash": overrides_hash,
                "privacy_boundary": self.config.raw.get("privacy_boundary", "client_embedding"),
            },
        )

    def _native_ckks_config(self) -> dict[str, Any]:
        """Merge experiment CKKS metadata with backend-specific OpenFHE params."""
        merged = dict(self.loaded.ckks)
        backend_ckks = self.config.backend_config.get("ckks")
        if isinstance(backend_ckks, dict):
            merged.update(backend_ckks)
        return merged

    def _case_spec(self, item: str | dict[str, Any]) -> CaseSpec:
        if isinstance(item, str):
            return self._builtin_case(item)
        name = str(item["name"])
        schedule_name = str(item.get("schedule_name", name))
        overrides = item.get("overrides_path")
        return CaseSpec(
            name=name,
            schedule_name=schedule_name,
            overrides_path=resolve_path(overrides, self.config.config_path.parent) if overrides else None,
            requires_overrides=bool(item.get("requires_overrides", False)),
        )

    def _builtin_case(self, name: str) -> CaseSpec:
        if name == "high":
            return CaseSpec(name="high", schedule_name="uniform_high")
        if name == "pre_distill":
            return CaseSpec(name="pre_distill", schedule_name="hetune_generated")
        if name == "post_distill":
            return CaseSpec(
                name="post_distill",
                schedule_name="hetune_generated",
                overrides_path=self.paths.distillation_dir() / "overrides.pt",
                requires_overrides=True,
            )
        raise ValueError(f"Unknown deployment case: {name}")

    def _metadata(self, availability: dict[str, Any]) -> dict[str, Any]:
        backend_metadata = getattr(self.backend, "metadata", lambda: {})()
        return {
            "deployment_id": self.config.deployment_id,
            "experiment_id": self.loaded.experiment["experiment_id"],
            "model_id": self.loaded.model.get("model_id", ""),
            "model_name_or_path": self.loaded.model.get("model_name_or_path", ""),
            "dataset_id": self.loaded.dataset.get("dataset_id", self.loaded.dataset.get("dataset_name", "")),
            "sequence_length": self.config.sequence_length,
            "sample_size": self.config.sample_size,
            "latency_repetitions": self.config.latency_repetitions,
            "deployment_config_path": str(self.config.config_path),
            "experiment_config_path": str(self.config.experiment_config_path),
            "backend_config_path": str(self.config.backend_config_path) if self.config.backend_config_path else "",
            "backend": backend_metadata,
            "backend_availability": availability,
            "runner_mode": self.config.runner_mode,
            "encrypted_sequence_length": self.config.encrypted_sequence_length,
            "encrypted_sample_size": self.config.encrypted_sample_size,
        }

    def _write_case_metrics(self, case_dir: Path, result: DeploymentCaseResult) -> None:
        pd.DataFrame([result.to_row()]).to_csv(case_dir / "metrics.csv", index=False)
        latency_rows = [
            {
                "case": result.case_name,
                "latency_ms": result.latency_ms,
                "latency_p50_ms": result.latency_p50_ms,
                "latency_p95_ms": result.latency_p95_ms,
                "feasible": result.feasible,
            }
        ]
        pd.DataFrame(latency_rows).to_csv(case_dir / "latency.csv", index=False)

    def _write_report(
        self,
        results: list[DeploymentCaseResult],
        comparison_path: Path,
        availability: dict[str, Any],
    ) -> None:
        lines = [
            f"# HE Deployment Report: {self.loaded.experiment['experiment_id']}",
            "",
            f"- Deployment id: `{self.config.deployment_id}`",
            f"- Backend available: `{availability.get('available')}`",
            f"- Backend status: `{availability.get('reason')}`",
            f"- Privacy boundary: `{self.config.raw.get('privacy_boundary', 'client_embedding')}`",
            f"- Sequence length: `{self.config.sequence_length}`",
            f"- Sample size: `{self.config.sample_size}`",
            f"- Runner mode: `{self.config.runner_mode}`",
            f"- Encrypted sequence length: `{self.config.encrypted_sequence_length}`",
            f"- Encrypted sample size: `{self.config.encrypted_sample_size}`",
            f"- Plaintext accuracy fallback allowed: `{not self.config.fail_on_plaintext_accuracy_fallback}`",
            f"- Comparison: `{comparison_path}`",
            "",
            "| Case | Schedule | Feasible | Accuracy | Latency ms | Error |",
            "|---|---|---:|---:|---:|---|",
        ]
        for result in results:
            accuracy = "" if result.accuracy is None else f"{result.accuracy:.6f}"
            latency = "" if result.latency_ms is None else f"{result.latency_ms:.3f}"
            error = result.error.replace("|", "/")
            lines.append(
                f"| {result.case_name} | {result.schedule_name} | {result.feasible} | {accuracy} | {latency} | {error} |"
            )
        (self.output_dir / "deployment_report.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _int_config(config: dict[str, Any], key: str, fallback: int) -> int:
    value = config.get(key, fallback)
    if value is None or value == "":
        return fallback
    return int(value)


def _bool_config(config: dict[str, Any], key: str, fallback: bool) -> bool:
    value = config.get(key, fallback)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _int_pair_config(config: dict[str, Any], key: str, fallback: tuple[int, int]) -> tuple[int, int]:
    value = config.get(key, fallback)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (int(value[0]), int(value[1]))
    return fallback
