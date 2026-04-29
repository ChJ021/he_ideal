from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from hetune.core.types import CostVector
from hetune.utils.paths import resolve_path


class HEBackendError(RuntimeError):
    """Base error for encrypted deployment backends."""


class HEBackendUnavailableError(HEBackendError):
    """Raised when the configured HE runtime is not installed or usable."""


class HECaseExecutionError(HEBackendError):
    """Raised when an installed backend cannot execute one deployment case."""


@dataclass(slots=True)
class BackendAvailability:
    available: bool
    reason: str
    checked_paths: dict[str, str]

    def require(self) -> None:
        if not self.available:
            raise HEBackendUnavailableError(self.reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "checked_paths": dict(self.checked_paths),
        }


@dataclass(slots=True)
class DeploymentCaseRequest:
    case_name: str
    schedule_name: str
    experiment_config_path: Path
    deployment_config_path: Path
    backend_config_path: Path | None
    schedule_path: Path
    output_dir: Path
    overrides_path: Path | None
    sample_size: int
    sequence_length: int
    latency_repetitions: int
    metadata: dict[str, Any]
    runner_mode: str = "openfhe_schedule_workload"
    forward_manifest_path: Path | None = None
    ckks_config: dict[str, Any] | None = None
    multiplicative_depth: int = 60
    scaling_mod_size: int = 59
    first_mod_size: int = 60
    poly_modulus_degree: int = 0
    linear_kernel: str = "bsgs_hoisted"
    bsgs_baby_step: int = 32
    fuse_qkv: bool = True
    packing_strategy: str = "row_packed"
    token_block_size: str = "auto"
    profile_native_stages: bool = True
    bootstrap_enabled: bool = False
    bootstrap_level_budget: tuple[int, int] = (4, 4)
    bootstrap_dim1: tuple[int, int] = (0, 0)
    bootstrap_levels_after: int = 10
    bootstrap_num_iterations: int = 1
    bootstrap_precision: int = 0
    bootstrap_auto_guard: bool = True
    bootstrap_guard_min_levels: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "schedule_name": self.schedule_name,
            "experiment_config_path": str(self.experiment_config_path),
            "deployment_config_path": str(self.deployment_config_path),
            "backend_config_path": str(self.backend_config_path) if self.backend_config_path else None,
            "schedule_path": str(self.schedule_path),
            "output_dir": str(self.output_dir),
            "overrides_path": str(self.overrides_path) if self.overrides_path else None,
            "sample_size": self.sample_size,
            "sequence_length": self.sequence_length,
            "latency_repetitions": self.latency_repetitions,
            "runner_mode": self.runner_mode,
            "forward_manifest_path": str(self.forward_manifest_path) if self.forward_manifest_path else None,
            "multiplicative_depth": self.multiplicative_depth,
            "scaling_mod_size": self.scaling_mod_size,
            "first_mod_size": self.first_mod_size,
            "poly_modulus_degree": self.poly_modulus_degree,
            "linear_kernel": self.linear_kernel,
            "bsgs_baby_step": self.bsgs_baby_step,
            "fuse_qkv": self.fuse_qkv,
            "packing_strategy": self.packing_strategy,
            "token_block_size": self.token_block_size,
            "profile_native_stages": self.profile_native_stages,
            "bootstrap_enabled": self.bootstrap_enabled,
            "bootstrap_level_budget": list(self.bootstrap_level_budget),
            "bootstrap_dim1": list(self.bootstrap_dim1),
            "bootstrap_levels_after": self.bootstrap_levels_after,
            "bootstrap_num_iterations": self.bootstrap_num_iterations,
            "bootstrap_precision": self.bootstrap_precision,
            "bootstrap_auto_guard": self.bootstrap_auto_guard,
            "bootstrap_guard_min_levels": self.bootstrap_guard_min_levels,
            "ckks_config": dict(self.ckks_config or {}),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class DeploymentCaseResult:
    case_name: str
    schedule_name: str
    feasible: bool
    accuracy: float | None = None
    sample_count: int = 0
    latency_ms: float | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    cost: CostVector | None = None
    predictions_path: Path | None = None
    logits_path: Path | None = None
    error: str = ""
    backend_metadata: dict[str, Any] | None = None

    @classmethod
    def infeasible(
        cls,
        case_name: str,
        schedule_name: str,
        error: str,
        *,
        backend_metadata: dict[str, Any] | None = None,
    ) -> "DeploymentCaseResult":
        return cls(
            case_name=case_name,
            schedule_name=schedule_name,
            feasible=False,
            error=error,
            backend_metadata=backend_metadata or {},
        )

    def to_row(self) -> dict[str, Any]:
        cost = self.cost or CostVector()
        metadata = self.backend_metadata or {}
        row = {
            "case": self.case_name,
            "schedule": self.schedule_name,
            "feasible": self.feasible,
            "accuracy": self.accuracy,
            "accuracy_source": metadata.get("accuracy_source", ""),
            "runner_mode": metadata.get("runner_mode", ""),
            "linear_kernel": metadata.get("linear_kernel", ""),
            "bsgs_baby_step": metadata.get("bsgs_baby_step", ""),
            "fuse_qkv": metadata.get("fuse_qkv", ""),
            "packing_strategy": metadata.get("packing_strategy", ""),
            "token_block_size": metadata.get("token_block_size", ""),
            "rotation_key_count": metadata.get("rotation_key_count", ""),
            "keygen_ms": metadata.get("keygen_ms", ""),
            "encrypt_ms": metadata.get("encrypt_ms", ""),
            "qkv_ms": metadata.get("qkv_ms", ""),
            "attention_ms": metadata.get("attention_ms", ""),
            "ffn_ms": metadata.get("ffn_ms", ""),
            "classifier_ms": metadata.get("classifier_ms", ""),
            "decrypt_ms": metadata.get("decrypt_ms", ""),
            "bootstrap_mode": metadata.get("bootstrap_mode", ""),
            "bootstrap_ms": metadata.get("bootstrap_ms", ""),
            "bootstrap_policy_count": metadata.get("bootstrap_policy_count", ""),
            "bootstrap_auto_count": metadata.get("bootstrap_auto_count", ""),
            "sample_count": self.sample_count,
            "latency_ms": self.latency_ms,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "predictions_path": str(self.predictions_path) if self.predictions_path else "",
            "logits_path": str(self.logits_path) if self.logits_path else "",
            "error": self.error,
        }
        for key, value in cost.to_dict().items():
            if key != "latency_ms":
                row[key] = value
        return row

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: Path) -> "DeploymentCaseResult":
        cost = CostVector.from_dict(data.get("cost") or data)
        predictions = data.get("predictions_path")
        predictions_path = resolve_path(predictions, base_dir) if predictions else None
        logits = data.get("logits_path")
        logits_path = resolve_path(logits, base_dir) if logits else None
        return cls(
            case_name=str(data.get("case_name", data.get("case", ""))),
            schedule_name=str(data.get("schedule_name", data.get("schedule", ""))),
            feasible=bool(data.get("feasible", False)),
            accuracy=_optional_float(data.get("accuracy")),
            sample_count=int(data.get("sample_count", 0)),
            latency_ms=_optional_float(data.get("latency_ms")),
            latency_p50_ms=_optional_float(data.get("latency_p50_ms")),
            latency_p95_ms=_optional_float(data.get("latency_p95_ms")),
            cost=cost,
            predictions_path=predictions_path,
            logits_path=logits_path,
            error=str(data.get("error", "")),
            backend_metadata=dict(data.get("backend_metadata", {})),
        )


class HEBackend(Protocol):
    def availability(self) -> BackendAvailability:
        ...

    def run_case(self, request: DeploymentCaseRequest) -> DeploymentCaseResult:
        ...


@dataclass(slots=True)
class OpenFHEExternalBackend:
    """Delegates real HE execution to an OpenFHE native runner.

    The Python side owns orchestration and reporting. The backend is considered
    unavailable unless both OpenFHE and the runner exist, so deployment never
    silently falls back to plaintext.
    """

    config: dict[str, Any]
    config_path: Path | None = None

    @property
    def root_dir(self) -> Path:
        return self._path("openfhe_root", Path("/home/cj/github/openfhe"))

    @property
    def source_dir(self) -> Path:
        return self._path("source_dir", self.root_dir / "src")

    @property
    def build_dir(self) -> Path:
        return self._path("build_dir", self.root_dir / "build")

    @property
    def install_dir(self) -> Path:
        return self._path("install_dir", self.root_dir / "install")

    @property
    def python_dir(self) -> Path:
        return self._path("python_dir", self.root_dir / "openfhe-python")

    @property
    def runner_path(self) -> Path:
        configured = self.config.get("runner_path")
        if configured:
            return self._path("runner_path", self.install_dir / "bin" / "hetune_openfhe_runner")
        return self.install_dir / "bin" / "hetune_openfhe_runner"

    def _path(self, key: str, default: Path) -> Path:
        raw = self.config.get(key)
        path = Path(raw) if raw is not None else default
        resolved = path if path.is_absolute() or self.config_path is None else self.config_path.parent / path
        return resolved.resolve(strict=False)

    def availability(self) -> BackendAvailability:
        checked = {
            "openfhe_root": str(self.root_dir),
            "source_dir": str(self.source_dir),
            "build_dir": str(self.build_dir),
            "install_dir": str(self.install_dir),
            "python_dir": str(self.python_dir),
            "runner_path": str(self.runner_path),
        }
        config_files = list(self.install_dir.glob("**/OpenFHEConfig.cmake"))
        if not self.root_dir.exists():
            return BackendAvailability(False, f"OpenFHE root not found: {self.root_dir}", checked)
        if not self.install_dir.exists():
            return BackendAvailability(False, f"OpenFHE install dir not found: {self.install_dir}", checked)
        if not config_files:
            return BackendAvailability(
                False,
                f"OpenFHEConfig.cmake not found under {self.install_dir}",
                checked,
            )
        if not self.runner_path.exists():
            return BackendAvailability(
                False,
                f"OpenFHE native runner not found: {self.runner_path}",
                checked,
            )
        if not os.access(self.runner_path, os.X_OK):
            return BackendAvailability(
                False,
                f"OpenFHE native runner is not executable: {self.runner_path}",
                checked,
            )
        return BackendAvailability(True, "available", checked)

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "openfhe_external",
            "backend_id": self.config.get("backend_id", "openfhe_local"),
            "openfhe_root": str(self.root_dir),
            "source_dir": str(self.source_dir),
            "build_dir": str(self.build_dir),
            "install_dir": str(self.install_dir),
            "python_dir": str(self.python_dir),
            "runner_path": str(self.runner_path),
            "config_path": str(self.config_path) if self.config_path else "",
        }

    def run_case(self, request: DeploymentCaseRequest) -> DeploymentCaseResult:
        self.availability().require()
        request.output_dir.mkdir(parents=True, exist_ok=True)
        request_path = request.output_dir / "request.json"
        result_path = request.output_dir / "result.json"
        request_path.write_text(json.dumps(request.to_dict(), indent=2), encoding="utf-8")
        command = [
            str(self.runner_path),
            "--request",
            str(request_path),
            "--output",
            str(result_path),
        ]
        env = dict(os.environ)
        openfhe_lib = str(self.install_dir / "lib")
        existing_ld_path = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = (
            f"{openfhe_lib}:{existing_ld_path}" if existing_ld_path else openfhe_lib
        )
        completed = subprocess.run(
            command,
            cwd=str(request.output_dir),
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise HECaseExecutionError(
                f"OpenFHE runner failed for {request.case_name}: {message}"
            )
        if not result_path.exists():
            raise HECaseExecutionError(
                f"OpenFHE runner did not write result JSON: {result_path}"
            )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return DeploymentCaseResult.from_dict(payload, request.output_dir)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
