from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hetune.core.serialization import load_yaml
from hetune.experiments.config import LoadedExperimentConfig, load_experiment_config
from hetune.utils.paths import project_root, resolve_path


@dataclass(slots=True)
class DeploymentConfig:
    root: Path
    config_path: Path
    raw: dict[str, Any]
    experiment_config_path: Path
    experiment: LoadedExperimentConfig
    backend_config_path: Path | None
    backend_config: dict[str, Any]

    @property
    def deployment_id(self) -> str:
        return str(
            self.raw.get(
                "deployment_id",
                f"{self.experiment.experiment['experiment_id']}_he_deployment",
            )
        )

    @property
    def cases(self) -> list[str | dict[str, Any]]:
        configured = self.raw.get("cases")
        if configured is None:
            configured = ["high", "pre_distill", "post_distill"]
        return list(configured)

    @property
    def sample_size(self) -> int:
        return int(self.raw.get("sample_size", self.raw.get("validation_size", 16)))

    @property
    def sequence_length(self) -> int:
        return int(
            self.raw.get(
                "sequence_length",
                self.experiment.experiment.get("sequence_length", 128),
            )
        )

    @property
    def latency_repetitions(self) -> int:
        return int(self.raw.get("latency_repetitions", 1))

    @property
    def runner_mode(self) -> str:
        return str(self.raw.get("runner_mode", "openfhe_schedule_workload"))

    @property
    def encrypted_sample_size(self) -> int:
        return int(self.raw.get("encrypted_sample_size", self.sample_size))

    @property
    def encrypted_sequence_length(self) -> int:
        return int(self.raw.get("encrypted_sequence_length", self.sequence_length))

    @property
    def forward_artifact_dir(self) -> Path | None:
        configured = self.raw.get("forward_artifact_dir")
        if configured is None:
            return None
        return resolve_path(configured, self.config_path.parent)

    @property
    def fail_on_plaintext_accuracy_fallback(self) -> bool:
        return bool(
            self.raw.get(
                "fail_on_plaintext_accuracy_fallback",
                self.runner_mode == "openfhe_distilbert_forward",
            )
        )

    @property
    def linear_kernel(self) -> str:
        return str(self.raw.get("linear_kernel", "bsgs_hoisted"))

    @property
    def bsgs_baby_step(self) -> int:
        return int(self.raw.get("bsgs_baby_step", 32))

    @property
    def fuse_qkv(self) -> bool:
        return bool(self.raw.get("fuse_qkv", True))

    @property
    def packing_strategy(self) -> str:
        return str(self.raw.get("packing_strategy", "row_packed"))

    @property
    def token_block_size(self) -> str:
        return str(self.raw.get("token_block_size", "auto"))

    @property
    def profile_native_stages(self) -> bool:
        return bool(self.raw.get("profile_native_stages", True))

    @property
    def fail_on_unavailable_backend(self) -> bool:
        return bool(self.raw.get("fail_on_unavailable_backend", True))

    @property
    def continue_on_case_failure(self) -> bool:
        return bool(self.raw.get("continue_on_case_failure", True))


def load_deployment_config(path: str | Path) -> DeploymentConfig:
    root = project_root()
    config_path = resolve_path(path, root)
    raw = load_yaml(config_path)
    base = config_path.parent

    if "experiment_config" in raw:
        experiment_config_path = resolve_path(raw["experiment_config"], base)
        deployment_raw = raw
    else:
        experiment_config_path = config_path
        deployment_raw = {
            "deployment_id": f"{raw.get('experiment_id', 'experiment')}_he_deployment",
            "cases": ["high", "pre_distill", "post_distill"],
        }

    backend_config_path: Path | None = None
    backend_config: dict[str, Any] = {}
    backend_ref = deployment_raw.get("he_backend_config")
    if backend_ref is not None:
        backend_config_path = resolve_path(backend_ref, base)
        backend_config = load_yaml(backend_config_path)

    return DeploymentConfig(
        root=root,
        config_path=config_path,
        raw=deployment_raw,
        experiment_config_path=experiment_config_path,
        experiment=load_experiment_config(experiment_config_path),
        backend_config_path=backend_config_path,
        backend_config=backend_config,
    )
