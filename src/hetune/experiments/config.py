from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hetune.core.serialization import load_yaml
from hetune.utils.paths import project_root, resolve_path


@dataclass(slots=True)
class LoadedExperimentConfig:
    root: Path
    experiment: dict[str, Any]
    model: dict[str, Any]
    dataset: dict[str, Any]
    approximations: dict[str, Any]
    ckks: dict[str, Any]


def load_experiment_config(path: str | Path) -> LoadedExperimentConfig:
    root = project_root()
    experiment_path = resolve_path(path, root)
    experiment = load_yaml(experiment_path)
    base = experiment_path.parent

    def load_ref(key: str) -> dict[str, Any]:
        ref = experiment.get(key)
        if ref is None:
            raise KeyError(f"Experiment config missing {key}")
        return load_yaml(resolve_path(ref, base))

    return LoadedExperimentConfig(
        root=root,
        experiment=experiment,
        model=load_ref("model_config"),
        dataset=load_ref("dataset_config"),
        approximations=load_ref("approximation_config"),
        ckks=load_ref("ckks_config"),
    )
