from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hetune.core.types import ExperimentPaths, SchedulePlan
from hetune.experiments.data import load_tokenized_dataset
from hetune.experiments.distillation import load_override_payload
from hetune.models.hf_adapter import HFSequenceClassifierAdapter
from hetune.operators.gelu import (
    gelu_calibrated_scale,
    gelu_chebyshev_coefficients,
    gelu_degree_for_candidate,
    gelu_power_coefficients,
)
from hetune.profiling.calibration import load_calibration_stats


@dataclass(slots=True)
class ForwardArtifact:
    manifest_path: Path
    sample_count: int
    sequence_length: int
    hidden_size: int


def export_distilbert_forward_artifact(
    *,
    output_dir: Path,
    loaded_experiment: Any,
    schedule: SchedulePlan,
    schedule_name: str,
    case_name: str,
    sample_size: int,
    sequence_length: int,
    overrides_path: Path | None,
    ckks_config: dict[str, Any],
) -> ForwardArtifact:
    """Export a native-runner manifest for encrypted DistilBERT forward.

    The privacy boundary is client-side embeddings: this exporter materializes
    the embeddings so the native runner can encrypt them locally for the
    reproducible experiment. A production split can replace this file with
    client-produced ciphertexts without changing the deployment orchestration.
    """

    import torch
    from torch.utils.data import DataLoader

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = HFSequenceClassifierAdapter(
        model_id=str(loaded_experiment.model.get("model_id", "model")),
        model_name_or_path=str(loaded_experiment.model["model_name_or_path"]),
        num_labels=int(loaded_experiment.model.get("num_labels", 2)),
        trust_remote_code=bool(loaded_experiment.model.get("trust_remote_code", False)),
        device=str(loaded_experiment.model.get("device", "cpu")),
    ).load()
    if adapter.model is None:
        raise RuntimeError("DistilBERT model failed to load")
    model = adapter.model
    model.eval()

    if overrides_path is not None:
        payload = load_override_payload(overrides_path)
        adapter.apply_parameter_overrides(payload.get("entries", []))

    tokenized = load_tokenized_dataset(
        loaded_experiment.dataset,
        adapter,
        split_key="validation_split",
        sample_size=sample_size,
        max_length=sequence_length,
    )
    loader = DataLoader(tokenized, batch_size=sample_size)
    try:
        batch = next(iter(loader))
    except StopIteration as exc:
        raise ValueError("Deployment validation split is empty") from exc

    labels = batch["labels"].detach().cpu().numpy().astype(np.int32)
    attention_mask = batch["attention_mask"].detach().cpu().numpy().astype(np.int32)
    input_ids = batch["input_ids"].to(torch.device(adapter.device))
    with torch.no_grad():
        embeddings = model.distilbert.embeddings(input_ids).detach().cpu().numpy().astype(np.float32)

    blobs: list[dict[str, Any]] = []

    def write_blob(name: str, array: np.ndarray, dtype: str) -> None:
        safe = name.replace(".", "_").replace("/", "_")
        suffix = "i32" if dtype == "int32" else "f32"
        path = output_dir / f"{safe}.{suffix}"
        contiguous = np.ascontiguousarray(array.astype(np.int32 if dtype == "int32" else np.float32))
        contiguous.tofile(path)
        blobs.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": list(contiguous.shape),
                "path": path.name,
            }
        )

    write_blob("inputs.embeddings", embeddings, "float32")
    write_blob("inputs.attention_mask", attention_mask, "int32")
    write_blob("inputs.labels", labels, "int32")

    distilbert = model.distilbert
    layers = distilbert.transformer.layer
    for layer_index, layer in enumerate(layers):
        prefix = f"layer.{layer_index}"
        for module_name in ["q_lin", "k_lin", "v_lin", "out_lin"]:
            module = getattr(layer.attention, module_name)
            write_blob(f"{prefix}.attention.{module_name}.weight", _tensor(module.weight), "float32")
            write_blob(f"{prefix}.attention.{module_name}.bias", _tensor(module.bias), "float32")
        for module_name, module in [
            ("sa_layer_norm", layer.sa_layer_norm),
            ("ffn.lin1", layer.ffn.lin1),
            ("ffn.lin2", layer.ffn.lin2),
            ("output_layer_norm", layer.output_layer_norm),
        ]:
            write_blob(f"{prefix}.{module_name}.weight", _tensor(module.weight), "float32")
            write_blob(f"{prefix}.{module_name}.bias", _tensor(module.bias), "float32")

    write_blob("pre_classifier.weight", _tensor(model.pre_classifier.weight), "float32")
    write_blob("pre_classifier.bias", _tensor(model.pre_classifier.bias), "float32")
    write_blob("classifier.weight", _tensor(model.classifier.weight), "float32")
    write_blob("classifier.bias", _tensor(model.classifier.bias), "float32")

    config = model.config
    calibration_stats = _load_forward_calibration_stats(loaded_experiment)
    gelu_approximations = _gelu_approximations(schedule, calibration_stats)
    manifest = {
        "format_version": 1,
        "runner_mode": "openfhe_distilbert_forward",
        "case_name": case_name,
        "schedule_name": schedule_name,
        "model_type": "distilbert",
        "sample_count": int(embeddings.shape[0]),
        "sequence_length": int(embeddings.shape[1]),
        "hidden_size": int(getattr(config, "dim", embeddings.shape[2])),
        "intermediate_size": int(getattr(config, "hidden_dim", 0)),
        "num_layers": int(getattr(config, "n_layers", len(layers))),
        "num_heads": int(getattr(config, "n_heads", 1)),
        "num_labels": int(getattr(config, "num_labels", 2)),
        "activation": str(getattr(config, "activation", "gelu")),
        "privacy_boundary": "client_embedding",
        "accuracy_source": "native_decrypted_logits",
        "ckks": _public_ckks_config(ckks_config),
        "gelu_approximations": gelu_approximations,
        "schedule_entries": [
            {
                "layer_index": entry.operator_key.layer_index,
                "operator_type": entry.operator_key.operator_type,
                "operator_name": entry.operator_key.name,
                "operator_id": entry.operator_key.id,
                "candidate_id": entry.candidate_id,
                "bootstrap_policy": entry.bootstrap_policy,
            }
            for entry in schedule.entries
        ],
        "blobs": blobs,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return ForwardArtifact(
        manifest_path=manifest_path,
        sample_count=int(embeddings.shape[0]),
        sequence_length=int(embeddings.shape[1]),
        hidden_size=int(embeddings.shape[2]),
    )


def _tensor(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float32)


def _load_forward_calibration_stats(loaded_experiment: Any) -> dict[str, dict[str, Any]]:
    experiment_id = loaded_experiment.experiment["experiment_id"]
    output_root = loaded_experiment.root / loaded_experiment.experiment.get("output_root", "outputs")
    stats_path = ExperimentPaths(experiment_id, root=output_root).profile_dir() / "operator_calibration_stats.csv"
    return load_calibration_stats(stats_path)


def _gelu_approximations(
    schedule: SchedulePlan,
    calibration_stats: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in schedule.entries:
        if entry.operator_key.operator_type != "gelu":
            continue
        rows.append(
            _gelu_approximation_row(
                layer_index=entry.operator_key.layer_index,
                operator_name=entry.operator_key.name,
                operator_id=entry.operator_key.id,
                candidate_id=entry.candidate_id,
                stats=calibration_stats.get(entry.operator_key.id, {}),
            )
        )
    rows.append(
        _gelu_approximation_row(
            layer_index=-1,
            operator_name="classifier_activation",
            operator_id="classifier_activation",
            candidate_id="gelu.exact.high.v1",
            stats={},
        )
    )
    return rows


def _gelu_approximation_row(
    *,
    layer_index: int,
    operator_name: str,
    operator_id: str,
    candidate_id: str,
    stats: dict[str, Any],
) -> dict[str, Any]:
    degree = gelu_degree_for_candidate(candidate_id)
    scale = gelu_calibrated_scale(stats)
    return {
        "layer_index": layer_index,
        "operator_name": operator_name,
        "operator_id": operator_id,
        "candidate_id": candidate_id,
        "approximation_family": "calibrated_chebyshev_gelu",
        "degree": degree,
        "scale": scale,
        "tail_policy": "negative_zero_positive_identity_plaintext_only",
        "chebyshev_coefficients": list(gelu_chebyshev_coefficients(scale=scale, degree=degree)),
        "power_coefficients": list(gelu_power_coefficients(scale=scale, degree=degree)),
    }


def _public_ckks_config(ckks_config: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "ckks_param_id",
        "backend",
        "backend_id",
        "security_bits",
        "poly_modulus_degree",
        "scaling_mod_size",
        "first_mod_size",
        "multiplicative_depth",
        "bootstrapping_supported",
        "default_scale_bits",
        "bootstrap_enabled",
        "bootstrap_level_budget",
        "bootstrap_dim1",
        "bootstrap_levels_after",
        "bootstrap_num_iterations",
        "bootstrap_precision",
        "bootstrap_auto_guard",
        "bootstrap_guard_min_levels",
    ]
    return {key: ckks_config[key] for key in keys if key in ckks_config}
