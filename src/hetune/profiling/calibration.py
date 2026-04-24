from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from hetune.core.ids import OperatorKey
from hetune.models.hf_adapter import HFSequenceClassifierAdapter, _get_attr


CALIBRATION_COLUMNS = [
    "operator_id",
    "operator_type",
    "layer_index",
    "operator_name",
    "operator_path",
    "sample_count",
    "mean",
    "std",
    "min",
    "max",
    "abs_p95",
    "abs_p99",
    "rms_mean",
    "var_min",
    "var_p01",
    "var_p05",
    "var_mean",
    "var_p95",
    "var_p99",
]


def collect_operator_calibration_stats(
    adapter: HFSequenceClassifierAdapter,
    dataset: Any,
    output_path: str | Path,
    batch_size: int = 16,
    max_batches: int | None = None,
) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader

    if adapter.model is None:
        raise RuntimeError("Model must be loaded before calibration")

    captured: list[dict[str, Any]] = []
    handles = []
    for operator in adapter.operators:
        if operator.operator_type not in {"gelu", "layernorm"}:
            continue
        handle = _register_operator_hook(adapter, operator, captured)
        if handle is not None:
            handles.append(handle)

    loader = DataLoader(dataset, batch_size=batch_size)
    device = torch.device(adapter.device)
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                batch.pop("labels", None)
                batch = {key: value.to(device) for key, value in batch.items()}
                adapter.model(**batch)
    finally:
        for handle in handles:
            handle.remove()

    stats = aggregate_calibration_rows(captured)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(target, index=False)
    adapter.set_calibration_stats(load_calibration_stats(target))
    return stats


def aggregate_calibration_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=CALIBRATION_COLUMNS)
    aggregated = []
    for operator_id, group in pd.DataFrame(rows).groupby("operator_id", sort=False):
        first = group.iloc[0]
        sample_count = float(group["sample_count"].sum())
        mean = _weighted_average(group, "mean", "sample_count")
        mean_square = _weighted_average(group, "_mean_square", "sample_count")
        std = math.sqrt(max(mean_square - mean * mean, 0.0))
        aggregated.append(
            {
                "operator_id": operator_id,
                "operator_type": first["operator_type"],
                "layer_index": int(first["layer_index"]),
                "operator_name": first["operator_name"],
                "operator_path": first["operator_path"],
                "sample_count": int(sample_count),
                "mean": mean,
                "std": std,
                "min": float(group["min"].min()),
                "max": float(group["max"].max()),
                "abs_p95": float(group["abs_p95"].max()),
                "abs_p99": float(group["abs_p99"].max()),
                "rms_mean": math.sqrt(max(mean_square, 0.0)),
                "var_min": float(group["var_min"].min()),
                "var_p01": float(group["var_p01"].min()),
                "var_p05": float(group["var_p05"].min()),
                "var_mean": _weighted_average(group, "var_mean", "_var_count"),
                "var_p95": float(group["var_p95"].max()),
                "var_p99": float(group["var_p99"].max()),
            }
        )
    return pd.DataFrame(aggregated, columns=CALIBRATION_COLUMNS)


def load_calibration_stats(path: str | Path) -> dict[str, dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return {}
    data = pd.read_csv(target)
    if data.empty or "operator_id" not in data.columns:
        return {}
    return {
        str(row["operator_id"]): {
            key: row[key]
            for key in data.columns
            if key != "operator_id" and not pd.isna(row[key])
        }
        for row in data.to_dict(orient="records")
    }


def calibration_coverage(
    operators: list[OperatorKey],
    stats: dict[str, dict[str, Any]],
    operator_types: set[str] | None = None,
) -> dict[str, int]:
    tracked = [
        operator
        for operator in operators
        if operator.operator_type in (operator_types or {"gelu", "layernorm"})
    ]
    covered = [operator for operator in tracked if operator.id in stats]
    return {
        "tracked": len(tracked),
        "covered": len(covered),
        "missing": len(tracked) - len(covered),
    }


def _register_operator_hook(
    adapter: HFSequenceClassifierAdapter,
    operator: OperatorKey,
    rows: list[dict[str, Any]],
):
    import torch

    try:
        target = _get_attr(adapter.model, operator.path)
    except (AttributeError, IndexError, KeyError, TypeError):
        return None

    if isinstance(target, torch.nn.Module):
        return target.register_forward_pre_hook(
            lambda module, inputs, op=operator: _capture_pre_hook(op, module, inputs, rows)
        )

    fallback_path = _fallback_activation_input_path(operator)
    if fallback_path is None:
        return None
    try:
        fallback = _get_attr(adapter.model, fallback_path)
    except (AttributeError, IndexError, KeyError, TypeError):
        return None
    if not isinstance(fallback, torch.nn.Module):
        return None
    return fallback.register_forward_hook(
        lambda module, inputs, output, op=operator: _capture_output_hook(op, module, output, rows)
    )


def _capture_pre_hook(
    operator: OperatorKey,
    module: Any,
    inputs: tuple[Any, ...],
    rows: list[dict[str, Any]],
) -> None:
    if not inputs:
        return
    _append_tensor_stats(rows, operator, inputs[0], module)


def _capture_output_hook(
    operator: OperatorKey,
    module: Any,
    output: Any,
    rows: list[dict[str, Any]],
) -> None:
    _append_tensor_stats(rows, operator, output, module)


def _append_tensor_stats(
    rows: list[dict[str, Any]],
    operator: OperatorKey,
    tensor: Any,
    module: Any,
) -> None:
    import torch

    if not torch.is_tensor(tensor):
        return
    values = tensor.detach().float()
    if values.numel() == 0:
        return
    flat = values.reshape(-1)
    sampled = _sample_flat(flat)
    abs_sampled = sampled.abs()
    row = {
        "operator_id": operator.id,
        "operator_type": operator.operator_type,
        "layer_index": operator.layer_index,
        "operator_name": operator.name,
        "operator_path": operator.path,
        "sample_count": int(flat.numel()),
        "mean": float(flat.mean().item()),
        "_mean_square": float(flat.pow(2).mean().item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "abs_p95": _quantile(abs_sampled, 0.95),
        "abs_p99": _quantile(abs_sampled, 0.99),
        "_var_count": 1,
        "var_min": 0.0,
        "var_p01": 0.0,
        "var_p05": 0.0,
        "var_mean": 0.0,
        "var_p95": 0.0,
        "var_p99": 0.0,
    }
    if operator.operator_type == "layernorm":
        variance = _layernorm_variance(values, module)
        var_flat = variance.reshape(-1)
        var_sampled = _sample_flat(var_flat)
        row.update(
            {
                "_var_count": int(var_flat.numel()),
                "var_min": float(var_flat.min().item()),
                "var_p01": _quantile(var_sampled, 0.01),
                "var_p05": _quantile(var_sampled, 0.05),
                "var_mean": float(var_flat.mean().item()),
                "var_p95": _quantile(var_sampled, 0.95),
                "var_p99": _quantile(var_sampled, 0.99),
            }
        )
    rows.append(row)


def _layernorm_variance(values, module):
    normalized_shape = getattr(module, "normalized_shape", None)
    if normalized_shape is None:
        dims = (-1,)
    else:
        dims = tuple(range(-len(tuple(normalized_shape)), 0))
    centered = values - values.mean(dim=dims, keepdim=True)
    return centered.pow(2).mean(dim=dims)


def _sample_flat(flat, max_values: int = 200_000):
    if flat.numel() <= max_values:
        return flat
    step = max(1, math.ceil(flat.numel() / max_values))
    return flat[::step]


def _quantile(values, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.quantile(q).item())


def _fallback_activation_input_path(operator: OperatorKey) -> str | None:
    if operator.operator_type != "gelu":
        return None
    if operator.path.endswith(".intermediate_act_fn"):
        return operator.path.removesuffix(".intermediate_act_fn") + ".dense"
    if operator.path.endswith(".activation"):
        return operator.path.removesuffix(".activation") + ".lin1"
    return None


def _weighted_average(group: pd.DataFrame, value_column: str, weight_column: str) -> float:
    weights = group[weight_column].astype(float)
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    return float((group[value_column].astype(float) * weights).sum() / total)
