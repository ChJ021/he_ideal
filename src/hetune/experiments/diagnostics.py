from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from hetune.core.ids import OperatorKey
from hetune.core.types import SchedulePlan
from hetune.execution.evaluator import EvaluationResult
from hetune.operators.registry import ApproximationRegistry
from hetune.profiling.metrics import label_flip_rate, logit_kl
from hetune.scheduling.policies import UniformPolicy


def write_combination_diagnostics(
    output_path: str | Path,
    operators: list[OperatorKey],
    registry: ApproximationRegistry,
    evaluate_schedule,
    metadata: dict[str, Any],
    constraints: dict[str, Any],
    ckks_param_id: str,
    quality: str = "low",
) -> Path:
    high_schedule = UniformPolicy(registry, ckks_param_id, "high").generate(
        operators,
        metadata={**metadata, "policy": "diagnostic_high"},
        constraints=constraints,
    )
    baseline = evaluate_schedule(high_schedule)
    rows = [
        _row("baseline_high", baseline, baseline, 0, []),
    ]
    layers = sorted({operator.layer_index for operator in operators})
    for layer in layers:
        selected = {operator.id for operator in operators if operator.layer_index == layer}
        schedule = _partial_quality_schedule(
            operators,
            registry,
            ckks_param_id,
            quality,
            selected,
            metadata={**metadata, "policy": f"diagnostic_layer_{layer}_{quality}"},
            constraints=constraints,
        )
        result = evaluate_schedule(schedule)
        rows.append(_row(f"layer_{layer}_{quality}", result, baseline, len(selected), sorted(selected)))

    for left, right in zip(layers, layers[1:]):
        selected = {
            operator.id
            for operator in operators
            if operator.layer_index in {left, right}
        }
        schedule = _partial_quality_schedule(
            operators,
            registry,
            ckks_param_id,
            quality,
            selected,
            metadata={
                **metadata,
                "policy": f"diagnostic_layers_{left}_{right}_{quality}",
            },
            constraints=constraints,
        )
        result = evaluate_schedule(schedule)
        rows.append(
            _row(
                f"layers_{left}_{right}_{quality}",
                result,
                baseline,
                len(selected),
                sorted(selected),
            )
        )

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target, index=False)
    return target


def _partial_quality_schedule(
    operators: list[OperatorKey],
    registry: ApproximationRegistry,
    ckks_param_id: str,
    quality: str,
    selected_operator_ids: set[str],
    metadata: dict[str, Any],
    constraints: dict[str, Any],
) -> SchedulePlan:
    high = UniformPolicy(registry, ckks_param_id, "high").generate(
        operators,
        metadata=metadata,
        constraints=constraints,
    )
    replacement = UniformPolicy(registry, ckks_param_id, quality).generate(
        operators,
        metadata=metadata,
        constraints=constraints,
    )
    replacement_by_id = {
        entry.operator_key.id: entry for entry in replacement.entries
    }
    entries = [
        replacement_by_id[entry.operator_key.id]
        if entry.operator_key.id in selected_operator_ids
        else entry
        for entry in high.entries
    ]
    return SchedulePlan(metadata=metadata, entries=entries, constraints=constraints)


def _row(
    name: str,
    result: EvaluationResult,
    baseline: EvaluationResult,
    downgraded_operator_count: int,
    operator_ids: list[str],
) -> dict[str, object]:
    return {
        "diagnostic": name,
        "accuracy": result.accuracy,
        "accuracy_drop": baseline.accuracy - result.accuracy,
        "logit_kl": logit_kl(baseline.logits, result.logits),
        "label_flip_rate": label_flip_rate(baseline.logits, result.logits),
        "downgraded_operator_count": downgraded_operator_count,
        "operator_ids": ";".join(operator_ids),
    }
