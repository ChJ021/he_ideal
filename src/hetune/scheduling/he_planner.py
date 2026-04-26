from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hetune.core.types import CostVector, ScheduleEntry, SchedulePlan


@dataclass(slots=True)
class HEFeasibilityResult:
    schedule: SchedulePlan
    breakdown_rows: list[dict[str, Any]]
    bootstrap_rows: list[dict[str, Any]]
    total_cost: CostVector
    feasible: bool
    unsupported_count: int
    estimated_bootstrap_count: int
    first_violation_reason: str | None = None


def available_levels(ckks_config: dict[str, Any]) -> int:
    return int(
        ckks_config.get(
            "available_levels",
            max(len(ckks_config.get("coefficient_modulus_chain", [])) - 2, 0),
        )
    )


def level_cost(cost: CostVector) -> int:
    return max(int(cost.depth), int(cost.rescale_count))


def bootstrap_cost(ckks_config: dict[str, Any]) -> CostVector:
    configured = ckks_config.get("bootstrap_cost")
    if isinstance(configured, dict):
        cost = CostVector.from_dict(configured)
        if cost.bootstrap_count <= 0:
            return CostVector(
                latency_ms=cost.latency_ms,
                rotations=cost.rotations,
                ct_ct_mults=cost.ct_ct_mults,
                ct_pt_mults=cost.ct_pt_mults,
                rescale_count=cost.rescale_count,
                relin_count=cost.relin_count,
                depth=cost.depth,
                bootstrap_count=1,
                memory_mb=cost.memory_mb,
            )
        return cost
    return CostVector(bootstrap_count=1)


def analyze_schedule_feasibility(
    schedule_name: str,
    schedule: SchedulePlan,
    cost_model: Any,
    ckks_config: dict[str, Any],
) -> HEFeasibilityResult:
    rows: list[dict[str, Any]] = []
    total = CostVector()
    initial_breakdown: list[dict[str, Any]] = []
    for order, entry in enumerate(schedule.entries):
        cost, source = _estimate_with_source(cost_model, entry)
        total += cost
        initial_breakdown.append(
            {
                "schedule": schedule_name,
                "entry_order": order,
                "operator_id": entry.operator_key.id,
                "layer_index": entry.operator_key.layer_index,
                "operator_type": entry.operator_key.operator_type,
                "operator_name": entry.operator_key.name,
                "candidate_id": entry.candidate_id,
                "ckks_param_id": entry.ckks_param_id,
                "scale_id": entry.scale_id,
                "level_budget": entry.level_budget,
                "bootstrap_policy": entry.bootstrap_policy,
                "layout_id": entry.layout_id,
                "cost_source": source,
                **cost.to_dict(),
                "level_cost": level_cost(cost),
                "weighted_cost": cost.weighted(getattr(cost_model, "weights", {})),
            }
        )

    bootstrap_rows, annotations, feasible, unsupported_count, estimated_bootstrap_count, first_reason = (
        build_bootstrap_plan_with_annotations(
            schedule_name=schedule_name,
            schedule=schedule,
            breakdown_rows=initial_breakdown,
            ckks_config=ckks_config,
        )
    )
    annotated_entries: list[ScheduleEntry] = []
    for entry, row in zip(schedule.entries, initial_breakdown, strict=True):
        annotation = annotations.get(entry.operator_key.id, {})
        bootstrap_policy = str(annotation.get("bootstrap_policy", entry.bootstrap_policy))
        level_budget = int(annotation.get("level_budget", entry.level_budget))
        layout_id = entry.layout_id
        if layout_id == "plaintext_sim":
            layout_id = "ckks_profiled"
        annotated_entry = ScheduleEntry(
            operator_key=entry.operator_key,
            candidate_id=entry.candidate_id,
            ckks_param_id=entry.ckks_param_id,
            scale_id=entry.scale_id,
            level_budget=level_budget,
            bootstrap_policy=bootstrap_policy,
            layout_id=layout_id,
        )
        annotated_entries.append(annotated_entry)
        row["level_budget"] = level_budget
        row["bootstrap_policy"] = bootstrap_policy
        row["layout_id"] = layout_id
        rows.append(row)

    if estimated_bootstrap_count:
        bootstrap = bootstrap_cost(ckks_config)
        for _ in range(estimated_bootstrap_count):
            total += bootstrap

    metadata = dict(schedule.metadata)
    metadata.update(
        {
            "he_feasible": feasible,
            "estimated_bootstrap_count": estimated_bootstrap_count,
            "unsupported_rows": unsupported_count,
            "he_backend_id": ckks_config.get("backend_id", ckks_config.get("backend", "unknown")),
            "ckks_param_id": ckks_config.get("ckks_param_id", "unknown"),
        }
    )
    if first_reason is not None:
        metadata["he_first_violation_reason"] = first_reason

    return HEFeasibilityResult(
        schedule=SchedulePlan(
            metadata=metadata,
            entries=annotated_entries,
            constraints=dict(schedule.constraints),
        ),
        breakdown_rows=rows,
        bootstrap_rows=bootstrap_rows,
        total_cost=total,
        feasible=feasible,
        unsupported_count=unsupported_count,
        estimated_bootstrap_count=estimated_bootstrap_count,
        first_violation_reason=first_reason,
    )


def build_bootstrap_plan(
    schedule_name: str,
    schedule: SchedulePlan,
    breakdown_rows: list[dict[str, Any]],
    ckks_config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows, _, _, _, _, _ = build_bootstrap_plan_with_annotations(
        schedule_name=schedule_name,
        schedule=schedule,
        breakdown_rows=breakdown_rows,
        ckks_config=ckks_config,
    )
    return rows


def build_bootstrap_plan_with_annotations(
    schedule_name: str,
    schedule: SchedulePlan,
    breakdown_rows: list[dict[str, Any]],
    ckks_config: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, int | str]],
    bool,
    int,
    int,
    str | None,
]:
    budget = available_levels(ckks_config)
    bootstrapping_supported = bool(ckks_config.get("bootstrapping_supported", False))
    if budget <= 0:
        return (
            [
                {
                    "schedule": schedule_name,
                    "segment_index": 0,
                    "required": False,
                    "status": "not_evaluated",
                    "reason": "available_levels_not_configured",
                    "available_levels": budget,
                    "bootstrap_supported": bootstrapping_supported,
                    "bootstrap_before_operator_id": "",
                    "layer_index": "",
                    "operator_type": "",
                    "candidate_id": "",
                    "levels_used_before": 0,
                    "operator_level_cost": 0,
                }
            ],
            {
                entry.operator_key.id: {
                    "level_budget": 0,
                    "bootstrap_policy": "none",
                }
                for entry in schedule.entries
            },
            False,
            1,
            0,
            "available_levels_not_configured",
        )

    rows: list[dict[str, Any]] = []
    annotations: dict[str, dict[str, int | str]] = {}
    segment_index = 0
    levels_used = 0
    feasible = True
    unsupported_count = 0
    estimated_bootstrap_count = 0
    first_violation_reason: str | None = None

    for entry, cost_row in zip(schedule.entries, breakdown_rows, strict=True):
        op_level_cost = int(cost_row["level_cost"])
        bootstrap_policy = "none"

        if op_level_cost > budget:
            row = {
                "schedule": schedule_name,
                "segment_index": segment_index,
                "required": True,
                "status": "unsupported",
                "reason": "single_operator_exceeds_available_levels",
                "available_levels": budget,
                "bootstrap_supported": bootstrapping_supported,
                "bootstrap_before_operator_id": entry.operator_key.id,
                "layer_index": entry.operator_key.layer_index,
                "operator_type": entry.operator_key.operator_type,
                "candidate_id": entry.candidate_id,
                "levels_used_before": levels_used,
                "operator_level_cost": op_level_cost,
            }
            rows.append(row)
            annotations[entry.operator_key.id] = {
                "level_budget": max(budget - levels_used, 0),
                "bootstrap_policy": bootstrap_policy,
            }
            feasible = False
            unsupported_count += 1
            first_violation_reason = first_violation_reason or str(row["reason"])
            segment_index += 1
            levels_used = 0
            continue

        if levels_used and levels_used + op_level_cost > budget:
            status = "supported" if bootstrapping_supported else "unsupported"
            row = {
                "schedule": schedule_name,
                "segment_index": segment_index,
                "required": True,
                "status": status,
                "reason": "level_budget_exceeded",
                "available_levels": budget,
                "bootstrap_supported": bootstrapping_supported,
                "bootstrap_before_operator_id": entry.operator_key.id,
                "layer_index": entry.operator_key.layer_index,
                "operator_type": entry.operator_key.operator_type,
                "candidate_id": entry.candidate_id,
                "levels_used_before": levels_used,
                "operator_level_cost": op_level_cost,
            }
            rows.append(row)
            if bootstrapping_supported:
                estimated_bootstrap_count += 1
                bootstrap_policy = "bootstrap_before"
                segment_index += 1
                levels_used = 0
            else:
                feasible = False
                unsupported_count += 1
                first_violation_reason = first_violation_reason or str(row["reason"])
                segment_index += 1
                levels_used = 0

        annotations[entry.operator_key.id] = {
            "level_budget": max(budget - levels_used, 0),
            "bootstrap_policy": bootstrap_policy,
        }
        levels_used += op_level_cost

    if not rows:
        rows.append(
            {
                "schedule": schedule_name,
                "segment_index": 0,
                "required": False,
                "status": "no_bootstrap_required",
                "reason": "within_level_budget",
                "available_levels": budget,
                "bootstrap_supported": bootstrapping_supported,
                "bootstrap_before_operator_id": "",
                "layer_index": "",
                "operator_type": "",
                "candidate_id": "",
                "levels_used_before": levels_used,
                "operator_level_cost": 0,
            }
        )
    return (
        rows,
        annotations,
        feasible,
        unsupported_count,
        estimated_bootstrap_count,
        first_violation_reason,
    )


def _estimate_with_source(cost_model: Any, entry: ScheduleEntry) -> tuple[CostVector, str]:
    if hasattr(cost_model, "estimate_with_source"):
        return cost_model.estimate_with_source(
            entry.operator_key,
            entry.candidate_id,
            entry.ckks_param_id,
        )
    cost = cost_model.estimate(entry.operator_key, entry.candidate_id, entry.ckks_param_id)
    return cost, "static_fallback"
