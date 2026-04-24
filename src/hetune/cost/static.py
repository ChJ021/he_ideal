from __future__ import annotations

from pathlib import Path

import pandas as pd

from hetune.core.ids import OperatorKey
from hetune.core.types import CostVector, SchedulePlan
from hetune.operators.registry import ApproximationRegistry


class StaticCostModel:
    """Static CKKS-style cost proxy based on candidate metadata."""

    def __init__(
        self,
        registry: ApproximationRegistry,
        ckks_param_id: str = "static_ckks_128",
        weights: dict[str, float] | None = None,
    ) -> None:
        self.registry = registry
        self.ckks_param_id = ckks_param_id
        self.weights = weights or {}

    def estimate(
        self,
        operator_key: OperatorKey,
        candidate_id: str,
        ckks_param_id: str | None = None,
    ) -> CostVector:
        provider = self.registry.get(candidate_id)
        cost = provider.he_cost()
        return CostVector(
            latency_ms=cost.latency_ms,
            rotations=cost.rotations,
            ct_ct_mults=cost.ct_ct_mults,
            ct_pt_mults=cost.ct_pt_mults,
            rescale_count=cost.rescale_count,
            relin_count=cost.relin_count,
            depth=cost.depth,
            bootstrap_count=cost.bootstrap_count,
            memory_mb=cost.memory_mb,
        )

    def estimate_schedule(self, schedule: SchedulePlan) -> CostVector:
        total = CostVector()
        for entry in schedule.entries:
            total += self.estimate(entry.operator_key, entry.candidate_id, entry.ckks_param_id)
        return total

    def weighted_cost(self, operator_key: OperatorKey, candidate_id: str) -> float:
        return self.estimate(operator_key, candidate_id).weighted(self.weights)

    def export_candidate_costs(self, output_path: str | Path) -> None:
        rows = []
        for provider in self.registry.all():
            cost = provider.he_cost()
            row = provider.spec.to_dict()
            row.pop("cost_hint", None)
            row.update(cost.to_dict())
            row["weighted_cost"] = cost.weighted(self.weights)
            rows.append(row)
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(target, index=False)
