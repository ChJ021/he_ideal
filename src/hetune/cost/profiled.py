from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from hetune.core.ids import OperatorKey
from hetune.core.types import CostVector, SchedulePlan
from hetune.cost.static import StaticCostModel
from hetune.operators.registry import ApproximationRegistry


PROFILE_COST_COLUMNS = (
    "latency_ms",
    "rotations",
    "ct_ct_mults",
    "ct_pt_mults",
    "rescale_count",
    "relin_count",
    "depth",
    "bootstrap_count",
    "memory_mb",
)


class ProfiledHECostModel:
    """HE cost model calibrated by imported OpenFHE/SEAL microbenchmarks."""

    def __init__(
        self,
        registry: ApproximationRegistry,
        profile_path: str | Path | None = None,
        ckks_param_id: str = "static_ckks_128",
        backend_id: str | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.registry = registry
        self.ckks_param_id = ckks_param_id
        self.backend_id = backend_id
        self.weights = weights or {}
        self.static = StaticCostModel(registry, ckks_param_id, self.weights)
        self.profile_path = Path(profile_path) if profile_path else None
        self.profile_costs = self._load_profile_costs(self.profile_path)

    def estimate(
        self,
        operator_key: OperatorKey,
        candidate_id: str,
        ckks_param_id: str | None = None,
    ) -> CostVector:
        cost = self.profile_costs.get(candidate_id)
        if cost is not None:
            return cost
        return self.static.estimate(operator_key, candidate_id, ckks_param_id)

    def estimate_with_source(
        self,
        operator_key: OperatorKey,
        candidate_id: str,
        ckks_param_id: str | None = None,
    ) -> tuple[CostVector, str]:
        cost = self.profile_costs.get(candidate_id)
        if cost is not None:
            return cost, "profile"
        return self.static.estimate(operator_key, candidate_id, ckks_param_id), "static_fallback"

    def estimate_schedule(self, schedule: SchedulePlan) -> CostVector:
        total = CostVector()
        for entry in schedule.entries:
            total += self.estimate(entry.operator_key, entry.candidate_id, entry.ckks_param_id)
        return total

    def weighted_cost(self, operator_key: OperatorKey, candidate_id: str) -> float:
        return self.estimate(operator_key, candidate_id).weighted(self.weights)

    def source_for(self, candidate_id: str) -> str:
        if candidate_id in self.profile_costs:
            return "profile"
        return "static_fallback"

    def _load_profile_costs(self, profile_path: Path | None) -> dict[str, CostVector]:
        if profile_path is None or not profile_path.exists():
            return {}
        rows = _read_profile_rows(profile_path)
        rows = _filter_profile_rows(rows, self.ckks_param_id, self.backend_id)
        costs: dict[str, CostVector] = {}
        for row in rows:
            candidate_id = str(row.get("candidate_id", ""))
            if not candidate_id:
                continue
            costs[candidate_id] = CostVector.from_dict(
                {column: row.get(column, 0) for column in PROFILE_COST_COLUMNS}
            )
        return costs


def _read_profile_rows(profile_path: Path) -> list[dict[str, Any]]:
    suffix = profile_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(profile_path).to_dict(orient="records")
    if suffix == ".json":
        with profile_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            data = data.get("records", data.get("profiles", []))
        if not isinstance(data, list):
            raise ValueError(f"HE profile JSON must contain a list of records: {profile_path}")
        return [dict(item) for item in data]
    raise ValueError(f"Unsupported HE profile format: {profile_path}")


def _filter_profile_rows(
    rows: list[dict[str, Any]],
    ckks_param_id: str,
    backend_id: str | None,
) -> list[dict[str, Any]]:
    filtered = rows
    if any("ckks_param_id" in row for row in filtered):
        matched = [
            row for row in filtered if str(row.get("ckks_param_id", "")) == ckks_param_id
        ]
        filtered = matched
    if backend_id and any("backend_id" in row for row in filtered):
        matched = [row for row in filtered if str(row.get("backend_id", "")) == backend_id]
        filtered = matched
    return filtered
