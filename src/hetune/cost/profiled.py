from __future__ import annotations

import json
from dataclasses import dataclass
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


class ProfileValidationError(ValueError):
    """Raised when strict profiled HE cost validation fails."""


@dataclass(slots=True)
class ProfileLoadSummary:
    profile_path: Path | None
    requested_ckks_param_id: str
    requested_backend_id: str | None
    profile_required: bool
    raw_rows: int = 0
    ckks_param_rows: int = 0
    backend_rows: int = 0
    profile_candidates_loaded: int = 0
    profile_exists: bool = False
    load_error: str | None = None


@dataclass(slots=True)
class ScheduleProfileCoverage:
    total_entries: int
    non_base_entries: int
    profile_entries: int
    static_fallback_entries: int
    profile_coverage_rate: float
    used_candidates_with_profile: int
    used_candidates_missing_profile: int
    used_candidate_ids_with_profile: tuple[str, ...]
    used_candidate_ids_missing_profile: tuple[str, ...]
    strict_profile_check_passed: bool
    strict_profile_check_reason: str | None = None


class ProfiledHECostModel:
    """HE cost model calibrated by imported OpenFHE/SEAL microbenchmarks."""

    def __init__(
        self,
        registry: ApproximationRegistry,
        profile_path: str | Path | None = None,
        ckks_param_id: str = "static_ckks_128",
        backend_id: str | None = None,
        weights: dict[str, float] | None = None,
        profile_required: bool = False,
        profile_min_coverage: float = 0.0,
    ) -> None:
        self.registry = registry
        self.ckks_param_id = ckks_param_id
        self.backend_id = backend_id
        self.weights = weights or {}
        self.profile_required = profile_required
        self.profile_min_coverage = float(profile_min_coverage)
        self.static = StaticCostModel(registry, ckks_param_id, self.weights)
        self.profile_path = Path(profile_path) if profile_path else None
        self.profile_costs, self.load_summary = self._load_profile_costs(self.profile_path)

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

    def coverage_for_schedule(self, schedule: SchedulePlan) -> ScheduleProfileCoverage:
        total_entries = len(schedule.entries)
        non_base_entries = [
            entry for entry in schedule.entries if not entry.candidate_id.endswith(".base")
        ]
        profile_entries = sum(
            1 for entry in non_base_entries if entry.candidate_id in self.profile_costs
        )
        static_fallback_entries = len(non_base_entries) - profile_entries
        used_candidate_ids = sorted({entry.candidate_id for entry in non_base_entries})
        used_with_profile = tuple(
            candidate_id for candidate_id in used_candidate_ids if candidate_id in self.profile_costs
        )
        used_missing_profile = tuple(
            candidate_id for candidate_id in used_candidate_ids if candidate_id not in self.profile_costs
        )
        profile_coverage_rate = (
            profile_entries / len(non_base_entries) if non_base_entries else 1.0
        )
        strict_passed = True
        strict_reason: str | None = None
        if self.profile_required and non_base_entries:
            if self.load_summary.load_error is not None:
                strict_passed = False
                strict_reason = self.load_summary.load_error
            elif used_missing_profile:
                strict_passed = False
                strict_reason = "missing_profile_for_used_candidates"
            elif profile_coverage_rate < self.profile_min_coverage:
                strict_passed = False
                strict_reason = "profile_coverage_below_minimum"
        return ScheduleProfileCoverage(
            total_entries=total_entries,
            non_base_entries=len(non_base_entries),
            profile_entries=profile_entries,
            static_fallback_entries=static_fallback_entries,
            profile_coverage_rate=profile_coverage_rate,
            used_candidates_with_profile=len(used_with_profile),
            used_candidates_missing_profile=len(used_missing_profile),
            used_candidate_ids_with_profile=used_with_profile,
            used_candidate_ids_missing_profile=used_missing_profile,
            strict_profile_check_passed=strict_passed,
            strict_profile_check_reason=strict_reason,
        )

    def require_schedule_coverage(self, schedule: SchedulePlan, schedule_name: str) -> None:
        coverage = self.coverage_for_schedule(schedule)
        if coverage.strict_profile_check_passed:
            return
        missing = ", ".join(coverage.used_candidate_ids_missing_profile) or "none"
        raise ProfileValidationError(
            "Strict HE profile validation failed for "
            f"{schedule_name}: {coverage.strict_profile_check_reason}; "
            f"missing_profile_candidates={missing}"
        )

    def export_candidate_costs(self, output_path: str | Path) -> None:
        rows = []
        for provider in self.registry.all():
            cost, source = self.estimate_with_source(
                OperatorKey("profile", -1, provider.operator_type, "candidate", "candidate"),
                provider.candidate_id,
                self.ckks_param_id,
            )
            row = provider.spec.to_dict()
            row.pop("cost_hint", None)
            row.update(cost.to_dict())
            row["weighted_cost"] = cost.weighted(self.weights)
            row["cost_source"] = source
            rows.append(row)
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(target, index=False)

    def _load_profile_costs(
        self,
        profile_path: Path | None,
    ) -> tuple[dict[str, CostVector], ProfileLoadSummary]:
        summary = ProfileLoadSummary(
            profile_path=profile_path,
            requested_ckks_param_id=self.ckks_param_id,
            requested_backend_id=self.backend_id,
            profile_required=self.profile_required,
        )
        if profile_path is None:
            if self.profile_required:
                summary.load_error = "profile_required_but_profile_path_missing"
                raise ProfileValidationError(
                    "Strict HE profile validation failed: no backend_profile_path is configured."
                )
            return {}, summary
        summary.profile_exists = profile_path.exists()
        if not summary.profile_exists:
            if self.profile_required:
                summary.load_error = "profile_required_but_profile_file_missing"
                raise ProfileValidationError(
                    "Strict HE profile validation failed: "
                    f"profile file does not exist: {profile_path}"
                )
            return {}, summary
        rows = _read_profile_rows(profile_path)
        summary.raw_rows = len(rows)
        rows, stats = _filter_profile_rows(rows, self.ckks_param_id, self.backend_id)
        summary.ckks_param_rows = stats["ckks_param_rows"]
        summary.backend_rows = stats["backend_rows"]
        costs: dict[str, CostVector] = {}
        for row in rows:
            candidate_id = str(row.get("candidate_id", ""))
            if not candidate_id:
                continue
            costs[candidate_id] = CostVector.from_dict(
                {column: row.get(column, 0) for column in PROFILE_COST_COLUMNS}
            )
        summary.profile_candidates_loaded = len(costs)
        if self.profile_required and summary.profile_candidates_loaded == 0:
            summary.load_error = "profile_rows_do_not_match_backend_and_ckks_config"
            raise ProfileValidationError(
                "Strict HE profile validation failed: "
                f"{profile_path} contains {summary.raw_rows} rows, but 0 matched "
                f"ckks_param_id={self.ckks_param_id!r} and backend_id={self.backend_id!r}."
            )
        return costs, summary


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
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    filtered = rows
    ckks_param_rows = len(filtered)
    if any("ckks_param_id" in row for row in filtered):
        matched = [
            row for row in filtered if str(row.get("ckks_param_id", "")) == ckks_param_id
        ]
        ckks_param_rows = len(matched)
        filtered = matched
    backend_rows = len(filtered)
    if backend_id and any("backend_id" in row for row in filtered):
        matched = [row for row in filtered if str(row.get("backend_id", "")) == backend_id]
        backend_rows = len(matched)
        filtered = matched
    return filtered, {
        "ckks_param_rows": ckks_param_rows,
        "backend_rows": backend_rows,
    }
