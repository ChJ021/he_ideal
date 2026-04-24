from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from hetune.core.serialization import load_yaml
from hetune.core.serialization import save_yaml
from hetune.core.types import CostVector, ExperimentPaths, ScheduleEntry, SchedulePlan
from hetune.cost.profiled import PROFILE_COST_COLUMNS, ProfiledHECostModel
from hetune.experiments.artifacts import (
    write_artifacts_index,
    write_manifest,
)
from hetune.experiments.config import load_experiment_config
from hetune.experiments.runner import ExperimentRunner
from hetune.operators.registry import build_default_registry
from hetune.utils.paths import resolve_path


class HEAnalysisRunner:
    """Analyze generated schedules with profiled HE cost data."""

    def __init__(self, config_path: str | Path, command_name: str = "run-he") -> None:
        self.loaded = load_experiment_config(config_path)
        self.operator_scope = self.loaded.experiment.get("operator_scope", "activation_norm")
        self.operator_types = ExperimentRunner._operator_types_for_scope(
            self.operator_scope,
            self.loaded.experiment.get("operator_types"),
        )
        self.command_name = command_name
        self.experiment_id = self.loaded.experiment["experiment_id"]
        self.paths = ExperimentPaths(
            experiment_id=self.experiment_id,
            root=self.loaded.root / self.loaded.experiment.get("output_root", "outputs"),
        )
        self.paths.ensure()

    def run(self) -> Path:
        registry = build_default_registry(self._enabled_candidates(self.loaded.approximations))
        self._ensure_base_schedule(registry)
        cost_model = ProfiledHECostModel(
            registry=registry,
            profile_path=self._profile_path(),
            ckks_param_id=self.loaded.ckks.get("ckks_param_id", "static_ckks_128"),
            backend_id=self.loaded.ckks.get("backend_id") or self.loaded.ckks.get("backend"),
            weights=self.loaded.experiment.get("cost_weights", {}),
        )
        schedule_paths = self._schedule_paths()
        if not schedule_paths:
            raise FileNotFoundError(
                f"No schedules found under {self.paths.schedule_dir()}; run tuning first."
            )

        metrics_rows: list[dict[str, Any]] = []
        breakdown_rows: list[dict[str, Any]] = []
        bootstrap_rows: list[dict[str, Any]] = []

        for schedule_name, schedule_path in schedule_paths.items():
            schedule = SchedulePlan.from_dict(load_yaml(schedule_path))
            entry_rows, total = self._breakdown_rows(schedule_name, schedule, cost_model)
            breakdown_rows.extend(entry_rows)
            bootstrap_rows.extend(
                build_bootstrap_plan(schedule_name, schedule, entry_rows, self.loaded.ckks)
            )
            profile_entries = sum(1 for row in entry_rows if row["cost_source"] == "profile")
            fallback_entries = len(entry_rows) - profile_entries
            metrics_rows.append(
                {
                    "schedule": schedule_name,
                    "entries": len(entry_rows),
                    "profile_entries": profile_entries,
                    "static_fallback_entries": fallback_entries,
                    "profile_coverage_rate": (
                        profile_entries / len(entry_rows) if entry_rows else 0.0
                    ),
                    **total.to_dict(),
                    "weighted_cost": total.weighted(cost_model.weights),
                }
            )

        metrics = pd.DataFrame(metrics_rows)
        breakdown = pd.DataFrame(breakdown_rows)
        bootstrap = pd.DataFrame(bootstrap_rows)
        coverage = build_profile_coverage(breakdown)

        he_dir = self.paths.he_analysis_dir()
        metrics_path = he_dir / "he_metrics.csv"
        breakdown_path = he_dir / "he_cost_breakdown.csv"
        coverage_path = he_dir / "profile_coverage.csv"
        bootstrap_path = he_dir / "bootstrap_plan.csv"
        metrics.to_csv(metrics_path, index=False)
        breakdown.to_csv(breakdown_path, index=False)
        coverage.to_csv(coverage_path, index=False)
        bootstrap.to_csv(bootstrap_path, index=False)

        figure_path = self.paths.figure_dir() / "he_cost_breakdown.png"
        write_he_cost_figure(metrics, self.loaded.experiment.get("cost_weights", {}), figure_path)
        report_path = self.paths.report_dir() / "he_report.md"
        write_he_report(
            report_path,
            self.experiment_id,
            self.loaded.ckks,
            cost_model,
            metrics,
            coverage,
            bootstrap,
            figure_path,
        )
        write_manifest(
            self.paths,
            self.experiment_id,
            self.operator_scope,
            self.operator_types,
            self.command_name,
            {
                "profile": self.paths.profile_dir() / "sensitivity_matrix.csv",
                "schedule": self.paths.schedule_dir() / "hetune_generated.yaml",
                "decisions": self.paths.schedule_dir() / "validated_greedy_decisions.csv",
                "metrics": self.paths.evaluation_dir() / "metrics.csv",
                "report": self.paths.report_dir() / "report.md",
                "he_metrics": metrics_path,
                "he_breakdown": breakdown_path,
                "he_profile_coverage": coverage_path,
                "bootstrap_plan": bootstrap_path,
                "he_report": report_path,
            },
        )
        write_artifacts_index(
            self.paths,
            self.experiment_id,
            self.operator_scope,
            self.operator_types,
        )
        return metrics_path

    def _breakdown_rows(
        self,
        schedule_name: str,
        schedule: SchedulePlan,
        cost_model: ProfiledHECostModel,
    ) -> tuple[list[dict[str, Any]], CostVector]:
        rows: list[dict[str, Any]] = []
        total = CostVector()
        for order, entry in enumerate(schedule.entries):
            cost, source = cost_model.estimate_with_source(
                entry.operator_key,
                entry.candidate_id,
                entry.ckks_param_id,
            )
            total += cost
            rows.append(
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
                    "level_cost": max(cost.depth, cost.rescale_count),
                    "weighted_cost": cost.weighted(cost_model.weights),
                }
            )
        return rows, total

    def _schedule_paths(self) -> dict[str, Path]:
        candidates = {
            "base": self.paths.schedule_dir() / "base_reference.yaml",
            "hetune_generated": self.paths.schedule_dir() / "hetune_generated.yaml",
            "uniform_low": self.paths.schedule_dir() / "uniform_low.yaml",
            "uniform_mid": self.paths.schedule_dir() / "uniform_mid.yaml",
            "uniform_high": self.paths.schedule_dir() / "uniform_high.yaml",
        }
        return {name: path for name, path in candidates.items() if path.exists()}

    def _ensure_base_schedule(self, registry) -> None:
        base_path = self.paths.schedule_dir() / "base_reference.yaml"
        generated_path = self.paths.schedule_dir() / "hetune_generated.yaml"
        if not generated_path.exists():
            return
        generated = SchedulePlan.from_dict(load_yaml(generated_path))
        entries: list[ScheduleEntry] = []
        for entry in generated.entries:
            candidate_id = f"{entry.operator_key.operator_type}.base"
            try:
                registry.get(candidate_id)
            except KeyError:
                continue
            entries.append(
                ScheduleEntry(
                    operator_key=entry.operator_key,
                    candidate_id=candidate_id,
                    ckks_param_id=entry.ckks_param_id,
                    scale_id=entry.scale_id,
                    level_budget=0,
                    bootstrap_policy="none",
                    layout_id="plaintext_reference",
                )
            )
        if entries:
            metadata = {
                key: value
                for key, value in generated.metadata.items()
                if key
                not in {
                    "policy",
                    "baseline_accuracy",
                    "baseline_policy",
                    "high_accuracy",
                    "high_accuracy_drop_from_base",
                    "accepted_downgrades",
                    "rejected_downgrades",
                }
            }
            base = SchedulePlan(
                metadata={
                    **metadata,
                    "policy": "base_reference",
                    "source_schedule": str(generated_path),
                },
                entries=entries,
                constraints={**generated.constraints, "input_independent": True},
            )
            save_yaml(base.to_dict(), base_path)

    def _profile_path(self) -> Path | None:
        if self.loaded.ckks.get("backend") == "static-only":
            return None
        profile = self.loaded.ckks.get("backend_profile_path")
        if not profile:
            return None
        return resolve_path(profile, self.loaded.root)

    @staticmethod
    def _enabled_candidates(config: dict[str, Any]) -> set[str]:
        candidates = config.get("candidates", [])
        return {item["candidate_id"] for item in candidates if item.get("enabled", True)}


def build_profile_coverage(breakdown: pd.DataFrame) -> pd.DataFrame:
    if breakdown.empty:
        return pd.DataFrame(
            columns=[
                "schedule",
                "operator_type",
                "candidate_id",
                "entries",
                "profile_entries",
                "static_fallback_entries",
                "profile_coverage_rate",
            ]
        )
    grouped = (
        breakdown.groupby(["schedule", "operator_type", "candidate_id"], dropna=False)
        .agg(
            entries=("candidate_id", "size"),
            profile_entries=("cost_source", lambda values: int((values == "profile").sum())),
            static_fallback_entries=(
                "cost_source",
                lambda values: int((values == "static_fallback").sum()),
            ),
        )
        .reset_index()
    )
    grouped["profile_coverage_rate"] = grouped["profile_entries"] / grouped["entries"]
    return grouped


def build_bootstrap_plan(
    schedule_name: str,
    schedule: SchedulePlan,
    breakdown_rows: list[dict[str, Any]],
    ckks_config: dict[str, Any],
) -> list[dict[str, Any]]:
    available_levels = int(
        ckks_config.get(
            "available_levels",
            max(len(ckks_config.get("coefficient_modulus_chain", [])) - 2, 0),
        )
    )
    bootstrapping_supported = bool(ckks_config.get("bootstrapping_supported", False))
    if available_levels <= 0:
        return [
            {
                "schedule": schedule_name,
                "segment_index": 0,
                "required": False,
                "status": "not_evaluated",
                "reason": "available_levels_not_configured",
                "available_levels": available_levels,
                "bootstrap_supported": bootstrapping_supported,
                "bootstrap_before_operator_id": "",
                "layer_index": "",
                "operator_type": "",
                "candidate_id": "",
                "levels_used_before": 0,
                "operator_level_cost": 0,
            }
        ]

    rows: list[dict[str, Any]] = []
    segment_index = 0
    levels_used = 0
    for entry, cost_row in zip(schedule.entries, breakdown_rows, strict=True):
        level_cost = int(cost_row["level_cost"])
        if level_cost > available_levels:
            rows.append(
                {
                    "schedule": schedule_name,
                    "segment_index": segment_index,
                    "required": True,
                    "status": "unsupported",
                    "reason": "single_operator_exceeds_available_levels",
                    "available_levels": available_levels,
                    "bootstrap_supported": bootstrapping_supported,
                    "bootstrap_before_operator_id": entry.operator_key.id,
                    "layer_index": entry.operator_key.layer_index,
                    "operator_type": entry.operator_key.operator_type,
                    "candidate_id": entry.candidate_id,
                    "levels_used_before": levels_used,
                    "operator_level_cost": level_cost,
                }
            )
            segment_index += 1
            levels_used = 0
            continue
        if levels_used and levels_used + level_cost > available_levels:
            rows.append(
                {
                    "schedule": schedule_name,
                    "segment_index": segment_index,
                    "required": True,
                    "status": "supported" if bootstrapping_supported else "unsupported",
                    "reason": "level_budget_exceeded",
                    "available_levels": available_levels,
                    "bootstrap_supported": bootstrapping_supported,
                    "bootstrap_before_operator_id": entry.operator_key.id,
                    "layer_index": entry.operator_key.layer_index,
                    "operator_type": entry.operator_key.operator_type,
                    "candidate_id": entry.candidate_id,
                    "levels_used_before": levels_used,
                    "operator_level_cost": level_cost,
                }
            )
            segment_index += 1
            levels_used = 0
        levels_used += level_cost

    if not rows:
        rows.append(
            {
                "schedule": schedule_name,
                "segment_index": 0,
                "required": False,
                "status": "no_bootstrap_required",
                "reason": "within_level_budget",
                "available_levels": available_levels,
                "bootstrap_supported": bootstrapping_supported,
                "bootstrap_before_operator_id": "",
                "layer_index": "",
                "operator_type": "",
                "candidate_id": "",
                "levels_used_before": levels_used,
                "operator_level_cost": 0,
            }
        )
    return rows


def write_he_cost_figure(
    metrics: pd.DataFrame,
    weights: dict[str, float],
    output_path: str | Path,
) -> None:
    if metrics.empty:
        return
    import matplotlib.pyplot as plt

    weighted = metrics[["schedule"]].copy()
    for column in PROFILE_COST_COLUMNS:
        weighted[column] = metrics[column] * float(weights.get(column, 1.0))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ax = weighted.set_index("schedule").plot(kind="bar", stacked=True, figsize=(10, 5))
    ax.set_ylabel("Weighted HE cost")
    ax.set_xlabel("Schedule")
    ax.set_title("HE cost breakdown by schedule")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def write_he_report(
    output_path: str | Path,
    experiment_id: str,
    ckks_config: dict[str, Any],
    cost_model: ProfiledHECostModel,
    metrics: pd.DataFrame,
    coverage: pd.DataFrame,
    bootstrap: pd.DataFrame,
    figure_path: Path,
) -> None:
    lines = [
        f"# HE Analysis Report: {experiment_id}",
        "",
        f"- Backend: `{ckks_config.get('backend', 'unknown')}`",
        f"- Backend id: `{ckks_config.get('backend_id', ckks_config.get('backend', 'unknown'))}`",
        f"- CKKS parameter id: `{ckks_config.get('ckks_param_id', 'unknown')}`",
        f"- Profile path: `{cost_model.profile_path or 'none'}`",
        f"- Profile candidates loaded: `{len(cost_model.profile_costs)}`",
        f"- Figure: `{figure_path}`",
        "",
        "## Schedule HE Metrics",
        "",
        _markdown_table(metrics),
        "",
        "## Profile Coverage",
        "",
        _markdown_table(coverage),
        "",
        "## Bootstrap / Level Plan",
        "",
        _markdown_table(bootstrap),
        "",
        "## Notes",
        "",
        "- `profile` means the candidate cost came from imported OpenFHE/SEAL microbenchmark data.",
        "- `static_fallback` means the candidate used built-in static CKKS-style metadata.",
        "- `base` is the plaintext reference schedule and is not a deployable HE schedule.",
        "- Unsupported bootstrap rows indicate the schedule exceeds the configured level budget while bootstrapping is disabled.",
    ]
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(data: pd.DataFrame, max_rows: int = 20) -> str:
    if data.empty:
        return "_No rows._"
    visible = data.head(max_rows)
    columns = list(visible.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for row in visible.to_dict(orient="records"):
        rows.append("| " + " | ".join(_format_cell(row[column]) for column in columns) + " |")
    if len(data) > max_rows:
        rows.append(f"| ... | {' | '.join('...' for _ in columns[1:])} |")
    return "\n".join([header, separator, *rows])


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
