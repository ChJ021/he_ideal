from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from hetune.core.serialization import load_yaml
from hetune.core.serialization import save_yaml
from hetune.core.types import CostVector, ExperimentPaths, ScheduleEntry, SchedulePlan
from hetune.cost.profiled import (
    PROFILE_COST_COLUMNS,
    ProfileValidationError,
    ProfiledHECostModel,
)
from hetune.experiments.artifacts import (
    write_artifacts_index,
    write_manifest,
)
from hetune.experiments.config import load_experiment_config
from hetune.experiments.runner import ExperimentRunner
from hetune.operators.registry import build_default_registry
from hetune.scheduling.he_planner import analyze_schedule_feasibility, build_bootstrap_plan
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
            profile_required=self._profile_required(),
            profile_min_coverage=self._profile_min_coverage(),
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
            analysis = analyze_schedule_feasibility(
                schedule_name,
                schedule,
                cost_model,
                self.loaded.ckks,
            )
            coverage = cost_model.coverage_for_schedule(schedule)
            breakdown_rows.extend(analysis.breakdown_rows)
            bootstrap_rows.extend(analysis.bootstrap_rows)
            entry_rows = analysis.breakdown_rows
            profile_entries = sum(1 for row in entry_rows if row["cost_source"] == "profile")
            fallback_entries = len(entry_rows) - profile_entries
            metrics_rows.append(
                {
                    "schedule": schedule_name,
                    "entries": len(entry_rows),
                    "profile_entries": profile_entries,
                    "static_fallback_entries": fallback_entries,
                    "profile_coverage_rate": coverage.profile_coverage_rate,
                    "schedule_feasible": analysis.feasible,
                    "unsupported_rows": analysis.unsupported_count,
                    "estimated_bootstrap_count": analysis.estimated_bootstrap_count,
                    "profile_candidates_loaded": cost_model.load_summary.profile_candidates_loaded,
                    "used_candidates_with_profile": coverage.used_candidates_with_profile,
                    "used_candidates_missing_profile": coverage.used_candidates_missing_profile,
                    "used_candidate_ids_missing_profile": ",".join(
                        coverage.used_candidate_ids_missing_profile
                    ),
                    "strict_profile_check_passed": coverage.strict_profile_check_passed,
                    "strict_profile_check_reason": coverage.strict_profile_check_reason or "",
                    **analysis.total_cost.to_dict(),
                    "weighted_cost": analysis.total_cost.weighted(cost_model.weights),
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
        final_row = metrics[metrics["schedule"] == "hetune_generated"]
        if (
            self._profile_required()
            and not final_row.empty
            and not bool(final_row.iloc[0]["strict_profile_check_passed"])
        ):
            reason = str(final_row.iloc[0].get("strict_profile_check_reason", "")) or "unknown"
            missing = str(final_row.iloc[0].get("used_candidate_ids_missing_profile", ""))
            raise ProfileValidationError(
                "Strict HE profile validation failed for hetune_generated: "
                f"{reason}; missing_profile_candidates={missing or 'none'}"
            )
        return metrics_path

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

    def _profile_required(self) -> bool:
        return bool(self.loaded.experiment.get("scheduler", {}).get("profile_required", False))

    def _profile_min_coverage(self) -> float:
        scheduler = self.loaded.experiment.get("scheduler", {})
        if "profile_min_coverage" in scheduler:
            return float(scheduler["profile_min_coverage"])
        return 1.0 if self._profile_required() else 0.0

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
    feasible = metrics[metrics["schedule_feasible"].astype(bool)] if "schedule_feasible" in metrics.columns else pd.DataFrame()
    infeasible = metrics[~metrics["schedule_feasible"].astype(bool)] if "schedule_feasible" in metrics.columns else pd.DataFrame()
    unsupported_total = int(bootstrap["status"].eq("unsupported").sum()) if not bootstrap.empty and "status" in bootstrap.columns else 0
    single_operator_infeasible = sorted(
        set(
            bootstrap.loc[
                bootstrap["reason"] == "single_operator_exceeds_available_levels",
                "candidate_id",
            ].astype(str)
        )
    ) if not bootstrap.empty and "reason" in bootstrap.columns else []
    lines = [
        f"# HE Analysis Report: {experiment_id}",
        "",
        f"- Backend: `{ckks_config.get('backend', 'unknown')}`",
        f"- Backend id: `{ckks_config.get('backend_id', ckks_config.get('backend', 'unknown'))}`",
        f"- CKKS parameter id: `{ckks_config.get('ckks_param_id', 'unknown')}`",
        f"- Profile path: `{cost_model.profile_path or 'none'}`",
        f"- Profile candidates loaded: `{cost_model.load_summary.profile_candidates_loaded}`",
        f"- Strict profile required: `{cost_model.profile_required}`",
        f"- Figure: `{figure_path}`",
        f"- Feasible schedules: `{', '.join(feasible['schedule'].tolist()) if not feasible.empty else 'none'}`",
        f"- Infeasible schedules: `{', '.join(infeasible['schedule'].tolist()) if not infeasible.empty else 'none'}`",
        f"- Unsupported rows: `{unsupported_total}`",
        f"- Single-operator infeasible candidates: `{', '.join(single_operator_infeasible) if single_operator_infeasible else 'none'}`",
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
        "- `strict_profile_check_passed` is the gate for reporting a deployable HE conclusion.",
        "- `schedule_feasible = true` means the schedule fits the configured level budget with modeled bootstrap placement.",
        "- Unsupported rows indicate the schedule exceeds the configured level budget or contains a single operator that cannot fit in the configured levels.",
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
