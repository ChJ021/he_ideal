from __future__ import annotations

from pathlib import Path

import pandas as pd

from hetune.core.types import CostVector, SchedulePlan


def _markdown_table(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return "_No metrics produced._"
    columns = [str(column) for column in metrics.columns]
    rows = [[str(value) for value in row] for row in metrics.to_numpy()]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def write_report(
    output_path: str | Path,
    experiment_id: str,
    schedule: SchedulePlan,
    metrics: pd.DataFrame,
    total_cost: CostVector,
    decision_log_path: str | Path | None = None,
    diagnostics_path: str | Path | None = None,
    calibration_stats_path: str | Path | None = None,
    calibration_coverage: dict[str, int] | None = None,
    distillation_summary_path: str | Path | None = None,
    distillation_report_path: str | Path | None = None,
    distillation_overrides_path: str | Path | None = None,
    operator_scope: str | None = None,
    operator_types: list[str] | None = None,
) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    decision_summary = _decision_summary(decision_log_path)
    diagnostics_summary = _diagnostics_summary(diagnostics_path)
    schedule_summary = _schedule_type_summary(schedule)
    calibration_summary = _calibration_summary(
        calibration_stats_path,
        calibration_coverage,
    )
    distillation_summary = _distillation_summary(
        distillation_summary_path,
        distillation_report_path,
        distillation_overrides_path,
        metrics,
        operator_scope or schedule.metadata.get("operator_scope"),
    )
    base_accuracy = _metric_value(metrics, "base", "accuracy")
    lines = [
        f"# HETune-LLM Report: {experiment_id}",
        "",
        "## Summary",
        "",
        f"- Operator scope: `{operator_scope or schedule.metadata.get('operator_scope', 'unknown')}`",
        f"- Optimized operator types: `{', '.join(operator_types or schedule.metadata.get('operator_types', []))}`",
        f"- Final policy: `{schedule.metadata.get('policy', 'unknown')}`",
        f"- Schedule entries: `{len(schedule.entries)}`",
        f"- Estimated latency: `{total_cost.latency_ms:.3f}`",
        f"- Estimated depth sum: `{total_cost.depth}`",
        f"- Estimated rotations: `{total_cost.rotations}`",
        f"- Base reference accuracy: `{base_accuracy:.6f}`"
        if base_accuracy is not None
        else "- Base reference accuracy: `not_available`",
        "",
        "Schedule entries by type:",
        "",
        schedule_summary,
        "",
        "Calibration stats:",
        "",
        calibration_summary,
        "",
        "Distillation:",
        "",
        distillation_summary,
        "",
        "## Metrics",
        "",
        _markdown_table(metrics),
        "",
        "## Validated Greedy Decisions",
        "",
        decision_summary,
        "",
        "## Combination Diagnostics",
        "",
        diagnostics_summary,
        "",
        "## Next-Stage Improvements",
        "",
        "- Import SEAL/OpenFHE microbenchmark results and calibrate static costs.",
        "- Add BERT-Mini, DistilBERT, MRPC, and MNLI subset configs.",
        "- Add lower-cost Softmax candidates and isolate their accuracy risk.",
        "- Add bootstrapping placement constraints to the greedy scheduler.",
        "- Run ablations that disable sensitivity, cost model, and layer-wise choices.",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _schedule_type_summary(schedule: SchedulePlan) -> str:
    rows: dict[str, int] = {}
    for entry in schedule.entries:
        rows[entry.operator_key.operator_type] = rows.get(entry.operator_key.operator_type, 0) + 1
    if not rows:
        return "_No schedule entries._"
    return _markdown_table(
        pd.DataFrame(
            [
                {"operator_type": operator_type, "entry_count": count}
                for operator_type, count in sorted(rows.items())
            ]
        )
    )


def _metric_value(metrics: pd.DataFrame, schedule: str, column: str) -> float | None:
    if metrics.empty or "schedule" not in metrics.columns or column not in metrics.columns:
        return None
    rows = metrics[metrics["schedule"] == schedule]
    if rows.empty:
        return None
    return float(rows.iloc[0][column])


def _decision_summary(path: str | Path | None) -> str:
    if path is None or not Path(path).exists():
        return "_No validated greedy decision log found._"
    decisions = pd.read_csv(path)
    if decisions.empty or "accepted" not in decisions.columns:
        return "_No downgrade candidates were evaluated._"
    accepted = int(decisions["accepted"].sum())
    rejected = int((~decisions["accepted"].astype(bool)).sum())
    rejected_rows = decisions[~decisions["accepted"].astype(bool)].copy()
    if "combined_accuracy_drop" in rejected_rows.columns:
        rejected_rows["_has_combined_metrics"] = rejected_rows[
            "combined_accuracy_drop"
        ].notna()
        rejected_rows = rejected_rows.sort_values(
            ["_has_combined_metrics", "combined_accuracy_drop"],
            ascending=[False, False],
        )
    rejected_preview = rejected_rows.head(8)
    lines = [
        f"- Accepted downgrades: `{accepted}`",
        f"- Rejected downgrades: `{rejected}`",
    ]
    if not rejected_preview.empty:
        lines.extend(
            [
                "",
                "Top rejected candidates:",
                "",
                _markdown_table(
                    rejected_preview[
                        [
                            "operator_id",
                            "to_candidate_id",
                            "reason",
                            "combined_accuracy_drop",
                            "combined_logit_kl",
                            "combined_label_flip_rate",
                        ]
                    ]
                ),
            ]
        )
    return "\n".join(lines)


def _diagnostics_summary(path: str | Path | None) -> str:
    if path is None or not Path(path).exists():
        return "_No combination diagnostics found._"
    diagnostics = pd.read_csv(path)
    if diagnostics.empty:
        return "_No combination diagnostics were produced._"
    ranked = diagnostics.sort_values("accuracy_drop", ascending=False).head(8)
    return _markdown_table(
        ranked[
            [
                "diagnostic",
                "accuracy",
                "accuracy_drop",
                "logit_kl",
                "label_flip_rate",
                "downgraded_operator_count",
            ]
        ]
    )


def _calibration_summary(
    path: str | Path | None,
    coverage: dict[str, int] | None,
) -> str:
    if path is None:
        return "- Stats file: `not_available`"
    target = Path(path)
    lines = [f"- Stats file: `{target}`"]
    if coverage:
        lines.append(f"- Covered operators: `{coverage.get('covered', 0)}`")
        lines.append(f"- Tracked operators: `{coverage.get('tracked', 0)}`")
        lines.append(f"- Missing operators: `{coverage.get('missing', 0)}`")
    if not target.exists():
        lines.append("- Status: `missing`")
    return "\n".join(lines)


def _distillation_summary(
    summary_path: str | Path | None,
    report_path: str | Path | None,
    overrides_path: str | Path | None,
    metrics: pd.DataFrame,
    operator_scope: str | None,
) -> str:
    if summary_path is None and report_path is None and overrides_path is None:
        return "- Status: `not_configured`"
    summary = Path(summary_path) if summary_path is not None else None
    report = Path(report_path) if report_path is not None else None
    overrides = Path(overrides_path) if overrides_path is not None else None
    distilled_accuracy = _metric_value(metrics, "hetune_generated_distilled", "accuracy")
    generated_accuracy = _metric_value(metrics, "hetune_generated", "accuracy")
    scope_label = (operator_scope or "current_scope").replace("_", "-")
    lines = []
    if summary is not None:
        lines.append(f"- Summary: `{summary}`")
    if report is not None:
        lines.append(f"- Report: `{report}`")
    if overrides is not None:
        lines.append(f"- Overrides: `{overrides}`")
    status = "available" if overrides is not None and overrides.exists() else "not_run"
    lines.append(f"- Status: `{status}`")
    if generated_accuracy is not None and distilled_accuracy is not None:
        delta = distilled_accuracy - generated_accuracy
        lines.extend(
            [
                "",
                f"{scope_label} accuracy before/after distillation:",
                "",
                _markdown_table(
                    pd.DataFrame(
                        [
                            {
                                "variant": f"{scope_label}_before_distill",
                                "schedule": "hetune_generated",
                                "accuracy": f"{generated_accuracy:.6f}",
                            },
                            {
                                "variant": f"{scope_label}_after_distill",
                                "schedule": "hetune_generated_distilled",
                                "accuracy": f"{distilled_accuracy:.6f}",
                            },
                            {
                                "variant": f"{scope_label}_delta",
                                "schedule": "after-before",
                                "accuracy": f"{delta:+.6f}",
                            },
                        ]
                    )
                ),
            ]
        )
    else:
        if generated_accuracy is not None:
            lines.append(f"- Generated accuracy: `{generated_accuracy:.6f}`")
        if distilled_accuracy is not None:
            lines.append(f"- Distilled accuracy: `{distilled_accuracy:.6f}`")
    return "\n".join(lines)
