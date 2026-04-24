from __future__ import annotations

from pathlib import Path
from typing import Any

from hetune.core.serialization import save_yaml
from hetune.core.types import ExperimentPaths


def write_config_snapshots(paths: ExperimentPaths, loaded) -> None:
    snapshots = {
        "experiment.yaml": loaded.experiment,
        "model.yaml": loaded.model,
        "dataset.yaml": loaded.dataset,
        "approximations.yaml": loaded.approximations,
        "ckks.yaml": loaded.ckks,
    }
    for filename, data in snapshots.items():
        save_yaml(data, paths.config_dir() / filename)


def write_manifest(
    paths: ExperimentPaths,
    experiment_id: str,
    operator_scope: str,
    operator_types: list[str],
    command: str,
    artifacts: dict[str, str | Path],
) -> None:
    save_yaml(
        {
            "experiment_id": experiment_id,
            "operator_scope": operator_scope,
            "operator_types": operator_types,
            "command": command,
            "run_dir": str(paths.run_dir()),
            "artifacts": {key: str(value) for key, value in artifacts.items()},
        },
        paths.manifest_path(),
    )


def write_artifacts_index(
    paths: ExperimentPaths,
    experiment_id: str,
    operator_scope: str,
    operator_types: list[str],
) -> Path:
    rows = [
        ("Sensitivity matrix", paths.profile_dir() / "sensitivity_matrix.csv"),
        ("Combination diagnostics", paths.profile_dir() / "combination_diagnostics.csv"),
        ("Base reference schedule", paths.schedule_dir() / "base_reference.yaml"),
        ("Generated schedule", paths.schedule_dir() / "hetune_generated.yaml"),
        ("Additive greedy schedule", paths.schedule_dir() / "hetune_additive_greedy.yaml"),
        ("Validated greedy decisions", paths.schedule_dir() / "validated_greedy_decisions.csv"),
        ("Softmax selection", paths.schedule_dir() / "softmax_selection.csv"),
        ("Metrics", paths.evaluation_dir() / "metrics.csv"),
        ("HE metrics", paths.he_analysis_dir() / "he_metrics.csv"),
        ("HE cost breakdown", paths.he_analysis_dir() / "he_cost_breakdown.csv"),
        ("HE bootstrap plan", paths.he_analysis_dir() / "bootstrap_plan.csv"),
        ("Sensitivity heatmap", paths.figure_dir() / "sensitivity_heatmap.png"),
        ("HE cost breakdown figure", paths.figure_dir() / "he_cost_breakdown.png"),
        ("Report", paths.report_dir() / "report.md"),
        ("HE report", paths.report_dir() / "he_report.md"),
        ("Manifest", paths.manifest_path()),
    ]
    lines = [
        f"# Artifacts: {experiment_id}",
        "",
        f"- Operator scope: `{operator_scope}`",
        f"- Operator types: `{', '.join(operator_types)}`",
        f"- Run directory: `{paths.run_dir()}`",
        "",
        "| Artifact | Path | Exists |",
        "| --- | --- | --- |",
    ]
    for label, path in rows:
        lines.append(f"| {label} | `{path}` | `{path.exists()}` |")
    output = paths.report_dir() / "artifacts_index.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
