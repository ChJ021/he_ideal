from pathlib import Path

from hetune.core.types import ExperimentPaths
from hetune.experiments.runner import ExperimentRunner


def test_experiment_paths_are_grouped_by_run():
    paths = ExperimentPaths("exp", root=Path("outputs"))
    assert paths.run_dir() == Path("outputs/runs/exp")
    assert paths.profile_dir() == Path("outputs/runs/exp/profiles")
    assert paths.schedule_dir() == Path("outputs/runs/exp/schedules")
    assert paths.evaluation_dir() == Path("outputs/runs/exp/evaluations")
    assert paths.distillation_dir() == Path("outputs/runs/exp/distillation")
    assert paths.report_dir() == Path("outputs/runs/exp/reports")


def test_operator_scope_defaults():
    assert ExperimentRunner._operator_types_for_scope("activation_norm", None) == [
        "gelu",
        "layernorm",
    ]
    assert ExperimentRunner._operator_types_for_scope("softmax_only", None) == [
        "softmax"
    ]
    assert ExperimentRunner._operator_types_for_scope("all_nonlinear", None) == [
        "gelu",
        "layernorm",
        "softmax",
    ]


def test_configured_operator_types_override_scope():
    assert ExperimentRunner._operator_types_for_scope("activation_norm", ["softmax"]) == [
        "softmax"
    ]
