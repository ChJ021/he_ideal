from pathlib import Path

import pandas as pd
import pytest

from hetune.core.ids import OperatorKey
from hetune.core.serialization import save_yaml
from hetune.core.types import ScheduleEntry, SchedulePlan
from hetune.cost.profiled import ProfileValidationError, ProfiledHECostModel
from hetune.experiments.he_analysis import HEAnalysisRunner, build_bootstrap_plan
from hetune.operators.registry import build_default_registry


def test_profiled_cost_model_prefers_profile_costs(tmp_path: Path):
    profile = tmp_path / "profile.csv"
    profile.write_text(
        "\n".join(
            [
                "backend_id,ckks_param_id,candidate_id,latency_ms,rotations,ct_ct_mults,ct_pt_mults,rescale_count,relin_count,depth,bootstrap_count,memory_mb",
                "openfhe_cpu,ckks_test,gelu.poly.degree2.v1,9.0,1,2,3,4,5,6,0,7.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    registry = build_default_registry()
    model = ProfiledHECostModel(
        registry,
        profile_path=profile,
        ckks_param_id="ckks_test",
        backend_id="openfhe_cpu",
    )
    operator = OperatorKey("mock", 0, "gelu", "ffn_activation", "layer.0.act")

    profiled, source = model.estimate_with_source(operator, "gelu.poly.degree2.v1")
    fallback, fallback_source = model.estimate_with_source(
        operator,
        "gelu.poly.degree3.v1",
    )

    assert profiled.latency_ms == 9.0
    assert profiled.depth == 6
    assert source == "profile"
    assert fallback.latency_ms > 0
    assert fallback_source == "static_fallback"
    assert model.load_summary.profile_candidates_loaded == 1


def test_profiled_cost_model_required_mismatch_raises_clear_error(tmp_path: Path):
    profile = tmp_path / "profile.csv"
    profile.write_text(
        "\n".join(
            [
                "backend_id,ckks_param_id,candidate_id,latency_ms,rotations,ct_ct_mults,ct_pt_mults,rescale_count,relin_count,depth,bootstrap_count,memory_mb",
                "openfhe_cpu,ckks_other,gelu.poly.degree2.v1,9.0,1,2,3,4,5,6,0,7.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ProfileValidationError, match="0 matched"):
        ProfiledHECostModel(
            build_default_registry(),
            profile_path=profile,
            ckks_param_id="ckks_test",
            backend_id="openfhe_cpu",
            profile_required=True,
            profile_min_coverage=1.0,
        )


def test_profiled_cost_model_strict_coverage_flags_missing_candidates(tmp_path: Path):
    profile = tmp_path / "profile.csv"
    profile.write_text(
        "\n".join(
            [
                "backend_id,ckks_param_id,candidate_id,latency_ms,rotations,ct_ct_mults,ct_pt_mults,rescale_count,relin_count,depth,bootstrap_count,memory_mb",
                "openfhe_cpu,ckks_test,gelu.poly.degree2.v1,1.0,0,1,2,1,0,1,0,8.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    registry = build_default_registry()
    model = ProfiledHECostModel(
        registry,
        profile_path=profile,
        ckks_param_id="ckks_test",
        backend_id="openfhe_cpu",
        profile_required=True,
        profile_min_coverage=1.0,
    )
    operator = OperatorKey("mock", 0, "gelu", "ffn_activation", "layer.0.act")
    schedule = SchedulePlan(
        metadata={"policy": "test"},
        entries=[ScheduleEntry(operator, "gelu.poly.degree3.v1", ckks_param_id="ckks_test")],
    )

    coverage = model.coverage_for_schedule(schedule)

    assert coverage.used_candidates_with_profile == 0
    assert coverage.used_candidates_missing_profile == 1
    assert not coverage.strict_profile_check_passed
    assert coverage.strict_profile_check_reason == "missing_profile_for_used_candidates"
    with pytest.raises(ProfileValidationError, match="missing_profile_candidates=gelu.poly.degree3.v1"):
        model.require_schedule_coverage(schedule, "test")


def test_bootstrap_plan_flags_level_overflow():
    operators = [
        OperatorKey("mock", index, "gelu", "ffn_activation", f"layer.{index}.act")
        for index in range(3)
    ]
    schedule = SchedulePlan(
        metadata={},
        entries=[
            ScheduleEntry(operator, "gelu.poly.degree3.v1")
            for operator in operators
        ],
    )
    breakdown_rows = [
        {"level_cost": 2},
        {"level_cost": 2},
        {"level_cost": 2},
    ]

    plan = build_bootstrap_plan(
        "mock_schedule",
        schedule,
        breakdown_rows,
        {"available_levels": 3, "bootstrapping_supported": False},
    )

    required = [row for row in plan if row["required"]]
    assert len(required) == 2
    assert required[0]["bootstrap_before_operator_id"] == operators[1].id
    assert required[0]["status"] == "unsupported"


def test_he_analysis_runner_writes_outputs_without_loading_model(tmp_path: Path):
    output_root = tmp_path / "outputs"
    profile = tmp_path / "profile.csv"
    profile.write_text(
        "\n".join(
            [
                "backend_id,ckks_param_id,candidate_id,latency_ms,rotations,ct_ct_mults,ct_pt_mults,rescale_count,relin_count,depth,bootstrap_count,memory_mb",
                "openfhe_cpu,ckks_test,gelu.poly.degree2.v1,1.0,0,1,2,1,0,1,0,8.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    save_yaml({"model_id": "mock", "model_name_or_path": "unused"}, tmp_path / "model.yaml")
    save_yaml(
        {
            "dataset_id": "mock",
            "dataset_name": "unused",
            "calibration_split": "unused",
            "validation_split": "unused",
            "text_fields": ["sentence"],
        },
        tmp_path / "dataset.yaml",
    )
    save_yaml(
        {"candidates": [{"candidate_id": "gelu.poly.degree2.v1", "enabled": True}]},
        tmp_path / "approximations.yaml",
    )
    save_yaml(
        {
            "ckks_param_id": "ckks_test",
            "backend": "openfhe-seal-profiled",
            "backend_id": "openfhe_cpu",
            "security_bits": 128,
            "coefficient_modulus_chain": [60, 40, 40, 60],
            "available_levels": 2,
            "bootstrapping_supported": False,
            "backend_profile_path": str(profile),
        },
        tmp_path / "ckks.yaml",
    )
    save_yaml(
        {
            "experiment_id": "mock_exp",
            "operator_scope": "activation_norm",
            "operator_types": ["gelu"],
            "model_config": "model.yaml",
            "dataset_config": "dataset.yaml",
            "approximation_config": "approximations.yaml",
            "ckks_config": "ckks.yaml",
            "output_root": str(output_root),
            "scheduler": {
                "he_aware": True,
                "profile_required": True,
                "profile_min_coverage": 1.0,
            },
        },
        tmp_path / "experiment.yaml",
    )

    schedule_dir = output_root / "runs" / "mock_exp" / "schedules"
    schedule_dir.mkdir(parents=True)
    operator = OperatorKey("mock", 0, "gelu", "ffn_activation", "layer.0.act")
    schedule = SchedulePlan(
        metadata={"experiment_id": "mock_exp"},
        entries=[ScheduleEntry(operator, "gelu.poly.degree2.v1")],
    )
    save_yaml(schedule.to_dict(), schedule_dir / "hetune_generated.yaml")

    metrics_path = HEAnalysisRunner(tmp_path / "experiment.yaml").run()
    metrics = pd.read_csv(metrics_path)
    generated = metrics.loc[metrics["schedule"] == "hetune_generated"].iloc[0]

    run_dir = output_root / "runs" / "mock_exp"
    assert metrics_path == run_dir / "he_analysis" / "he_metrics.csv"
    assert (run_dir / "he_analysis" / "he_cost_breakdown.csv").exists()
    assert (run_dir / "he_analysis" / "profile_coverage.csv").exists()
    assert (run_dir / "he_analysis" / "bootstrap_plan.csv").exists()
    assert (run_dir / "reports" / "he_report.md").exists()
    assert bool(generated["strict_profile_check_passed"])
    assert int(generated["profile_candidates_loaded"]) == 1
    assert float(generated["profile_coverage_rate"]) == 1.0


def test_he_analysis_runner_raises_on_strict_profile_failure(tmp_path: Path):
    output_root = tmp_path / "outputs"
    profile = tmp_path / "profile.csv"
    profile.write_text(
        "\n".join(
            [
                "backend_id,ckks_param_id,candidate_id,latency_ms,rotations,ct_ct_mults,ct_pt_mults,rescale_count,relin_count,depth,bootstrap_count,memory_mb",
                "openfhe_cpu,ckks_test,gelu.poly.degree2.v1,1.0,0,1,2,1,0,1,0,8.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    save_yaml({"model_id": "mock", "model_name_or_path": "unused"}, tmp_path / "model.yaml")
    save_yaml(
        {
            "dataset_id": "mock",
            "dataset_name": "unused",
            "calibration_split": "unused",
            "validation_split": "unused",
            "text_fields": ["sentence"],
        },
        tmp_path / "dataset.yaml",
    )
    save_yaml(
        {
            "candidates": [
                {"candidate_id": "gelu.poly.degree2.v1", "enabled": True},
                {"candidate_id": "gelu.poly.degree3.v1", "enabled": True},
            ]
        },
        tmp_path / "approximations.yaml",
    )
    save_yaml(
        {
            "ckks_param_id": "ckks_test",
            "backend": "openfhe-seal-profiled",
            "backend_id": "openfhe_cpu",
            "security_bits": 128,
            "coefficient_modulus_chain": [60, 40, 40, 60],
            "available_levels": 2,
            "bootstrapping_supported": False,
            "backend_profile_path": str(profile),
        },
        tmp_path / "ckks.yaml",
    )
    save_yaml(
        {
            "experiment_id": "mock_exp_fail",
            "operator_scope": "activation_norm",
            "operator_types": ["gelu"],
            "model_config": "model.yaml",
            "dataset_config": "dataset.yaml",
            "approximation_config": "approximations.yaml",
            "ckks_config": "ckks.yaml",
            "output_root": str(output_root),
            "scheduler": {
                "he_aware": True,
                "profile_required": True,
                "profile_min_coverage": 1.0,
            },
        },
        tmp_path / "experiment.yaml",
    )

    schedule_dir = output_root / "runs" / "mock_exp_fail" / "schedules"
    schedule_dir.mkdir(parents=True)
    operator = OperatorKey("mock", 0, "gelu", "ffn_activation", "layer.0.act")
    schedule = SchedulePlan(
        metadata={"experiment_id": "mock_exp_fail"},
        entries=[ScheduleEntry(operator, "gelu.poly.degree3.v1", ckks_param_id="ckks_test")],
    )
    save_yaml(schedule.to_dict(), schedule_dir / "hetune_generated.yaml")

    with pytest.raises(ProfileValidationError, match="missing_profile_candidates=gelu.poly.degree3.v1"):
        HEAnalysisRunner(tmp_path / "experiment.yaml").run()
