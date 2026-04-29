from hetune.core.ids import OperatorKey
from hetune.core.types import ScheduleEntry, SchedulePlan
from hetune.cost.static import StaticCostModel
from hetune.operators.registry import build_default_registry
from hetune.scheduling.he_planner import analyze_schedule_feasibility
from hetune.scheduling.policies import HEUniformPolicy


def test_he_uniform_policy_picks_highest_feasible_candidate():
    registry = build_default_registry()
    cost_model = StaticCostModel(registry, ckks_param_id="ckks_test")
    operators = [OperatorKey("mock", 0, "gelu", "ffn_activation", "layer.0.act")]

    schedule = HEUniformPolicy(
        registry=registry,
        cost_model=cost_model,
        ckks_config={"available_levels": 3, "bootstrapping_supported": True},
        ckks_param_id="ckks_test",
        quality="high",
    ).generate(operators)

    assert schedule.entries[0].candidate_id == "gelu.chebyshev.degree5.v1"


def test_analyze_schedule_feasibility_inserts_bootstrap_and_counts_cost():
    registry = build_default_registry()
    cost_model = StaticCostModel(registry, ckks_param_id="ckks_test")
    operators = [
        OperatorKey("mock", 0, "gelu", "ffn_activation", "layer.0.act"),
        OperatorKey("mock", 1, "gelu", "ffn_activation", "layer.1.act"),
    ]
    schedule = SchedulePlan(
        metadata={"policy": "test"},
        entries=[
            ScheduleEntry(operators[0], "gelu.chebyshev.degree5.v1", ckks_param_id="ckks_test"),
            ScheduleEntry(operators[1], "gelu.chebyshev.degree5.v1", ckks_param_id="ckks_test"),
        ],
    )

    result = analyze_schedule_feasibility(
        "test",
        schedule,
        cost_model,
        {
            "available_levels": 4,
            "bootstrapping_supported": True,
            "bootstrap_cost": {"latency_ms": 10.0, "bootstrap_count": 1},
        },
    )

    assert result.feasible
    assert result.estimated_bootstrap_count == 1
    assert result.schedule.entries[1].bootstrap_policy == "bootstrap_before"
    assert result.total_cost.latency_ms == 15.2
    assert result.total_cost.bootstrap_count == 1


def test_analyze_schedule_feasibility_rejects_single_operator_over_budget():
    registry = build_default_registry()
    cost_model = StaticCostModel(registry, ckks_param_id="ckks_test")
    operator = OperatorKey("mock", 0, "gelu", "ffn_activation", "layer.0.act")
    schedule = SchedulePlan(
        metadata={"policy": "test"},
        entries=[
            ScheduleEntry(operator, "gelu.exact.high.v1", ckks_param_id="ckks_test"),
        ],
    )

    result = analyze_schedule_feasibility(
        "test",
        schedule,
        cost_model,
        {"available_levels": 4, "bootstrapping_supported": True},
    )

    assert not result.feasible
    assert result.first_violation_reason == "single_operator_exceeds_available_levels"
    assert result.unsupported_count == 1
