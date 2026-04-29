from hetune.core.ids import OperatorKey
from hetune.core.types import SensitivityRecord
from hetune.cost.static import StaticCostModel
from hetune.operators.registry import build_default_registry
from hetune.scheduling.policies import GreedyDowngradePolicy, UniformPolicy
from hetune.security.validators import SecurityValidator


def test_offline_schedule_pipeline_without_huggingface():
    registry = build_default_registry()
    operators = [
        OperatorKey("mock", 0, "gelu", "ffn_activation", "layer.0.act"),
        OperatorKey("mock", 0, "layernorm", "ln", "layer.0.ln"),
    ]
    high = UniformPolicy(registry, quality="high").generate(operators)
    assert not SecurityValidator().validate(high)

    records = [
        SensitivityRecord(op, candidate, 1.0, 0.999, 0.001, 0.0, 0.0)
        for op, candidate in [
            (operators[0], "gelu.chebyshev.degree9.v1"),
            (operators[1], "layernorm.newton.low_iter.v1"),
        ]
    ]
    greedy = GreedyDowngradePolicy(
        registry=registry,
        cost_model=StaticCostModel(registry),
        max_accuracy_drop=0.01,
    ).generate(operators, records)
    assert not SecurityValidator().validate(greedy)
    assert len(greedy.entries) == len(operators)


def test_softmax_scope_pipeline_without_huggingface():
    registry = build_default_registry()
    operators = [
        OperatorKey("mock", 0, "gelu", "ffn_activation", "layer.0.act"),
        OperatorKey("mock", 0, "softmax", "attention_softmax", "layer.0.attn"),
    ]
    scoped = [operator for operator in operators if operator.operator_type == "softmax"]
    schedule = UniformPolicy(registry, quality="low").generate(scoped)
    assert len(schedule.entries) == 1
    assert schedule.entries[0].operator_key.operator_type == "softmax"
