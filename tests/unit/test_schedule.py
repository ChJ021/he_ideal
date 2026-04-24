from hetune.core.ids import OperatorKey
from hetune.core.types import SensitivityRecord
from hetune.cost.static import StaticCostModel
from hetune.operators.registry import build_default_registry
from hetune.scheduling.policies import (
    BasePolicy,
    GreedyDowngradePolicy,
    UniformPolicy,
    ValidatedGreedyDowngradePolicy,
)


def test_uniform_policy_generates_one_entry_per_operator():
    registry = build_default_registry()
    operators = [
        OperatorKey("m", 0, "gelu", "a", "a"),
        OperatorKey("m", 0, "layernorm", "b", "b"),
    ]
    schedule = UniformPolicy(registry, quality="low").generate(operators)
    assert len(schedule.entries) == 2
    assert schedule.entries[0].candidate_id == "gelu.poly.degree2.v1"


def test_base_policy_is_separate_from_uniform_high():
    registry = build_default_registry()
    operators = [OperatorKey("m", 0, "gelu", "a", "a")]
    base = BasePolicy(registry).generate(operators)
    high = UniformPolicy(registry, quality="high").generate(operators)

    assert base.entries[0].candidate_id == "gelu.base"
    assert high.entries[0].candidate_id == "gelu.exact.high.v1"


def test_greedy_policy_respects_accuracy_budget():
    registry = build_default_registry()
    operators = [OperatorKey("m", 0, "gelu", "a", "a")]
    records = [
        SensitivityRecord(
            operator_key=operators[0],
            candidate_id="gelu.poly.degree2.v1",
            baseline_accuracy=1.0,
            candidate_accuracy=0.98,
            accuracy_drop=0.02,
            logit_kl=0.0,
            label_flip_rate=0.0,
        ),
        SensitivityRecord(
            operator_key=operators[0],
            candidate_id="gelu.poly.degree3.v1",
            baseline_accuracy=1.0,
            candidate_accuracy=0.995,
            accuracy_drop=0.005,
            logit_kl=0.0,
            label_flip_rate=0.0,
        ),
    ]
    schedule = GreedyDowngradePolicy(
        registry=registry,
        cost_model=StaticCostModel(registry),
        max_accuracy_drop=0.01,
    ).generate(operators, records)
    assert schedule.entries[0].candidate_id != "gelu.poly.degree2.v1"


def test_validated_greedy_rejects_bad_combination():
    import numpy as np
    from types import SimpleNamespace

    enabled = {"gelu.exact.high.v1", "gelu.poly.degree2.v1"}
    registry = build_default_registry(enabled)
    operators = [
        OperatorKey("m", 0, "gelu", "a", "a"),
        OperatorKey("m", 1, "gelu", "b", "b"),
    ]
    records = [
        SensitivityRecord(
            operator_key=operator,
            candidate_id="gelu.poly.degree2.v1",
            baseline_accuracy=1.0,
            candidate_accuracy=1.0,
            accuracy_drop=0.0,
            logit_kl=0.0,
            label_flip_rate=0.0,
        )
        for operator in operators
    ]

    def evaluate(schedule):
        downgraded = sum(
            entry.candidate_id != "gelu.exact.high.v1" for entry in schedule.entries
        )
        accuracy = 1.0 if downgraded < 2 else 0.75
        logits = np.array([[0.0, 1.0], [0.0, 1.0]])
        labels = np.array([1, 1])
        return SimpleNamespace(logits=logits, labels=labels, accuracy=accuracy)

    result = ValidatedGreedyDowngradePolicy(
        registry=registry,
        cost_model=StaticCostModel(registry),
        evaluate_schedule=evaluate,
        max_accuracy_drop=0.01,
        max_downgrades_per_layer=2,
    ).generate(operators, records)
    accepted = [decision for decision in result.decisions if decision.accepted]
    rejected = [decision for decision in result.decisions if not decision.accepted]
    assert len(accepted) == 1
    assert any("max_accuracy_drop" in decision.reason for decision in rejected)
    assert sum(
        entry.candidate_id == "gelu.poly.degree2.v1" for entry in result.schedule.entries
    ) == 1
