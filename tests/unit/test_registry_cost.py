from hetune.core.ids import OperatorKey
from hetune.cost.static import StaticCostModel
from hetune.operators.registry import build_default_registry


def test_registry_queries_gelu_candidates():
    registry = build_default_registry()
    gelu = registry.query("gelu")
    assert gelu[0].candidate_id == "gelu.exact.high.v1"
    assert gelu[-1].candidate_id == "gelu.poly.degree2.v1"
    assert all(not provider.candidate_id.endswith(".base") for provider in gelu)
    assert registry.get("gelu.base").candidate_id == "gelu.base"


def test_static_cost_prefers_low_degree_gelu():
    registry = build_default_registry()
    cost_model = StaticCostModel(registry)
    operator = OperatorKey("m", 0, "gelu", "ffn_activation", "x.y")
    high = cost_model.weighted_cost(operator, "gelu.exact.high.v1")
    low = cost_model.weighted_cost(operator, "gelu.poly.degree2.v1")
    assert low < high


def test_ckks_only_registry_filters_to_he_supported_candidates():
    registry = build_default_registry(ckks_only=True)

    assert registry.get("gelu.poly.degree2.v1").candidate_id == "gelu.poly.degree2.v1"
    assert registry.get("softmax.power.degree2.v1").candidate_id == "softmax.power.degree2.v1"
    try:
        registry.get("softmax.clipped.stable.v1")
    except KeyError:
        pass
    else:
        raise AssertionError("softmax.clipped.stable.v1 should not be in ckks_only registry")
