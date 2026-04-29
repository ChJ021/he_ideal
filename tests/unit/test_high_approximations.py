import torch

from hetune.operators.registry import build_default_registry


def test_gelu_base_is_not_part_of_default_search():
    registry = build_default_registry()
    queried = [provider.candidate_id for provider in registry.query("gelu")]
    assert "gelu.base" not in queried
    assert registry.get("gelu.base").candidate_id == "gelu.base"


def test_gelu_high_approximation_is_closer_than_lower_degree():
    registry = build_default_registry()
    x = torch.linspace(-4.0, 4.0, 1000)
    base = registry.get("gelu.base").plaintext_impl()(x)
    high = registry.get("gelu.exact.high.v1").plaintext_impl()(x)
    degree5 = registry.get("gelu.chebyshev.degree5.v1").plaintext_impl()(x)
    assert torch.max(torch.abs(high - base)) < torch.max(torch.abs(degree5 - base))


def test_higher_degree_gelu_candidate_improves_lower_degree():
    registry = build_default_registry()
    x = torch.linspace(-4.0, 4.0, 1000)
    base = registry.get("gelu.base").plaintext_impl()(x)
    degree9 = registry.get("gelu.chebyshev.degree9.v1").plaintext_impl()(x)
    degree5 = registry.get("gelu.chebyshev.degree5.v1").plaintext_impl()(x)

    assert torch.mean((degree9 - base).pow(2)) < torch.mean((degree5 - base).pow(2))


def test_layernorm_high_approximation_is_closer_than_affine():
    registry = build_default_registry()
    torch.manual_seed(0)
    original = torch.nn.LayerNorm(8)
    x = torch.randn(3, 4, 8)
    base = registry.get("layernorm.base").build_layernorm_module(original)(x)
    high = registry.get("layernorm.exact.high.v1").build_layernorm_module(original)(x)
    affine = registry.get("layernorm.affine.low_cost.v1").build_layernorm_module(original)(x)
    assert torch.isfinite(high).all()
    assert torch.mean((high - base).pow(2)) < torch.mean((affine - base).pow(2))
