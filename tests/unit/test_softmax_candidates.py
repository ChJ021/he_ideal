import torch

from hetune.operators.registry import build_default_registry


def test_softmax_candidates_return_normalized_probabilities():
    registry = build_default_registry()
    scores = torch.tensor([[[[1.0, 0.0, -20000.0]]]])
    for provider in registry.query("softmax"):
        probs = provider.plaintext_impl()(scores)
        assert probs.shape == scores.shape
        assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)))
        assert torch.all(probs >= 0)


def test_softmax_low_cost_candidates_are_cheaper_than_exact():
    registry = build_default_registry()
    exact = registry.get("softmax.exact.high.v1").he_cost().weighted()
    low = registry.get("softmax.power.degree2.v1").he_cost().weighted()
    assert low < exact


def test_softmax_high_approximation_is_close_to_base():
    registry = build_default_registry()
    torch.manual_seed(0)
    scores = torch.randn(2, 3, 4, 8)
    base = registry.get("softmax.base").plaintext_impl()(scores)
    high = registry.get("softmax.exact.high.v1").plaintext_impl()(scores)
    low = registry.get("softmax.power.degree2.v1").plaintext_impl()(scores)
    assert torch.mean((high - base).pow(2)) < torch.mean((low - base).pow(2))
