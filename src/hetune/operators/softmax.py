from __future__ import annotations

from hetune.core.types import ApproximationSpec, CostVector
from hetune.operators.base import ApproximationProvider, TensorFunction


class SoftmaxProvider(ApproximationProvider):
    def plaintext_impl(self, context: dict | None = None) -> TensorFunction:
        candidate = self.candidate_id

        def base(scores):
            import torch

            return torch.softmax(scores, dim=-1)

        def high_poly_exp_degree4(scores):
            import torch

            shifted, mask = _shift_for_attention(scores)
            weights = (
                0.94736725056556614
                + 0.76620517093679286 * shifted
                + 0.23476745560170914 * shifted.pow(2)
                + 0.031358179019374724 * shifted.pow(3)
                + 0.0015232991710078347 * shifted.pow(4)
            )
            weights = torch.clamp(weights, min=0.0)
            weights = weights.masked_fill(mask, 0.0)
            return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        def clipped(scores):
            import torch

            mask = scores < -1e4
            clipped_scores = torch.clamp(scores, min=-8.0, max=8.0)
            clipped_scores = clipped_scores.masked_fill(mask, torch.finfo(scores.dtype).min)
            return torch.softmax(clipped_scores, dim=-1)

        def poly_exp_degree2(scores):
            import torch

            shifted, mask = _shift_for_attention(scores)
            # A bounded non-negative degree-2 surrogate for exp(x), x <= 0.
            weights = torch.relu(1.0 + shifted / 4.0).pow(2)
            weights = weights.masked_fill(mask, 0.0)
            return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        def power_degree2(scores):
            import torch

            shifted, mask = _shift_for_attention(scores)
            weights = torch.relu(1.0 + shifted / 2.0).pow(2)
            weights = weights.masked_fill(mask, 0.0)
            return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        functions = {
            "softmax.base": base,
            "softmax.exact.high.v1": high_poly_exp_degree4,
            "softmax.clipped.stable.v1": clipped,
            "softmax.poly_exp.degree2.v1": poly_exp_degree2,
            "softmax.power.degree2.v1": power_degree2,
        }
        if candidate not in functions:
            raise KeyError(f"Unknown Softmax candidate: {candidate}")
        return functions[candidate]

    def build_attention_module(self, original_module, context: dict | None = None):
        from hetune.models.attention_wrappers import build_attention_wrapper

        return build_attention_wrapper(original_module, self.plaintext_impl(context))


def _shift_for_attention(scores):
    import torch

    mask = scores < -1e4
    finite_scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    max_scores = finite_scores.max(dim=-1, keepdim=True).values
    shifted = torch.clamp(finite_scores - max_scores, min=-8.0, max=0.0)
    return shifted, mask


def softmax_providers() -> list[SoftmaxProvider]:
    return [
        SoftmaxProvider(
            ApproximationSpec(
                operator_type="softmax",
                candidate_id="softmax.base",
                approximation_family="plaintext_reference",
                quality_rank=1000,
                depth=0,
                supports_plaintext_simulation=True,
                supports_ckks_backend=False,
                expected_accuracy_risk=0.0,
                implementation_backend="torch_reference",
                cost_hint=CostVector(),
            )
        ),
        SoftmaxProvider(
            ApproximationSpec(
                operator_type="softmax",
                candidate_id="softmax.exact.high.v1",
                approximation_family="high_poly_exp",
                quality_rank=100,
                degree=4,
                valid_input_range=(-8.0, 0.0),
                depth=4,
                supports_ckks_backend=True,
                expected_accuracy_risk=0.0,
                cost_hint=CostVector(latency_ms=8.0, rotations=3, ct_ct_mults=4, ct_pt_mults=4, depth=4, rescale_count=4),
            )
        ),
        SoftmaxProvider(
            ApproximationSpec(
                operator_type="softmax",
                candidate_id="softmax.clipped.stable.v1",
                approximation_family="clipped_exact",
                quality_rank=80,
                depth=4,
                supports_ckks_backend=False,
                expected_accuracy_risk=0.02,
                cost_hint=CostVector(latency_ms=5.0, rotations=2, ct_ct_mults=4, depth=4, rescale_count=4),
            )
        ),
        SoftmaxProvider(
            ApproximationSpec(
                operator_type="softmax",
                candidate_id="softmax.poly_exp.degree2.v1",
                approximation_family="poly_exp",
                quality_rank=50,
                degree=2,
                valid_input_range=(-8.0, 0.0),
                depth=2,
                supports_ckks_backend=True,
                expected_accuracy_risk=0.08,
                cost_hint=CostVector(latency_ms=2.0, rotations=1, ct_ct_mults=2, ct_pt_mults=2, depth=2, rescale_count=2),
            )
        ),
        SoftmaxProvider(
            ApproximationSpec(
                operator_type="softmax",
                candidate_id="softmax.power.degree2.v1",
                approximation_family="power_attention",
                quality_rank=35,
                degree=2,
                valid_input_range=(-4.0, 0.0),
                depth=1,
                supports_ckks_backend=True,
                expected_accuracy_risk=0.12,
                cost_hint=CostVector(latency_ms=1.2, rotations=1, ct_ct_mults=1, ct_pt_mults=2, depth=1, rescale_count=1),
            )
        ),
    ]
