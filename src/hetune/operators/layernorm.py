from __future__ import annotations

import copy
import math
from functools import lru_cache

from hetune.core.types import ApproximationSpec, CostVector
from hetune.operators.base import ApproximationProvider, TensorFunction


class ApproxLayerNorm:
    """Torch module wrapper with HE-oriented LayerNorm variants."""

    def __init__(
        self,
        original_module,
        mode: str,
        rsqrt_coeffs: tuple[float, ...] | None = None,
        rsqrt_domain: tuple[float, float] | None = None,
    ):
        import torch

        self.mode = mode
        self.normalized_shape = original_module.normalized_shape
        self.eps = original_module.eps
        self.weight = torch.nn.Parameter(original_module.weight.detach().clone())
        self.bias = torch.nn.Parameter(original_module.bias.detach().clone())
        self.rsqrt_coeffs = rsqrt_coeffs
        self.rsqrt_domain = rsqrt_domain

    def as_module(self):
        import torch

        parent = self

        class _Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.normalized_shape = parent.normalized_shape
                self.eps = parent.eps
                self.weight = parent.weight
                self.bias = parent.bias
                self.mode = parent.mode
                self.rsqrt_coeffs = parent.rsqrt_coeffs
                self.rsqrt_domain = parent.rsqrt_domain

            def forward(self, x):
                if self.mode == "affine":
                    return x * self.weight + self.bias
                mean = x.mean(dim=-1, keepdim=True)
                centered = x - mean
                if self.mode == "centered_affine":
                    return centered * self.weight + self.bias
                var = centered.pow(2).mean(dim=-1, keepdim=True)
                if self.mode == "rsqrt_low_iter":
                    inv_std = 1.0 / (1.0 + 0.5 * (var - 1.0))
                elif self.mode == "rsqrt_poly_calibrated":
                    inv_std = _eval_chebyshev(
                        var + self.eps,
                        self.rsqrt_coeffs,
                        self.rsqrt_domain,
                    )
                else:
                    inv_std = torch.rsqrt(var + self.eps)
                return centered * inv_std * self.weight + self.bias

        return _Module()


class LayerNormProvider(ApproximationProvider):
    def plaintext_impl(self, context: dict | None = None) -> TensorFunction:
        raise NotImplementedError("LayerNorm candidates replace modules, not functions")

    def build_layernorm_module(self, original_module, context: dict | None = None):
        if self.candidate_id == "layernorm.base":
            return copy.deepcopy(original_module)
        if self.candidate_id == "layernorm.exact.high.v1":
            lower, upper = _calibrated_layernorm_range(
                _stats_from_context(context),
                float(getattr(original_module, "eps", 1e-12)),
            )
            coeffs = _rsqrt_coefficients(lower, upper, degree=9)
            return ApproxLayerNorm(
                original_module,
                "rsqrt_poly_calibrated",
                rsqrt_coeffs=coeffs,
                rsqrt_domain=(lower, upper),
            ).as_module()
        mode = {
            "layernorm.affine.low_cost.v1": "affine",
            "layernorm.centered.mid_cost.v1": "centered_affine",
            "layernorm.newton.low_iter.v1": "rsqrt_low_iter",
        }.get(self.candidate_id)
        if mode is None:
            raise KeyError(f"Unknown LayerNorm candidate: {self.candidate_id}")
        return ApproxLayerNorm(original_module, mode).as_module()


def layernorm_providers() -> list[LayerNormProvider]:
    return [
        LayerNormProvider(
            ApproximationSpec(
                operator_type="layernorm",
                candidate_id="layernorm.base",
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
        LayerNormProvider(
            ApproximationSpec(
                operator_type="layernorm",
                candidate_id="layernorm.exact.high.v1",
                approximation_family="calibrated_clipped_rsqrt_polynomial",
                quality_rank=100,
                degree=9,
                valid_input_range=(0.05, 8.0),
                depth=6,
                supports_ckks_backend=True,
                expected_accuracy_risk=0.0,
                cost_hint=CostVector(latency_ms=8.5, rotations=2, ct_ct_mults=6, ct_pt_mults=7, depth=6, rescale_count=6),
            )
        ),
        LayerNormProvider(
            ApproximationSpec(
                operator_type="layernorm",
                candidate_id="layernorm.newton.low_iter.v1",
                approximation_family="newton",
                quality_rank=70,
                degree=2,
                depth=3,
                supports_ckks_backend=True,
                expected_accuracy_risk=0.04,
                cost_hint=CostVector(latency_ms=3.0, rotations=2, ct_ct_mults=3, depth=3, rescale_count=3),
            )
        ),
        LayerNormProvider(
            ApproximationSpec(
                operator_type="layernorm",
                candidate_id="layernorm.centered.mid_cost.v1",
                approximation_family="centered_affine",
                quality_rank=45,
                depth=1,
                supports_ckks_backend=True,
                expected_accuracy_risk=0.07,
                cost_hint=CostVector(latency_ms=1.2, rotations=1, ct_ct_mults=1, ct_pt_mults=2, depth=1, rescale_count=1),
            )
        ),
        LayerNormProvider(
            ApproximationSpec(
                operator_type="layernorm",
                candidate_id="layernorm.affine.low_cost.v1",
                approximation_family="affine",
                quality_rank=20,
                depth=0,
                supports_ckks_backend=True,
                expected_accuracy_risk=0.12,
                cost_hint=CostVector(latency_ms=0.4, ct_pt_mults=1, depth=0),
            )
        ),
    ]


def _stats_from_context(context: dict | None) -> dict:
    if not context:
        return {}
    stats = context.get("calibration_stats", {})
    return stats if isinstance(stats, dict) else {}


def _calibrated_layernorm_range(stats: dict, eps: float) -> tuple[float, float]:
    low_raw = stats.get("var_p01") or stats.get("var_min") or 0.05
    high_raw = stats.get("var_p99") or stats.get("var_p95") or 4.0
    high_p95_raw = stats.get("var_p95") or high_raw
    mean_raw = stats.get("var_mean") or high_p95_raw
    try:
        lower = float(low_raw)
        upper = float(high_raw)
        upper_p95 = float(high_p95_raw)
        mean = float(mean_raw)
    except (TypeError, ValueError):
        lower, upper, upper_p95, mean = 0.05, 4.0, 4.0, 1.0
    if not math.isfinite(lower) or lower <= 0:
        lower = 0.05
    if not math.isfinite(upper) or upper <= lower:
        upper = max(4.0, lower + 1.0)
    if not math.isfinite(upper_p95) or upper_p95 <= lower:
        upper_p95 = upper
    if not math.isfinite(mean) or mean <= lower:
        mean = upper_p95
    lower = max(eps, 1e-4, lower * 0.8)
    if upper > upper_p95 * 4.0:
        upper = upper_p95 * 2.0
    else:
        upper = upper * 1.2
    upper = max(upper, mean * 10.0, lower + 0.25, 1.0)
    if upper <= lower:
        upper = lower + 0.25
    return round(lower, 6), round(upper, 6)


@lru_cache(maxsize=128)
def _rsqrt_coefficients(lower: float, upper: float, degree: int) -> tuple[float, ...]:
    import numpy as np

    lower = round(float(lower), 6)
    upper = round(float(upper), 6)
    xs = np.linspace(lower, upper, max(4096, degree * 512))
    ys = 1.0 / np.sqrt(xs)
    fitted = np.polynomial.Chebyshev.fit(
        xs,
        ys,
        degree,
        domain=[lower, upper],
    )
    return tuple(float(value) for value in fitted.coef)


def _eval_chebyshev(
    x,
    coeffs: tuple[float, ...] | None,
    domain: tuple[float, float] | None,
):
    if not coeffs or domain is None:
        import torch

        return torch.rsqrt(x)
    lower, upper = domain
    x = x.clamp(min=lower, max=upper)
    scaled = (2.0 * x - (upper + lower)) / (upper - lower)
    result_next = x.new_zeros(x.shape)
    result_next_next = x.new_zeros(x.shape)
    for coeff in reversed(coeffs[1:]):
        result = 2.0 * scaled * result_next - result_next_next + coeff
        result_next_next = result_next
        result_next = result
    return scaled * result_next - result_next_next + coeffs[0]
