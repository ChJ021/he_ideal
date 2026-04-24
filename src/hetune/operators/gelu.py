from __future__ import annotations

import math
from functools import lru_cache

from hetune.core.types import ApproximationSpec, CostVector
from hetune.operators.base import ApproximationProvider, TensorFunction


class GeluProvider(ApproximationProvider):
    def plaintext_impl(self, context: dict | None = None) -> TensorFunction:
        candidate = self.candidate_id

        def base(x):
            import torch

            return torch.nn.functional.gelu(x)

        def calibrated_high(x):
            import torch

            stats = _stats_from_context(context)
            scale = _calibrated_gelu_scale(stats)
            coeffs = _gelu_coefficients(scale=scale, degree=13)
            central = _eval_chebyshev(x, coeffs, -scale, scale)
            # Plaintext simulation of a sign-gated polynomial: GELU tends to
            # 0 on the negative tail and identity on the positive tail.
            return torch.where(
                x < -scale,
                torch.zeros_like(x),
                torch.where(x > scale, x, central),
            )

        def degree2(x):
            return 0.5 * x + 0.25 * x * x

        def degree3(x):
            return 0.5 * x + 0.197 * x * x + 0.035677 * x * x * x

        def degree5(x):
            x2 = x * x
            return 0.5 * x + 0.197 * x2 + 0.035677 * x2 * x - 0.0012 * x2 * x2 * x

        functions = {
            "gelu.base": base,
            "gelu.exact.high.v1": calibrated_high,
            "gelu.poly.degree2.v1": degree2,
            "gelu.poly.degree3.v1": degree3,
            "gelu.poly.degree5.v1": degree5,
        }
        if candidate not in functions:
            raise KeyError(f"Unknown GELU candidate: {candidate}")
        return functions[candidate]


def gelu_providers() -> list[GeluProvider]:
    return [
        GeluProvider(
            ApproximationSpec(
                operator_type="gelu",
                candidate_id="gelu.base",
                approximation_family="plaintext_reference",
                quality_rank=1000,
                degree=0,
                depth=0,
                supports_plaintext_simulation=True,
                supports_ckks_backend=False,
                expected_accuracy_risk=0.0,
                implementation_backend="torch_reference",
                cost_hint=CostVector(),
            )
        ),
        GeluProvider(
            ApproximationSpec(
                operator_type="gelu",
                candidate_id="gelu.exact.high.v1",
                approximation_family="calibrated_chebyshev_gelu",
                quality_rank=100,
                degree=13,
                valid_input_range=(-8.0, 8.0),
                depth=5,
                supports_ckks_backend=True,
                expected_accuracy_risk=0.0,
                cost_hint=CostVector(latency_ms=7.0, ct_ct_mults=6, ct_pt_mults=7, depth=5, rescale_count=5),
            )
        ),
        GeluProvider(
            ApproximationSpec(
                operator_type="gelu",
                candidate_id="gelu.poly.degree5.v1",
                approximation_family="polynomial",
                quality_rank=80,
                degree=5,
                valid_input_range=(-4.0, 4.0),
                depth=3,
                expected_accuracy_risk=0.01,
                cost_hint=CostVector(latency_ms=2.5, ct_ct_mults=3, ct_pt_mults=4, depth=3, rescale_count=3),
            )
        ),
        GeluProvider(
            ApproximationSpec(
                operator_type="gelu",
                candidate_id="gelu.poly.degree3.v1",
                approximation_family="polynomial",
                quality_rank=60,
                degree=3,
                valid_input_range=(-3.0, 3.0),
                depth=2,
                expected_accuracy_risk=0.03,
                cost_hint=CostVector(latency_ms=1.5, ct_ct_mults=2, ct_pt_mults=3, depth=2, rescale_count=2),
            )
        ),
        GeluProvider(
            ApproximationSpec(
                operator_type="gelu",
                candidate_id="gelu.poly.degree2.v1",
                approximation_family="polynomial",
                quality_rank=30,
                degree=2,
                valid_input_range=(-2.5, 2.5),
                depth=1,
                expected_accuracy_risk=0.08,
                cost_hint=CostVector(latency_ms=0.8, ct_ct_mults=1, ct_pt_mults=2, depth=1, rescale_count=1),
            )
        ),
    ]


def _stats_from_context(context: dict | None) -> dict:
    if not context:
        return {}
    stats = context.get("calibration_stats", {})
    return stats if isinstance(stats, dict) else {}


def _calibrated_gelu_scale(stats: dict) -> float:
    raw = stats.get("abs_p99") or stats.get("abs_p95") or 4.0
    try:
        scale = float(raw)
    except (TypeError, ValueError):
        scale = 4.0
    if not math.isfinite(scale) or scale <= 0:
        scale = 4.0
    return max(3.0, min(10.0, scale * 1.1))


@lru_cache(maxsize=128)
def _gelu_coefficients(scale: float, degree: int) -> tuple[float, ...]:
    import numpy as np

    scale = round(float(scale), 3)
    xs = np.linspace(-scale, scale, max(4096, degree * 512))
    erf_values = np.vectorize(math.erf)(xs / math.sqrt(2.0))
    ys = 0.5 * xs * (1.0 + erf_values)
    fitted = np.polynomial.Chebyshev.fit(
        xs,
        ys,
        degree,
        domain=[-scale, scale],
    )
    return tuple(float(value) for value in fitted.coef)


def _eval_chebyshev(x, coeffs: tuple[float, ...], lower: float, upper: float):
    scaled = (2.0 * x - (upper + lower)) / (upper - lower)
    result_next = x.new_zeros(x.shape)
    result_next_next = x.new_zeros(x.shape)
    for coeff in reversed(coeffs[1:]):
        result = 2.0 * scaled * result_next - result_next_next + coeff
        result_next_next = result_next
        result_next = result
    return scaled * result_next - result_next_next + coeffs[0]
