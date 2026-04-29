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

        def calibrated_chebyshev(x, degree: int):
            import torch

            stats = _stats_from_context(context)
            scale = gelu_calibrated_scale(stats)
            coeffs = gelu_chebyshev_coefficients(scale=scale, degree=degree)
            central = _eval_chebyshev(x, coeffs, -scale, scale)
            # Plaintext simulation of a sign-gated polynomial: GELU tends to
            # 0 on the negative tail and identity on the positive tail.
            return torch.where(
                x < -scale,
                torch.zeros_like(x),
                torch.where(x > scale, x, central),
            )

        functions = {
            "gelu.base": base,
            "gelu.exact.high.v1": lambda x: calibrated_chebyshev(x, degree=13),
            "gelu.chebyshev.degree11.v1": lambda x: calibrated_chebyshev(x, degree=11),
            "gelu.chebyshev.degree9.v1": lambda x: calibrated_chebyshev(x, degree=9),
            "gelu.chebyshev.degree5.v1": lambda x: calibrated_chebyshev(x, degree=5),
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
                candidate_id="gelu.chebyshev.degree11.v1",
                approximation_family="calibrated_chebyshev_gelu",
                quality_rank=90,
                degree=11,
                valid_input_range=(-8.0, 8.0),
                depth=5,
                supports_ckks_backend=True,
                expected_accuracy_risk=0.002,
                cost_hint=CostVector(latency_ms=5.8, ct_ct_mults=5, ct_pt_mults=6, depth=5, rescale_count=5),
            )
        ),
        GeluProvider(
            ApproximationSpec(
                operator_type="gelu",
                candidate_id="gelu.chebyshev.degree9.v1",
                approximation_family="calibrated_chebyshev_gelu",
                quality_rank=75,
                degree=9,
                valid_input_range=(-8.0, 8.0),
                depth=4,
                supports_ckks_backend=True,
                expected_accuracy_risk=0.006,
                cost_hint=CostVector(latency_ms=4.4, ct_ct_mults=4, ct_pt_mults=5, depth=4, rescale_count=4),
            )
        ),
        GeluProvider(
            ApproximationSpec(
                operator_type="gelu",
                candidate_id="gelu.chebyshev.degree5.v1",
                approximation_family="calibrated_chebyshev_gelu",
                quality_rank=55,
                degree=5,
                valid_input_range=(-8.0, 8.0),
                depth=3,
                supports_ckks_backend=True,
                expected_accuracy_risk=0.02,
                cost_hint=CostVector(latency_ms=2.6, ct_ct_mults=3, ct_pt_mults=4, depth=3, rescale_count=3),
            )
        ),
    ]


def _stats_from_context(context: dict | None) -> dict:
    if not context:
        return {}
    stats = context.get("calibration_stats", {})
    return stats if isinstance(stats, dict) else {}


def gelu_degree_for_candidate(candidate_id: str) -> int:
    degrees = {
        "gelu.exact.high.v1": 13,
        "gelu.chebyshev.degree11.v1": 11,
        "gelu.chebyshev.degree9.v1": 9,
        "gelu.chebyshev.degree5.v1": 5,
    }
    try:
        return degrees[candidate_id]
    except KeyError as exc:
        raise KeyError(f"Unknown GELU Chebyshev candidate: {candidate_id}") from exc


def gelu_calibrated_scale(stats: dict) -> float:
    raw = stats.get("abs_p99") or stats.get("abs_p95") or 4.0
    try:
        scale = float(raw)
    except (TypeError, ValueError):
        scale = 4.0
    if not math.isfinite(scale) or scale <= 0:
        scale = 4.0
    return max(3.0, min(10.0, scale * 1.1))


@lru_cache(maxsize=128)
def gelu_chebyshev_coefficients(scale: float, degree: int) -> tuple[float, ...]:
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


@lru_cache(maxsize=128)
def gelu_power_coefficients(scale: float, degree: int) -> tuple[float, ...]:
    import numpy as np

    scale = round(float(scale), 3)
    cheb = np.polynomial.Chebyshev(
        gelu_chebyshev_coefficients(scale=scale, degree=degree),
        domain=[-scale, scale],
    )
    power = cheb.convert(kind=np.polynomial.Polynomial)
    return tuple(float(value) for value in power.coef)


def _eval_chebyshev(x, coeffs: tuple[float, ...], lower: float, upper: float):
    scaled = (2.0 * x - (upper + lower)) / (upper - lower)
    result_next = x.new_zeros(x.shape)
    result_next_next = x.new_zeros(x.shape)
    for coeff in reversed(coeffs[1:]):
        result = 2.0 * scaled * result_next - result_next_next + coeff
        result_next_next = result_next
        result_next = result
    return scaled * result_next - result_next_next + coeffs[0]
