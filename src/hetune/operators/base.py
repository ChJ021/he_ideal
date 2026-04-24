from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from hetune.core.types import ApproximationSpec, CostVector


TensorFunction = Callable[[Any], Any]


@dataclass(slots=True)
class ApproximationProvider:
    spec: ApproximationSpec

    @property
    def operator_type(self) -> str:
        return self.spec.operator_type

    @property
    def candidate_id(self) -> str:
        return self.spec.candidate_id

    def supports(self, context: dict[str, Any] | None = None) -> bool:
        return True

    def plaintext_impl(self, context: dict[str, Any] | None = None) -> TensorFunction:
        raise NotImplementedError

    def build_layernorm_module(
        self,
        original_module: Any,
        context: dict[str, Any] | None = None,
    ) -> Any:
        raise NotImplementedError(f"{self.candidate_id} is not a LayerNorm provider")

    def build_attention_module(
        self,
        original_module: Any,
        context: dict[str, Any] | None = None,
    ) -> Any:
        raise NotImplementedError(f"{self.candidate_id} is not an attention provider")

    def he_cost(self) -> CostVector:
        return self.spec.cost_hint
