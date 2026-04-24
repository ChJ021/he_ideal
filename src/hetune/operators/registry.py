from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from hetune.operators.base import ApproximationProvider
from hetune.operators.gelu import gelu_providers
from hetune.operators.layernorm import layernorm_providers
from hetune.operators.softmax import softmax_providers


class ApproximationRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, ApproximationProvider] = {}
        self._by_type: dict[str, list[ApproximationProvider]] = defaultdict(list)

    def register(self, provider: ApproximationProvider) -> None:
        if provider.candidate_id in self._providers:
            raise ValueError(f"Duplicate candidate id: {provider.candidate_id}")
        self._providers[provider.candidate_id] = provider
        self._by_type[provider.operator_type].append(provider)
        self._by_type[provider.operator_type].sort(
            key=lambda item: item.spec.quality_rank,
            reverse=True,
        )

    def register_many(self, providers: Iterable[ApproximationProvider]) -> None:
        for provider in providers:
            self.register(provider)

    def get(self, candidate_id: str) -> ApproximationProvider:
        return self._providers[candidate_id]

    def query(
        self,
        operator_type: str,
        enabled_ids: set[str] | None = None,
        include_base: bool = False,
    ) -> list[ApproximationProvider]:
        providers = list(self._by_type.get(operator_type, []))
        if not include_base:
            providers = [
                provider for provider in providers if not provider.candidate_id.endswith(".base")
            ]
        if enabled_ids is not None:
            providers = [provider for provider in providers if provider.candidate_id in enabled_ids]
        return providers

    def all(self) -> list[ApproximationProvider]:
        return list(self._providers.values())


def build_default_registry(enabled_ids: set[str] | None = None) -> ApproximationRegistry:
    registry = ApproximationRegistry()
    providers: list[ApproximationProvider] = []
    providers.extend(gelu_providers())
    providers.extend(layernorm_providers())
    providers.extend(softmax_providers())
    for provider in providers:
        if enabled_ids is None or provider.candidate_id in enabled_ids:
            registry.register(provider)
    return registry
