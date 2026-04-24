from __future__ import annotations

from hetune.core.ids import OperatorKey
from hetune.operators.registry import ApproximationRegistry


def build_search_space(
    operators: list[OperatorKey],
    registry: ApproximationRegistry,
) -> dict[str, list[str]]:
    space: dict[str, list[str]] = {}
    for operator in operators:
        candidates = registry.query(operator.operator_type)
        space[operator.id] = [candidate.candidate_id for candidate in candidates]
    return space
