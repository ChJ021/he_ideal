from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from hetune.core.ids import OperatorKey
from hetune.core.types import ScheduleEntry, SchedulePlan, SensitivityRecord
from hetune.cost.static import StaticCostModel
from hetune.operators.registry import ApproximationRegistry
from hetune.profiling.metrics import label_flip_rate, logit_kl


def _quality_sorted(registry: ApproximationRegistry, operator_type: str) -> list[str]:
    return [provider.candidate_id for provider in registry.query(operator_type)]


def _base_candidate_id(registry: ApproximationRegistry, operator_type: str) -> str | None:
    candidate_id = f"{operator_type}.base"
    try:
        registry.get(candidate_id)
    except KeyError:
        return None
    return candidate_id


def _schedule_from_entries(
    entries_by_id: dict[str, ScheduleEntry],
    operators: list[OperatorKey],
    metadata: dict[str, object] | None,
    constraints: dict[str, object] | None,
) -> SchedulePlan:
    return SchedulePlan(
        metadata=metadata or {},
        entries=[
            entries_by_id[operator.id]
            for operator in operators
            if operator.id in entries_by_id
        ],
        constraints=constraints or {},
    )


@dataclass(slots=True)
class UniformPolicy:
    registry: ApproximationRegistry
    ckks_param_id: str = "static_ckks_128"
    quality: str = "high"

    def generate(
        self,
        operators: list[OperatorKey],
        metadata: dict[str, object] | None = None,
        constraints: dict[str, object] | None = None,
    ) -> SchedulePlan:
        entries: list[ScheduleEntry] = []
        for operator in operators:
            candidates = _quality_sorted(self.registry, operator.operator_type)
            if not candidates:
                continue
            if self.quality == "low":
                candidate_id = candidates[-1]
            elif self.quality == "mid":
                candidate_id = candidates[len(candidates) // 2]
            else:
                candidate_id = candidates[0]
            entries.append(
                ScheduleEntry(
                    operator_key=operator,
                    candidate_id=candidate_id,
                    ckks_param_id=self.ckks_param_id,
                )
            )
        return SchedulePlan(
            metadata={**(metadata or {}), "policy": f"uniform_{self.quality}"},
            entries=entries,
            constraints=constraints or {},
        )


@dataclass(slots=True)
class BasePolicy:
    registry: ApproximationRegistry
    ckks_param_id: str = "static_ckks_128"

    def generate(
        self,
        operators: list[OperatorKey],
        metadata: dict[str, object] | None = None,
        constraints: dict[str, object] | None = None,
    ) -> SchedulePlan:
        entries: list[ScheduleEntry] = []
        for operator in operators:
            candidate_id = _base_candidate_id(self.registry, operator.operator_type)
            if candidate_id is None:
                continue
            entries.append(
                ScheduleEntry(
                    operator_key=operator,
                    candidate_id=candidate_id,
                    ckks_param_id=self.ckks_param_id,
                )
            )
        return SchedulePlan(
            metadata={**(metadata or {}), "policy": "base_reference"},
            entries=entries,
            constraints=constraints or {},
        )


@dataclass(slots=True)
class GreedyDowngradePolicy:
    registry: ApproximationRegistry
    cost_model: StaticCostModel
    max_accuracy_drop: float = 0.01
    ckks_param_id: str = "static_ckks_128"

    def generate(
        self,
        operators: list[OperatorKey],
        sensitivity_records: list[SensitivityRecord],
        metadata: dict[str, object] | None = None,
        constraints: dict[str, object] | None = None,
    ) -> SchedulePlan:
        constraints = dict(constraints or {})
        constraints.setdefault("max_accuracy_drop", self.max_accuracy_drop)
        high = UniformPolicy(self.registry, self.ckks_param_id, "high").generate(
            operators,
            metadata={**(metadata or {}), "policy": "hetune_greedy"},
            constraints=constraints,
        )
        current: dict[str, ScheduleEntry] = {
            entry.operator_key.id: entry for entry in high.entries
        }
        sensitivity = {
            (record.operator_key.id, record.candidate_id): record
            for record in sensitivity_records
        }
        downgrades: list[tuple[float, float, OperatorKey, str]] = []
        for operator in operators:
            candidates = _quality_sorted(self.registry, operator.operator_type)
            if len(candidates) <= 1:
                continue
            current_candidate = candidates[0]
            current_cost = self.cost_model.weighted_cost(operator, current_candidate)
            for candidate_id in candidates[1:]:
                new_cost = self.cost_model.weighted_cost(operator, candidate_id)
                saving = max(current_cost - new_cost, 0.0)
                record = sensitivity.get((operator.id, candidate_id))
                if record is None:
                    provider = self.registry.get(candidate_id)
                    penalty = provider.spec.expected_accuracy_risk
                else:
                    penalty = max(record.sensitivity_score, 1e-9)
                benefit = saving / max(penalty, 1e-9)
                downgrades.append((benefit, penalty, operator, candidate_id))
        downgrades.sort(key=lambda item: item[0], reverse=True)

        accumulated_drop = 0.0
        for _, penalty, operator, candidate_id in downgrades:
            existing = current[operator.id]
            existing_rank = self.registry.get(existing.candidate_id).spec.quality_rank
            new_rank = self.registry.get(candidate_id).spec.quality_rank
            if new_rank >= existing_rank:
                continue
            record = sensitivity.get((operator.id, candidate_id))
            accuracy_drop = (
                max(record.accuracy_drop, 0.0)
                if record is not None
                else self.registry.get(candidate_id).spec.expected_accuracy_risk
            )
            if accumulated_drop + accuracy_drop > self.max_accuracy_drop:
                continue
            current[operator.id] = ScheduleEntry(
                operator_key=operator,
                candidate_id=candidate_id,
                ckks_param_id=self.ckks_param_id,
            )
            accumulated_drop += accuracy_drop

        schedule = SchedulePlan(
            metadata={
                **(metadata or {}),
                "policy": "hetune_greedy",
                "estimated_additive_accuracy_drop": accumulated_drop,
            },
            entries=[current[operator.id] for operator in operators if operator.id in current],
            constraints=constraints,
        )
        return schedule


@dataclass(slots=True)
class DowngradeDecision:
    operator_id: str
    layer_index: int
    operator_type: str
    operator_name: str
    from_candidate_id: str
    to_candidate_id: str
    accepted: bool
    reason: str
    benefit: float
    cost_saving: float
    single_accuracy_drop: float
    combined_accuracy: float | None = None
    combined_accuracy_drop: float | None = None
    combined_logit_kl: float | None = None
    combined_label_flip_rate: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "operator_id": self.operator_id,
            "layer_index": self.layer_index,
            "operator_type": self.operator_type,
            "operator_name": self.operator_name,
            "from_candidate_id": self.from_candidate_id,
            "to_candidate_id": self.to_candidate_id,
            "accepted": self.accepted,
            "reason": self.reason,
            "benefit": self.benefit,
            "cost_saving": self.cost_saving,
            "single_accuracy_drop": self.single_accuracy_drop,
            "combined_accuracy": self.combined_accuracy,
            "combined_accuracy_drop": self.combined_accuracy_drop,
            "combined_logit_kl": self.combined_logit_kl,
            "combined_label_flip_rate": self.combined_label_flip_rate,
        }


@dataclass(slots=True)
class ValidatedGreedyResult:
    schedule: SchedulePlan
    decisions: list[DowngradeDecision]


@dataclass(slots=True)
class ValidatedGreedyDowngradePolicy:
    registry: ApproximationRegistry
    cost_model: StaticCostModel
    evaluate_schedule: Callable[[SchedulePlan], Any]
    baseline_schedule: SchedulePlan | None = None
    max_accuracy_drop: float = 0.01
    max_logit_kl: float = 0.02
    max_label_flip_rate: float = 0.01
    max_downgrades_per_layer: int = 1
    protected_operator_types: tuple[str, ...] = ("softmax",)
    min_quality_rank_by_operator_type: dict[str, int] | None = None
    ckks_param_id: str = "static_ckks_128"

    def generate(
        self,
        operators: list[OperatorKey],
        sensitivity_records: list[SensitivityRecord],
        metadata: dict[str, object] | None = None,
        constraints: dict[str, object] | None = None,
    ) -> ValidatedGreedyResult:
        constraints = dict(constraints or {})
        constraints.setdefault("max_accuracy_drop", self.max_accuracy_drop)
        constraints.setdefault("max_logit_kl", self.max_logit_kl)
        constraints.setdefault("max_label_flip_rate", self.max_label_flip_rate)
        constraints.setdefault("max_downgrades_per_layer", self.max_downgrades_per_layer)
        constraints.setdefault("protected_operator_types", list(self.protected_operator_types))

        high = UniformPolicy(self.registry, self.ckks_param_id, "high").generate(
            operators,
            metadata={**(metadata or {}), "policy": "hetune_validated_greedy"},
            constraints=constraints,
        )
        baseline_schedule = self.baseline_schedule or BasePolicy(
            self.registry,
            self.ckks_param_id,
        ).generate(
            operators,
            metadata={**(metadata or {}), "policy": "base_reference"},
            constraints=constraints,
        )
        baseline = self.evaluate_schedule(baseline_schedule)
        high_result = self.evaluate_schedule(high)
        current: dict[str, ScheduleEntry] = {
            entry.operator_key.id: entry for entry in high.entries
        }
        sensitivity = {
            (record.operator_key.id, record.candidate_id): record
            for record in sensitivity_records
        }
        downgrades = self._rank_downgrades(operators, sensitivity)
        decisions: list[DowngradeDecision] = []
        layer_downgrade_counts: dict[int, int] = {}
        downgraded_operator_ids: set[str] = set()

        for benefit, cost_saving, operator, candidate_id in downgrades:
            existing = current[operator.id]
            single_record = sensitivity.get((operator.id, candidate_id))
            single_accuracy_drop = (
                max(single_record.accuracy_drop, 0.0)
                if single_record is not None
                else self.registry.get(candidate_id).spec.expected_accuracy_risk
            )
            precheck_reason = self._precheck_candidate(
                operator,
                existing.candidate_id,
                candidate_id,
                layer_downgrade_counts,
                downgraded_operator_ids,
            )
            if precheck_reason is not None:
                decisions.append(
                    self._decision(
                        operator,
                        existing.candidate_id,
                        candidate_id,
                        accepted=False,
                        reason=precheck_reason,
                        benefit=benefit,
                        cost_saving=cost_saving,
                        single_accuracy_drop=single_accuracy_drop,
                    )
                )
                continue

            trial = dict(current)
            trial[operator.id] = ScheduleEntry(
                operator_key=operator,
                candidate_id=candidate_id,
                ckks_param_id=self.ckks_param_id,
            )
            trial_schedule = _schedule_from_entries(
                trial,
                operators,
                metadata={**(metadata or {}), "policy": "hetune_validated_greedy"},
                constraints=constraints,
            )
            result = self.evaluate_schedule(trial_schedule)
            combined_accuracy_drop = baseline.accuracy - result.accuracy
            combined_logit_kl = logit_kl(baseline.logits, result.logits)
            combined_label_flip_rate = label_flip_rate(baseline.logits, result.logits)
            violations = []
            if combined_accuracy_drop > self.max_accuracy_drop:
                violations.append("max_accuracy_drop")
            if combined_logit_kl > self.max_logit_kl:
                violations.append("max_logit_kl")
            if combined_label_flip_rate > self.max_label_flip_rate:
                violations.append("max_label_flip_rate")

            if violations:
                decisions.append(
                    self._decision(
                        operator,
                        existing.candidate_id,
                        candidate_id,
                        accepted=False,
                        reason="violates_" + "+".join(violations),
                        benefit=benefit,
                        cost_saving=cost_saving,
                        single_accuracy_drop=single_accuracy_drop,
                        combined_accuracy=result.accuracy,
                        combined_accuracy_drop=combined_accuracy_drop,
                        combined_logit_kl=combined_logit_kl,
                        combined_label_flip_rate=combined_label_flip_rate,
                    )
                )
                continue

            current = trial
            if operator.id not in downgraded_operator_ids:
                layer_downgrade_counts[operator.layer_index] = (
                    layer_downgrade_counts.get(operator.layer_index, 0) + 1
                )
                downgraded_operator_ids.add(operator.id)
            decisions.append(
                self._decision(
                    operator,
                    existing.candidate_id,
                    candidate_id,
                    accepted=True,
                    reason="accepted",
                    benefit=benefit,
                    cost_saving=cost_saving,
                    single_accuracy_drop=single_accuracy_drop,
                    combined_accuracy=result.accuracy,
                    combined_accuracy_drop=combined_accuracy_drop,
                    combined_logit_kl=combined_logit_kl,
                    combined_label_flip_rate=combined_label_flip_rate,
                )
            )

        schedule = _schedule_from_entries(
            current,
            operators,
            metadata={
                **(metadata or {}),
                "policy": "hetune_validated_greedy",
                "baseline_accuracy": baseline.accuracy,
                "baseline_policy": baseline_schedule.metadata.get("policy", "base_reference"),
                "high_accuracy": high_result.accuracy,
                "high_accuracy_drop_from_base": baseline.accuracy - high_result.accuracy,
                "accepted_downgrades": sum(1 for decision in decisions if decision.accepted),
                "rejected_downgrades": sum(1 for decision in decisions if not decision.accepted),
            },
            constraints=constraints,
        )
        return ValidatedGreedyResult(schedule=schedule, decisions=decisions)

    def _rank_downgrades(
        self,
        operators: list[OperatorKey],
        sensitivity: dict[tuple[str, str], SensitivityRecord],
    ) -> list[tuple[float, float, OperatorKey, str]]:
        downgrades: list[tuple[float, float, OperatorKey, str]] = []
        for operator in operators:
            candidates = _quality_sorted(self.registry, operator.operator_type)
            if len(candidates) <= 1:
                continue
            current_candidate = candidates[0]
            current_cost = self.cost_model.weighted_cost(operator, current_candidate)
            for candidate_id in candidates[1:]:
                new_cost = self.cost_model.weighted_cost(operator, candidate_id)
                cost_saving = max(current_cost - new_cost, 0.0)
                record = sensitivity.get((operator.id, candidate_id))
                if record is None:
                    penalty = self.registry.get(candidate_id).spec.expected_accuracy_risk
                else:
                    penalty = max(record.sensitivity_score, 1e-9)
                benefit = cost_saving / max(penalty, 1e-9)
                downgrades.append((benefit, cost_saving, operator, candidate_id))
        downgrades.sort(key=lambda item: item[0], reverse=True)
        return downgrades

    def _precheck_candidate(
        self,
        operator: OperatorKey,
        existing_candidate_id: str,
        candidate_id: str,
        layer_downgrade_counts: dict[int, int],
        downgraded_operator_ids: set[str],
    ) -> str | None:
        if operator.operator_type in self.protected_operator_types:
            return "protected_operator_type"
        existing_rank = self.registry.get(existing_candidate_id).spec.quality_rank
        new_rank = self.registry.get(candidate_id).spec.quality_rank
        if new_rank >= existing_rank:
            return "not_a_downgrade"
        min_ranks = self.min_quality_rank_by_operator_type or {"layernorm": 45}
        min_rank = min_ranks.get(operator.operator_type)
        if min_rank is not None and new_rank < min_rank:
            return "below_min_quality_rank"
        if (
            operator.id not in downgraded_operator_ids
            and layer_downgrade_counts.get(operator.layer_index, 0)
            >= self.max_downgrades_per_layer
        ):
            return "max_downgrades_per_layer"
        return None

    @staticmethod
    def _decision(
        operator: OperatorKey,
        from_candidate_id: str,
        to_candidate_id: str,
        accepted: bool,
        reason: str,
        benefit: float,
        cost_saving: float,
        single_accuracy_drop: float,
        combined_accuracy: float | None = None,
        combined_accuracy_drop: float | None = None,
        combined_logit_kl: float | None = None,
        combined_label_flip_rate: float | None = None,
    ) -> DowngradeDecision:
        return DowngradeDecision(
            operator_id=operator.id,
            layer_index=operator.layer_index,
            operator_type=operator.operator_type,
            operator_name=operator.name,
            from_candidate_id=from_candidate_id,
            to_candidate_id=to_candidate_id,
            accepted=accepted,
            reason=reason,
            benefit=benefit,
            cost_saving=cost_saving,
            single_accuracy_drop=single_accuracy_drop,
            combined_accuracy=combined_accuracy,
            combined_accuracy_drop=combined_accuracy_drop,
            combined_logit_kl=combined_logit_kl,
            combined_label_flip_rate=combined_label_flip_rate,
        )
