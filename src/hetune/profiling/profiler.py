from __future__ import annotations

from pathlib import Path

import pandas as pd

from hetune.core.types import ScheduleEntry, SchedulePlan, SensitivityRecord
from hetune.execution.evaluator import PlaintextEvaluator
from hetune.operators.registry import ApproximationRegistry
from hetune.profiling.metrics import label_flip_rate, logit_kl


class SensitivityProfiler:
    def __init__(
        self,
        evaluator: PlaintextEvaluator,
        registry: ApproximationRegistry,
        metadata: dict[str, object],
    ) -> None:
        self.evaluator = evaluator
        self.registry = registry
        self.metadata = metadata

    def profile_all(self, dataset) -> list[SensitivityRecord]:
        baseline = self.evaluator.run(dataset)
        records: list[SensitivityRecord] = []
        for operator in self.evaluator.adapter.operators:
            for provider in self.registry.query(operator.operator_type):
                schedule = SchedulePlan(
                    metadata={**self.metadata, "policy": "single_replacement"},
                    entries=[
                        ScheduleEntry(
                            operator_key=operator,
                            candidate_id=provider.candidate_id,
                        )
                    ],
                    constraints={"input_independent": True},
                )
                candidate = self.evaluator.run(
                    dataset,
                    schedule=schedule,
                    operator_filter={operator.id},
                )
                records.append(
                    SensitivityRecord(
                        operator_key=operator,
                        candidate_id=provider.candidate_id,
                        baseline_accuracy=baseline.accuracy,
                        candidate_accuracy=candidate.accuracy,
                        accuracy_drop=baseline.accuracy - candidate.accuracy,
                        logit_kl=logit_kl(baseline.logits, candidate.logits),
                        label_flip_rate=label_flip_rate(baseline.logits, candidate.logits),
                    )
                )
        return records

    @staticmethod
    def save(records: list[SensitivityRecord], path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([record.to_dict() for record in records]).to_csv(target, index=False)

    @staticmethod
    def load(path: str | Path) -> list[dict[str, object]]:
        return pd.read_csv(path).to_dict(orient="records")
