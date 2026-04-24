from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hetune.core.types import SchedulePlan
from hetune.models.hf_adapter import HFSequenceClassifierAdapter
from hetune.operators.registry import ApproximationRegistry
from hetune.profiling.metrics import accuracy


@dataclass(slots=True)
class EvaluationResult:
    logits: np.ndarray
    labels: np.ndarray
    accuracy: float


class PlaintextEvaluator:
    def __init__(
        self,
        adapter: HFSequenceClassifierAdapter,
        registry: ApproximationRegistry,
        batch_size: int = 16,
        max_batches: int | None = None,
    ) -> None:
        self.adapter = adapter
        self.registry = registry
        self.batch_size = batch_size
        self.max_batches = max_batches

    def run(
        self,
        dataset: Any,
        schedule: SchedulePlan | None = None,
        operator_filter: set[str] | None = None,
    ) -> EvaluationResult:
        import torch
        from torch.utils.data import DataLoader

        if self.adapter.model is None:
            raise RuntimeError("Model must be loaded before evaluation")
        if schedule is not None:
            self.adapter.apply_schedule(schedule, self.registry, operator_filter)
        else:
            self.adapter.restore_original()

        loader = DataLoader(dataset, batch_size=self.batch_size)
        logits_list: list[np.ndarray] = []
        labels_list: list[np.ndarray] = []
        device = torch.device(self.adapter.device)
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if self.max_batches is not None and batch_index >= self.max_batches:
                    break
                labels = batch.pop("labels")
                batch = {key: value.to(device) for key, value in batch.items()}
                output = self.adapter.model(**batch)
                logits = output.logits.detach().cpu().numpy()
                logits_list.append(logits)
                labels_list.append(labels.numpy())
        self.adapter.restore_original()
        logits_np = np.concatenate(logits_list, axis=0)
        labels_np = np.concatenate(labels_list, axis=0)
        return EvaluationResult(
            logits=logits_np,
            labels=labels_np,
            accuracy=accuracy(logits_np, labels_np),
        )
