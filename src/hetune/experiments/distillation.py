from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from hetune.core.serialization import load_yaml
from hetune.core.types import ExperimentPaths, SchedulePlan
from hetune.execution.evaluator import PlaintextEvaluator
from hetune.experiments.config import load_experiment_config
from hetune.experiments.data import load_tokenized_dataset
from hetune.models.hf_adapter import HFSequenceClassifierAdapter, _get_attr
from hetune.operators.registry import build_default_registry
from hetune.profiling.calibration import load_calibration_stats
from hetune.profiling.metrics import accuracy


def save_override_payload(payload: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return target


def load_override_payload(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def build_override_payload(
    adapter: HFSequenceClassifierAdapter,
    schedule: SchedulePlan,
    schedule_name: str,
    operator_types: tuple[str, ...] = ("layernorm",),
    parameter_names: tuple[str, ...] = ("weight", "bias"),
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if adapter.model is None:
        raise RuntimeError("Model must be loaded before collecting overrides")
    entries: list[dict[str, Any]] = []
    for entry in schedule.entries:
        if entry.operator_key.operator_type not in operator_types:
            continue
        if entry.candidate_id.endswith(".base"):
            continue
        module = _get_attr(adapter.model, entry.operator_key.path)
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if parameter_name not in parameter_names:
                continue
            entries.append(
                {
                    "operator_id": entry.operator_key.id,
                    "operator_path": entry.operator_key.path,
                    "candidate_id": entry.candidate_id,
                    "parameter_name": parameter_name,
                    "tensor": parameter.detach().cpu().clone(),
                }
            )
    return {
        "schedule_name": schedule_name,
        "metadata": metadata or {},
        "entries": entries,
    }


def override_summary_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in payload.get("entries", []):
        tensor = entry["tensor"].detach().float()
        rows.append(
            {
                "operator_id": entry["operator_id"],
                "candidate_id": entry["candidate_id"],
                "parameter_name": entry["parameter_name"],
                "shape": list(tensor.shape),
                "mean_abs": float(tensor.abs().mean().item()),
                "max_abs": float(tensor.abs().max().item()),
            }
        )
    return rows


class DistillationRunner:
    def __init__(self, config_path: str | Path, command_name: str = "distill") -> None:
        self.config_path = Path(config_path)
        self.loaded = load_experiment_config(config_path)
        self.operator_scope = self.loaded.experiment.get("operator_scope", "activation_norm")
        self.operator_types = _operator_types_for_scope(
            self.operator_scope,
            self.loaded.experiment.get("operator_types"),
        )
        self.command_name = command_name
        self.experiment_id = self.loaded.experiment["experiment_id"]
        self.paths = ExperimentPaths(
            experiment_id=self.experiment_id,
            root=self.loaded.root / self.loaded.experiment.get("output_root", "outputs"),
        )
        self.paths.ensure()

    def is_enabled(self) -> bool:
        return bool(self._config().get("enabled", False))

    def run(self, force: bool = False) -> Path:
        if not force and not self.is_enabled():
            raise ValueError("Distillation is disabled in experiment config")

        registry = build_default_registry(self._enabled_candidates(self.loaded.approximations))
        schedule_name = str(self._config().get("schedule_name", "hetune_generated"))
        schedule = self._load_schedule(schedule_name)

        teacher = self._build_adapter()
        student = self._build_adapter()
        train_dataset = self._load_dataset(
            teacher,
            split_name=str(
                self._config().get(
                    "train_split",
                    self.loaded.dataset.get("calibration_split", "train"),
                )
            ),
            sample_size=int(self._config().get("train_size", 2000)),
            shuffle_seed=0,
        )
        val_dataset = self._load_dataset(
            teacher,
            split_name=str(self.loaded.dataset.get("validation_split", "validation")),
            sample_size=int(self._config().get("val_size", 512)),
            shuffle_seed=None,
        )
        evaluator = PlaintextEvaluator(
            student,
            registry,
            batch_size=int(self._config().get("batch_size", 16)),
        )
        pre_result = evaluator.run(val_dataset, schedule=schedule)

        student.apply_schedule(schedule, registry)
        teacher.model.eval()
        student.model.eval()

        trained_entries = [
            entry
            for entry in schedule.entries
            if entry.operator_key.operator_type == "layernorm"
            and not entry.candidate_id.endswith(".base")
        ]
        trainable_params = self._configure_trainable_student(student, trained_entries)
        if not trainable_params:
            raise ValueError("No trainable LayerNorm replacement parameters found for distillation")

        hidden_entries = [
            entry
            for entry in trained_entries
            if entry.operator_key.operator_type
            in set(self._config().get("target_operator_types", ["layernorm"]))
            and entry.operator_key.name
            in set(self._config().get("target_operator_names", ["ffn_layernorm"]))
        ]
        teacher_hidden: dict[str, torch.Tensor] = {}
        student_hidden: dict[str, torch.Tensor] = {}
        teacher_handles = self._register_hidden_hooks(teacher, hidden_entries, teacher_hidden, detach=True)
        student_handles = self._register_hidden_hooks(student, hidden_entries, student_hidden, detach=False)

        optimizer = torch.optim.Adam(
            trainable_params,
            lr=float(self._config().get("lr", 5e-5)),
            weight_decay=float(self._config().get("weight_decay", 0.0)),
        )
        best_accuracy = pre_result.accuracy
        best_payload = build_override_payload(
            student,
            schedule,
            schedule_name,
            metadata={
                "pre_distill_accuracy": pre_result.accuracy,
                "distillation_enabled": True,
            },
        )
        history_rows = [
            {
                "epoch": 0,
                "train_loss": 0.0,
                "train_kl": 0.0,
                "train_ce": 0.0,
                "train_hidden": 0.0,
                "val_accuracy": pre_result.accuracy,
                "best_val_accuracy": best_accuracy,
            }
        ]
        patience = int(self._config().get("patience", 1))
        stale_epochs = 0
        for epoch in range(1, int(self._config().get("epochs", 2)) + 1):
            epoch_stats = self._train_epoch(
                teacher,
                student,
                train_dataset,
                optimizer,
                teacher_hidden,
                student_hidden,
            )
            val_accuracy = self._evaluate_loaded_model(student, val_dataset)
            history_rows.append(
                {
                    "epoch": epoch,
                    **epoch_stats,
                    "val_accuracy": val_accuracy,
                    "best_val_accuracy": max(best_accuracy, val_accuracy),
                }
            )
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_payload = build_override_payload(
                    student,
                    schedule,
                    schedule_name,
                    metadata={
                        "pre_distill_accuracy": pre_result.accuracy,
                        "best_val_accuracy": best_accuracy,
                        "epoch": epoch,
                        "trained_operator_count": len(trained_entries),
                        "hidden_target_operator_count": len(hidden_entries),
                    },
                )
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs > patience:
                    break

        for handle in teacher_handles + student_handles:
            handle.remove()

        student.apply_parameter_overrides(best_payload["entries"])
        final_accuracy = self._evaluate_loaded_model(student, val_dataset)
        best_payload["metadata"]["final_val_accuracy"] = final_accuracy

        summary_path = self.paths.distillation_dir() / "summary.csv"
        pd.DataFrame(history_rows).to_csv(summary_path, index=False)
        overrides_path = save_override_payload(
            best_payload,
            self.paths.distillation_dir() / "overrides.pt",
        )
        report_path = self.paths.distillation_dir() / "report.md"
        self._write_report(
            report_path,
            schedule_name,
            trained_entries,
            hidden_entries,
            pre_result.accuracy,
            final_accuracy,
            summary_path,
            overrides_path,
        )
        pd.DataFrame(override_summary_rows(best_payload)).to_csv(
            self.paths.distillation_dir() / "overrides_summary.csv",
            index=False,
        )
        return overrides_path

    def _train_epoch(
        self,
        teacher: HFSequenceClassifierAdapter,
        student: HFSequenceClassifierAdapter,
        dataset,
        optimizer,
        teacher_hidden: dict[str, torch.Tensor],
        student_hidden: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        from torch.utils.data import DataLoader

        loader = DataLoader(
            dataset,
            batch_size=int(self._config().get("batch_size", 16)),
            shuffle=False,
        )
        device = torch.device(student.device)
        total_loss = 0.0
        total_kl = 0.0
        total_ce = 0.0
        total_hidden = 0.0
        batch_count = 0
        temperature = float(self._config().get("temperature", 2.0))
        alpha_kl = float(self._config().get("alpha_kl", 0.7))
        alpha_ce = float(self._config().get("alpha_ce", 0.3))
        alpha_hidden = float(self._config().get("alpha_hidden", 1.0))
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            teacher_hidden.clear()
            student_hidden.clear()
            with torch.no_grad():
                teacher_logits = teacher.model(**batch).logits.detach()
            optimizer.zero_grad(set_to_none=True)
            student_logits = student.model(**batch).logits
            kl = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature**2)
            ce = F.cross_entropy(student_logits, labels)
            hidden_losses = [
                F.mse_loss(student_hidden[key], teacher_hidden[key])
                for key in teacher_hidden
                if key in student_hidden
            ]
            hidden = (
                torch.stack(hidden_losses).mean()
                if hidden_losses
                else student_logits.new_tensor(0.0)
            )
            loss = alpha_kl * kl + alpha_ce * ce + alpha_hidden * hidden
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_kl += float(kl.item())
            total_ce += float(ce.item())
            total_hidden += float(hidden.item())
            batch_count += 1
        divisor = max(batch_count, 1)
        return {
            "train_loss": total_loss / divisor,
            "train_kl": total_kl / divisor,
            "train_ce": total_ce / divisor,
            "train_hidden": total_hidden / divisor,
        }

    def _evaluate_loaded_model(
        self,
        adapter: HFSequenceClassifierAdapter,
        dataset,
    ) -> float:
        import numpy as np
        from torch.utils.data import DataLoader

        loader = DataLoader(
            dataset,
            batch_size=int(self._config().get("batch_size", 16)),
            shuffle=False,
        )
        device = torch.device(adapter.device)
        logits_list: list[np.ndarray] = []
        labels_list: list[np.ndarray] = []
        adapter.model.eval()
        with torch.no_grad():
            for batch in loader:
                labels = batch.pop("labels")
                batch = {key: value.to(device) for key, value in batch.items()}
                output = adapter.model(**batch)
                logits_list.append(output.logits.detach().cpu().numpy())
                labels_list.append(labels.numpy())
        return accuracy(
            np.concatenate(logits_list, axis=0),
            np.concatenate(labels_list, axis=0),
        )

    def _configure_trainable_student(
        self,
        adapter: HFSequenceClassifierAdapter,
        entries,
    ) -> list[torch.nn.Parameter]:
        assert adapter.model is not None
        for parameter in adapter.model.parameters():
            parameter.requires_grad = False
        trainable: list[torch.nn.Parameter] = []
        for entry in entries:
            module = _get_attr(adapter.model, entry.operator_key.path)
            for name, parameter in module.named_parameters(recurse=False):
                if name not in {"weight", "bias"}:
                    continue
                parameter.requires_grad = True
                trainable.append(parameter)
        return trainable

    def _register_hidden_hooks(
        self,
        adapter: HFSequenceClassifierAdapter,
        entries,
        storage: dict[str, torch.Tensor],
        detach: bool,
    ) -> list[Any]:
        handles: list[Any] = []
        if adapter.model is None:
            return handles
        for entry in entries:
            module = _get_attr(adapter.model, entry.operator_key.path)

            def hook(_module, _inputs, output, key=entry.operator_key.id):
                value = output[0] if isinstance(output, tuple) else output
                storage[key] = value.detach() if detach else value

            handles.append(module.register_forward_hook(hook))
        return handles

    def _build_adapter(self) -> HFSequenceClassifierAdapter:
        model_cfg = self.loaded.model
        adapter = HFSequenceClassifierAdapter(
            model_id=model_cfg["model_id"],
            model_name_or_path=model_cfg["model_name_or_path"],
            num_labels=int(model_cfg.get("num_labels", 2)),
            device=self.loaded.experiment.get("device", "cpu"),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        ).load()
        adapter.operators = [
            operator
            for operator in adapter.operators
            if operator.operator_type in set(self.operator_types)
        ]
        adapter.set_calibration_stats(
            load_calibration_stats(self.paths.profile_dir() / "operator_calibration_stats.csv")
        )
        return adapter

    def _load_dataset(
        self,
        adapter: HFSequenceClassifierAdapter,
        split_name: str,
        sample_size: int | None,
        shuffle_seed: int | None,
    ):
        return load_tokenized_dataset(
            self.loaded.dataset,
            adapter,
            split_key="validation_split",
            sample_size=sample_size,
            max_length=int(self.loaded.experiment.get("sequence_length", 128)),
            split_name=split_name,
            shuffle_seed=shuffle_seed,
        )

    def _load_schedule(self, schedule_name: str) -> SchedulePlan:
        path = self.paths.schedule_dir() / f"{schedule_name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Schedule not found: {path}")
        return SchedulePlan.from_dict(load_yaml(path))

    def _config(self) -> dict[str, Any]:
        return dict(self.loaded.experiment.get("distillation", {}))

    def _enabled_candidates(self, config: dict[str, Any]) -> set[str]:
        candidates = config.get("candidates", [])
        enabled: set[str] = set()
        for item in candidates:
            if item.get("enabled", True):
                enabled.add(item["candidate_id"])
        return enabled

    def _write_report(
        self,
        output_path: Path,
        schedule_name: str,
        trained_entries,
        hidden_entries,
        pre_accuracy: float,
        final_accuracy: float,
        summary_path: Path,
        overrides_path: Path,
    ) -> None:
        lines = [
            f"# Distillation Report: {self.experiment_id}",
            "",
            f"- Schedule: `{schedule_name}`",
            f"- Trainable operators: `{len(trained_entries)}`",
            f"- Hidden target operators: `{len(hidden_entries)}`",
            f"- Pre-distill validation accuracy: `{pre_accuracy:.6f}`",
            f"- Distilled validation accuracy: `{final_accuracy:.6f}`",
            f"- Summary: `{summary_path}`",
            f"- Overrides: `{overrides_path}`",
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _operator_types_for_scope(
    operator_scope: str,
    configured: list[str] | None,
) -> list[str]:
    if configured:
        return list(configured)
    scopes = {
        "activation_norm": ["gelu", "layernorm"],
        "softmax_only": ["softmax"],
        "all_nonlinear": ["gelu", "layernorm", "softmax"],
    }
    if operator_scope not in scopes:
        raise ValueError(f"Unknown operator_scope: {operator_scope}")
    return scopes[operator_scope]
