from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from hetune.core.ids import OperatorKey
from hetune.core.types import SchedulePlan
from hetune.operators.registry import ApproximationRegistry


class FunctionalActivationModule:
    def __init__(self, function):
        import torch

        class _Module(torch.nn.Module):
            def forward(self, x):
                return function(x)

        self.module = _Module()

    def as_module(self):
        return self.module


def _get_parent_and_attr(root: Any, dotted_path: str) -> tuple[Any, str]:
    parts = dotted_path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def _get_attr(root: Any, dotted_path: str) -> Any:
    parent, attr = _get_parent_and_attr(root, dotted_path)
    return parent[int(attr)] if attr.isdigit() else getattr(parent, attr)


def _set_attr(root: Any, dotted_path: str, value: Any) -> None:
    parent, attr = _get_parent_and_attr(root, dotted_path)
    if attr.isdigit():
        parent[int(attr)] = value
    else:
        setattr(parent, attr, value)


@dataclass
class HFSequenceClassifierAdapter:
    model_id: str
    model_name_or_path: str
    num_labels: int = 2
    device: str = "cpu"
    trust_remote_code: bool = False
    model: Any | None = None
    tokenizer: Any | None = None
    operators: list[OperatorKey] = field(default_factory=list)
    calibration_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    _originals: dict[str, Any] = field(default_factory=dict)

    def load(self) -> "HFSequenceClassifierAdapter":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            num_labels=self.num_labels,
            trust_remote_code=self.trust_remote_code,
        )
        self.model.to(torch.device(self.device))
        self.model.eval()
        self.operators = self.discover_operators()
        return self

    def discover_operators(self) -> list[OperatorKey]:
        if self.model is None:
            raise RuntimeError("Model must be loaded before operator discovery")
        config = getattr(self.model, "config", None)
        model_type = getattr(config, "model_type", "")
        if model_type in {"bert", "roberta", "albert"} and hasattr(self.model, "bert"):
            return self._discover_bert_like("bert.encoder.layer")
        if model_type == "distilbert" and hasattr(self.model, "distilbert"):
            return self._discover_distilbert()
        return self._discover_by_module_scan()

    def _discover_bert_like(self, layer_prefix: str) -> list[OperatorKey]:
        layers = _get_attr(self.model, layer_prefix)
        operators: list[OperatorKey] = []
        for index, _ in enumerate(layers):
            operators.append(
                OperatorKey(
                    model_id=self.model_id,
                    layer_index=index,
                    operator_type="gelu",
                    name="ffn_activation",
                    path=f"{layer_prefix}.{index}.intermediate.intermediate_act_fn",
                )
            )
            operators.extend(
                [
                    OperatorKey(
                        model_id=self.model_id,
                        layer_index=index,
                        operator_type="softmax",
                        name="attention_softmax",
                        path=f"{layer_prefix}.{index}.attention.self",
                    ),
                    OperatorKey(
                        model_id=self.model_id,
                        layer_index=index,
                        operator_type="layernorm",
                        name="attention_output_layernorm",
                        path=f"{layer_prefix}.{index}.attention.output.LayerNorm",
                    ),
                    OperatorKey(
                        model_id=self.model_id,
                        layer_index=index,
                        operator_type="layernorm",
                        name="ffn_output_layernorm",
                        path=f"{layer_prefix}.{index}.output.LayerNorm",
                    ),
                ]
            )
        return operators

    def _discover_distilbert(self) -> list[OperatorKey]:
        layers = self.model.distilbert.transformer.layer
        operators: list[OperatorKey] = []
        for index, _ in enumerate(layers):
            operators.append(
                OperatorKey(
                    model_id=self.model_id,
                    layer_index=index,
                    operator_type="gelu",
                    name="ffn_activation",
                    path=f"distilbert.transformer.layer.{index}.ffn.activation",
                )
            )
            operators.extend(
                [
                    OperatorKey(
                        model_id=self.model_id,
                        layer_index=index,
                        operator_type="softmax",
                        name="attention_softmax",
                        path=f"distilbert.transformer.layer.{index}.attention",
                    ),
                    OperatorKey(
                        model_id=self.model_id,
                        layer_index=index,
                        operator_type="layernorm",
                        name="attention_layernorm",
                        path=f"distilbert.transformer.layer.{index}.sa_layer_norm",
                    ),
                    OperatorKey(
                        model_id=self.model_id,
                        layer_index=index,
                        operator_type="layernorm",
                        name="ffn_layernorm",
                        path=f"distilbert.transformer.layer.{index}.output_layer_norm",
                    ),
                ]
            )
        return operators

    def _discover_by_module_scan(self) -> list[OperatorKey]:
        import torch

        operators: list[OperatorKey] = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                layer_index = _extract_first_integer(name)
                operators.append(
                    OperatorKey(
                        model_id=self.model_id,
                        layer_index=layer_index,
                        operator_type="layernorm",
                        name=name.replace(".", "_"),
                        path=name,
                    )
                )
        return operators

    def apply_schedule(
        self,
        schedule: SchedulePlan,
        registry: ApproximationRegistry,
        operator_filter: set[str] | None = None,
    ) -> None:
        if self.model is None:
            raise RuntimeError("Model must be loaded before applying schedules")
        self.restore_original()
        known = {operator.id: operator for operator in self.operators}
        for entry in schedule.entries:
            if operator_filter is not None and entry.operator_key.id not in operator_filter:
                continue
            operator = known.get(entry.operator_key.id, entry.operator_key)
            provider = registry.get(entry.candidate_id)
            if entry.candidate_id.endswith(".base"):
                continue
            context = self._operator_context(operator)
            if operator.path not in self._originals:
                self._originals[operator.path] = copy.deepcopy(_get_attr(self.model, operator.path))
            if operator.operator_type == "gelu":
                import torch

                original = self._originals[operator.path]
                function = provider.plaintext_impl(context)
                replacement = (
                    FunctionalActivationModule(function).as_module()
                    if isinstance(original, torch.nn.Module)
                    else function
                )
                _set_attr(self.model, operator.path, replacement)
            elif operator.operator_type == "layernorm":
                original = self._originals[operator.path]
                _set_attr(
                    self.model,
                    operator.path,
                    provider.build_layernorm_module(original, context),
                )
            elif operator.operator_type == "softmax":
                original = self._originals[operator.path]
                _set_attr(
                    self.model,
                    operator.path,
                    provider.build_attention_module(original, context),
                )

    def set_calibration_stats(self, stats: dict[str, dict[str, Any]]) -> None:
        self.calibration_stats = stats

    def apply_parameter_overrides(self, overrides: list[dict[str, Any]]) -> None:
        import torch

        if self.model is None:
            raise RuntimeError("Model must be loaded before applying parameter overrides")
        by_id = {operator.id: operator for operator in self.operators}
        for override in overrides:
            operator_id = str(override["operator_id"])
            operator_path = str(
                override.get("operator_path") or by_id[operator_id].path
            )
            parameter_name = str(override["parameter_name"])
            tensor = override["tensor"]
            module = _get_attr(self.model, operator_path)
            parameter = getattr(module, parameter_name, None)
            if not isinstance(parameter, torch.nn.Parameter):
                raise KeyError(
                    f"Missing parameter {parameter_name} on scheduled module {operator_path}"
                )
            with torch.no_grad():
                parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))

    def _operator_context(self, operator: OperatorKey) -> dict[str, Any]:
        return {
            "operator": operator,
            "operator_id": operator.id,
            "operator_path": operator.path,
            "operator_type": operator.operator_type,
            "calibration_stats": self.calibration_stats.get(operator.id, {}),
        }

    def restore_original(self) -> None:
        if self.model is None:
            return
        for path, value in self._originals.items():
            _set_attr(self.model, path, value)
        self._originals.clear()

    def tokenize_batch(
        self,
        examples: dict[str, list[Any]],
        text_fields: list[str],
        max_length: int,
    ) -> dict[str, Any]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        if len(text_fields) == 1:
            return self.tokenizer(
                examples[text_fields[0]],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
        return self.tokenizer(
            examples[text_fields[0]],
            examples[text_fields[1]],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )


def _extract_first_integer(text: str) -> int:
    for part in text.split("."):
        if part.isdigit():
            return int(part)
    return 0
