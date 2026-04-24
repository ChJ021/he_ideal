from pathlib import Path

import torch

from hetune.core.ids import OperatorKey
from hetune.core.types import ScheduleEntry, SchedulePlan
from hetune.experiments.distillation import (
    build_override_payload,
    load_override_payload,
    override_summary_rows,
    save_override_payload,
)
from hetune.models.hf_adapter import HFSequenceClassifierAdapter


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.LayerNorm(4)])


def test_override_payload_roundtrip_and_apply(tmp_path: Path):
    model = DummyModel()
    operator = OperatorKey("mock", 0, "layernorm", "ffn_layernorm", "blocks.0")
    adapter = HFSequenceClassifierAdapter(
        model_id="mock",
        model_name_or_path="unused",
        model=model,
        operators=[operator],
    )
    schedule = SchedulePlan(
        metadata={"policy": "validated_greedy"},
        entries=[ScheduleEntry(operator, "layernorm.exact.high.v1")],
    )

    original_weight = model.blocks[0].weight.detach().clone()
    original_bias = model.blocks[0].bias.detach().clone()
    payload = build_override_payload(adapter, schedule, "hetune_generated")

    assert payload["schedule_name"] == "hetune_generated"
    assert len(payload["entries"]) == 2

    output_path = save_override_payload(payload, tmp_path / "overrides.pt")
    restored = load_override_payload(output_path)
    rows = override_summary_rows(restored)

    assert output_path.exists()
    assert len(rows) == 2
    assert {row["parameter_name"] for row in rows} == {"weight", "bias"}

    with torch.no_grad():
        model.blocks[0].weight.zero_()
        model.blocks[0].bias.zero_()

    adapter.apply_parameter_overrides(restored["entries"])

    assert torch.allclose(model.blocks[0].weight, original_weight)
    assert torch.allclose(model.blocks[0].bias, original_bias)


def test_build_override_payload_skips_base_candidates():
    model = DummyModel()
    operator = OperatorKey("mock", 0, "layernorm", "ffn_layernorm", "blocks.0")
    adapter = HFSequenceClassifierAdapter(
        model_id="mock",
        model_name_or_path="unused",
        model=model,
        operators=[operator],
    )
    schedule = SchedulePlan(
        metadata={},
        entries=[ScheduleEntry(operator, "layernorm.base")],
    )

    payload = build_override_payload(adapter, schedule, "base_reference")

    assert payload["entries"] == []
