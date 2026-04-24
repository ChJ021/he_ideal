from types import SimpleNamespace

import torch

from hetune.core.ids import OperatorKey
from hetune.models.hf_adapter import HFSequenceClassifierAdapter
from hetune.profiling.calibration import (
    calibration_coverage,
    collect_operator_calibration_stats,
    load_calibration_stats,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = torch.nn.GELU()
        self.norm = torch.nn.LayerNorm(4)

    def forward(self, input_ids):
        hidden = self.gelu(input_ids.float())
        hidden = self.norm(hidden)
        logits = hidden.mean(dim=1)
        return SimpleNamespace(logits=logits)


def test_collect_operator_calibration_stats(tmp_path):
    operators = [
        OperatorKey("tiny", 0, "gelu", "ffn_activation", "gelu"),
        OperatorKey("tiny", 0, "layernorm", "ffn_layernorm", "norm"),
    ]
    adapter = HFSequenceClassifierAdapter(
        model_id="tiny",
        model_name_or_path="unused",
        model=TinyModel(),
        operators=operators,
    )
    dataset = [
        {
            "input_ids": torch.randn(3, 4),
            "labels": torch.tensor(1),
        }
        for _ in range(4)
    ]
    output = tmp_path / "operator_calibration_stats.csv"

    stats = collect_operator_calibration_stats(
        adapter,
        dataset,
        output,
        batch_size=2,
    )
    loaded = load_calibration_stats(output)

    assert output.exists()
    assert set(stats["operator_type"]) == {"gelu", "layernorm"}
    assert all(operator.id in loaded for operator in operators)
    assert loaded[operators[0].id]["abs_p99"] > 0
    assert loaded[operators[1].id]["var_p99"] > 0
    assert calibration_coverage(operators, loaded) == {
        "tracked": 2,
        "covered": 2,
        "missing": 0,
    }
