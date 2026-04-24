from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_sensitivity_heatmap(csv_path: str | Path, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = pd.read_csv(csv_path)
    if data.empty:
        return
    data["operator"] = (
        data["layer_index"].astype(str)
        + ":"
        + data["operator_type"].astype(str)
        + ":"
        + data["operator_name"].astype(str)
    )
    pivot = data.pivot_table(
        index="operator",
        columns="candidate_id",
        values="sensitivity_score",
        aggfunc="mean",
        fill_value=0.0,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    height = max(4, min(18, 0.35 * len(pivot.index) + 2))
    width = max(8, min(24, 0.25 * len(pivot.columns) + 8))
    plt.figure(figsize=(width, height))
    sns.heatmap(pivot, cmap="viridis", linewidths=0.2)
    plt.tight_layout()
    plt.savefig(target, dpi=180)
    plt.close()
