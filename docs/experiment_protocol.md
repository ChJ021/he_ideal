# Experiment Protocol

1. Activate the virtual environment.
2. Sync dependencies with `uv sync --extra dev`.
3. Run `hetune run-activate --config configs/experiments/distilbert_sst2.yaml`.
4. Run `hetune run-softmax --config configs/experiments/distilbert_sst2_softmax.yaml`.
5. Run `hetune run-all --config configs/experiments/distilbert_sst2_all_nonlinear.yaml`.
6. Run `hetune run-he --config configs/experiments/distilbert_sst2_all_nonlinear_he.yaml`
   after schedules have been generated.
7. Inspect `outputs/runs/<experiment_id>/reports/artifacts_index.md`.
8. Inspect `outputs/runs/<experiment_id>/schedules/hetune_generated.yaml` and
   `outputs/runs/<experiment_id>/schedules/validated_greedy_decisions.csv`.
9. Inspect metrics, HE analysis, figures, and report outputs under the same run
   directory.

Use BERT-Tiny configs for quick smoke tests and DistilBERT configs for meaningful
SST-2 accuracy.
