# HETune-LLM

MVP experiment framework for layer-wise error budgeting and approximation
scheduling in CKKS-friendly Transformer inference.

This repository currently implements the first experimental stage:

- HuggingFace sequence-classification model loading.
- GLUE/SST-2 calibration and evaluation pipeline.
- GELU, LayerNorm, and conservative Softmax approximation candidates.
- Layer-wise single-replacement sensitivity profiling.
- Static CKKS-style cost modeling.
- Uniform baselines, additive greedy scheduling, and validated greedy scheduling.
- Three clear experiment entries: activation/norm, Softmax-only, and all nonlinear operators.
- Profiled HE cost import for OpenFHE/SEAL microbenchmark analysis.
- Strict profile coverage checks for deployable HE conclusions.
- Independent SEAL profile generation via `bench-seal-profile`.
- OpenFHE deployment orchestration via `deploy-he`.
- Schedule, metric, report, and figure artifacts under `outputs/`.

The scheduling pipeline remains plaintext PyTorch simulation plus static/profiled
HE cost analysis. The deployment pipeline is separate and requires a real
OpenFHE native runner; it fails fast rather than substituting plaintext latency.

## Setup

On this Linux workspace:

```bash
source .venv/bin/activate
uv sync --extra dev
```

On Windows, activate with `.venv\Scripts\activate`.

## Run

```bash
hetune run-activate --config configs/experiments/bert_tiny_sst2.yaml
```

Useful subcommands:

```bash
hetune run-activate --config configs/experiments/distilbert_sst2.yaml
hetune run-softmax --config configs/experiments/distilbert_sst2_softmax.yaml
hetune run-all --config configs/experiments/distilbert_sst2_all_nonlinear.yaml
hetune run-all-he --config configs/experiments/distilbert_sst2_all_nonlinear_he.yaml
hetune run-he --config configs/experiments/distilbert_sst2_all_nonlinear_he.yaml
hetune bench-seal-profile --ckks-config configs/ckks/seal_profiled.yaml --output benchmarks/ckks_operator_benchmarks/example_seal_profile.csv
hetune deploy-he --config configs/deployment/distilbert_sst2_openfhe.yaml
```

The default experiment is small and intended to validate the full pipeline.
For meaningful SST-2 accuracy, point `configs/models/bert_tiny.yaml` or a new
model config at a fine-tuned sequence-classification checkpoint.

Candidate ids ending in `.base` are plaintext references and are excluded from
automatic schedule search. The `*.exact.high.v1` candidates are high-quality
HE-friendly approximation fallbacks for HETune schedules.
Reports include a `base` row so approximation accuracy can still be compared
against the original plaintext model.

`run-he` expects schedules from a previous `run-all` or `tune` run. It imports
OpenFHE/SEAL-style microbenchmark costs from the CKKS config, then analyzes the
generated schedule and uniform baselines without re-running model inference.
When `scheduler.profile_required: true`, `run-all-he` and `run-he` fail if the
profile file does not match the configured `backend_id + ckks_param_id` or if
the final non-base schedule uses any candidate without profile coverage.

`deploy-he` compares three deployment cases without changing existing experiment
interfaces: `high` uses `uniform_high`, `pre_distill` uses `hetune_generated`,
and `post_distill` uses `hetune_generated` plus
`distillation/overrides.pt` LayerNorm parameters. OpenFHE is expected under
`/home/cj/github/openfhe` with `src`, `build`, `install`, and optional
`openfhe-python` subdirectories. Use `scripts/install_openfhe.sh` to build the
third-party library in that single root. The HETune native runner is a project
artifact and is built inside this repository:

```bash
scripts/build_openfhe_runner.sh
```

The default runner path is `build/openfhe_runner/hetune_openfhe_runner`, which
links to `/home/cj/github/openfhe/install/lib` at runtime. If it is missing,
deployment stops by default. Pass `--allow-unavailable-backend` only to write an
explicit `feasible=false` report for environment validation.

## Outputs

For an experiment id such as `distilbert_sst2_activation_norm`, the pipeline writes:

- `outputs/runs/distilbert_sst2_activation_norm/manifest.yaml`
- `outputs/runs/distilbert_sst2_activation_norm/configs/`
- `outputs/runs/distilbert_sst2_activation_norm/profiles/sensitivity_matrix.csv`
- `outputs/runs/distilbert_sst2_activation_norm/profiles/combination_diagnostics.csv`
- `outputs/runs/distilbert_sst2_activation_norm/schedules/hetune_generated.yaml`
- `outputs/runs/distilbert_sst2_activation_norm/schedules/validated_greedy_decisions.csv`
- `outputs/runs/distilbert_sst2_activation_norm/evaluations/metrics.csv`
- `outputs/runs/distilbert_sst2_activation_norm/he_analysis/he_metrics.csv`
- `outputs/runs/distilbert_sst2_activation_norm/he_analysis/he_cost_breakdown.csv`
- `outputs/runs/distilbert_sst2_activation_norm/he_analysis/bootstrap_plan.csv`
- `outputs/runs/distilbert_sst2_activation_norm/he_deployment/comparison.csv`
- `outputs/runs/distilbert_sst2_activation_norm/he_deployment/deployment_report.md`
- `outputs/runs/distilbert_sst2_activation_norm/figures/sensitivity_heatmap.png`
- `outputs/runs/distilbert_sst2_activation_norm/figures/he_cost_breakdown.png`
- `outputs/runs/distilbert_sst2_activation_norm/reports/report.md`
- `outputs/runs/distilbert_sst2_activation_norm/reports/he_report.md`
- `outputs/runs/distilbert_sst2_activation_norm/reports/artifacts_index.md`
- `outputs/cost_tables/static_ckks_128/candidate_costs.csv`

## Next Stage

See `docs/next_stage_improvements.md` for the post-MVP work list: CKKS
microbenchmarks, SEAL/OpenFHE import, Softmax alternatives, bootstrapping
placement, MRPC/MNLI, and ablations.
