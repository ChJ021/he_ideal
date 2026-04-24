# Project Structure

```text
hetuned_llm/
  configs/        YAML configuration for models, datasets, candidates, CKKS, and experiments.
  src/hetune/     Python package implementing the HETune-LLM MVP.
  tests/          Unit and integration tests.
  docs/           Architecture, protocol, schedule format, and next-stage notes.
  outputs/        Generated runs and global cost tables.
  benchmarks/     Reserved for CKKS and plaintext approximation benchmark scripts.
  data/           Optional local raw, processed, and calibration data.
```

Important package directories:

- `core/`: stable ids, cost vectors, schedules, sensitivity records, YAML helpers.
- `operators/`: approximation providers and candidate registry.
- `models/`: HuggingFace adapter and operator discovery/replacement.
- `profiling/`: metric functions and single-replacement sensitivity profiler.
- `cost/`: static CKKS-style costs, profiled HE cost import, and candidate cost export.
- `scheduling/`: uniform baselines and greedy HETune policy.
- `execution/`: plaintext PyTorch schedule evaluator.
- `experiments/`: config loading, data loading, orchestration, reports, figures.
- `security/`: MVP input-independence schedule audit.

Output runs are grouped by experiment:

```text
outputs/
  runs/<experiment_id>/
    manifest.yaml
    configs/
    profiles/
    schedules/
    evaluations/
    he_analysis/
    figures/
    reports/
    logs/
  cost_tables/
```
