# Architecture

HETune-LLM MVP follows this data flow:

```text
configs
  -> HuggingFace model adapter
  -> approximation registry
  -> plaintext sensitivity profiler
  -> static CKKS cost model
  -> uniform and greedy scheduling
  -> plaintext evaluator
  -> profiled HE schedule analysis
  -> reports and figures
```

The schedule is the main decoupling layer. Model code discovers stable
`OperatorKey` values. The registry maps `candidate_id` values to plaintext
implementations and cost metadata. The scheduler only emits
`OperatorKey -> candidate_id` choices.

This keeps per-layer approximation choices out of model code.

The default scheduler is validated greedy: single-replacement profiling is used
to rank candidate downgrades, but each tentative downgrade is accepted only after
running the full current combination schedule on the calibration set. This
prevents local errors from silently accumulating into a large end-to-end drop.

The CLI exposes three experiment scopes:

- `run-activate`: GELU and LayerNorm.
- `run-softmax`: attention Softmax only.
- `run-all`: GELU, LayerNorm, and Softmax jointly.

Each run writes all artifacts under `outputs/runs/<experiment_id>/`.

The second-stage HE analysis entry is `run-he`. It reads existing schedule
artifacts, imports OpenFHE/SEAL-style microbenchmark cost profiles when
available, falls back to static costs for missing candidates, and writes HE cost
breakdowns under `he_analysis/`.
