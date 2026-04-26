# HE Analysis Report: distilbert_sst2_all_nonlinear_he

- Backend: `seal-profiled`
- Backend id: `seal_cpu`
- CKKS parameter id: `seal_profiled_bootstrap`
- Profile path: `/home/cj/he_ideal/hetuned_llm/benchmarks/ckks_operator_benchmarks/example_seal_profile_bootstrap.csv`
- Profile candidates loaded: `11`
- Strict profile required: `True`
- Figure: `/home/cj/he_ideal/hetuned_llm/outputs/runs/distilbert_sst2_all_nonlinear_he/figures/he_cost_breakdown.png`
- Feasible schedules: `base, hetune_generated, uniform_low, uniform_mid, uniform_high`
- Infeasible schedules: `none`
- Unsupported rows: `0`
- Single-operator infeasible candidates: `none`

## Schedule HE Metrics

| schedule | entries | profile_entries | static_fallback_entries | profile_coverage_rate | schedule_feasible | unsupported_rows | estimated_bootstrap_count | profile_candidates_loaded | used_candidates_with_profile | used_candidates_missing_profile | used_candidate_ids_missing_profile | strict_profile_check_passed | strict_profile_check_reason | latency_ms | rotations | ct_ct_mults | ct_pt_mults | rescale_count | relin_count | depth | bootstrap_count | memory_mb | weighted_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 24 | 0 | 24 | 1 | True | 0 | 0 | 11 | 0 | 0 |  | True |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| hetune_generated | 24 | 24 | 0 | 1 | True | 0 | 21 | 11 | 5 | 0 |  | True |  | 2715.97 | 544 | 455 | 385 | 281 | 84 | 113 | 21 | 5376 | 4675.18 |
| uniform_low | 24 | 24 | 0 | 1 | True | 0 | 1 | 11 | 3 | 0 |  | True |  | 139.152 | 30 | 28 | 48 | 20 | 4 | 12 | 1 | 256 | 256.012 |
| uniform_mid | 24 | 24 | 0 | 1 | True | 0 | 5 | 11 | 3 | 0 |  | True |  | 640.356 | 138 | 116 | 114 | 76 | 20 | 36 | 5 | 1280 | 1135.56 |
| uniform_high | 24 | 24 | 0 | 1 | True | 0 | 23 | 11 | 3 | 0 |  | True |  | 2978.88 | 594 | 500 | 426 | 310 | 92 | 126 | 23 | 5888 | 5131.16 |

## Profile Coverage

| schedule | operator_type | candidate_id | entries | profile_entries | static_fallback_entries | profile_coverage_rate |
| --- | --- | --- | --- | --- | --- | --- |
| base | gelu | gelu.base | 6 | 0 | 6 | 0 |
| base | layernorm | layernorm.base | 12 | 0 | 12 | 0 |
| base | softmax | softmax.base | 6 | 0 | 6 | 0 |
| hetune_generated | gelu | gelu.exact.high.v1 | 6 | 6 | 0 | 1 |
| hetune_generated | layernorm | layernorm.centered.mid_cost.v1 | 2 | 2 | 0 | 1 |
| hetune_generated | layernorm | layernorm.exact.high.v1 | 9 | 9 | 0 | 1 |
| hetune_generated | layernorm | layernorm.newton.low_iter.v1 | 1 | 1 | 0 | 1 |
| hetune_generated | softmax | softmax.exact.high.v1 | 6 | 6 | 0 | 1 |
| uniform_high | gelu | gelu.exact.high.v1 | 6 | 6 | 0 | 1 |
| uniform_high | layernorm | layernorm.exact.high.v1 | 12 | 12 | 0 | 1 |
| uniform_high | softmax | softmax.exact.high.v1 | 6 | 6 | 0 | 1 |
| uniform_low | gelu | gelu.poly.degree2.v1 | 6 | 6 | 0 | 1 |
| uniform_low | layernorm | layernorm.affine.low_cost.v1 | 12 | 12 | 0 | 1 |
| uniform_low | softmax | softmax.power.degree2.v1 | 6 | 6 | 0 | 1 |
| uniform_mid | gelu | gelu.poly.degree3.v1 | 6 | 6 | 0 | 1 |
| uniform_mid | layernorm | layernorm.centered.mid_cost.v1 | 12 | 12 | 0 | 1 |
| uniform_mid | softmax | softmax.poly_exp.degree2.v1 | 6 | 6 | 0 | 1 |

## Bootstrap / Level Plan

| schedule | segment_index | required | status | reason | available_levels | bootstrap_supported | bootstrap_before_operator_id | layer_index | operator_type | candidate_id | levels_used_before | operator_level_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0 | False | no_bootstrap_required | within_level_budget | 6 | True |  |  |  |  | 0 | 0 |
| hetune_generated | 0 | True | supported | level_budget_exceeded | 6 | True | distilbert_sst2.layer0.softmax.attention_softmax | 0 | softmax | softmax.exact.high.v1 | 5 | 4 |
| hetune_generated | 1 | True | supported | level_budget_exceeded | 6 | True | distilbert_sst2.layer0.layernorm.ffn_layernorm | 0 | layernorm | layernorm.exact.high.v1 | 5 | 6 |
| hetune_generated | 2 | True | supported | level_budget_exceeded | 6 | True | distilbert_sst2.layer1.gelu.ffn_activation | 1 | gelu | gelu.exact.high.v1 | 6 | 5 |
| hetune_generated | 3 | True | supported | level_budget_exceeded | 6 | True | distilbert_sst2.layer1.softmax.attention_softmax | 1 | softmax | softmax.exact.high.v1 | 5 | 4 |
| hetune_generated | 4 | True | supported | level_budget_exceeded | 6 | True | distilbert_sst2.layer1.layernorm.attention_layernorm | 1 | layernorm | layernorm.exact.high.v1 | 4 | 6 |
| hetune_generated | 5 | True | supported | level_budget_exceeded | 6 | True | distilbert_sst2.layer1.layernorm.ffn_layernorm | 1 | layernorm | layernorm.exact.high.v1 | 6 | 6 |
| hetune_generated | 6 | True | supported | level_budget_exceeded | 6 | True | distilbert_sst2.layer2.gelu.ffn_activation | 2 | gelu | gelu.exact.high.v1 | 6 | 5 |
| hetune_generated | 7 | True | supported | level_budget_exceeded | 6 | True | distilbert_sst2.layer2.softmax.attention_softmax | 2 | softmax | softmax.exact.high.v1 | 5 | 4 |
| hetune_generated | 8 | True | supported | level_budget_exceeded | 6 | True | distilbert_sst2.layer2.layernorm.attention_layernorm | 2 | layernorm | layernorm.exact.high.v1 | 4 | 6 |
| hetune_generated | 9 | True | supported | level_budget_exceeded | 6 | True | distilbert_sst2.layer2.layernorm.ffn_layernorm | 2 | layernorm | layernorm.newton.low_iter.v1 | 6 | 3 |
| hetune_generated | 10 | True | supported | level_budget_exceeded | 6 | True | distilbert_sst2.layer3.gelu.ffn_activation | 3 | gelu | gelu.exact.high.v1 | 3 | 5 |
| hetune_generated | 11 | True | supported | level_budget_exceeded | 6 | True | distilbert_sst2.layer3.softmax.attention_softmax | 3 | softmax | softmax.exact.high.v1 | 5 | 4 |
| hetune_generated | 12 | True | supported | level_budget_exceeded | 6 | True | distilbert_sst2.layer3.layernorm.attention_layernorm | 3 | layernorm | layernorm.exact.high.v1 | 4 | 6 |
| hetune_generated | 13 | True | supported | level_budget_exceeded | 6 | True | distilbert_sst2.layer3.layernorm.ffn_layernorm | 3 | layernorm | layernorm.centered.mid_cost.v1 | 6 | 1 |
| hetune_generated | 14 | True | supported | level_budget_exceeded | 6 | True | distilbert_sst2.layer4.softmax.attention_softmax | 4 | softmax | softmax.exact.high.v1 | 6 | 4 |
| hetune_generated | 15 | True | supported | level_budget_exceeded | 6 | True | distilbert_sst2.layer4.layernorm.attention_layernorm | 4 | layernorm | layernorm.exact.high.v1 | 4 | 6 |
| hetune_generated | 16 | True | supported | level_budget_exceeded | 6 | True | distilbert_sst2.layer4.layernorm.ffn_layernorm | 4 | layernorm | layernorm.exact.high.v1 | 6 | 6 |
| hetune_generated | 17 | True | supported | level_budget_exceeded | 6 | True | distilbert_sst2.layer5.gelu.ffn_activation | 5 | gelu | gelu.exact.high.v1 | 6 | 5 |
| hetune_generated | 18 | True | supported | level_budget_exceeded | 6 | True | distilbert_sst2.layer5.softmax.attention_softmax | 5 | softmax | softmax.exact.high.v1 | 5 | 4 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

## Notes

- `profile` means the candidate cost came from imported OpenFHE/SEAL microbenchmark data.
- `static_fallback` means the candidate used built-in static CKKS-style metadata.
- `base` is the plaintext reference schedule and is not a deployable HE schedule.
- `strict_profile_check_passed` is the gate for reporting a deployable HE conclusion.
- `schedule_feasible = true` means the schedule fits the configured level budget with modeled bootstrap placement.
- Unsupported rows indicate the schedule exceeds the configured level budget or contains a single operator that cannot fit in the configured levels.
