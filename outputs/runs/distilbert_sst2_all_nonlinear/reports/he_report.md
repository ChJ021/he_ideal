# HE Analysis Report: distilbert_sst2_all_nonlinear

- Backend: `openfhe-seal-profiled`
- Backend id: `openfhe_cpu`
- CKKS parameter id: `ckks_128_profiled`
- Profile path: `/home/cj/he_ideal/hetuned_llm/benchmarks/ckks_operator_benchmarks/example_openfhe_profile.csv`
- Profile candidates loaded: `12`
- Figure: `/home/cj/he_ideal/hetuned_llm/outputs/runs/distilbert_sst2_all_nonlinear/figures/he_cost_breakdown.png`

## Schedule HE Metrics

| schedule | entries | profile_entries | static_fallback_entries | profile_coverage_rate | latency_ms | rotations | ct_ct_mults | ct_pt_mults | rescale_count | relin_count | depth | bootstrap_count | memory_mb | weighted_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 24 | 0 | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| hetune_generated | 24 | 24 | 0 | 1 | 277.7 | 216 | 124 | 26 | 124 | 56 | 124 | 0 | 1488 | 689.28 |
| uniform_low | 24 | 24 | 0 | 1 | 45.6 | 48 | 12 | 36 | 12 | 0 | 12 | 0 | 624 | 94.44 |
| uniform_mid | 24 | 24 | 0 | 1 | 94.2 | 108 | 36 | 54 | 36 | 12 | 36 | 0 | 888 | 225.78 |
| uniform_high | 24 | 24 | 0 | 1 | 298.2 | 228 | 132 | 24 | 132 | 60 | 132 | 0 | 1548 | 735.48 |

## Profile Coverage

| schedule | operator_type | candidate_id | entries | profile_entries | static_fallback_entries | profile_coverage_rate |
| --- | --- | --- | --- | --- | --- | --- |
| base | gelu | gelu.base | 6 | 0 | 6 | 0 |
| base | layernorm | layernorm.base | 12 | 0 | 12 | 0 |
| base | softmax | softmax.base | 6 | 0 | 6 | 0 |
| hetune_generated | gelu | gelu.exact.high.v1 | 6 | 6 | 0 | 1 |
| hetune_generated | layernorm | layernorm.centered.mid_cost.v1 | 1 | 1 | 0 | 1 |
| hetune_generated | layernorm | layernorm.exact.high.v1 | 10 | 10 | 0 | 1 |
| hetune_generated | layernorm | layernorm.newton.low_iter.v1 | 1 | 1 | 0 | 1 |
| hetune_generated | softmax | softmax.clipped.stable.v1 | 1 | 1 | 0 | 1 |
| hetune_generated | softmax | softmax.exact.high.v1 | 5 | 5 | 0 | 1 |
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
| base | 0 | False | no_bootstrap_required | within_level_budget | 4 | False |  |  |  |  | 0 | 0 |
| hetune_generated | 0 | True | unsupported | single_operator_exceeds_available_levels | 4 | False | distilbert_sst2.layer0.gelu.ffn_activation | 0 | gelu | gelu.exact.high.v1 | 0 | 6 |
| hetune_generated | 1 | True | unsupported | single_operator_exceeds_available_levels | 4 | False | distilbert_sst2.layer0.softmax.attention_softmax | 0 | softmax | softmax.exact.high.v1 | 0 | 6 |
| hetune_generated | 2 | True | unsupported | single_operator_exceeds_available_levels | 4 | False | distilbert_sst2.layer0.layernorm.ffn_layernorm | 0 | layernorm | layernorm.exact.high.v1 | 1 | 5 |
| hetune_generated | 3 | True | unsupported | single_operator_exceeds_available_levels | 4 | False | distilbert_sst2.layer1.gelu.ffn_activation | 1 | gelu | gelu.exact.high.v1 | 0 | 6 |
| hetune_generated | 4 | True | unsupported | single_operator_exceeds_available_levels | 4 | False | distilbert_sst2.layer1.softmax.attention_softmax | 1 | softmax | softmax.exact.high.v1 | 0 | 6 |
| hetune_generated | 5 | True | unsupported | single_operator_exceeds_available_levels | 4 | False | distilbert_sst2.layer1.layernorm.attention_layernorm | 1 | layernorm | layernorm.exact.high.v1 | 0 | 5 |
| hetune_generated | 6 | True | unsupported | single_operator_exceeds_available_levels | 4 | False | distilbert_sst2.layer1.layernorm.ffn_layernorm | 1 | layernorm | layernorm.exact.high.v1 | 0 | 5 |
| hetune_generated | 7 | True | unsupported | single_operator_exceeds_available_levels | 4 | False | distilbert_sst2.layer2.gelu.ffn_activation | 2 | gelu | gelu.exact.high.v1 | 0 | 6 |
| hetune_generated | 8 | True | unsupported | single_operator_exceeds_available_levels | 4 | False | distilbert_sst2.layer2.softmax.attention_softmax | 2 | softmax | softmax.exact.high.v1 | 0 | 6 |
| hetune_generated | 9 | True | unsupported | single_operator_exceeds_available_levels | 4 | False | distilbert_sst2.layer2.layernorm.attention_layernorm | 2 | layernorm | layernorm.exact.high.v1 | 0 | 5 |
| hetune_generated | 10 | True | unsupported | single_operator_exceeds_available_levels | 4 | False | distilbert_sst2.layer3.gelu.ffn_activation | 3 | gelu | gelu.exact.high.v1 | 3 | 6 |
| hetune_generated | 11 | True | unsupported | single_operator_exceeds_available_levels | 4 | False | distilbert_sst2.layer3.layernorm.attention_layernorm | 3 | layernorm | layernorm.exact.high.v1 | 4 | 5 |
| hetune_generated | 12 | True | unsupported | single_operator_exceeds_available_levels | 4 | False | distilbert_sst2.layer3.layernorm.ffn_layernorm | 3 | layernorm | layernorm.exact.high.v1 | 0 | 5 |
| hetune_generated | 13 | True | unsupported | single_operator_exceeds_available_levels | 4 | False | distilbert_sst2.layer4.gelu.ffn_activation | 4 | gelu | gelu.exact.high.v1 | 0 | 6 |
| hetune_generated | 14 | True | unsupported | single_operator_exceeds_available_levels | 4 | False | distilbert_sst2.layer4.softmax.attention_softmax | 4 | softmax | softmax.exact.high.v1 | 0 | 6 |
| hetune_generated | 15 | True | unsupported | single_operator_exceeds_available_levels | 4 | False | distilbert_sst2.layer4.layernorm.attention_layernorm | 4 | layernorm | layernorm.exact.high.v1 | 0 | 5 |
| hetune_generated | 16 | True | unsupported | single_operator_exceeds_available_levels | 4 | False | distilbert_sst2.layer4.layernorm.ffn_layernorm | 4 | layernorm | layernorm.exact.high.v1 | 0 | 5 |
| hetune_generated | 17 | True | unsupported | single_operator_exceeds_available_levels | 4 | False | distilbert_sst2.layer5.gelu.ffn_activation | 5 | gelu | gelu.exact.high.v1 | 0 | 6 |
| hetune_generated | 18 | True | unsupported | single_operator_exceeds_available_levels | 4 | False | distilbert_sst2.layer5.softmax.attention_softmax | 5 | softmax | softmax.exact.high.v1 | 0 | 6 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

## Notes

- `profile` means the candidate cost came from imported OpenFHE/SEAL microbenchmark data.
- `static_fallback` means the candidate used built-in static CKKS-style metadata.
- `base` is the plaintext reference schedule and is not a deployable HE schedule.
- Unsupported bootstrap rows indicate the schedule exceeds the configured level budget while bootstrapping is disabled.
