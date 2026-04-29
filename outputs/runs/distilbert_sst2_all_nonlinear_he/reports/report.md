# HETune-LLM Report: distilbert_sst2_all_nonlinear_he

## Summary

- Operator scope: `all_nonlinear`
- Optimized operator types: `gelu, layernorm, softmax`
- Final policy: `hetune_validated_greedy`
- Schedule entries: `24`
- Estimated latency: `2715.966`
- Estimated depth sum: `113`
- Estimated rotations: `544`
- Base reference accuracy: `0.914062`

Schedule entries by type:

| operator_type | entry_count |
| --- | --- |
| gelu | 6 |
| layernorm | 12 |
| softmax | 6 |

Calibration stats:

- Stats file: `/home/cj/he_ideal/hetuned_llm/outputs/runs/distilbert_sst2_all_nonlinear_he/profiles/operator_calibration_stats.csv`
- Covered operators: `18`
- Tracked operators: `18`
- Missing operators: `0`

HE deployment:

- HE-aware tuning: `True`
- Feasible: `True`
- CKKS parameter id: `seal_profiled_bootstrap`
- Backend id: `seal_cpu`
- Profile candidates loaded: `11`
- Used candidates with profile: `5`
- Used candidates missing profile: `0`
- Strict profile check passed: `True`
- Profile coverage rate: `1.000000`
- Estimated bootstrap count: `21`
- Unsupported rows: `0`

Distillation:

- Summary: `/home/cj/he_ideal/hetuned_llm/outputs/runs/distilbert_sst2_all_nonlinear_he/distillation/summary.csv`
- Report: `/home/cj/he_ideal/hetuned_llm/outputs/runs/distilbert_sst2_all_nonlinear_he/distillation/report.md`
- Overrides: `/home/cj/he_ideal/hetuned_llm/outputs/runs/distilbert_sst2_all_nonlinear_he/distillation/overrides.pt`
- Status: `available`

all-nonlinear accuracy before/after distillation:

| variant | schedule | accuracy |
| --- | --- | --- |
| all-nonlinear_before_distill | hetune_generated | 0.875000 |
| all-nonlinear_after_distill | hetune_generated_distilled | 0.886719 |
| all-nonlinear_delta | after-before | +0.011719 |

## Metrics

| schedule | accuracy | profile_candidates_loaded | profile_entries | static_fallback_entries | profile_coverage_rate | used_candidates_with_profile | used_candidates_missing_profile | used_candidate_ids_missing_profile | strict_profile_check_passed | strict_profile_check_reason | latency_ms | rotations | ct_ct_mults | ct_pt_mults | rescale_count | relin_count | depth | bootstrap_count | memory_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0.9140625 | 11 | 0 | 0 | 1.0 | 0 | 0 |  | True |  | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| hetune_generated | 0.875 | 11 | 24 | 0 | 1.0 | 5 | 0 |  | True |  | 2715.966 | 544 | 455 | 385 | 281 | 84 | 113 | 21 | 5376.0 |
| hetune_generated_distilled | 0.88671875 | 11 | 24 | 0 | 1.0 | 5 | 0 |  | True |  | 2715.966 | 544 | 455 | 385 | 281 | 84 | 113 | 21 | 5376.0 |
| uniform_low | 0.51171875 | 11 | 24 | 0 | 1.0 | 3 | 0 |  | True |  | 629.28 | 126 | 104 | 108 | 64 | 20 | 24 | 5 | 1280.0 |
| uniform_mid | 0.51171875 | 11 | 24 | 0 | 1.0 | 3 | 0 |  | True |  | 1016.496 | 210 | 176 | 162 | 112 | 32 | 48 | 8 | 2048.0 |
| uniform_high | 0.87890625 | 11 | 24 | 0 | 1.0 | 3 | 0 |  | True |  | 2978.88 | 594 | 500 | 426 | 310 | 92 | 126 | 23 | 5888.0 |

## Validated Greedy Decisions

- Accepted downgrades: `3`
- Rejected downgrades: `63`

Top rejected candidates:

| operator_id | to_candidate_id | reason | combined_accuracy_drop | combined_logit_kl | combined_label_flip_rate |
| --- | --- | --- | --- | --- | --- |
| distilbert_sst2.layer2.layernorm.ffn_layernorm | layernorm.centered.mid_cost.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.5 | 1.1177252531051636 | 0.5078125 |
| distilbert_sst2.layer1.layernorm.ffn_layernorm | layernorm.centered.mid_cost.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.3984375 | 0.6389824151992798 | 0.40625 |
| distilbert_sst2.layer0.gelu.ffn_activation | gelu.chebyshev.degree5.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.1328125 | 0.353337287902832 | 0.140625 |
| distilbert_sst2.layer0.layernorm.ffn_layernorm | layernorm.centered.mid_cost.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.1171875 | 0.3358473479747772 | 0.125 |
| distilbert_sst2.layer1.layernorm.attention_layernorm | layernorm.centered.mid_cost.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.06640625 | 0.2372560054063797 | 0.08203125 |
| distilbert_sst2.layer3.softmax.attention_softmax | softmax.power.degree2.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.02734375 | 0.1308044195175171 | 0.04296875 |
| distilbert_sst2.layer1.softmax.attention_softmax | softmax.power.degree2.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.0234375 | 0.0794122442603111 | 0.0390625 |
| distilbert_sst2.layer4.layernorm.ffn_layernorm | layernorm.centered.mid_cost.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.0234375 | 0.0941793620586395 | 0.0390625 |

## Combination Diagnostics

| diagnostic | accuracy | accuracy_drop | logit_kl | label_flip_rate | downgraded_operator_count |
| --- | --- | --- | --- | --- | --- |
| layer_1_low | 0.4921875 | 0.484375 | 1.1751807928085327 | 0.4921875 | 4 |
| layers_2_3_low | 0.4921875 | 0.484375 | 2.284071922302246 | 0.4921875 | 8 |
| layers_1_2_low | 0.4921875 | 0.484375 | 0.8378996253013611 | 0.4921875 | 8 |
| layers_0_1_low | 0.4921875 | 0.484375 | 1.085676193237305 | 0.4921875 | 8 |
| layer_2_low | 0.6875 | 0.2890625 | 0.7614694833755493 | 0.296875 | 4 |
| layers_3_4_low | 0.7421875 | 0.234375 | 0.657001793384552 | 0.2421875 | 8 |
| layer_3_low | 0.8203125 | 0.15625 | 0.4629559516906738 | 0.1640625 | 4 |
| layer_0_low | 0.83203125 | 0.14453125 | 0.4096512794494629 | 0.14453125 | 4 |

## Next-Stage Improvements

- Import SEAL/OpenFHE microbenchmark results and calibrate static costs.
- Add BERT-Mini, DistilBERT, MRPC, and MNLI subset configs.
- Add lower-cost Softmax candidates and isolate their accuracy risk.
- Add bootstrapping placement constraints to the greedy scheduler.
- Run ablations that disable sensitivity, cost model, and layer-wise choices.
