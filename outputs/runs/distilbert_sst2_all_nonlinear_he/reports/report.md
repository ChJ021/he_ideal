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
| uniform_low | 0.48828125 | 11 | 24 | 0 | 1.0 | 3 | 0 |  | True |  | 139.152 | 30 | 28 | 48 | 20 | 4 | 12 | 1 | 256.0 |
| uniform_mid | 0.48828125 | 11 | 24 | 0 | 1.0 | 3 | 0 |  | True |  | 640.356 | 138 | 116 | 114 | 76 | 20 | 36 | 5 | 1280.0 |
| uniform_high | 0.87890625 | 11 | 24 | 0 | 1.0 | 3 | 0 |  | True |  | 2978.88 | 594 | 500 | 426 | 310 | 92 | 126 | 23 | 5888.0 |

## Validated Greedy Decisions

- Accepted downgrades: `3`
- Rejected downgrades: `63`

Top rejected candidates:

| operator_id | to_candidate_id | reason | combined_accuracy_drop | combined_logit_kl | combined_label_flip_rate |
| --- | --- | --- | --- | --- | --- |
| distilbert_sst2.layer1.gelu.ffn_activation | gelu.poly.degree2.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.53125 | 1.7962393760681152 | 0.5390625 |
| distilbert_sst2.layer2.layernorm.ffn_layernorm | layernorm.centered.mid_cost.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.5 | 1.1177252531051636 | 0.5078125 |
| distilbert_sst2.layer3.gelu.ffn_activation | gelu.poly.degree2.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.5 | 0.98957622051239 | 0.5 |
| distilbert_sst2.layer2.gelu.ffn_activation | gelu.poly.degree2.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.48828125 | 1.2349497079849243 | 0.49609375 |
| distilbert_sst2.layer0.gelu.ffn_activation | gelu.poly.degree5.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.48828125 | 0.8144989609718323 | 0.49609375 |
| distilbert_sst2.layer1.gelu.ffn_activation | gelu.poly.degree5.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.484375 | 1.1745336055755615 | 0.4921875 |
| distilbert_sst2.layer1.gelu.ffn_activation | gelu.poly.degree3.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.48046875 | 0.8526496291160583 | 0.48828125 |
| distilbert_sst2.layer0.gelu.ffn_activation | gelu.poly.degree2.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.48046875 | 0.9455373883247375 | 0.48828125 |

## Combination Diagnostics

| diagnostic | accuracy | accuracy_drop | logit_kl | label_flip_rate | downgraded_operator_count |
| --- | --- | --- | --- | --- | --- |
| layers_2_3_low | 0.44921875 | 0.52734375 | 5.836549282073975 | 0.55078125 | 8 |
| layer_0_low | 0.4921875 | 0.484375 | 0.8437407612800598 | 0.4921875 | 4 |
| layer_2_low | 0.4921875 | 0.484375 | 0.7238026857376099 | 0.4921875 | 4 |
| layer_1_low | 0.4921875 | 0.484375 | 1.2469854354858398 | 0.4921875 | 4 |
| layers_3_4_low | 0.4921875 | 0.484375 | 10.121749877929688 | 0.4921875 | 8 |
| layer_3_low | 0.4921875 | 0.484375 | 0.9212669134140016 | 0.4921875 | 4 |
| layers_0_1_low | 0.4921875 | 0.484375 | 2.067066192626953 | 0.4921875 | 8 |
| layers_1_2_low | 0.5390625 | 0.4375 | 9.623184204101562 | 0.46875 | 8 |

## Next-Stage Improvements

- Import SEAL/OpenFHE microbenchmark results and calibrate static costs.
- Add BERT-Mini, DistilBERT, MRPC, and MNLI subset configs.
- Add lower-cost Softmax candidates and isolate their accuracy risk.
- Add bootstrapping placement constraints to the greedy scheduler.
- Run ablations that disable sensitivity, cost model, and layer-wise choices.
