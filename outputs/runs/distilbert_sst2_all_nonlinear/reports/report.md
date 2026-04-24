# HETune-LLM Report: distilbert_sst2_all_nonlinear

## Summary

- Operator scope: `all_nonlinear`
- Optimized operator types: `gelu, layernorm, softmax`
- Final policy: `hetune_validated_greedy`
- Schedule entries: `24`
- Estimated latency: `176.200`
- Estimated depth sum: `118`
- Estimated rotations: `40`
- Base reference accuracy: `0.914062`

Schedule entries by type:

| operator_type | entry_count |
| --- | --- |
| gelu | 6 |
| layernorm | 12 |
| softmax | 6 |

Calibration stats:

- Stats file: `/home/cj/he_ideal/hetuned_llm/outputs/runs/distilbert_sst2_all_nonlinear/profiles/operator_calibration_stats.csv`
- Covered operators: `18`
- Tracked operators: `18`
- Missing operators: `0`

Distillation:

- Summary: `/home/cj/he_ideal/hetuned_llm/outputs/runs/distilbert_sst2_all_nonlinear/distillation/summary.csv`
- Report: `/home/cj/he_ideal/hetuned_llm/outputs/runs/distilbert_sst2_all_nonlinear/distillation/report.md`
- Overrides: `/home/cj/he_ideal/hetuned_llm/outputs/runs/distilbert_sst2_all_nonlinear/distillation/overrides.pt`
- Status: `available`

all-nonlinear accuracy before/after distillation:

| variant | schedule | accuracy |
| --- | --- | --- |
| all-nonlinear_before_distill | hetune_generated | 0.875000 |
| all-nonlinear_after_distill | hetune_generated_distilled | 0.886719 |
| all-nonlinear_delta | after-before | +0.011719 |

## Metrics

| schedule | accuracy | latency_ms | rotations | ct_ct_mults | ct_pt_mults | rescale_count | relin_count | depth | bootstrap_count | memory_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0.9140625 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| hetune_generated | 0.875 | 176.2 | 40 | 124 | 134 | 118 | 0 | 118 | 0 | 0.0 |
| hetune_generated_distilled | 0.88671875 | 176.2 | 40 | 124 | 134 | 118 | 0 | 118 | 0 | 0.0 |
| uniform_low | 0.48828125 | 16.8 | 6 | 12 | 36 | 12 | 0 | 12 | 0 | 0.0 |
| uniform_mid | 0.48828125 | 35.400000000000006 | 18 | 36 | 54 | 36 | 0 | 36 | 0 | 0.0 |
| uniform_high | 0.87890625 | 192.0 | 42 | 132 | 150 | 126 | 0 | 126 | 0 | 0.0 |

## Validated Greedy Decisions

- Accepted downgrades: `3`
- Rejected downgrades: `69`

Top rejected candidates:

| operator_id | to_candidate_id | reason | combined_accuracy_drop | combined_logit_kl | combined_label_flip_rate |
| --- | --- | --- | --- | --- | --- |
| distilbert_sst2.layer1.gelu.ffn_activation | gelu.poly.degree2.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.51171875 | 1.1932690143585205 | 0.51953125 |
| distilbert_sst2.layer2.layernorm.ffn_layernorm | layernorm.centered.mid_cost.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.5 | 1.2792794704437256 | 0.5078125 |
| distilbert_sst2.layer3.gelu.ffn_activation | gelu.poly.degree2.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.4921875 | 0.9794045686721802 | 0.5 |
| distilbert_sst2.layer0.gelu.ffn_activation | gelu.poly.degree2.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.484375 | 0.7847380042076111 | 0.4921875 |
| distilbert_sst2.layer2.gelu.ffn_activation | gelu.poly.degree2.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.484375 | 0.9364246129989624 | 0.4921875 |
| distilbert_sst2.layer1.gelu.ffn_activation | gelu.poly.degree5.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.484375 | 0.882723867893219 | 0.4921875 |
| distilbert_sst2.layer0.gelu.ffn_activation | gelu.poly.degree5.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.484375 | 0.7944474220275879 | 0.4921875 |
| distilbert_sst2.layer2.gelu.ffn_activation | gelu.poly.degree5.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.48046875 | 0.8553111553192139 | 0.48828125 |

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
