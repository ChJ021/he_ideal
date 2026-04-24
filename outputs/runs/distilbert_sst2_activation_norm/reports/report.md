# HETune-LLM Report: distilbert_sst2_activation_norm

## Summary

- Operator scope: `activation_norm`
- Optimized operator types: `gelu, layernorm`
- Final policy: `hetune_validated_greedy`
- Schedule entries: `18`
- Estimated latency: `129.400`
- Estimated depth sum: `92`
- Estimated rotations: `22`
- Base reference accuracy: `0.914062`

Schedule entries by type:

| operator_type | entry_count |
| --- | --- |
| gelu | 6 |
| layernorm | 12 |

Calibration stats:

- Stats file: `/home/cj/he_ideal/hetuned_llm/outputs/runs/distilbert_sst2_activation_norm/profiles/operator_calibration_stats.csv`
- Covered operators: `18`
- Tracked operators: `18`
- Missing operators: `0`

## Metrics

| schedule | accuracy | latency_ms | rotations | ct_ct_mults | ct_pt_mults | rescale_count | relin_count | depth | bootstrap_count | memory_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0.9140625 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| hetune_generated | 0.875 | 129.4 | 22 | 98 | 116 | 92 | 0 | 92 | 0 | 0.0 |
| uniform_low | 0.48828125 | 9.600000000000003 | 0 | 6 | 24 | 6 | 0 | 6 | 0 | 0.0 |
| uniform_mid | 0.48828125 | 23.399999999999995 | 12 | 24 | 42 | 24 | 0 | 24 | 0 | 0.0 |
| uniform_high | 0.84765625 | 144.0 | 24 | 108 | 126 | 102 | 0 | 102 | 0 | 0.0 |

## Validated Greedy Decisions

- Accepted downgrades: `2`
- Rejected downgrades: `52`

Top rejected candidates:

| operator_id | to_candidate_id | reason | combined_accuracy_drop | combined_logit_kl | combined_label_flip_rate |
| --- | --- | --- | --- | --- | --- |
| distilbert_sst2.layer0.layernorm.ffn_layernorm | layernorm.centered.mid_cost.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.5 | 2.0940756797790527 | 0.5078125 |
| distilbert_sst2.layer3.gelu.ffn_activation | gelu.poly.degree5.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.5 | 1.115190505981445 | 0.5078125 |
| distilbert_sst2.layer1.layernorm.ffn_layernorm | layernorm.centered.mid_cost.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.5 | 1.6766090393066406 | 0.5078125 |
| distilbert_sst2.layer3.gelu.ffn_activation | gelu.poly.degree2.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.5 | 1.040092706680298 | 0.5078125 |
| distilbert_sst2.layer0.gelu.ffn_activation | gelu.poly.degree3.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.5 | 1.8474500179290767 | 0.5078125 |
| distilbert_sst2.layer0.gelu.ffn_activation | gelu.poly.degree2.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.5 | 0.7582893371582031 | 0.5078125 |
| distilbert_sst2.layer1.gelu.ffn_activation | gelu.poly.degree2.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.5 | 0.8889484405517578 | 0.5078125 |
| distilbert_sst2.layer1.gelu.ffn_activation | gelu.poly.degree5.v1 | violates_max_accuracy_drop+max_logit_kl+max_label_flip_rate | 0.5 | 10.37446403503418 | 0.5078125 |

## Combination Diagnostics

| diagnostic | accuracy | accuracy_drop | logit_kl | label_flip_rate | downgraded_operator_count |
| --- | --- | --- | --- | --- | --- |
| layers_1_2_low | 0.41015625 | 0.55859375 | 11.809460639953612 | 0.57421875 | 6 |
| layers_2_3_low | 0.4609375 | 0.5078125 | 8.807790756225586 | 0.515625 | 6 |
| layer_0_low | 0.4921875 | 0.4765625 | 0.9831048250198364 | 0.4765625 | 3 |
| layer_1_low | 0.4921875 | 0.4765625 | 1.1814225912094116 | 0.4765625 | 3 |
| layers_3_4_low | 0.4921875 | 0.4765625 | 9.787178993225098 | 0.4765625 | 6 |
| layer_2_low | 0.4921875 | 0.4765625 | 0.6927176117897034 | 0.4765625 | 3 |
| layer_3_low | 0.4921875 | 0.4765625 | 0.885296106338501 | 0.4765625 | 3 |
| layers_0_1_low | 0.4921875 | 0.4765625 | 2.6069109439849854 | 0.4765625 | 6 |

## Next-Stage Improvements

- Import SEAL/OpenFHE microbenchmark results and calibrate static costs.
- Add BERT-Mini, DistilBERT, MRPC, and MNLI subset configs.
- Add lower-cost Softmax candidates and isolate their accuracy risk.
- Add bootstrapping placement constraints to the greedy scheduler.
- Run ablations that disable sensitivity, cost model, and layer-wise choices.
