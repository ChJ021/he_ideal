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

HE deployment:

- Status: `not_enabled`

Distillation:

- Summary: `/home/cj/he_ideal/hetuned_llm/outputs/runs/distilbert_sst2_activation_norm/distillation/summary.csv`
- Report: `/home/cj/he_ideal/hetuned_llm/outputs/runs/distilbert_sst2_activation_norm/distillation/report.md`
- Overrides: `/home/cj/he_ideal/hetuned_llm/outputs/runs/distilbert_sst2_activation_norm/distillation/overrides.pt`
- Status: `available`

activation-norm accuracy before/after distillation:

| variant | schedule | accuracy |
| --- | --- | --- |
| activation-norm_before_distill | hetune_generated | 0.875000 |
| activation-norm_after_distill | hetune_generated_distilled | 0.894531 |
| activation-norm_delta | after-before | +0.019531 |

## Metrics

| schedule | accuracy | latency_ms | rotations | ct_ct_mults | ct_pt_mults | rescale_count | relin_count | depth | bootstrap_count | memory_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0.9140625 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| hetune_generated | 0.875 | 129.4 | 22 | 98 | 116 | 92 | 0 | 92 | 0 | 0.0 |
| hetune_generated_distilled | 0.89453125 | 129.4 | 22 | 98 | 116 | 92 | 0 | 92 | 0 | 0.0 |
| uniform_low | 0.51171875 | 20.4 | 0 | 18 | 36 | 18 | 0 | 18 | 0 | 0.0 |
| uniform_mid | 0.51171875 | 40.800000000000004 | 12 | 36 | 54 | 36 | 0 | 36 | 0 | 0.0 |
| uniform_high | 0.84765625 | 144.0 | 24 | 108 | 126 | 102 | 0 | 102 | 0 | 0.0 |

## Validated Greedy Decisions

- Accepted downgrades: `2`
- Rejected downgrades: `52`

Top rejected candidates:

| operator_id | to_candidate_id | reason | combined_accuracy_drop | combined_logit_kl | combined_label_flip_rate |
| --- | --- | --- | --- | --- | --- |
| distilbert_sst2.layer0.layernorm.ffn_layernorm | layernorm.centered.mid_cost.v1 | violates_max_logit_kl+max_label_flip_rate | 0.5 | 2.0940756797790527 | 0.5078125 |
| distilbert_sst2.layer1.layernorm.ffn_layernorm | layernorm.centered.mid_cost.v1 | violates_max_logit_kl+max_label_flip_rate | 0.5 | 1.6766090393066406 | 0.5078125 |
| distilbert_sst2.layer1.layernorm.attention_layernorm | layernorm.centered.mid_cost.v1 | violates_max_logit_kl+max_label_flip_rate | 0.26953125 | 0.8792555928230286 | 0.27734375 |
| distilbert_sst2.layer3.layernorm.ffn_layernorm | layernorm.centered.mid_cost.v1 | violates_max_logit_kl+max_label_flip_rate | 0.0859375 | 0.2466151714324951 | 0.09375 |
| distilbert_sst2.layer1.layernorm.ffn_layernorm | layernorm.newton.low_iter.v1 | violates_max_logit_kl+max_label_flip_rate | 0.05078125 | 0.110220581293106 | 0.05859375 |
| distilbert_sst2.layer4.gelu.ffn_activation | gelu.chebyshev.degree5.v1 | violates_max_logit_kl+max_label_flip_rate | 0.02734375 | 0.1311751753091812 | 0.04296875 |
| distilbert_sst2.layer5.gelu.ffn_activation | gelu.chebyshev.degree11.v1 | violates_max_logit_kl+max_label_flip_rate | 0.0234375 | 0.1237257495522499 | 0.0390625 |
| distilbert_sst2.layer5.gelu.ffn_activation | gelu.chebyshev.degree9.v1 | violates_max_logit_kl+max_label_flip_rate | 0.0234375 | 0.1236826479434967 | 0.0390625 |

## Combination Diagnostics

| diagnostic | accuracy | accuracy_drop | logit_kl | label_flip_rate | downgraded_operator_count |
| --- | --- | --- | --- | --- | --- |
| layer_1_low | 0.4921875 | 0.4765625 | 1.189694046974182 | 0.4765625 | 3 |
| layers_2_3_low | 0.4921875 | 0.4765625 | 2.060921192169189 | 0.4765625 | 6 |
| layers_1_2_low | 0.4921875 | 0.4765625 | 0.8898296356201172 | 0.4765625 | 6 |
| layers_0_1_low | 0.4921875 | 0.4765625 | 1.1356278657913208 | 0.4765625 | 6 |
| layer_2_low | 0.61328125 | 0.35546875 | 0.6745226383209229 | 0.36328125 | 3 |
| layer_0_low | 0.69140625 | 0.27734375 | 0.6373891234397888 | 0.28515625 | 3 |
| layer_3_low | 0.7109375 | 0.2578125 | 0.8416025638580322 | 0.2578125 | 3 |
| layers_3_4_low | 0.73046875 | 0.23828125 | 0.70766282081604 | 0.23828125 | 6 |

## Next-Stage Improvements

- Import SEAL/OpenFHE microbenchmark results and calibrate static costs.
- Add BERT-Mini, DistilBERT, MRPC, and MNLI subset configs.
- Add lower-cost Softmax candidates and isolate their accuracy risk.
- Add bootstrapping placement constraints to the greedy scheduler.
- Run ablations that disable sensitivity, cost model, and layer-wise choices.
