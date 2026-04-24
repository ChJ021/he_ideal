# HETune-LLM Report: distilbert_sst2_softmax

## Summary

- Operator scope: `softmax_only`
- Optimized operator types: `softmax`
- Final policy: `hetune_validated_greedy`
- Schedule entries: `6`
- Estimated latency: `7.200`
- Estimated depth sum: `6`
- Estimated rotations: `6`
- Base reference accuracy: `0.914062`

Schedule entries by type:

| operator_type | entry_count |
| --- | --- |
| softmax | 6 |

Calibration stats:

- Stats file: `/home/cj/he_ideal/hetuned_llm/outputs/runs/distilbert_sst2_softmax/profiles/operator_calibration_stats.csv`
- Covered operators: `0`
- Tracked operators: `0`
- Missing operators: `0`

## Metrics

| schedule | accuracy | latency_ms | rotations | ct_ct_mults | ct_pt_mults | rescale_count | relin_count | depth | bootstrap_count | memory_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0.9140625 | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| hetune_generated | 0.8984375 | 7.2 | 6 | 6 | 12 | 6 | 0 | 6 | 0 | 0.0 |
| uniform_low | 0.8984375 | 7.2 | 6 | 6 | 12 | 6 | 0 | 6 | 0 | 0.0 |
| uniform_mid | 0.90625 | 12.0 | 6 | 12 | 12 | 12 | 0 | 12 | 0 | 0.0 |
| uniform_high | 0.90234375 | 48.0 | 18 | 24 | 24 | 24 | 0 | 24 | 0 | 0.0 |

## Validated Greedy Decisions

- Accepted downgrades: `15`
- Rejected downgrades: `3`

Top rejected candidates:

| operator_id | to_candidate_id | reason | combined_accuracy_drop | combined_logit_kl | combined_label_flip_rate |
| --- | --- | --- | --- | --- | --- |
| distilbert_sst2.layer2.softmax.attention_softmax | softmax.clipped.stable.v1 | not_a_downgrade | nan | nan | nan |
| distilbert_sst2.layer1.softmax.attention_softmax | softmax.clipped.stable.v1 | not_a_downgrade | nan | nan | nan |
| distilbert_sst2.layer3.softmax.attention_softmax | softmax.clipped.stable.v1 | not_a_downgrade | nan | nan | nan |

## Combination Diagnostics

| diagnostic | accuracy | accuracy_drop | logit_kl | label_flip_rate | downgraded_operator_count |
| --- | --- | --- | --- | --- | --- |
| layer_5_low | 0.98828125 | 0.00390625 | 0.0016523351659998 | 0.00390625 | 1 |
| baseline_high | 0.9921875 | 0.0 | 0.0 | 0.0 | 0 |
| layer_1_low | 0.9921875 | 0.0 | 0.0036608020309358 | 0.0 | 1 |
| layer_0_low | 0.9921875 | 0.0 | 0.0004039507475681 | 0.0 | 1 |
| layer_2_low | 0.9921875 | 0.0 | 0.0018265941180288 | 0.0 | 1 |
| layer_3_low | 0.9921875 | 0.0 | 0.0034235001076012 | 0.0 | 1 |
| layer_4_low | 0.9921875 | 0.0 | 0.0021927247289568 | 0.0 | 1 |
| layers_0_1_low | 0.9921875 | 0.0 | 0.0035352779086679 | 0.0 | 2 |

## Next-Stage Improvements

- Import SEAL/OpenFHE microbenchmark results and calibrate static costs.
- Add BERT-Mini, DistilBERT, MRPC, and MNLI subset configs.
- Add lower-cost Softmax candidates and isolate their accuracy risk.
- Add bootstrapping placement constraints to the greedy scheduler.
- Run ablations that disable sensitivity, cost model, and layer-wise choices.
