# HE Deployment Report: distilbert_sst2_all_nonlinear_he

- Deployment id: `distilbert_sst2_openfhe`
- Backend available: `True`
- Backend status: `available`
- Privacy boundary: `client_embedding`
- Sequence length: `128`
- Sample size: `16`
- Runner mode: `openfhe_distilbert_forward`
- Encrypted sequence length: `16`
- Encrypted sample size: `1`
- Plaintext accuracy fallback allowed: `False`
- Comparison: `/home/cj/he_ideal/hetuned_llm/outputs/runs/distilbert_sst2_all_nonlinear_he/he_deployment/comparison.csv`

| Case | Schedule | Feasible | Accuracy | Latency ms | Error |
|---|---|---:|---:|---:|---|
| high | uniform_high | False |  |  | OpenFHE runner failed for high:  |
| pre_distill | hetune_generated | False |  |  | OpenFHE runner failed for pre_distill:  |
| post_distill | hetune_generated | False |  |  | OpenFHE runner failed for post_distill:  |
