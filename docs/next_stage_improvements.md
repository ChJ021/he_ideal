# Next-Stage Improvements

After the MVP experiment finishes, record the actual gaps and push the next
stage in this order:

1. CKKS microbenchmark import
   - Add SEAL/OpenFHE benchmark result schemas.
   - Measure degree, scale, depth, rotation, rescale, relinearization, latency.
   - Replace static cost hints with calibrated backend-specific costs.

2. Model and dataset expansion
   - Add BERT-Mini and DistilBERT configs as first-class experiments.
   - Add MRPC and MNLI subset loaders.
   - Report F1 for MRPC and matched/mismatched accuracy for MNLI.

3. Softmax approximation
   - Add low-order polynomial Softmax, power attention, or kernel attention.
   - Evaluate Softmax in isolation before enabling it in greedy search.
   - Keep a conservative high-precision fallback in all schedules.

4. Bootstrapping placement
   - Track cumulative level budget across schedule segments.
   - Add bootstrap placement to the greedy policy.
   - Report bootstrap count and estimated latency separately.

5. Ablations and robustness
   - Disable sensitivity scoring, cost modeling, and layer-wise choices one at a time.
   - Vary calibration size, sequence length bucket, and accuracy tolerance.
   - Save all configs, random seeds, package versions, and git status.
   - Compare additive greedy against validated greedy to quantify interaction risk.

6. Visualization
   - Add Pareto frontier plots.
   - Add cost breakdown bars for rotation, multiplication, rescale, and bootstrap.
   - Add per-layer schedule color strips.

7. Engineering hardening
   - Cache baseline logits and tokenized datasets.
   - Add resume support when profile CSV already exists.
   - Add a true backend adapter interface for SEAL/OpenFHE operator execution.
