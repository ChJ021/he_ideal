# Schedule Format

Schedules are YAML files with three top-level fields:

```yaml
metadata:
  experiment_id: distilbert_sst2_activation_norm
  model_id: bert_tiny
  dataset_id: sst2
  operator_scope: activation_norm
  operator_types:
    - gelu
    - layernorm
  policy: hetune_greedy
entries:
  - operator_key:
      model_id: bert_tiny
      layer_index: 0
      operator_type: gelu
      name: ffn_activation
      path: bert.encoder.layer.0.intermediate.intermediate_act_fn
    candidate_id: gelu.chebyshev.degree9.v1
    ckks_param_id: static_ckks_128
    scale_id: scale_2_40
    level_budget: 0
    bootstrap_policy: none
    layout_id: plaintext_sim
constraints:
  max_accuracy_drop: 0.01
  input_independent: true
  min_security_bits: 128
```

To change a layer's approximation manually, edit only `candidate_id` in the
matching schedule entry.

`*.base` candidates are plaintext references that call the original PyTorch
function or module. They are registered for comparison and manual debugging but
are excluded from automatic schedule search. `*.exact.high.v1` candidates are
the highest-quality approximation options used by HETune schedules, not
mathematical exact HE implementations.
