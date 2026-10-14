# Operator presets

Five points between a signed linear layer and a shunting excitatory/inhibitory
(E/I) dendritic tree. Each file is the body of a `model.core` section for the
structured `einet` core: a layer of 32 excitatory units on 64 inputs, without
the `type` key. To train one, put its contents under `model.core` with
`type: einet`:

```yaml
model:
  core:
    type: einet
    # contents of one preset file
```

`tests/test_operator_presets.py` builds every preset with
`ConfigurableEINetwork(config=..., input_dim=64)` and runs a forward pass.

| Preset | Weights | Inhibition | Branch factors | Integration | Gates (initialization) |
|---|---|---|---|---|---|
| `00_mlp_flat` | signed | none | `[1]` | additive | none |
| `01_signed_sparse_tree` | signed | none | `[2, 2]` | additive | none |
| `02_positive_ei_flat` | positive (`softplus`) | 8 inhibitory cells | `[1]` | additive | `param_tanh` (`occupancy_quantile`) |
| `03_positive_ei_tree` | positive (`softplus`) | 8 inhibitory cells | `[2, 2]` | additive | `param_tanh` (`occupancy_quantile`) |
| `04_shunting_tree` | positive (`softplus`) | 8 inhibitory cells | `[2, 2]` | shunting | `param_tanh` (`analytical`, the default) |

The synapse counts per branch differ between the flat and the tree presets.
Adjacent presets differ as follows:

- `00` to `01`: the branch factors.
- `01` to `02`: the weight sign, the inhibitory cells, the branch factors, the
  gates and a ReLU on the core output.
- `02` to `03`: the branch factors.
- `03` to `04`: the integration rule, together with the gate initialization
  policy (`04` sets none, so it uses `analytical`).

The positive presets (`02` to `04`) declare `biological_neuron: true`, so
building them also checks the biological constraints (nonnegative weights,
separate excitation and inhibition, a rectified input and nonnegative
outputs). With a branch-to-soma coupling of one, the operator of `00_mlp_flat`
(signed weights, one compartment, additive integration, no gates) is a
bias-free linear layer
(`tests/test_operator_extensions.py::test_signed_flat_dendrinet_equals_linear_layer`).
