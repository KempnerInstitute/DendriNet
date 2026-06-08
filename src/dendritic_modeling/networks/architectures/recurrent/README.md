# Recurrent Architectures

This package contains recurrent-capable network cores and baseline RNN wrappers.
The recurrent unified-EI path is intentionally built on top of the same
``DendriticBranchLayer`` and ``initialize.py`` logic used by the feedforward
EI stack, so synapse semantics, reactivation policies, and initialization
rules stay aligned unless recurrence genuinely requires something extra
(temporal traces, recurrent synapse banks, or sequence reduction).

## Modules

- `ei_config.py`: dataclasses for unified E-I recurrent configuration
- `ei_layer.py`: one recurrent-capable E-I layer
- `ei_network.py`: stacked unified E-I network
- `ei_state.py`: recurrent state containers
- `stateful_dendrinet.py`: dendritic state and trace logic
- `baseline_rnn.py`: GRU, LSTM, and vanilla-RNN wrappers with the same
  interface shape as the unified E-I core

## Supported `model.core.type` Values

Unified E-I:

- `ei_unified`
- `unified_ei`
- recurrent-default aliases: `rnn_dendritic_shunting`, `rnn_dendritic_additive`,
  `rnn_flat_shunting`, `rnn_flat_additive`

Baseline recurrent models:

- `gru`
- `lstm`
- `vanilla_rnn`
- alias: `rnn`

## Training Path

Use the recurrent trainer for sequence experiments:

```yaml
model:
  core:
    type: ei_unified
training:
  main:
    strategy: recurrent
```

Unified E-I configuration lives under `model.core.unified_ei`.

## Notes

- Feedforward E-I models remain under
  `networks/architectures/excitation_inhibition/`.
- Output reduction is controlled by the recurrent core config
  (`last`, `mean`, or `all`).
- Structured EI defaults stay on the smoother positive transform (`softplus`);
  biology-facing recurrent runs can opt into `exp` explicitly when needed.
- Recurrent additive aliases (`rnn_dendritic_additive`,
  `rnn_flat_additive`) enable additive voltage normalization and use a fixed
  reactivation gate (`m=1.0`, `b=0.0`) so normalized additive voltages keep
  useful gate contrast without the non-leaf analytical-init saturation trap.
- For paper-style recurrent runs, a common recipe is `weight_transform: exp`,
  `allow_self_recurrence: false`, and review of the post-build
  `init_gate_stats.json` artifact when using data-driven gate initialization.
