# Recurrent Architectures

This package contains recurrent-capable network cores and baseline RNN wrappers.

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
- aliases: `unified_ei`, `ei_net`, `unified_einet`

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
