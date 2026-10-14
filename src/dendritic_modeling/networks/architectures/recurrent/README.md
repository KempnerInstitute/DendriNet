# Recurrent architectures

This package holds the population networks, the unified E/I networks, the
recurrent form of the structured E/I cores and the recurrent baselines. The
recurrent dendritic populations reuse the `DendriticBranchLayer` and
initialization code of the feedforward E/I stack
(`networks/architectures/excitation_inhibition/dendritic`), so synapses, gates
and initialization behave the same with and without recurrence; recurrence
adds temporal traces, recurrent synapses and sequence handling.

## Modules

- Population networks: `population_network.py` (the network and its sequence
  state), `population_layer.py` (named populations and directed connections
  of one layer), `population_configs.py`, `population_common.py`,
  `population_sources.py`, `population_state.py`, `population_constants.py`.
- Stateful dendritic populations: `stateful_dendrinet.py` (branch layers and
  the per-step update) with `stateful_layers.py`, `stateful_setup.py`,
  `stateful_step.py`, `stateful_feedback.py` and `stateful_typed_dynamics.py`
  (receptor traces, short-term plasticity, typed currents).
- Population options and dynamics: `ei_config.py` (`PopulationConfig`,
  `EILayerConfig`, `EINetworkConfig`), `synapse_types.py`, `spiking_soma.py`,
  `dendritic_spikes.py`, `dendritic_geometry.py`.
- Unified E/I networks: `ei_layer.py`, `ei_network.py`, `ei_state.py`.
- Recurrent structured E/I cores: `structured_einet_factory.py` and the
  `structured_*` builders, which translate the structured E/I sections into a
  recurrent network.
- Baselines: `baseline_rnn.py` (GRU, LSTM, vanilla RNN),
  `heterogeneous_leak_ctrnn.py`, `legendre_memory.py`.
- `protocol.py`: the interfaces trainers and analysis tools use for recurrent
  cores.

## Core types built here

- `population_network`
- `unified_ei` (alias `ei_unified`; `ei_net` and `unified_einet` build it
  with a deprecation warning) and the aliases `rnn_dendritic_shunting`,
  `rnn_dendritic_additive`, `rnn_dendritic_normalized_additive`,
  `rnn_flat_shunting`, `rnn_flat_additive`, `rnn_flat_normalized_additive`
- `gru`, `lstm`, `vanilla_rnn` (alias `rnn`)
- `heterogeneous_leak_ctrnn` (alias `heterogeneous_ctrnn`)
- `legendre_memory`
- structured E/I types (`einet`, `dendritic_*`, `flat_*`) with recurrent
  synapse counts or temporal options, through `structured_einet_factory.py`

Sequence models train with `training.main.strategy: recurrent`. The user
documentation is in `docs/population_networks.rst`,
`docs/structured_ei_cores.rst`, `docs/temporal_dynamics.rst` and
`docs/comparison_models.rst`; `docs/available_components.rst` lists every
core type.

## Notes

- Population blocks (`population_network` and `unified_ei` populations)
  default to `weight_transform: exp`; the structured E/I sections default to
  `softplus`. Set `weight_transform` explicitly to use the same
  parameterization in both.
- `output_mode` (`last`, `mean` or `all`) chooses which time steps a
  recurrent core returns.
- `rnn_dendritic_additive` and `rnn_flat_additive` integrate raw additively
  (`V = E - I`); the `*_normalized_additive` aliases add voltage
  normalization. The aliases set no gate initialization; set
  `reactivation_init_policy: occupancy_quantile` on the populations for gates
  calibrated on data.
- `allow_self_recurrence: false` removes each population's previous-step
  input to itself and keeps the recurrence between populations.
- Every training run writes `init_gate_stats.json`; check it when gates are
  initialized from data.
