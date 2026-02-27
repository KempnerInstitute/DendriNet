# Recurrent Architectures

This package provides the unified E-I network architecture (`EINetwork`) supporting both
feedforward and recurrent modes via the same class hierarchy.

## Components

- `PopulationConfig` / `EILayerConfig` / `EINetworkConfig` — dataclass configs
- `StatefulDendriNet` — single-population dendritic module (FF `forward()` + recurrent `step()`)
- `EILayer` — one E-I layer with optional I population and optional recurrence
- `EINetwork` — multi-layer stack with time-unrolling and output mode selection
- `DendriNetState` / `EIState` — per-level trace containers for recurrent mode

## Usage

### Feedforward mode

```yaml
model:
  core:
    type: ei_unified
    unified_ei:
      layers:
        - recurrent: false
          dt: 1.0
          excitatory:
            n_neurons: 64
            branch_factors: [3, 3]
            ff_excitatory_synapses: 12
            ff_inhibitory_synapses: 6
          inhibitory:
            n_neurons: 16
            branch_factors: [1]
            ff_excitatory_synapses: 8
            ff_inhibitory_synapses: 4
```

### Recurrent mode

Set `recurrent: true` per layer and add recurrent synapse counts:

```yaml
model:
  core:
    type: ei_unified
    unified_ei:
      output_mode: all   # last | all | mean
      layers:
        - recurrent: true
          dt: 1.0
          excitatory:
            n_neurons: 64
            branch_factors: [3, 3]
            ff_excitatory_synapses: 12
            ff_inhibitory_synapses: 6
            rec_excitatory_synapses: 8
            rec_inhibitory_synapses: 4
            tau_base: 10.0
            tau_ratio: 3.0
          inhibitory:
            n_neurons: 16
            branch_factors: [1]
            ff_excitatory_synapses: 8
            ff_inhibitory_synapses: 4
            rec_excitatory_synapses: 4
            rec_inhibitory_synapses: 4
```

## Notes

- Factory type aliases: `ei_unified`, `unified_ei`, `ei_net`, `unified_einet`.
- `model.core.unified_ei` is ignored unless `core.type` is one of those aliases.
- Per-level time constants are auto-generated geometrically from `tau_base` and `tau_ratio`
  (distal = longest, soma = shortest). Override with explicit `level_taus` list.
- The old `dendritic_rnn` / `dendrnn` / `vanilla_rnn` types have been removed.
  Use `ei_unified` with the `EINetworkConfig` format instead.
