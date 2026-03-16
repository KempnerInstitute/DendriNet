# Unified Sweep Implementation

This directory contains the implementation used by
`src/dendritic_modeling/scripts/sweeps/sweep_manager.py`.

## Files

- `base_sweep.py`: shared sweep generator/analyzer base classes and
  `SweepJobManager` support code.
- `sweep_types/unified_sweep.py`: the current config expander and filter logic
  used for sweep generation.

## Responsibilities

The unified generator is responsible for:

- expanding `sweep_config`
- applying `filter_config`
- generating per-config output directories
- writing config files
- preparing job scripts for SLURM execution

## Usage

Treat this as implementation code, not a separate public CLI.
Use it through:

```bash
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py --config <yaml>
```

## Notes

- Current schema and examples are documented in `docs/sweeps.rst`.
- Use `configs/sweep_config.yaml` and `configs/sweeps/*.yaml` as current
  examples rather than this README.
