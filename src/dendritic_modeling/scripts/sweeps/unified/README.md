# Unified sweep implementation

The implementation behind `dendritic-sweep`
(`dendritic_modeling.scripts.sweeps.sweep_manager`). It is not a separate
command-line tool. The user documentation is `docs/sweeps.rst`.

## Files

- `sweep_types/unified_sweep.py`: `UnifiedSweepGenerator`, which expands a
  sweep file into training configs, and `expand_axis`, which reads one axis
  value (literal list, scalar or explicit `range`/`linspace`/`geomspace`/
  `values` form).
- `base_sweep.py`: the generator and analyzer base classes, the sweep
  directory and config writer, and `SweepJobManager`, which writes the array,
  sequential and rerun job scripts and `submit_all_jobs.sh`.

## Responsibilities

The generator:

- expands the `sweep_config` axes and crosses them with the named
  `sweep_variants`;
- applies `filter_config`: the E/I pair filter and the branch sweeps;
- expands each condition into `experiment_settings.seeds_per_condition`
  replicates (`experiment.seed = base_seed + r`, also written to
  `seed_paths`);
- writes the configs, each with its own `outputs.results_dir`
  (`results/config_<i>` or `<execution_results_dir>/config_<i>`).

`SweepManager` composes the sweep file with its `extends` chain, loads every
generated config through the strict loader, checks
`sweep_contract.expected_config_count`, writes the job scripts with
`SweepJobManager` and writes `frozen_sweep_manifest.json`.

## Usage

```bash
dendritic-sweep --config <sweep.yaml> --generate-only
```

Use `configs/sweep_config.yaml` as the generic template, and keep
project-specific sweep files outside the library's `configs/` tree.
