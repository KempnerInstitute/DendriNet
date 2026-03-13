# Sweep Scripts

This directory contains the CLI and support code for parameter sweeps.

## Entry Points

- `sweep_manager.py`: generate configs, create job scripts, and optionally
  analyze a finished sweep directory.
- `result_analyzer.py`: analyze an existing sweep results directory.
- `rerun_missing_configs.py`: requeue missing or failed sweep configs.

## Internal Structure

- `analyzers/`: sweep-type detection and grouping logic
- `collectors/`: result loading and config extraction
- `plotters/`: sweep plots
- `unified/`: config expansion and SLURM job generation
- `utils/`: aggregation and plotting metadata

## Supported Analyzer Types

The current registry in `analyzers/__init__.py` supports:

- `ei`
- `stoch`
- `local_learning`
- `branch`
- `general`

## Common Commands

Generate configs only:

```bash
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py \
    --config configs/EI_sweeps_example.yaml --generate-only
```

Dry-run submission:

```bash
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py \
    --config configs/EI_sweeps_example.yaml --dry-run
```

Analyze a completed sweep:

```bash
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py \
    --analyze --results-dir /path/to/sweep/results
```

Use the analyzer directly:

```bash
python src/dendritic_modeling/scripts/sweeps/result_analyzer.py \
    /path/to/sweep/results --type ei
```

## Expected Directory Layout

```
<sweep_dir>/
|-- configs/
|-- results/
|-- plots/
`-- jobs/
```

Each generated config writes its outputs under `results/config_<n>/`.

## Notes

- The maintained user documentation is in `docs/sweeps.rst`.
- Use the example configs under `configs/` and `configs/sweeps/` as the
  source of truth for current schema.
