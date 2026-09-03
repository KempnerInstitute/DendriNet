# Sweep Scripts

This directory contains the CLI and support code for parameter sweeps.

## Entry Points

- `sweep_manager.py`: generate configs, create job scripts, and optionally
  analyze a finished sweep directory.
- `result_analyzer.py`: analyze an existing sweep results directory.
- `rerun_missing_configs.py`: requeue missing or failed sweep configs.
  Prefer `--folder /path/to/{run_name}_{timestamp}`. With `--timestamp`,
  the script uses the same `run_name` / output-dir lookup as
  `sweep_manager.py`.

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
    --config configs/sweep_config.yaml --generate-only
```

Dry-run submission:

```bash
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py \
    --config configs/sweep_config.yaml --dry-run
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
- Use `configs/sweep_config.yaml` as the maintained generic schema example.
- Keep project- or paper-specific sweep packs outside the main `configs/`
  tree (for example under `drafts/*/configs/sweeps/` or local ignored folders).
