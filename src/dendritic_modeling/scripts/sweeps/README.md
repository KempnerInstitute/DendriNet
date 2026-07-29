# Sweep Scripts

This directory contains the CLI and support code for parameter sweeps.

## Entry Points

- `sweep_manager.py`: generate configs, create job scripts, and optionally
  analyze a finished sweep directory.
- `result_analyzer.py`: analyze an existing sweep results directory.
- `rerun_missing_configs.py`: requeue missing or failed sweep configs.
- `summarize_initializer_components.py`: validation-only selector and
  center/slope contrasts for the analytical/occupancy 2-by-2 initializer
  factorial. It deliberately never reads held-out test endpoints.
- `collect_sweep_metrics.py`: manifest-driven, hash-recorded collection of
  multiple complete sweep arms into one checkpoint-level table. It performs no
  aggregation across trained seeds.

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

Collect several explicitly labeled, complete sweep arms for a reproducible
seed-level summary:

```bash
dendritic-collect-sweep-metrics --manifest path/to/collection.yaml
```

The collection manifest declares each sweep root, its expected config count,
scalar design labels, required output columns, and an output directory. The
release records hashes of every consumed generated config and held-out
performance file.

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

Every successful generation also writes `frozen_sweep_manifest.json` at the
sweep root. The manifest records the raw input-YAML hash, the fully resolved
`original_config.yaml` hash, every generated-config hash, the expected config
count, source/Git identity when available, and the scheduler profile. Declare
the factorial size in publication sweeps so generation fails closed:

```yaml
sweep_contract:
  expected_config_count: 120
```

The scheduler control plane accepts exactly these account/partition pairs:

- `kempner_dev` / `kempner_priority`
- `kempner_dev` / `kempner_eng`
- `kempner_dev` / `kempner_requeue`
- `kempner_bsabatini_lab` / `kempner_h100`
- `kempner_bsabatini_lab` / `kempner_h100_priority`

Missing scheduler values become `PLACEHOLDER` only with `--generate-only` or
`--dry-run`. Explicit values are validated in every mode, and a placeholder
manifest does not authorize later submission.

## Notes

- The maintained user documentation is in `docs/sweeps.rst`.
- Use `configs/sweep_config.yaml` as the maintained generic schema example.
- Keep project- or paper-specific sweep packs outside the main `configs/`
  tree (for example under `drafts/*/configs/sweeps/` or local ignored folders).
