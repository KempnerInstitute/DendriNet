# Sweep scripts

Command-line tools and support code for parameter sweeps. The user
documentation is `docs/sweeps.rst`; `configs/sweep_config.yaml` is the
annotated sweep template.

## Entry points

- `sweep_manager.py` (`dendritic-sweep`): generates the configs, job scripts
  and frozen manifest of a sweep, submits them unless `--generate-only` or
  `--dry-run` is given, and with `--analyze` runs the result analyzer on a
  finished sweep.
- `result_analyzer.py` (`dendritic-analyze`): collects, aggregates and plots
  a finished sweep directory.
- `collect_sweep_metrics.py` (`dendritic-collect-sweep-metrics`):
  manifest-driven collection of several complete sweeps into one table with
  one row per trained run. It performs no aggregation across seeds; the
  collection records hashes of every generated config and held-out
  performance file it reads.
- `paired_comparison.py`
  (`python -m dendritic_modeling.scripts.sweeps.paired_comparison`): pairs
  two arms of one sweep seed by seed and writes the paired units and a
  bootstrap summary.
- `rerun_missing_configs.py`
  (`python -m dendritic_modeling.scripts.sweeps.rerun_missing_configs`):
  lists the configs of a sweep without finished results and writes the
  sweep's array script restricted to them.
- `control_plane.py`: the site scheduler policy (approved account/partition
  pairs) and the frozen sweep manifest.
- `slurm_profile.py`: `SlurmProfile`, the `#SBATCH` header, environment and
  launch command that every job script of a sweep is rendered from.

## Internal structure

- `unified/`: config expansion and job-script generation (see
  `unified/README.md`)
- `collectors/`: result loading (`DataCollector`) and extraction of
  configuration columns (`ConfigExtractor`)
- `analyzers/`: the analyzer registry, sweep-type detection and grouping
- `plotters/`: sweep plots
- `utils/`: seed aggregation, ablation columns, metric labels and formatting

## Analyzer types

`analyzers/__init__.py` registers `ei`, `stoch`, `local_learning`, `branch`,
`general` and `noise`. Without `--type`, `auto_detect_sweep_type` tries them
in that order on the first five files of `configs/`; `general` accepts any
directory with `configs/` and `results/`, so `noise` runs only when requested.

## Common commands

```bash
dendritic-sweep --config configs/sweep_config.yaml --generate-only
dendritic-sweep --config configs/sweep_config.yaml --dry-run
dendritic-analyze <sweep_dir>
python -m dendritic_modeling.scripts.sweeps.rerun_missing_configs <sweep_dir> --write-script
dendritic-collect-sweep-metrics --manifest path/to/collection.yaml
```

## Notes

- `SweepManager._validate_generated_configs` loads every generated config
  through the strict config loader before any job script is written. The
  sweep file's own sections and keys are read with `.get` and are not
  validated.
- `frozen_sweep_manifest.json` records the input and resolved YAML hashes,
  the files of the `extends` chain, every generated-config hash, the expected
  config count, the scheduler profile and the source identity. The job
  scripts' source guard exits with code 66, 67 or 68 when the commit, the
  tracked changes or the untracked `src/` files differ from that identity.
- Missing scheduler values become `PLACEHOLDER` only with `--generate-only` or
  `--dry-run`. Explicit values are validated against the site policy in every
  mode, and a placeholder sweep cannot be submitted or rerun.
- `utils/aggregation.aggregate_over_seeds` separates conditions only by the
  columns `ConfigExtractor` extracts; runs that differ only in another swept
  path are averaged as replicates.
- Keep project-specific sweep files in the project's own repository, outside
  this library's `configs/` tree.
