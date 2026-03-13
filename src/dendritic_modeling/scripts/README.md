# Scripts

This directory contains CLI entry points. Reusable logic should live in package
modules under `dendritic_modeling.*`, not in these scripts.

## Layout

- `training/`: experiment runners
- `tools/`: SLURM, GPU estimation, and utility commands
- `sweeps/`: sweep generation and sweep analysis
- `analysis/`: task-specific analysis entry points
- `script_utils/`: helpers used by the CLI scripts

## Main Entry Points

Training:

```bash
python src/dendritic_modeling/scripts/training/train_experiments.py \
    configs/config_exp.yaml
python src/dendritic_modeling/scripts/training/train_experiments_fsdp.py \
    configs/config_exp_fsdp.yaml
```

Tools:

```bash
python src/dendritic_modeling/scripts/tools/submit_fsdp_job.py \
    configs/config_exp_fsdp.yaml --generate-only
python src/dendritic_modeling/scripts/tools/estimate_gpu_requirements.py \
    configs/config_exp.yaml
```

Sweeps:

```bash
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py \
    --config configs/EI_sweeps_example.yaml --generate-only
python src/dendritic_modeling/scripts/sweeps/result_analyzer.py \
    /path/to/sweep/results
```

## Notes

- Prefer running scripts from the repository root.
- The detailed workflow docs live in `docs/`.
