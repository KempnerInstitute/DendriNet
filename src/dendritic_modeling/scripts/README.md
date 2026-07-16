# Scripts

This directory contains CLI entry points. Reusable logic should live in package
modules under `dendritic_modeling.*`, not in these scripts.
Paper-specific artifact release workflows belong in their paper repositories.

## Layout

- `training/`: experiment runners
- `tools/`: SLURM, GPU estimation, and utility commands
- `sweeps/`: sweep generation and sweep analysis
- `analysis/`: task-specific analysis entry points
- `script_utils/`: helpers used by the CLI scripts

## Main Entry Points

Training (single-GPU):

```bash
python src/dendritic_modeling/scripts/training/train_experiments.py \
    configs/config_exp.yaml
```

Distributed training (DDP / FSDP) — set `distributed.mode` in the YAML:

```bash
sbatch scripts/train.sh configs/my_config.yaml        # auto-selects from YAML
sbatch scripts/train_ddp.sh configs/my_config.yaml     # DDP
sbatch scripts/train_fsdp.sh configs/my_config.yaml    # FSDP
```

Tools:

```bash
python src/dendritic_modeling/scripts/tools/estimate_gpu_requirements.py \
    configs/config_exp.yaml
```

Sweeps:

```bash
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py \
    --config configs/sweep_config.yaml --generate-only
python src/dendritic_modeling/scripts/sweeps/result_analyzer.py \
    /path/to/sweep/results
```

## Notes

- Prefer running scripts from the repository root.
- The detailed workflow docs live in `docs/`.
