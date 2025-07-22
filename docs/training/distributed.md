# FSDP Training Guide

## Overview

The dendritic modeling framework supports Fully Sharded Data Parallel (FSDP) training for large models that don't fit on a single GPU. This guide covers the different approaches and when to use each.

## Training Scripts Comparison

| Script | Use Case | FSDP Support | Environment |
|--------|----------|--------------|-------------|
| `train_experiments.py` | General training | **Intelligent delegation** | Single GPU, torchrun, SLURM |
| `train_experiments_fsdp.py` | Dedicated FSDP | **Full implementation** | torchrun, SLURM only |
| `generate_slurm_job.py` | Auto-job generation | **Smart selection** | Generates SLURM scripts |

## Usage Patterns

### 1. Single GPU Training (Standard)

```bash
# Standard single GPU training
python src/dendritic_modeling/scripts/train_experiments.py config.yaml
```

### 2. Single Node Multi-GPU (FSDP)

```bash
# Option A: Using train_experiments.py (recommended)
torchrun --nproc_per_node=4 \
  src/dendritic_modeling/scripts/train_experiments.py config.yaml --use_fsdp

# Option B: Using dedicated FSDP script directly
torchrun --nproc_per_node=4 \
  src/dendritic_modeling/scripts/train_experiments_fsdp.py config.yaml
```

### 3. Multi-Node Training (SLURM)

```bash
# Generate SLURM script automatically
python src/dendritic_modeling/scripts/generate_slurm_job.py config.yaml

# Submit generated job
sbatch generated_script.sbatch
```

## Configuration

Set `use_fsdp: false` in your config to force single-GPU mode:

```yaml
train:
  use_fsdp: false  # Forces single-GPU training
  # ... other settings
```

For FSDP-specific configuration:

```yaml
train:
  use_fsdp: true   # Enable FSDP (only used in config, not required for torchrun)
  fsdp_config:
    mixed_precision: true
    sharding_strategy: "full"
    cpu_offload: false
```

## How the Intelligent Delegation Works

When you use `train_experiments.py` with `--use_fsdp`:

1. **Environment Check**: Detects if running under torchrun/SLURM
2. **Error Guidance**: If not distributed, shows proper usage
3. **Smart Delegation**: If distributed, calls `train_experiments_fsdp.py`
4. **Seamless Transition**: Same config file, full FSDP functionality

## Error Messages and Solutions

### "FSDP requires distributed environment"
**Problem**: Using `--use_fsdp` without torchrun/SLURM
**Solution**: Use `torchrun --nproc_per_node=<gpus>` prefix

### "FSDP script not found"
**Problem**: Missing `train_experiments_fsdp.py`
**Solution**: Ensure all scripts are in the same directory

## Best Practices

### For Development
- Use `train_experiments.py` without `--use_fsdp` for single GPU
- Use smaller models and datasets
- Set `use_fsdp: false` in config

### For Single-Node Production
- Use `torchrun` with `train_experiments.py --use_fsdp`
- Leverages intelligent delegation
- Maintains familiar interface

### For Multi-Node Production
- Use `generate_slurm_job.py` for automatic script generation
- Handles complex distributed setups
- Optimizes resource allocation

## Migration from Old Approach

If you were using the old stub implementation:

```bash
# OLD (stub - didn't work)
python train_experiments.py config.yaml --use_fsdp

# NEW (intelligent delegation)
torchrun --nproc_per_node=4 train_experiments.py config.yaml --use_fsdp
```

The new implementation provides:
- ✅ **Real FSDP training** (not just a stub)
- ✅ **Clear error messages** with usage instructions
- ✅ **Seamless delegation** to full FSDP implementation
- ✅ **Backward compatibility** with existing configs 