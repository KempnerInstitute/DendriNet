# Configuration Validation and FSDP Training Guide

## Overview

This guide covers the reorganized validation system and automatic FSDP training capabilities introduced in the dendritic modeling framework.

## Configuration Validation

The validation system has been reorganized into focused, purpose-specific modules:

### File Structure

- `src/dendritic_modeling/utils/model_architecture_validator.py` - Validates model structural parameters
- `src/dendritic_modeling/utils/training_stability_validator.py` - Validates training configuration stability
- `src/dendritic_modeling/utils/config_validation.py` - Unified validation entry point

### Usage

```python
# Complete validation (recommended)
from dendritic_modeling.utils import validate_full_config
is_valid = validate_full_config(config, strict=False)

# Architecture-only validation (for sweeps/tests)
from dendritic_modeling.utils import validate_architecture_only
arch_valid = validate_architecture_only(config)

# Training stability validation (for training scripts)
from dendritic_modeling.utils import validate_training_only
training_valid = validate_training_only(config)
```

### Legacy Compatibility

All existing imports continue to work:

```python
# These still work
from dendritic_modeling.utils import validate_config, ConfigValidator, validate_optimization_combinations
```

### Validation Scope

#### Model Architecture Validation
- Layer dimension compatibility
- Synapse counts vs layer configurations
- TopK constraints and branch factors
- Transfer learning parameter alignment
- Biological plausibility constraints

#### Training Stability Validation
- AMP + gradient hooks compatibility
- High NaN risk optimization combinations
- FSDP configuration recommendations
- Learning rate vs other parameter interactions
- Memory vs speed trade-off suggestions

## FSDP Training

### Automatic GPU Allocation

The system automatically determines optimal GPU configuration based on model size:

```bash
# Automatic resource allocation and submission
python src/dendritic_modeling/scripts/tools/submit_fsdp_job.py configs/config_exp_fsdp.yaml --submit
```

### Configuration

Remove manual GPU/node specifications from config. The system automatically selects:

```yaml
slurm_config:
  account: kempner_dev
  partition: kempner_h100
  # nodes and gpu_per_node auto-determined by GPU estimator
```

### GPU Estimation Output

Simplified output focuses on H100/H200 compatibility:

```
GPU MEMORY ESTIMATION
Model: 41,923,802,474 parameters
Single GPU memory: 386.51 GB
4-GPU FSDP memory per GPU: 176.85 GB
[NO FIT] Requires multi-GPU setup
[RECOMMENDED] 8x H200 FSDP
[CONFIG] 2 nodes × 4 GPUs/node
```

### Multi-Node FSDP Fixes

The system includes networking and synchronization fixes:

- IPv4 networking: `NCCL_SOCKET_FAMILY=AF_INET`
- Extended timeouts: `TORCHELASTIC_RENDEZVOUS_TIMEOUT=1800`
- Better process coordination with staggered node startup
- Improved torchrun configuration with explicit backends

### Resource Scaling

The system automatically scales from 1 to 32 GPUs based on model requirements:

- Single GPU: H100/H200 if model fits
- 2-4 GPUs: Single-node FSDP
- 8+ GPUs: Multi-node FSDP with optimized communication

## Workflow Changes

### Training Script Selection

The system automatically selects the appropriate training script:

- Models requiring FSDP → `train_experiments_fsdp.py`
- Standard models → `train_experiments.py`

### SLURM Script Generation

Generated scripts include:

- Optimal resource allocation based on model size
- Networking fixes for multi-node communication
- Conditional logic for single vs multi-node execution
- Enhanced debugging and error handling

### Error Handling

Common FSDP errors are automatically prevented:

- Socket initialization failures (IPv6/IPv4 mismatch)
- Rendezvous timeouts in multi-node setups
- Process synchronization issues
- Memory estimation errors

## Migration Guide

### From Manual to Automatic Configuration

1. Remove hardcoded GPU/node settings from configs
2. Use `submit_fsdp_job.py` instead of manual SLURM scripts
3. Update imports to use new validation functions (optional)

### Backward Compatibility

All existing code continues to work without modification. New features are additive and use sensible defaults.

## Best Practices

### Configuration Validation

- Use `validate_full_config()` in training scripts
- Use `validate_architecture_only()` for parameter sweeps
- Set `strict=False` for training to allow warnings

### FSDP Training

- Start with `--generate-only` to review resource allocation
- Monitor initial logs for networking issues
- Use single-node FSDP for models under 10B parameters
- Use multi-node FSDP for models over 10B parameters

### Debugging

- Set `NCCL_DEBUG=INFO` for detailed networking logs
- Check GPU estimation output for memory requirements
- Verify node connectivity if rendezvous fails
- Review validation warnings for stability issues