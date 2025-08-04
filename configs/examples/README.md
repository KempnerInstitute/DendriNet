# Sweep Configuration Examples

Example configurations for the unified sweep framework.

## Quick Start

```bash
# Use the comprehensive example
cp configs/unified_all_sweeps_example.yaml configs/my_sweep.yaml

# Edit for your needs, then run (default behavior)
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py \
    --config configs/my_sweep.yaml
```

## Main Configuration

`unified_all_sweeps_example.yaml` - Complete example with all options:
- EI filtering modes
- Branch sweep patterns  
- Mixed sweep combinations
- SLURM settings
- All configuration sections

## Legacy Examples

Individual sweep examples (use unified example instead):
- `ei_sweep_example.yaml` - EI parameter sweeps
- `branch_sweep_example.yaml` - Branch architecture sweeps
- `noise_sweep_example.yaml` - Noise robustness sweeps
- `general_sweep_example.yaml` - General parameter sweeps

## Configuration Structure

```yaml
# Basic setup
output_dir: "results"
slurm_config:
  account: your_account      # SLURM account name
  partition: gpu_partition   # GPU partition name
  gpu_per_node: 4            # Maximum GPUs per node (nodes computed automatically)
  max_concurrent_jobs: 15    # Concurrent job limit

# Parameters to sweep
sweep_config:
  parameter.path: [val1, val2, val3]

# Optional filtering/patterns
filter_config:
  mode: "ei-sum"             # EI filtering
  branch_sweep_mode: "..."   # Branch patterns

# Base configuration (shared)
base_config:
  task: {...}
  model: {...}
  train: {...}
```

## Usage Patterns

Test first:
```bash
# Generate configs only
python sweep_manager.py --config my_sweep.yaml --generate-only

# Dry run (default behavior with dry-run flag)
python sweep_manager.py --config my_sweep.yaml --dry-run
```

Scale appropriately:
- Small exploratory: 10-50 configs, `max_concurrent_jobs: 5-10`
- Production runs: 100-500 configs, `max_concurrent_jobs: 15-25`
- Use 3+ seeds for statistics

Dataset management:
```yaml
outputs:
  dataset_dir: "/shared/datasets/"  # Shared location prevents duplicates
```

## Analysis

```bash
# Analyze results
python src/dendritic_modeling/scripts/sweeps/result_analyzer.py \
    --results-dir results/my_sweep_20240101_120000
```

See `configs/unified_all_sweeps_example.yaml` for complete examples and documentation.