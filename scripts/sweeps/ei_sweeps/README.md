# EI Parameter Sweeps

Specialized tools for exploring excitatory-inhibitory parameter spaces in dendritic models.

## Scripts Overview

### `generate_ei_sweep.py` (45KB, 1066 lines)
**Purpose**: Generate EI sweep configurations with advanced filtering options.

**Key Features**:
- Custom directory structure with timestamped folders
- SLURM job configuration with output organization
- Multiple filtering modes for E-I combinations
- Network type comparison (EINet vs MLP)
- Job splitting for large sweeps

**Usage**:
```bash
# Basic sweep generation
python generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml

# Split into multiple jobs
python generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml --split-jobs 10

# Filter combinations where E+I=52
python generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml --filter-mode ei-sum --filter-value 52

# Filter combinations where E/I ratio ≈ 2.0
python generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml --filter-mode ei-ratio --filter-value 2.0
```

### `analyze_ei_sweep.py` (162KB, 4209 lines)
**Purpose**: Comprehensive analysis and visualization of EI sweep results.

**Key Features**:
- Automatic result collection from timestamped directories
- Network type grouping (EINet Shunting, EINet No Shunting, MLP)
- Heatmaps of performance metrics vs E/I parameters
- Line plots with representative parameter lines
- Branch information analysis with E/I ratio plots
- Missing/unfinished run detection

**Usage**:
```bash
# Analyze specific experiment
python analyze_ei_sweep.py --timestamp 20250411144518

# Analyze using folder path
python analyze_ei_sweep.py --folder /path/to/results/ei_20250411144518

# Analyze all ei_ folders
python analyze_ei_sweep.py --all

# Show plots in addition to saving
python analyze_ei_sweep.py --timestamp 20250411144518 --show-plots

# Report missing runs
python analyze_ei_sweep.py --timestamp 20250411144518 --report-missing
```

### `rerun_missing_configs.py` (24KB, 688 lines)
**Purpose**: Identify and rerun failed or missing configurations from EI sweeps.

**Key Features**:
- Automatic detection of missing results
- Generation of rerun scripts for failed jobs
- Dry-run mode for testing
- SLURM job submission for reruns

**Usage**:
```bash
# Identify missing configs (dry run)
python rerun_missing_configs.py --ei-folder ei_20250411144518 --dry-run

# Generate rerun scripts
python rerun_missing_configs.py --ei-folder ei_20250411144518

# Automatically submit rerun jobs
python rerun_missing_configs.py --ei-folder ei_20250411144518 --submit
```

## Configuration Requirements

### EI Sweep Configuration File
Create `configs/sweep_params_seeds_ei.yaml`:

```yaml
# Sweep parameters
sweep_config:
  model.network.parameters.ee_synapses_per_branch_per_layer: [10, 20, 30, 40, 50]
  model.network.parameters.ie_synapses_per_branch_per_layer: [10, 20, 30, 40, 50]
  model.network.parameters.use_shunting: [true, false, "MLP"]

# Experiment settings
experiment_settings:
  seeds_per_condition: 3
  base_seed: 42

# SLURM configuration
slurm_config:
  run_name: "ei_sweep"
  account: "kempner_dev"
  partition: "kempner"
  time: "02:00:00"
  nodes: 1
  ntasks_per_node: 1
  gpus_per_node: 1
  cpus_per_task: 4
  mem: "32G"
  max_concurrent_jobs: 50
  modules_to_load: ["cuda/12.1.0-fasrc01"]
  conda_env_path: "/path/to/conda/envs/dendritic_env_ts"
  training_script: "src/dendritic_modeling/scripts/train_experiments.py"

# Base training configuration
train_config:
  task:
    dataset: "multixor"
    parameters:
      num_samples: 5000
  
  model:
    network:
      type: "EINet"
      parameters:
        num_layers: 3
        num_branches_per_layer: 10
        
  train:
    optimizer: "adam"
    learning_rate: 0.001
    batch_size: 64
    num_epochs: 100
    seed: 42
```

## Workflow Examples

### Complete EI Sweep Pipeline

```bash
# 1. Generate sweep with E/I ratio filtering
python generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml \
    --filter-mode ei-ratio --filter-value 1.5 --filter-tolerance 0.2

# 2. Submit SLURM job
sbatch run.sh

# 3. Monitor job progress
squeue -u $USER

# 4. Analyze results when complete
python analyze_ei_sweep.py --timestamp $(date +%Y%m%d%H%M%S) --show-plots

# 5. Rerun any failed configs
python rerun_missing_configs.py --ei-folder ei_$(date +%Y%m%d%H%M%S)
```

### Large-Scale Sweep with Job Splitting

```bash
# Generate sweep split into 20 jobs
python generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml --split-jobs 20

# Submit all job scripts
for script in job_*.sh; do
    sbatch "$script"
done

# Wait for completion and analyze
python analyze_ei_sweep.py --all --report-missing
```

## Filtering Modes

### Available Filter Types

1. **`all`** (default): Generate all E-I combinations
2. **`ei-sum`**: Only combinations where E+I equals constant
3. **`e-const`**: Only combinations with constant E value
4. **`i-const`**: Only combinations with constant I value
5. **`ei-ratio`**: Only combinations where E/I approximates constant ratio

### Filter Examples

```bash
# Only combinations where E+I=52
python generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml \
    --filter-mode ei-sum --filter-value 52

# Only combinations where E=26
python generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml \
    --filter-mode e-const --filter-value 26

# Only combinations where I=26
python generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml \
    --filter-mode i-const --filter-value 26

# Only combinations where E/I ≈ 2.0 (±0.1)
python generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml \
    --filter-mode ei-ratio --filter-value 2.0 --filter-tolerance 0.1
```

## Network Type Comparison

The scripts support comparing different network architectures:

- **`true`**: EINet with shunting inhibition enabled
- **`false`**: EINet with shunting inhibition disabled  
- **`"MLP"`**: Standard MLP network using same EI parameters

This enables direct comparison between dendritic (EINet) and non-dendritic (MLP) architectures in the same sweep.

## Output Structure

```
output/
└── ei_20250411144518/
    ├── configs/
    │   ├── ei_sweep_config_0.yaml
    │   ├── ei_sweep_config_1.yaml
    │   └── ...
    ├── results/
    │   ├── config_0/
    │   │   ├── performance/
    │   │   ├── analysis/
    │   │   └── plots/
    │   └── ...
    ├── plots/
    │   ├── heatmaps/
    │   ├── line_plots/
    │   └── network_comparison/
    ├── run.sh
    ├── job_0.sh (if split)
    └── rerun_missing.sh (if needed)
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `max_concurrent_jobs` or use job splitting
2. **Missing Results**: Use `rerun_missing_configs.py` to identify and rerun
3. **Plot Generation Errors**: Check that NetworkX is available for network plots
4. **SLURM Failures**: Verify account, partition, and resource limits

### Performance Tips

- Use filtering modes to reduce computational load
- Split large sweeps into multiple jobs
- Monitor disk space usage for large sweeps
- Use `--dry-run` to preview configurations before submission

---

*Compatible with dendritic-modeling framework v2.0+* 