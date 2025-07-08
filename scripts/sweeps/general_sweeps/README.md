# General Sweeps - Universal Parameter Sweep Utilities

This directory contains general-purpose parameter sweep utilities that can be used for any model parameters in the dendritic modeling framework.

## 📁 Scripts Overview

### `generate_sweep_configs.py` (21KB, 614 lines)
**Purpose**: Universal sweep configuration generator for any parameter combinations.

**Key Features**:
- Flexible parameter specification via YAML configuration
- SLURM job script generation
- Multiple seeds per condition support
- Nested parameter path support (e.g., `model.network.parameters.layers`)
- Custom output directory organization
- Array job and single job modes

**Usage**:
```bash
# Basic sweep generation
python generate_sweep_configs.py configs sweep_params.yaml

# Specify custom directories
python generate_sweep_configs.py --config-dir configs --param-filename sweep_params.yaml

# Output to specific directory
python generate_sweep_configs.py configs sweep_params.yaml --output-dir custom_sweep
```

### `analyze_sweep_results.py` (19KB, 610 lines)
**Purpose**: General analysis and visualization for any parameter sweep results.

**Key Features**:
- Automatic result collection from sweep directories
- Performance metric extraction and analysis
- Parameter correlation analysis
- Statistical summary generation
- Customizable visualization options

**Usage**:
```bash
# Analyze sweep results
python analyze_sweep_results.py --results-dir output/sweep_results

# Custom output directory for plots
python analyze_sweep_results.py --results-dir output/sweep_results --output-dir plots/analysis

# Include statistical analysis
python analyze_sweep_results.py --results-dir output/sweep_results --statistical-analysis
```

## 🔧 Configuration Format

### Sweep Parameters Configuration

Create `configs/sweep_params.yaml`:

```yaml
# SLURM job configuration
slurm_config:
  run_name: "parameter_sweep"
  account: "kempner_dev"
  partition: "kempner"
  time: "01:00:00"
  nodes: 1
  ntasks_per_node: 1
  gpus_per_node: 1
  cpus_per_task: 4
  mem: "16G"
  max_concurrent_jobs: 20
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

# Parameters to sweep
sweep_config:
  train.learning_rate: [0.001, 0.01, 0.1]
  train.batch_size: [32, 64, 128]
  model.network.parameters.num_layers: [2, 3, 4, 5]
  model.network.parameters.num_branches_per_layer: [5, 10, 15, 20]

# Experiment settings
experiment_settings:
  seeds_per_condition: 5
  base_seed: 42
```

## 🚀 Usage Examples

### Learning Rate and Architecture Sweep

```yaml
sweep_config:
  train.learning_rate: [1e-4, 1e-3, 1e-2]
  train.optimizer: ["adam", "sgd", "adamw"]
  model.network.parameters.num_layers: [2, 3, 4]
  model.network.parameters.num_branches_per_layer: [8, 16, 32]
```

### Dataset Comparison Sweep

```yaml
sweep_config:
  task.dataset: ["multixor", "center_surround", "orientation_bars"]
  task.parameters.num_samples: [2000, 5000, 10000]
  model.network.parameters.use_shunting: [true, false]
```

### Training Strategy Sweep

```yaml
sweep_config:
  train.strategy: ["mle", "two_step", "regression"]
  train.batch_size: [32, 64, 128]
  train.num_epochs: [50, 100, 200]
```

## 📊 Complete Workflow

### 1. Generate Sweep

```bash
# Create sweep configurations
python generate_sweep_configs.py configs sweep_params.yaml

# Submit SLURM jobs
sbatch run.sh

# Monitor progress
squeue -u $USER
```

### 2. Analyze Results

```bash
# Wait for completion, then analyze
python analyze_sweep_results.py \
    --results-dir output/parameter_sweep_* \
    --output-dir plots/parameter_analysis \
    --statistical-analysis
```

### 3. Extract Best Parameters

```bash
# Find optimal configurations
python analyze_sweep_results.py \
    --results-dir output/parameter_sweep_* \
    --extract-best \
    --metric test_accuracy \
    --output-file best_configs.json
```

## 📈 Output Structure

```
output/
└── parameter_sweep_20250412/
    ├── configs/
    │   ├── config_0.yaml
    │   ├── config_1.yaml
    │   └── ...
    ├── results/
    │   ├── config_0/
    │   ├── config_1/
    │   └── ...
    ├── analysis/
    │   ├── parameter_effects.png
    │   ├── correlation_analysis.png
    │   ├── best_configurations.json
    │   └── statistical_summary.txt
    └── run.sh
```

## 🔬 Advanced Features

### Nested Parameter Specification

Support for complex nested parameter paths:

```yaml
sweep_config:
  model.network.parameters.excitatory_cells.activation: ["tanh", "relu", "sigmoid"]
  model.network.parameters.inhibitory_cells.learning_rate: [0.001, 0.01]
  train.optimizer_params.weight_decay: [0.0, 1e-5, 1e-4]
```

### Conditional Parameter Sweeps

```yaml
sweep_config:
  model.network.type: ["EINet", "MLP"]
  # EINet-specific parameters
  model.network.parameters.use_shunting: [true, false]  # Only for EINet
  # MLP-specific parameters  
  model.network.parameters.hidden_sizes: [[64, 32], [128, 64], [256, 128]]  # Only for MLP
```

### Statistical Analysis Options

```yaml
analysis_config:
  metrics_to_analyze: ["train_accuracy", "test_accuracy", "convergence_time"]
  statistical_tests: ["anova", "tukey_hsd", "correlation"]
  significance_level: 0.05
  multiple_comparison_correction: "bonferroni"
```

## 🐛 Troubleshooting

### Common Issues

1. **Parameter Path Errors**: Ensure nested paths match configuration structure
2. **Memory Issues**: Reduce `max_concurrent_jobs` or parameter combinations
3. **SLURM Failures**: Check account permissions and resource limits
4. **Analysis Errors**: Verify all sweep jobs completed successfully

### Performance Tips

- Start with smaller parameter ranges for testing
- Use appropriate number of seeds for statistical significance
- Monitor disk space for large sweeps
- Consider parameter importance when designing sweeps

---

*Compatible with dendritic-modeling framework v2.0+* 