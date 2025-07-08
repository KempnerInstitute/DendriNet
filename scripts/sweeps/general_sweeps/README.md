# General Parameter Sweeps

Tools for running general parameter sweeps on dendritic models.

## Scripts Overview

### `generate_sweep_configs.py`
Generates configuration files for parameter sweeps based on a YAML specification.

Features:
- Grid search over multiple parameters
- Cartesian product of all parameter combinations
- Automatic SLURM job script generation
- Flexible configuration format

### `analyze_sweep_results.py`
Analyzes results from parameter sweeps and generates visualizations.

Features:
- Aggregates results across multiple runs
- Creates performance heatmaps
- Identifies best parameter combinations
- Generates summary statistics

### `sweep_utils.py`
Utility functions for parameter sweeps.

Features:
- Configuration file parsing
- Result file management
- Data aggregation helpers

## Configuration Format

Example sweep configuration (`sweep_params.yaml`):

```yaml
# SLURM configuration
slurm_config:
  run_name: "general_sweep"
  account: "kempner_dev"
  partition: "kempner"
  time: "02:00:00"
  nodes: 1
  ntasks_per_node: 1
  gpus_per_node: 1
  cpus_per_task: 4
  mem: "32G"
  modules_to_load: ["cuda/12.1.0-fasrc01"]
  conda_env_path: "/path/to/conda/envs/dendritic_env"
  training_script: "src/dendritic_modeling/scripts/train_experiments.py"

# Base training configuration
train_config:
  task:
    dataset: "mnist"
  
  model:
    network:
      type: "EINet"
      parameters:
        num_layers: 3
        
  train:
    optimizer: "adam"
    batch_size: 64
    num_epochs: 100

# Parameters to sweep
sweep_config:
  train.learning_rate: [0.0001, 0.001, 0.01]
  train.batch_size: [32, 64, 128]
  model.network.parameters.num_layers: [2, 3, 4, 5]
  model.network.parameters.use_shunting: [true, false]

# Experiment settings
experiment_settings:
  seeds_per_condition: 3
  base_seed: 42
```

## Usage Examples

### Basic Parameter Sweep

```bash
# Generate sweep configurations
python scripts/general_sweeps/generate_sweep_configs.py configs sweep_params.yaml

# Run the sweep (via SLURM)
bash run.sh

# Analyze results
python scripts/general_sweeps/analyze_sweep_results.py --results-dir output/general_sweep_20240101_120000
```

### Custom Output Directory

```bash
python scripts/general_sweeps/generate_sweep_configs.py \
    --config-dir configs \
    --param-filename sweep_params.yaml \
    --output-dir custom_sweep
```

### Parallel Job Submission

```bash
# Generate configs with job splitting (for large sweeps)
python scripts/general_sweeps/generate_sweep_configs.py configs sweep_params.yaml --split-jobs 10
```

## Complete Workflow

1. **Define sweep parameters** in a YAML file
2. **Generate configurations**:
   ```bash
   python scripts/general_sweeps/generate_sweep_configs.py configs sweep_params.yaml
   ```
3. **Submit jobs** (SLURM):
   ```bash
   sbatch run.sh
   ```
4. **Monitor progress**:
   ```bash
   squeue -u $USER
   ```
5. **Analyze results**:
   ```bash
   python scripts/general_sweeps/analyze_sweep_results.py --results-dir output/sweep_results
   ```

## Output Structure

```
output/
└── general_sweep_20240101_120000/
    ├── configs/
    │   ├── config_0.yaml
    │   ├── config_1.yaml
    │   └── ...
    ├── results/
    │   ├── config_0/
    │   ├── config_1/
    │   └── ...
    ├── plots/
    │   ├── parameter_heatmap.png
    │   ├── best_configs.png
    │   └── sweep_summary.pdf
    └── run.sh

## Advanced Features

### Parameter Dependencies

Define parameter combinations that should be tested together:

```yaml
sweep_config:
  # Regular parameters
  train.learning_rate: [0.001, 0.01]
  
  # Grouped parameters (tested together)
  _groups:
    architecture:
      - model.network.parameters.num_layers: 3
        model.network.parameters.num_branches_per_layer: 10
      - model.network.parameters.num_layers: 5
        model.network.parameters.num_branches_per_layer: 5
```

### Conditional Parameters

```yaml
sweep_config:
  model.network.type: ["EINet", "MLP"]
  
  # Only applies when network type is EINet
  _conditional:
    - when:
        model.network.type: "EINet"
      parameters:
        model.network.parameters.use_shunting: [true, false]
        model.network.parameters.ee_synapses_per_branch_per_layer: [10, 20]
```

## Troubleshooting

### Common Issues

1. **Configuration not found**:
   - Ensure config file path is correct
   - Check YAML syntax with `python -c "import yaml; yaml.safe_load(open('config.yaml'))"`

2. **SLURM job failures**:
   - Check output logs in `slurm-*.out` files
   - Verify conda environment path
   - Ensure sufficient resources allocated

3. **Missing results**:
   - Check if jobs completed: `sacct -j <job_id>`
   - Look for error files in output directory
   - Verify training script path is correct

### Performance Tips

- Use `--split-jobs` for large parameter sweeps
- Adjust `seeds_per_condition` based on variance needs
- Consider using array jobs for very large sweeps

---

For more information, see the main documentation or contact the development team. 