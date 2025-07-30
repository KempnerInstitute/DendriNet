# Sweep Configuration Examples

This directory contains comprehensive example configurations for all sweep types in the unified sweep framework. Each file shows ALL available options with detailed comments.

## Available Sweep Types

### 1. EI Sweep (`ei_sweep_example.yaml`)
- **Purpose**: Explore Excitation-Inhibition balance in dendritic networks
- **Key Parameters**: 
  - `ee_synapses_per_branch_per_layer` (excitatory connections)
  - `ie_synapses_per_branch_per_layer` (inhibitory connections)
  - `use_shunting` (inhibition type)
- **Filter Modes**: 
  - `all`: Generate all E/I combinations
  - `ei-sum`: Fix E+I to constant sum
  - `e-const`: Fix excitation, vary inhibition
  - `i-const`: Fix inhibition, vary excitation  
  - `ei-ratio`: Fix E/I ratio with tolerance

### 2. Branch Sweep (`branch_sweep_example.yaml`)
- **Purpose**: Study dendritic branching effects on learning
- **Key Parameters**:
  - `num_branches_per_layer` (dendrites per neuron)
  - `excitatory_branch_factors` (synapses per branch)
  - `num_layers` (network depth)
- **Use Cases**: Architecture optimization, dendrite importance

### 3. Noise Sweep (`noise_sweep_example.yaml`)
- **Purpose**: Test model robustness under noise conditions
- **Key Parameters**:
  - `noise.type` (gaussian, uniform, poisson, frequency)
  - `noise.level` (intensity from 0.0 to 1.0)
  - `noise.application_mode` (network_wide, branch_wise, layer_wise, input_only)
- **Use Cases**: Robustness analysis, noise tolerance studies

### 4. General Sweep (`general_sweep_example.yaml`)
- **Purpose**: Flexible sweeps over ANY configuration parameters
- **Key Features**:
  - Can sweep any config parameter using dot notation
  - Hyperparameter optimization
  - Architecture comparisons (EINet vs MLP)
  - Multi-dataset studies
- **Use Cases**: Custom experiments, hyperparameter tuning

## Usage Instructions

### 1. Choose and Copy
```bash
# Copy the relevant example to your working directory
cp configs/examples/ei_sweep_example.yaml configs/my_ei_sweep.yaml
```

### 2. Customize
- Edit the copied file to match your needs
- Uncomment/comment sections as needed
- Adjust parameter ranges and values
- Update SLURM settings for your cluster

### 3. Run the Sweep
```bash
# Generate configs only (for testing)
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py ei --config configs/my_ei_sweep.yaml --generate-only

# Full run (generate + submit to SLURM)
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py ei --config configs/my_ei_sweep.yaml --run

# Dry run (show what would be submitted)
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py ei --config configs/my_ei_sweep.yaml --run --dry-run

# Analyze existing results
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py ei --analyze --results-dir output/ei_20240101120000
```

### 4. Split Large Sweeps
For sweeps with many parameter combinations:
```bash
# Split into 10 parallel jobs
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py general --config configs/my_general_sweep.yaml --run --split-jobs 10
```

## Configuration Structure

All sweep configs follow the same structure:

```yaml
output_dir: "output"              # Results directory
slurm_config: {...}              # Cluster job settings
experiment_settings: {...}       # Seeds and experiment params
sweep_config: {...}              # Parameters to sweep over
filter_config: {...}             # Optional: filter combinations
base_config: {...}               # Base training configuration
```

## Tips and Best Practices

1. **Start Small**: Begin with a subset of parameters and few values
2. **Test First**: Always use `--generate-only` or `--dry-run` before full submission
3. **Monitor Resources**: Adjust `time`, `mem`, and `cpus_per_task` based on your experiments
4. **Use Filtering**: For EI sweeps, use filter modes to focus on specific hypotheses
5. **Multiple Seeds**: Use 3+ seeds for reliable statistics
6. **Save Space**: Set `save_model: false` unless you need trained models

## Common Parameter Patterns

### Small Exploratory Sweeps
- 2-4 values per parameter
- 1-3 seeds per condition
- ~10-50 total configurations

### Production Sweeps  
- 5-10 values per parameter
- 3-5 seeds per condition
- ~100-1000 total configurations
- Use job splitting (`--split-jobs`)

### Ablation Studies
- Fix most parameters, vary 1-2 key ones
- Many seeds for statistical power
- Use appropriate filter modes for focused comparisons