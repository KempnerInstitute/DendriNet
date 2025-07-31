# Unified Sweep Framework

Single YAML interface for all parameter sweeps in dendritic modeling experiments.

## Quick Start

```bash
# Run a sweep
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py \
    --config configs/unified_all_sweeps_example.yaml --run

# Generate configs only  
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py \
    --config configs/unified_all_sweeps_example.yaml --generate-only

# Analyze results (65+ comprehensive plots)
python src/dendritic_modeling/scripts/sweeps/result_analyzer.py \
    --results-dir results/unified_20240101120000 --auto-detect
```

## Sweep Types

### Available Sweep Types
- **`unified`** - General parameter sweeps with filtering (recommended)
- **`ei`** - E/I balance focused sweeps  
- **`branch`** - Dendritic architecture sweeps
- **`noise`** - Noise robustness studies
- **`general`** - Generic parameter exploration

All sweep types use the same unified configuration format below.

## Sweep Options

### 1. Parameter Sweeps (`sweep_config`)
Sweep any configuration parameter:

```yaml
sweep_config:
  # EI Parameters
  model.core_network.parameters.ee_synapses_per_branch_per_layer: [[1],[2],[3]]
  model.core_network.parameters.ie_synapses_per_branch_per_layer: [[1],[2],[3]]  
  model.core_network.parameters.use_shunting: [true, false]
  
  # Training Parameters
  train.main.param_groups.lr: [0.001, 0.01, 0.1]
  train.main.trainer.batch_size: [32, 64, 128]
  
  # Model Architecture
  model.core_network.parameters.layer_sizes: [[32,10], [64,32,10], [128,64,32,10]]
  model.encoder_network.parameters.hidden_sizes: [[32], [64], [128]]
  
  # Use ranges: [start, end, step]
  some_parameter: [0.1, 1.0, 0.1]  # Creates [0.1, 0.2, ..., 1.0]
```

### 2. EI Filtering (`filter_config.mode`)
Constrain excitatory/inhibitory combinations:

```yaml
filter_config:
  mode: "ei-sum"      # E + I = value
  mode: "e-const"     # E = value (fixed excitatory)
  mode: "i-const"     # I = value (fixed inhibitory)  
  mode: "ei-ratio"    # E/I = value (fixed ratio)
  value: 10           # Target value
  tolerance: 0        # Exact match (default: 1)
```

### 3. Branch Architecture Sweeps (`filter_config.branch_sweep_mode`)
Systematic dendritic architecture exploration:

#### `uniform_sweep` (Default)
Vary both depth and branch count:
```yaml
filter_config:
  branch_sweep_mode: "uniform_sweep"
  dendrite_tree_depth: [1, 4, 1]            # Depths: 1,2,3,4
  dendrite_branches_per_layer: [2, 6, 2]    # Branches: 2,4,6
  # Result: 4×3 = 12 combinations
```

#### `depth_sweep`
Vary depth, fixed branch count:
```yaml
filter_config:
  branch_sweep_mode: "depth_sweep"
  dendrite_tree_depth: [1, 5, 1]            # Depths: 1,2,3,4,5
  uniform_branches_per_layer: 4             # Fixed 4 branches per layer
  # Result: [4], [4,4], [4,4,4], [4,4,4,4], [4,4,4,4,4]
```

#### `width_sweep`
Fixed depth, vary branch count:
```yaml
filter_config:
  branch_sweep_mode: "width_sweep"
  fixed_dendrite_tree_depth: 3              # Fixed depth=3
  dendrite_branches_per_layer: [1, 8, 1]    # Branches: 1,2,3,4,5,6,7,8
  # Result: [1,1,1], [2,2,2], [3,3,3], ..., [8,8,8]
```

#### `pattern_sweep`
Specific architectural patterns:
```yaml
filter_config:
  branch_sweep_mode: "pattern_sweep"
  dendrite_tree_depth: [2, 4, 1]            # Depths: 2,3,4
  base_branches: 4                          # Base branch count
  branch_patterns: ["uniform", "decreasing", "increasing", "pyramid"]
  # Result: 3 depths × 4 patterns = 12 configurations
```

**Available patterns:**
- `uniform`: [4,4,4,4] - Same branches per layer
- `decreasing`: [4,3,2,1] - Decreasing with depth  
- `increasing`: [1,2,3,4] - Increasing with depth
- `pyramid`: [1,2,2,1] - Pyramid shape

#### `custom_configs`
User-defined configurations:
```yaml
filter_config:
  branch_sweep_mode: "custom_configs"
  custom_branch_configs:
    - {depth: 2, branches: [2, 4]}
    - {depth: 3, branches: [1, 2, 1]}
    - {depth: 4, branches: [2, 4, 4, 2]}
```

### 4. Combined Sweeps
Mix EI filtering with branch sweeps:

```yaml
sweep_config:
  model.core_network.parameters.ee_synapses_per_branch_per_layer: [[1],[2],[3]]
  model.core_network.parameters.ie_synapses_per_branch_per_layer: [[1],[2],[3]]
  train.main.param_groups.lr: [0.001, 0.01]

filter_config:
  # EI Filtering: Only E+I=4 combinations
  mode: "ei-sum"
  value: 4
  tolerance: 0
  
  # Branch Patterns: 2 depths × 2 patterns
  branch_sweep_mode: "pattern_sweep"
  dendrite_tree_depth: [2, 3, 1]
  base_branches: 3
  branch_patterns: ["uniform", "decreasing"]
  
# Result: 3 EI combos × 2 LR × 2 depths × 2 patterns = 24 configurations
```

## Configuration Structure

```yaml
# Basic setup
output_dir: "output"
slurm_config:
  max_concurrent_jobs: 15
  job_name: "my_sweep"
  time: "02:00:00"
  gpus_per_node: 1

# Experiment settings
experiment_settings:
  seeds_per_condition: 3
  base_seed: 42

# Parameter sweeps (required)
sweep_config:
  model.core_network.parameters.ee_synapses_per_branch_per_layer: [[1],[2],[3]]
  # ... any other parameters

# Filtering (optional)
filter_config:
  mode: "ei-sum"
  value: 4
  branch_sweep_mode: "pattern_sweep"
  # ... filtering options

# Base configuration (required)
base_config:
  model:
    core_network:
      type: "EINet"
  task:
    dataset: "cifar10"
  train:
    main:
      param_groups:
        lr: 0.001
```

## Examples

**EI Sweep Only:**
```yaml
sweep_config:
  model.core_network.parameters.ee_synapses_per_branch_per_layer: [[1],[2],[3]]
  model.core_network.parameters.ie_synapses_per_branch_per_layer: [[1],[2],[3]]
filter_config:
  mode: "ei-sum"
  value: 4
# Result: 3 combinations where E+I=4
```

**Branch Sweep Only:**
```yaml
filter_config:
  branch_sweep_mode: "pattern_sweep"
  dendrite_tree_depth: [2, 3, 1]
  base_branches: 4
  branch_patterns: ["uniform", "decreasing"]
# Result: 2 depths × 2 patterns = 4 configurations
```

**Mixed Parameters:**
```yaml
sweep_config:
  model.core_network.parameters.ee_synapses_per_branch_per_layer: [[2],[3]]
  model.core_network.parameters.ie_synapses_per_branch_per_layer: [[1],[2]]
  train.main.param_groups.lr: [0.001, 0.01]
  train.main.trainer.batch_size: [32, 64]
# Result: 2×2×2×2 = 16 combinations
```

## SLURM Integration

Automatic SLURM array job submission:
```bash
# 24 configurations → sbatch --array=0-23%15
# - 24 tasks total (indices 0-23)
# - Maximum 15 running concurrently
# - Each task gets dedicated GPU/resources
```

## Analysis

### Comprehensive Analysis (65+ plots)
```bash
# Auto-analyze sweep results (generates all plot types)
python src/dendritic_modeling/scripts/sweeps/result_analyzer.py \
    --results-dir results/unified_20240101120000 --auto-detect

# Analyze specific sweep type
python src/dendritic_modeling/scripts/sweeps/result_analyzer.py \
    --results-dir results/unified_20240101120000 --sweep-type unified
```

### Generated Plot Types
**Performance Analysis:**
- Metric vs E/I ratio plots (shunting vs non-shunting)
- Parameter scaling analysis 
- Performance heatmaps (EE vs IE parameter space)

**Robustness Analysis:**
- Noise perturbation analysis (gaussian/uniform)
- Ablation analysis (layer-wise component contributions)

**Information Theory Analysis:**
- Global mutual information trends: I(E;C), I(I;C), I(E;I|C), etc.
- Layer-specific information flow: MI vs E/I ratio by dendritic layer
- Information cascade patterns through dendritic tree

**Output Structure:**
```
results/unified_20240101120000/plots/
├── metric_vs_ei_ratio_plots/      # Performance trends
├── metric_vs_parameter_plots/     # Scaling relationships  
├── noise_vs_ei_ratio_plots/       # Robustness analysis
├── performance_heatmaps/          # Parameter space visualization
├── ablation_analysis_plots/       # Component contributions
├── information_theory_plots/      # Global MI analysis
└── layer_information_plots/       # Layer-specific MI analysis
```

## Help

```bash
# Sweep management help
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py --help

# Analysis help  
python src/dendritic_modeling/scripts/sweeps/result_analyzer.py --help
```

**Configuration Examples:**
- `configs/unified_all_sweeps_example.yaml` - Complete configuration template
- `.vscode/unified_all_sweeps_example.yaml` - Development configuration