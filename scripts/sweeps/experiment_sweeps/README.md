# Experiment Sweeps - Specialized Experimental Conditions

This directory contains specialized parameter sweep scripts for specific experimental conditions including noise robustness, branch architecture variations, and EI parameter exploration.

## 📁 Scripts Overview

### `sweep_noise_experiment.py` (15KB, 462 lines)
**Purpose**: Comprehensive noise robustness analysis for dendritic networks.

**Key Features**:
- Multiple noise types (Gaussian, uniform, Poisson, frequency)
- Noise level sweeps across different intensities
- Branch-wise and network-wide noise application
- Performance degradation analysis
- Robustness metric computation

**Usage**:
```bash
# Basic noise sweep
python sweep_noise_experiment.py --config configs/noise_sweep.yaml

# Specific noise types
python sweep_noise_experiment.py --config configs/noise_sweep.yaml --noise-types gaussian uniform

# Branch-specific analysis
python sweep_noise_experiment.py --config configs/noise_sweep.yaml --branch-sweep

# Custom noise levels
python sweep_noise_experiment.py --config configs/noise_sweep.yaml --noise-range 0.0 0.5 0.05
```

### `sweep_branch_experiment.py` (13KB, 364 lines)
**Purpose**: Branch architecture parameter exploration for dendritic networks.

**Key Features**:
- Branch count and organization sweeps
- Layer-wise branch configuration
- Branch connectivity patterns
- Synaptic distribution analysis
- Architecture optimization

**Usage**:
```bash
# Basic branch sweep
python sweep_branch_experiment.py --config configs/branch_sweep.yaml

# Multiple seeds for statistical significance
python sweep_branch_experiment.py --config configs/branch_sweep.yaml --seeds 5

# Custom branch ranges
python sweep_branch_experiment.py --config configs/branch_sweep.yaml --branch-range 5 25 5

# Layer-specific analysis
python sweep_branch_experiment.py --config configs/branch_sweep.yaml --layer-wise
```

### `sweep_ei_experiment.py` (13KB, 382 lines)
**Purpose**: Excitation-inhibition parameter exploration (simplified version of EI sweeps).

**Key Features**:
- E/I ratio exploration
- Synaptic strength sweeps
- Balance point identification
- Performance vs. E/I relationship analysis
- Network dynamics characterization

**Usage**:
```bash
# Basic EI sweep
python sweep_ei_experiment.py --config configs/ei_sweep.yaml

# Custom output directory
python sweep_ei_experiment.py --config configs/ei_sweep.yaml --output-dir custom_output

# Specific E/I ranges
python sweep_ei_experiment.py --config configs/ei_sweep.yaml --ei-range 0.5 3.0 0.25

# Include dynamics analysis
python sweep_ei_experiment.py --config configs/ei_sweep.yaml --analyze-dynamics
```

## 🔧 Configuration Examples

### Noise Experiment Configuration

Create `configs/noise_sweep.yaml`:

```yaml
experiment:
  type: "noise_robustness"
  noise_types: ["gaussian", "uniform", "poisson", "frequency"]
  noise_levels: [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
  
model_config:
  network:
    type: "EINet"
    parameters:
      num_layers: 3
      num_branches_per_layer: 10
      use_shunting: true
      
task_config:
  dataset: "multixor"
  parameters:
    num_samples: 5000
    
training_config:
  num_epochs: 100
  batch_size: 64
  learning_rate: 0.001
  seeds: [42, 43, 44, 45, 46]
  
analysis_config:
  compute_robustness_metrics: true
  plot_degradation_curves: true
  save_detailed_results: true
```

### Branch Architecture Configuration

Create `configs/branch_sweep.yaml`:

```yaml
experiment:
  type: "branch_architecture"
  
parameter_sweeps:
  num_branches_per_layer: [5, 10, 15, 20, 25]
  num_layers: [2, 3, 4, 5]
  branch_connectivity: ["full", "sparse", "layered"]
  synaptic_distribution: ["uniform", "random", "structured"]
  
model_config:
  network:
    type: "EINet"
    parameters:
      use_shunting: true
      ee_synapses_per_branch_per_layer: 10
      ie_synapses_per_branch_per_layer: 10
      
training_config:
  num_epochs: 100
  seeds_per_condition: 3
  
analysis_config:
  analyze_information_flow: true
  compute_capacity_metrics: true
  plot_architecture_effects: true
```

### EI Balance Configuration

Create `configs/ei_sweep.yaml`:

```yaml
experiment:
  type: "ei_balance"
  
parameter_sweeps:
  ee_synapses: [5, 10, 15, 20, 25, 30]
  ie_synapses: [5, 10, 15, 20, 25, 30]
  ei_ratios: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
  
model_config:
  network:
    type: "EINet"
    parameters:
      num_layers: 3
      num_branches_per_layer: 12
      use_shunting: [true, false]
      
analysis_config:
  compute_balance_metrics: true
  analyze_dynamics: true
  plot_phase_space: true
  statistical_analysis: true
```

## 🚀 Experimental Workflows

### Noise Robustness Analysis

```bash
# 1. Run comprehensive noise analysis
python sweep_noise_experiment.py \
    --config configs/noise_sweep.yaml \
    --noise-types gaussian uniform poisson \
    --output-dir results/noise_analysis

# 2. Analyze robustness patterns
python sweep_noise_experiment.py \
    --config configs/noise_sweep.yaml \
    --analyze-only \
    --results-dir results/noise_analysis

# 3. Generate robustness report
python sweep_noise_experiment.py \
    --config configs/noise_sweep.yaml \
    --generate-report \
    --results-dir results/noise_analysis
```

### Branch Architecture Optimization

```bash
# 1. Systematic architecture exploration
python sweep_branch_experiment.py \
    --config configs/branch_sweep.yaml \
    --branch-range 5 30 5 \
    --layer-range 2 6 1

# 2. Analyze information capacity
python sweep_branch_experiment.py \
    --config configs/branch_sweep.yaml \
    --analyze-capacity \
    --results-dir results/branch_analysis

# 3. Identify optimal architectures
python sweep_branch_experiment.py \
    --config configs/branch_sweep.yaml \
    --find-optimal \
    --metric information_capacity
```

### E/I Balance Characterization

```bash
# 1. E/I parameter space exploration
python sweep_ei_experiment.py \
    --config configs/ei_sweep.yaml \
    --ei-mode full_grid \
    --output-dir results/ei_balance

# 2. Dynamics analysis
python sweep_ei_experiment.py \
    --config configs/ei_sweep.yaml \
    --analyze-dynamics \
    --results-dir results/ei_balance

# 3. Optimal balance identification
python sweep_ei_experiment.py \
    --config configs/ei_sweep.yaml \
    --find-balance-point \
    --criterion stability
```

## 📊 Analysis Features

### Noise Robustness Metrics

- **Signal-to-Noise Ratio (SNR)**: Input signal quality measurement
- **Performance Degradation**: Accuracy loss vs. noise level
- **Robustness Index**: Normalized performance under noise
- **Critical Noise Level**: Threshold for significant performance loss

### Branch Architecture Metrics

- **Information Capacity**: Maximum information processing capability
- **Computational Efficiency**: Performance per architectural complexity
- **Resource Utilization**: Synaptic usage efficiency
- **Scalability Index**: Performance scaling with network size

### E/I Balance Metrics

- **Balance Point**: Optimal E/I ratio for performance
- **Stability Region**: E/I range with stable dynamics
- **Dynamic Range**: Input range with linear response
- **Oscillation Index**: Tendency for network oscillations

## 📈 Output Structure

```
output/
├── noise_analysis/
│   ├── noise_robustness_results.csv
│   ├── robustness_curves/
│   │   ├── gaussian_noise.png
│   │   ├── uniform_noise.png
│   │   └── poisson_noise.png
│   ├── degradation_analysis/
│   └── robustness_report.pdf
├── branch_analysis/
│   ├── architecture_results.csv
│   ├── capacity_analysis/
│   ├── optimal_architectures.json
│   └── architecture_comparison.png
└── ei_balance/
    ├── ei_sweep_results.csv
    ├── balance_analysis/
    ├── dynamics_plots/
    └── optimal_balance.json
```

## 🔬 Advanced Analysis Options

### Noise Type Characterization

```python
# Custom noise models
noise_models = {
    'gaussian': {'std_range': [0.05, 0.5], 'mean': 0.0},
    'uniform': {'range': [0.1, 0.6]}, 
    'poisson': {'lambda_range': [0.1, 2.0]},
    'frequency': {'cutoff_range': [0.1, 0.9]}
}
```

### Architectural Constraints

```python
# Branch organization constraints
constraints = {
    'max_branches_per_layer': 50,
    'min_synapses_per_branch': 3,
    'connectivity_sparsity': 0.8,
    'layer_size_ratios': [1.0, 0.8, 0.6]  # Decreasing layer sizes
}
```

### E/I Balance Analysis

```python
# Dynamic stability analysis
stability_metrics = {
    'eigenvalue_analysis': True,
    'lyapunov_exponents': True,
    'phase_space_analysis': True,
    'oscillation_detection': True
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Noise Artifacts**: Ensure noise models are appropriate for data type
2. **Architecture Constraints**: Verify branch configurations are feasible
3. **E/I Instability**: Check for divergent dynamics in extreme E/I ratios
4. **Memory Usage**: Monitor memory for large parameter sweeps

### Performance Tips

- Start with coarse parameter grids, then refine
- Use appropriate noise level ranges for meaningful analysis
- Consider computational cost for large architectural sweeps
- Validate E/I ranges to avoid unstable dynamics

---

*Compatible with dendritic-modeling framework v2.0+* 