# DendriNet Simulations

Specialized tools for running DendriNet architecture simulations with comprehensive parameter sweeps and analysis.

## Scripts Overview

### `simulate_dendrinet.py` (54KB, 1651 lines)
**Purpose**: Comprehensive dendritic network simulations with correlation enumeration and information theory analysis.

**Key Features**:
- Correlation enumeration across 27 combinations (cEE, cII, cEI)
- 9 group assignments for excitatory/inhibitory connections
- Synergy and redundancy analysis
- E/I ratio sweep capabilities
- Both shunting and linear dendritic models
- Information theory metrics (mutual information, PID)

**Usage**:
```bash
# Basic simulation
python simulate_dendrinet.py --D 30 --C 3 --N_E 5 --N_I 5 --rho 0.3

# E/I ratio sweep
python simulate_dendrinet.py --D 50 --C 3 --total-synapses 20 --rho 0.3 --ei-ratio-sweep

# Custom configuration
python simulate_dendrinet.py --config configs/dendrinet_sim.yaml
```

### `generate_dendrinet_sweep.py` (28KB, 899 lines)
**Purpose**: Generate parameter sweep configurations for dendrinet simulations.

**Key Features**:
- SLURM job generation for large-scale sweeps
- Parameter space exploration
- Array job support for parallel execution
- Custom output organization

**Usage**:
```bash
# Generate sweep configuration
python generate_dendrinet_sweep.py --config configs/dendrinet_sweep.yaml

# Array job generation
python generate_dendrinet_sweep.py --config configs/dendrinet_sweep.yaml --array-jobs

# Custom parameter ranges
python generate_dendrinet_sweep.py --D-range 20 50 10 --rho-range 0.1 0.5 0.1
```

### `simulate_dendrinet_array.py` (5.9KB, 183 lines)
**Purpose**: Array job version for distributed dendrinet simulations.

**Key Features**:
- SLURM array job compatibility
- Automatic parameter assignment based on job ID
- Resource-efficient execution
- Error handling and logging

**Usage**:
```bash
# Run specific array job
python simulate_dendrinet_array.py --job-id ${SLURM_ARRAY_TASK_ID}

# Test with specific parameters
python simulate_dendrinet_array.py --config configs/dendrinet_array.yaml --job-id 1
```

### `analyze_dendrinet_results.py` (31KB, 945 lines)
**Purpose**: Comprehensive analysis and visualization of dendrinet simulation results.

**Key Features**:
- Correlation effect analysis
- Synergy vs. redundancy plotting
- E/I ratio relationship analysis
- Performance metric visualization
- Statistical analysis and reporting

**Usage**:
```bash
# Basic analysis
python analyze_dendrinet_results.py --results-dir output/dendrinet_results

# Include correlation plots
python analyze_dendrinet_results.py --results-dir output/dendrinet_results --plot-correlations

# Custom output directory
python analyze_dendrinet_results.py --results-dir input/results --output-dir custom_plots
```

## Simulation Features

### Correlation Analysis

The simulations systematically explore correlations between:
- **Excitatory-Excitatory (cEE)**: Correlations within excitatory inputs
- **Inhibitory-Inhibitory (cII)**: Correlations within inhibitory inputs  
- **Excitatory-Inhibitory (cEI)**: Cross-correlations between E and I inputs

Each can take values: `{-ρ, 0, +ρ}` resulting in 27 total combinations.

### Information Group Assignments

The D-dimensional feature space is partitioned into 3 groups:
- **No-info group**: Dimensions with no class information
- **Medium-info group**: Dimensions with moderate class information
- **High-info group**: Dimensions with high class information

### Model Variants

1. **Shunting Model**: `output = E / (E + I + ε)`
2. **Linear Model**: `output = E - I`

Both models are compared across all parameter combinations.

## Usage Examples

### Complete Simulation Pipeline

```bash
# 1. Generate sweep configurations
python generate_dendrinet_sweep.py \
    --config configs/dendrinet_sweep.yaml \
    --D-range 30 60 10 \
    --rho-range 0.2 0.4 0.1

# 2. Submit array jobs
sbatch dendrinet_array_job.sh

# 3. Run analysis when complete
python analyze_dendrinet_results.py \
    --results-dir output/dendrinet_results \
    --plot-correlations \
    --output-dir plots/dendrinet_analysis
```

### Single Simulation Example

```bash
# Run comprehensive simulation
python simulate_dendrinet.py \
    --D 40 \
    --C 3 \
    --N_E 8 \
    --N_I 6 \
    --rho 0.3 \
    --max-iter 30 \
    --seeds 0 1 2 3 4 \
    --output-dir results/single_sim
```

### E/I Ratio Analysis

```bash
# Systematic E/I ratio exploration
python simulate_dendrinet.py \
    --ei-ratio-sweep \
    --total-synapses 20 \
    --ratio-range 0.5 3.0 \
    --ratio-steps 20 \
    --rho 0.25
```

## Configuration Files

### Dendrinet Sweep Configuration

Create `configs/dendrinet_sweep.yaml`:

```yaml
simulation_params:
  D: [30, 40, 50]  # Dimensionality
  C: [3]           # Number of classes
  N_E: [5, 8, 10]  # Excitatory synapses
  N_I: [5, 8, 10]  # Inhibitory synapses
  rho: [0.2, 0.3, 0.4]  # Correlation strength
  max_iter: 25
  
seeds:
  base_seed: 42
  num_seeds: 5
  
slurm_config:
  account: "kempner_dev"
  partition: "kempner" 
  time: "04:00:00"
  cpus_per_task: 4
  mem: "16G"
  array_limit: 100

output:
  base_dir: "output/dendrinet_sweep"
  save_detailed: true
  plot_results: true
```

### Single Simulation Configuration

Create `configs/dendrinet_sim.yaml`:

```yaml
simulation:
  D: 40
  C: 3
  N_E: 8
  N_I: 6
  rho: 0.3
  max_iter: 30
  apply_sigmoid: true
  
correlation_analysis:
  enable: true
  correlation_values: [-0.3, 0.0, 0.3]
  
information_analysis:
  enable: true
  num_bins: 10
  
output:
  dir: "results/dendrinet_sim"
  save_matrices: true
  generate_plots: true
```

## Output Structure

```
output/
└── dendrinet_results/
    ├── simulation_results.csv
    ├── correlation_analysis/
    │   ├── cEE_effects.png
    │   ├── cII_effects.png
    │   ├── cEI_effects.png
    │   └── correlation_heatmaps.png
    ├── synergy_analysis/
    │   ├── synergy_vs_redundancy.png
    │   ├── ei_ratio_synergy.png
    │   └── group_assignment_effects.png
    ├── performance_analysis/
    │   ├── accuracy_by_correlation.png
    │   ├── model_comparison.png
    │   └── statistical_summary.txt
    └── raw_data/
        ├── detailed_results.json
        └── parameter_logs/
```

## Advanced Features

### Information Theory Metrics

Computed for each simulation:

1. **Mutual Information**: Between inputs and outputs
2. **Synergy**: Information unique to input combinations
3. **Redundancy**: Shared information across inputs
4. **Unique Information**: Information specific to individual inputs

### Statistical Analysis

- Performance comparisons across correlation conditions
- Model variant analysis (shunting vs. linear)
- Group assignment effectiveness
- E/I ratio optimization

### Visualization Options

- Correlation effect heatmaps
- Synergy-redundancy scatter plots
- E/I ratio performance curves
- Statistical distribution plots

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce dimensionality or number of seeds
2. **Convergence Problems**: Increase `max_iter` or adjust learning rate
3. **Array Job Failures**: Check SLURM limits and resource allocation
4. **Correlation Matrix Errors**: Ensure correlation values create valid covariance matrices

### Performance Tips

- Use array jobs for large parameter sweeps
- Start with smaller dimensional spaces for testing
- Monitor convergence during optimization
- Use appropriate number of seeds for statistical reliability

---

*Compatible with dendritic-modeling framework v2.0+* 