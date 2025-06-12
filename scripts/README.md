# Scripts Directory - Dendritic Modeling Framework

This directory contains utility scripts for running experiments, parameter sweeps, simulations, and analysis in the dendritic modeling framework. Scripts are organized into functional subfolders for better organization and maintainability.

## 📁 Directory Structure

```
scripts/
├── ei_sweeps/              # Excitation-Inhibition parameter sweeps
├── general_sweeps/         # General parameter sweep utilities
├── experiment_sweeps/      # Specialized experimental sweeps
├── dendrinet_simulations/  # Advanced dendritic simulations
├── analysis/               # Analysis and visualization tools
└── README.md              # This file
```

## 📋 Script Categories

### 🔥 **EI Sweeps** (`ei_sweeps/`)

**Purpose**: Comprehensive parameter sweeps over excitation-inhibition (EI) parameters for dendritic networks.

- **`generate_ei_sweep.py`** (45KB, 1066 lines) - Generate EI sweep configurations
  ```bash
  python scripts/ei_sweeps/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml
  python scripts/ei_sweeps/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml --split-jobs 10
  python scripts/ei_sweeps/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml --filter-mode ei-sum --filter-value 52
  ```

- **`analyze_ei_sweep.py`** (162KB, 4209 lines) - Analyze EI sweep results
  ```bash
  python scripts/ei_sweeps/analyze_ei_sweep.py --timestamp 20250411144518
  python scripts/ei_sweeps/analyze_ei_sweep.py --folder /path/to/results/ei_20250411144518
  python scripts/ei_sweeps/analyze_ei_sweep.py --all     # Analyze all ei_ folders
  ```

- **`rerun_missing_configs.py`** (24KB, 688 lines) - Rerun failed/missing EI configs
  ```bash
  python scripts/ei_sweeps/rerun_missing_configs.py --ei-folder ei_20250411144518
  python scripts/ei_sweeps/rerun_missing_configs.py --ei-folder ei_20250411144518 --dry-run
  ```

### ⚙️ **General Sweeps** (`general_sweeps/`)

**Purpose**: General-purpose parameter sweep utilities for any model parameters.

- **`generate_sweep_configs.py`** (21KB, 614 lines) - General sweep configuration generator
  ```bash
  python scripts/general_sweeps/generate_sweep_configs.py configs sweep_params.yaml
  python scripts/general_sweeps/generate_sweep_configs.py --config-dir configs --param-filename sweep_params.yaml
  ```

- **`analyze_sweep_results.py`** (19KB, 610 lines) - General sweep results analyzer
  ```bash
  python scripts/general_sweeps/analyze_sweep_results.py --results-dir output/sweep_results
  python scripts/general_sweeps/analyze_sweep_results.py --results-dir output/sweep_results --output-dir plots
  ```

### 🧪 **Experiment Sweeps** (`experiment_sweeps/`)

**Purpose**: Specialized sweeps for specific experimental conditions (noise, branches, EI variations).

- **`sweep_noise_experiment.py`** (15KB, 462 lines) - Noise experiment sweeps
  ```bash
  python scripts/experiment_sweeps/sweep_noise_experiment.py --config configs/noise_sweep.yaml --noise-types gaussian uniform
  python scripts/experiment_sweeps/sweep_noise_experiment.py --config configs/noise_sweep.yaml --branch-sweep
  ```

- **`sweep_branch_experiment.py`** (13KB, 364 lines) - Branch experiment sweeps
  ```bash
  python scripts/experiment_sweeps/sweep_branch_experiment.py --config configs/branch_sweep.yaml
  python scripts/experiment_sweeps/sweep_branch_experiment.py --config configs/branch_sweep.yaml --seeds 5
  ```

- **`sweep_ei_experiment.py`** (13KB, 382 lines) - EI experiment sweeps
  ```bash
  python scripts/experiment_sweeps/sweep_ei_experiment.py --config configs/ei_sweep.yaml
  python scripts/experiment_sweeps/sweep_ei_experiment.py --config configs/ei_sweep.yaml --output-dir custom_output
  ```

### 🌿 **Dendrinet Simulations** (`dendrinet_simulations/`)

**Purpose**: Advanced dendritic network simulations with correlation analysis and information theory.

- **`generate_dendrinet_sweep.py`** (28KB, 899 lines) - Generate dendrinet parameter sweeps
  ```bash
  python scripts/dendrinet_simulations/generate_dendrinet_sweep.py --config configs/dendrinet_sweep.yaml
  python scripts/dendrinet_simulations/generate_dendrinet_sweep.py --config configs/dendrinet_sweep.yaml --array-jobs
  ```

- **`simulate_dendrinet.py`** (54KB, 1651 lines) - Run dendrinet simulations
  ```bash
  python scripts/dendrinet_simulations/simulate_dendrinet.py --D 30 --C 3 --N_E 5 --N_I 5 --rho 0.3
  python scripts/dendrinet_simulations/simulate_dendrinet.py --config configs/dendrinet_sim.yaml --ei-ratio-sweep
  ```

- **`simulate_dendrinet_array.py`** (5.9KB, 183 lines) - Array job version of dendrinet simulation
  ```bash
  python scripts/dendrinet_simulations/simulate_dendrinet_array.py --job-id ${SLURM_ARRAY_TASK_ID}
  python scripts/dendrinet_simulations/simulate_dendrinet_array.py --config configs/dendrinet_array.yaml --job-id 1
  ```

- **`analyze_dendrinet_results.py`** (31KB, 945 lines) - Analyze dendrinet simulation results
  ```bash
  python scripts/dendrinet_simulations/analyze_dendrinet_results.py --results-dir output/dendrinet_results
  python scripts/dendrinet_simulations/analyze_dendrinet_results.py --results-dir output/dendrinet_results --plot-correlations
  ```

### 📊 **Analysis Tools** (`analysis/`)

**Purpose**: Specialized analysis tools for information theory, pixel analysis, and visualization.

- **`analyze_pixel_information.py`** (33KB, 841 lines) - Analyze pixel-level information
  ```bash
  python scripts/analysis/analyze_pixel_information.py --dataset mnist --config configs/analysis.yaml
  python scripts/analysis/analyze_pixel_information.py --dataset cifar10 --pid-mode all_pairs --workers 8
  ```

- **`run_pixel_analysis.sh`** (4.3KB, 113 lines) - Shell script for pixel analysis
  ```bash
  bash scripts/analysis/run_pixel_analysis.sh mnist
  bash scripts/analysis/run_pixel_analysis.sh cifar10 --parallel
  ```

## 🚀 **Common Usage Patterns**

### Basic EI Sweep Workflow

1. **Generate sweep configurations**:
   ```bash
   python scripts/ei_sweeps/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml
   ```

2. **Run the sweep** (via SLURM):
   ```bash
   sbatch run.sh
   ```

3. **Analyze results**:
   ```bash
   python scripts/ei_sweeps/analyze_ei_sweep.py --timestamp $(date +%Y%m%d%H%M%S)
   ```

### Advanced EI Sweep with Filtering

```bash
# Generate only E+I=52 combinations 
python scripts/ei_sweeps/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml \
    --filter-mode ei-sum --filter-value 52

# Generate only E/I ratio ≈ 2.0 combinations
python scripts/ei_sweeps/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml \
    --filter-mode ei-ratio --filter-value 2.0 --filter-tolerance 0.1
```

### General Parameter Sweep Workflow

```bash
# 1. Generate general sweep configurations
python scripts/general_sweeps/generate_sweep_configs.py configs sweep_params.yaml

# 2. Run the sweep (via generated SLURM script)
sbatch run.sh

# 3. Analyze results
python scripts/general_sweeps/analyze_sweep_results.py --results-dir output/sweep_results
```

### Simulation and Analysis Pipeline

```bash
# 1. Generate dendrinet sweep
python scripts/dendrinet_simulations/generate_dendrinet_sweep.py --config configs/dendrinet_sweep.yaml

# 2. Run simulations  
python scripts/dendrinet_simulations/simulate_dendrinet.py --D 50 --C 3 --total-synapses 20 --rho 0.3

# 3. Analyze results
python scripts/dendrinet_simulations/analyze_dendrinet_results.py --results-dir output/dendrinet_results
```

## 📋 **Configuration Requirements**

### EI Sweep Configuration (`configs/sweep_params_seeds_ei.yaml`)

```yaml
sweep_config:
  model.network.parameters.ee_synapses_per_branch_per_layer: [10, 20, 30, 40, 50]
  model.network.parameters.ie_synapses_per_branch_per_layer: [10, 20, 30, 40, 50]
  model.network.parameters.use_shunting: [true, false, "MLP"]

experiment_settings:
  seeds_per_condition: 3
  base_seed: 42

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

train_config:
  # Base training configuration
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

### General Sweep Configuration (`configs/sweep_params.yaml`)

```yaml
slurm_config:
  run_name: "parameter_sweep"
  account: "kempner_dev" 
  partition: "kempner"
  time: "01:00:00"
  training_script: "src/dendritic_modeling/scripts/train_experiments.py"
  # ... other SLURM settings

train_config:
  # Base configuration for training
  
sweep_config:
  train.learning_rate: [0.001, 0.01, 0.1]
  train.batch_size: [32, 64, 128]
  model.network.parameters.num_layers: [2, 3, 4, 5]
  
experiment_settings:
  seeds_per_condition: 5
  base_seed: 42
```

## 🔧 **Script Dependencies**

### Updated Import Structure

All scripts have been updated to use the new codebase structure:

- ✅ `from dendritic_modeling.datasets import get_unified_datasets` (updated from `synthetic_datasets`)
- ✅ `from dendritic_modeling.training.strategies import TrainerMLE` (updated from `learning_strategies`)
- ✅ `from dendritic_modeling.training.factory import get_trainer` (correct location)
- ✅ `from dendritic_modeling.plotting.utils import plot_*` (updated from `plot_utils`)

### Required Packages

Scripts may require additional packages:
```bash
# For PID analysis (analysis/analyze_pixel_information.py)
pip install dit broja

# For visualization
pip install seaborn matplotlib

# For parallel processing
pip install tqdm multiprocessing
```

## 🐛 **Troubleshooting**

### Common Issues

1. **Import Errors**: Ensure you're in the root directory and the package is installed:
   ```bash
   cd /path/to/dendritic-modeling
   pip install -e .
   ```

2. **Missing Configs**: Check that config files exist and have correct YAML format:
   ```bash
   python -c "import yaml; yaml.safe_load(open('configs/sweep_params.yaml'))"
   ```

3. **SLURM Job Failures**: Check output logs and ensure paths are correct:
   ```bash
   # Check if training script exists
   ls -la src/dendritic_modeling/scripts/train_experiments.py
   ```

4. **Memory Issues**: For large sweeps, consider:
   - Using `--split-jobs` flag for EI sweeps
   - Reducing `max_concurrent_jobs` in SLURM config
   - Using `--filter-mode` to reduce combinations

### Performance Tips

- **EI Sweeps**: Use filtering modes to reduce computational load
- **Parallel Analysis**: Use `--workers` flag for pixel information analysis
- **Array Jobs**: Use `simulate_dendrinet_array.py` for large-scale simulations
- **Incremental Analysis**: Use `--timestamp` flags to analyze specific runs

## 📈 **Output Structure**

Scripts generate organized output directories:

```
output/
├── ei_20250411144518/          # EI sweep results
│   ├── configs/                # Generated configuration files
│   ├── results/                # Training results
│   │   ├── config_0/
│   │   ├── config_1/
│   │   └── ...
│   ├── plots/                  # Analysis plots
│   └── run.sh                  # Generated SLURM script
├── dendrinet_results/          # Dendrinet simulation results
│   ├── correlation_analysis/
│   ├── synergy_plots/
│   └── ei_ratio_analysis/
└── pixel_analysis/             # Pixel information analysis
    ├── mnist_single_pixel_mi.png
    ├── cifar10_pairwise_pid/
    └── information_matrices/
```

## 🔬 **Advanced Features**

### EI Sweep Filtering Modes

- **`all`**: Generate all E-I combinations (default)
- **`ei-sum`**: Only combinations where E+I equals constant
- **`e-const`**: Only combinations with constant E value  
- **`i-const`**: Only combinations with constant I value
- **`ei-ratio`**: Only combinations where E/I approximates constant ratio

### Network Type Comparison

Scripts support comparing different architectures:
- **`true`**: EINet with shunting inhibition
- **`false`**: EINet without shunting inhibition  
- **`"MLP"`**: Standard MLP using same EI parameters

### Dendrinet Simulations

Advanced features include:
- **Correlation enumeration**: 27 correlation combinations (cEE, cII, cEI)
- **Group assignments**: 9 ways (3 info groups × 3 group assignments)
- **Information theory**: Synergy, redundancy, unique information analysis
- **E/I ratio sweeps**: Systematic exploration of excitation/inhibition balance

---

*Last updated: January 2025*
*Compatible with dendritic-modeling framework v2.0+* 