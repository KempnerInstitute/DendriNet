# Dendritic Modeling Framework

<p align="center">
  <a href="https://github.com/KempnerInstitute/dendritic_modeling/actions/workflows/ci.yml">
    <img src="https://github.com/KempnerInstitute/dendritic_modeling/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://codecov.io/gh/KempnerInstitute/dendritic-modeling">
    <img src="https://codecov.io/gh/KempnerInstitute/dendritic-modeling/graph/badge.svg?token=HUJRFX92DF" alt="Coverage"/>
 </a>
</p>

<p align="center">
  <img src="dendritic_diagram.png" alt="Dendritic Neural Network Architecture" width="600">
</p>

A framework for modeling dendritic neural networks with biologically-inspired excitation-inhibition dynamics and sophisticated synaptic processing.

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/KempnerInstitute/dendritic-modeling.git
cd dendritic-modeling

# 2. Create conda environment
conda env create -f environment.yml
conda activate dendritic_env

# 3. Install the package
pip install -e .
```

## Quick Start

Run a simple training experiment:

```bash
python src/dendritic_modeling/scripts/training/train_experiments.py configs/config_exp.yaml
```

This trains an EINet model on MNIST and saves results in the `results/` directory.

## Parameter Sweeps & Analysis

Run comprehensive parameter sweeps with automatic analysis:

```bash
# 1. Run a sweep (E/I balance + architecture combinations)
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py \
    --config configs/unified_all_sweeps_example.yaml --run

# 2. Analyze results (generates 65+ plots)
python src/dendritic_modeling/scripts/sweeps/result_analyzer.py \
    --results-dir results/unified_20250101120000 --auto-detect
```

**Generates comprehensive analysis:**
- Performance trends vs E/I ratios
- Dendritic architecture comparisons  
- Noise robustness analysis
- Layer-specific information theory analysis (I(E;C), I(I;E|C), etc.)

See [`src/dendritic_modeling/scripts/sweeps/README.md`](src/dendritic_modeling/scripts/sweeps/README.md) for details.

## Documentation

For detailed usage, configuration options, and advanced features, see the [full documentation](docs/index.html).

## Core Components

- **Models**: Classifier and Regressor with encoder-core-decoder architecture
- **Networks**: EINet (excitation-inhibition), DendriNet (dendritic branching), and classical networks
- **Training**: Standard training, FSDP for large models, and parameter sweeps
- **Analysis**: Information-theoretic analysis, correlation studies, and visualization tools

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
