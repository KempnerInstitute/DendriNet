# Dendritic Modeling Framework

<p align="center">
  <a href="https://github.com/KempnerInstitute/dendritic_modeling/actions/workflows/ci.yml">
    <img src="https://github.com/KempnerInstitute/dendritic_modeling/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://codecov.io/gh/KempnerInstitute/dendritic-modeling">
    <img src="https://codecov.io/gh/KempnerInstitute/dendritic-modeling/graph/badge.svg?token=HUJRFX92DF" alt="Coverage"/>
 </a>
  <img src="https://img.shields.io/badge/Python-29,692%20lines-blue" alt="Lines of Code">
  <img src="https://img.shields.io/badge/Code%20Quality-Optimized-green" alt="Code Quality">
</p>

<p align="center">
  <img src="docs/_static/dendritic_diagram.png" alt="Dendritic Neural Network Architecture" width="600">
</p>

A comprehensive framework for modeling dendritic neural networks with biologically-inspired excitation-inhibition dynamics, sophisticated synaptic processing, and advanced training capabilities including pruning, regularization, and synaptic turnover analysis.

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

### Training

Run a simple training experiment:

```bash
python src/dendritic_modeling/scripts/training/train_experiments.py configs/config_exp.yaml
```

Use FSDP for large models with automatic GPU allocation:

```bash
# Auto-configured multi-node FSDP training
python src/dendritic_modeling/scripts/tools/submit_fsdp_job.py configs/config_exp_fsdp.yaml --submit

# Manual generation only
python src/dendritic_modeling/scripts/tools/submit_fsdp_job.py configs/config_exp_fsdp.yaml --generate-only
```

Use advanced training with pruning, regularization, and synaptic analysis:

```bash
# Edit config_exp.yaml and set:
# train.main.learning_strategy: enhanced
# train.main.pruning.enabled: true
# analysis.synapse_turnover_analysis.synaptic_analysis: true

python src/dendritic_modeling/scripts/training/train_experiments.py configs/config_exp.yaml
```

Enhanced features include:
- Automatic GPU allocation and FSDP scaling (1-32 GPUs)
- Multi-node distributed training with improved synchronization
- Multiple optimizers (Adam, SGD, RMSprop, AdamW)
- L1/L2 regularization with selective application
- Weight pruning with magnitude thresholds
- Synaptic turnover analysis with active synapse tracking
- Advanced reporting with branch-level statistics

See [ENHANCED_TRAINING_GUIDE.md](ENHANCED_TRAINING_GUIDE.md) for detailed usage and [VALIDATION_AND_FSDP_GUIDE.md](VALIDATION_AND_FSDP_GUIDE.md) for validation and FSDP configuration.

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

Generates comprehensive analysis:
- Performance trends vs E/I ratios
- Dendritic architecture comparisons  
- Noise robustness analysis
- Layer-specific information theory analysis (I(E;C), I(I;E|C), etc.)

See [`src/dendritic_modeling/scripts/sweeps/README.md`](src/dendritic_modeling/scripts/sweeps/README.md) for details.

## Documentation

For detailed usage, configuration options, and advanced features, see the [full documentation](docs/index.html).

## Core Components

- Models: Classifier and Regressor with encoder-core-decoder architecture
- Networks: EINet (excitation-inhibition), DendriNet (dendritic branching), and classical networks
- Training: Standard & enhanced training, auto-scaling FSDP for large models, pruning, regularization
- Validation: Unified configuration validation with model architecture and training stability checks
- GPU Estimation: Automatic resource allocation with H100/H200 optimization and memory estimation
- Analysis: Information theory, synaptic turnover, correlation studies, and 65+ visualization plots
- Sweeps: Automated parameter exploration with intelligent result analysis

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
