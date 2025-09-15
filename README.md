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

### Automated Installation
```bash
git clone https://github.com/KempnerInstitute/dendritic-modeling.git
cd dendritic-modeling
bash install.sh
```

### Manual Installation
```bash
conda env create -f setup/environment.yml
conda activate dendritic_env
pip install -e .
```

### Troubleshooting
```bash
bash install.sh --fix-only    # Fix existing installation
bash install.sh --verify      # Verify installation
```

See [INSTALL.md](INSTALL.md) for detailed troubleshooting.

## Quick Start

### Training

Run a simple training experiment:

```bash
python src/dendritic_modeling/scripts/training/train_experiments.py configs/config_exp.yaml
```

Use FSDP for large models with automatic GPU allocation:

```bash
# Generate and submit FSDP job (default behavior)
python src/dendritic_modeling/scripts/tools/submit_fsdp_job.py configs/config_exp_fsdp.yaml

# Generate SLURM script only
python src/dendritic_modeling/scripts/tools/submit_fsdp_job.py configs/config_exp_fsdp.yaml --generate-only

# Dry run to see what would be submitted
python src/dendritic_modeling/scripts/tools/submit_fsdp_job.py configs/config_exp_fsdp.yaml --dry-run
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
- Automatic GPU allocation with optimal node distribution
- Multi-node distributed training with improved synchronization
- Weight pruning with magnitude thresholds
- Synaptic turnover analysis with active synapse tracking
- Advanced reporting with branch-level statistics
- Configurable information analysis with computation levels (single_branch/layer_branch/all_branch)
- Enhanced EI network analysis with per-neuron, per-EI-network, and per-layer computation
- Multivariate aggregation for joint information computation
- Hash-based encoder network caching for efficiency
- Optional training optimizations (NaN detection, adaptive clipping)

See [ENHANCED_TRAINING_GUIDE.md](ENHANCED_TRAINING_GUIDE.md) for detailed usage and [VALIDATION_AND_FSDP_GUIDE.md](VALIDATION_AND_FSDP_GUIDE.md) for validation and FSDP configuration.

## Parameter Sweeps & Analysis

Run comprehensive parameter sweeps with automatic analysis:

```bash
# Generate and run sweep (default behavior)
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py --config configs/unified_all_sweeps_example.yaml

# Generate configs only (no job submission)
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py --config configs/unified_all_sweeps_example.yaml --generate-only

# Analyze results with auto-detection
python src/dendritic_modeling/scripts/sweeps/result_analyzer.py --results-dir output/sweep_20240101120000 --auto-detect
```

Generates comprehensive analysis:
- Performance trends vs E/I ratios
- Dendritic architecture comparisons
- Noise robustness analysis
- Configurable information theory analysis at branch, layer, or network levels
- Enhanced EI network analysis with separate excitatory/inhibitory and per-layer computation
- Per-neuron information analysis for detailed neural-level insights
- Multivariate joint information computation with flexible aggregation methods

See [`src/dendritic_modeling/scripts/sweeps/README.md`](src/dendritic_modeling/scripts/sweeps/README.md) for details.

## Documentation

For detailed usage, configuration options, and advanced features, see the [full documentation](docs/index.html).

## Core Components

- Models: Classifier and Regressor with encoder-core-decoder architecture
- Networks: EINet (excitation-inhibition), DendriNet (dendritic branching), and classical networks
- Training: Dual trainer structure (encoder/main), auto-scaling FSDP, pruning, regularization, hash-based caching
- Validation: Unified configuration validation with model architecture and training stability checks
- GPU Estimation: Automatic resource allocation with A100/H100 optimization and optimal node distribution
- Analysis: Multi-level information theory, synaptic turnover, correlation studies, and 65+ visualization plots
- Sweeps: Automated parameter exploration with intelligent result analysis

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
