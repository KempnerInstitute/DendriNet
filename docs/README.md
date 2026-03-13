# Documentation

This directory contains the Sphinx documentation for the Dendritic Modeling Framework.

## Structure

### Getting Started
- `installation.rst` -- Installation guide
- `quickstart.rst` -- First experiment walkthrough

### User Guide
- `network_architectures.rst` -- All network types: EINet, DendriNet, recurrent E-I, RNNs, MLP
- `training_strategies.rst` -- Training strategies: standard, FA/DFA, local learning, multi-stage, FSDP
- `datasets.rst` -- Standard, synthetic, and sequence datasets
- `analysis.rst` -- Information-theoretic, ablation, correlation, and other analyses
- `CONFIGURATION_GUIDE.rst` -- Complete YAML configuration reference
- `sweeps.rst` -- Parameter sweep setup and analysis
- `FSDP_TROUBLESHOOTING.rst` -- Distributed training guide

### Reference
- `available_components.rst` -- Quick reference of all type strings and options
- `contributing.rst` -- Development setup and code style

## Building

```bash
conda activate dendritic_env
cd docs
make html
```

Open `_build/html/index.html` to view.
