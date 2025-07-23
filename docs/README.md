# Documentation

This directory contains the comprehensive documentation for the Dendritic Modeling Framework.

## Structure

### Core Documentation
- `index.rst` - Main documentation homepage
- `installation.rst` - Simple conda-based installation guide
- `tutorials.rst` - Step-by-step tutorials including quickstart with config_exp.yaml
- `examples.rst` - Practical examples and use cases

### Technical Guides
- `NETWORK_ARCHITECTURES.md` - Detailed guide to all network components (EINet, DendriNet, etc.)
- `CONFIGURATION_GUIDE.md` - Complete configuration reference with examples
- `TRAINING_STRATEGIES.md` - All training strategies explained
- `MODEL_ARCHITECTURES.md` - Model architecture details

### Reference
- `api_reference.rst` - Complete API documentation
- `theory.rst` - Scientific background and theory
- `available_components.md` - Quick reference of all implemented components
- `contributing.rst` - Contribution guidelines

## Figures

All figures are stored in the `figures/` directory:
- `figures/architecture/` - Network architecture diagrams including:
  - EINet architecture diagrams
  - Dendritic neuron schematics
  - Shunting inhibition mechanisms
  - Sparse connectivity patterns
  - Block linear architecture
  - Computation flow diagrams

## Building Documentation

To build the documentation locally:

```bash
# Make sure you're in the dendritic_env conda environment
conda activate dendritic_env

# Navigate to docs directory
cd docs

# Build HTML documentation
make html
```

The built documentation will be in `_build/html/`. Open `_build/html/index.html` in your browser to view.

## Quick Start Example

The documentation includes a comprehensive quickstart guide that demonstrates:
1. Setting up the environment with conda
2. Running your first experiment with `config_exp.yaml`
3. Understanding the output and results
4. Customizing configurations

