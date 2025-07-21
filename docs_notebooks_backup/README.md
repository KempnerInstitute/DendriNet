# Interactive Notebooks

This directory contains Jupyter notebooks that provide interactive versions of the documentation tutorials and examples.

## Available Notebooks

### Getting Started
- `quickstart_interactive.ipynb` - Interactive version of the quickstart tutorial with runnable code
- `installation_verification.ipynb` - Verify your installation and test basic functionality

### Architecture Examples  
- `architecture_comparison.ipynb` - Compare EINet, DendriNet, and MLP architectures
- `einet_deep_dive.ipynb` - Detailed exploration of Excitation-Inhibition Networks
- `dendritic_computation.ipynb` - Understanding dendritic processing and shunting inhibition

### Training and Analysis
- `training_strategies.ipynb` - Explore different training approaches (standard, layer-wise, branch-wise)
- `information_analysis.ipynb` - Interactive information theory analysis tools
- `parameter_sweeps.ipynb` - Learn to run systematic parameter exploration

### Advanced Topics
- `fsdp_distributed_training.ipynb` - Large-scale distributed training examples
- `custom_architectures.ipynb` - Building custom network architectures
- `biological_realism.ipynb` - Analyzing biological plausibility of models

## Running Notebooks

### Local Installation

```bash
# Activate environment
conda activate dendritic_env

# Start Jupyter
jupyter notebook docs/notebooks/

# Or use Jupyter Lab
jupyter lab docs/notebooks/
```

### Google Colab

Each notebook includes a Colab badge for one-click access:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KempnerInstitute/dendritic-modeling/blob/main/docs/notebooks/)

### Requirements

Most notebooks require:
- PyTorch >= 1.12
- Dendritic Modeling Framework
- Matplotlib, NumPy, Pandas
- Jupyter notebook or JupyterLab

Additional requirements are specified in each notebook.

## Contributing

To contribute new notebooks:

1. Follow the existing structure and style
2. Include clear explanations and documentation
3. Test all code cells execute without errors
4. Add appropriate metadata and tags
5. Include Colab compatibility setup cells

## Support

If you encounter issues with notebooks:
- Check the [FAQ](../faq.rst) for common problems
- Ensure all dependencies are installed
- Report issues on [GitHub](https://github.com/KempnerInstitute/dendritic-modeling/issues)

Happy experimenting! 🧠📊 