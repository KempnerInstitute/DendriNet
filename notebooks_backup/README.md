# Dendritic Modeling Notebooks

This directory contains Jupyter notebooks demonstrating various aspects of dendritic modeling and experiments.

## Organization

### vision/
Notebooks focused on visual neuroscience experiments:
- `dendritic_visual_neurons.ipynb` - **Latest visual neuron experiments** demonstrating orientation tuning, center-surround receptive fields, and motion detection
- `dendritic_vs_visual_system.ipynb` - Comparison between dendritic models and biological visual systems

### convnets/
CNN implementations using dendritic layers:
- `dendritic_conv2d_demo.ipynb` - **Latest CNN implementation** showing how to use dendritic convolution layers in CNNs
- `dendritic_conv2d_explained.ipynb` - Detailed explanation of dendritic convolution mathematics and properties

### analysis/
Analysis and visualization tools:
- `analyze_dendritic_filters.ipynb` - Tools for analyzing learned filters in dendritic networks

## Key Experiments

### Visual Neuron Experiments (vision/dendritic_visual_neurons.ipynb)
- Creates Gabor-like receptive fields using E/I patterns
- Tests orientation selectivity with tuning curves
- Implements center-surround receptive fields (ON/OFF cells)
- Demonstrates translation invariance
- Creates motion-sensitive neurons

### CNN with Dendritic Layers (convnets/dendritic_conv2d_demo.ipynb)
- Drop-in replacement for standard Conv2d layers
- Implements divisive normalization: v_out = E/(E+I+ε)
- Shows orientation-selective filters
- Tests contrast response functions
- Builds complete CNN architectures

## Running the Notebooks

1. Ensure the dendritic-modeling package is installed:
   ```bash
   pip install -e .
   ```

2. Start Jupyter:
   ```bash
   jupyter notebook
   ```

3. Navigate to the desired notebook and run all cells.

## Requirements
- PyTorch >= 1.12
- NumPy
- Matplotlib
- SciPy
- Jupyter/IPython 