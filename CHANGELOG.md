# Framework Overview

This document provides an overview of the Dendritic Modeling Framework capabilities.

## Core Features

### Network Architectures
- EINet (Excitation-Inhibition Networks) with dendritic computation
- DendriNet (Pure dendritic networks) 
- MLP variants for baseline comparisons
- Hybrid architectures combining multiple network types

### Sparse Connectivity
- Standard TopK selection for deterministic pruning
- Variance-based selection for adaptive connectivity
- Stochastic TopK with temperature control
- DeepST (Dynamic Sparse Training) for evolving connectivity

### Training Strategies  
- Standard backpropagation training
- Layer-wise sequential training
- Two-step local-global training
- Voltage stabilization for shunting inhibition
- FSDP support for large-scale distributed training

### Analysis Tools
- Performance metrics and convergence analysis
- Information-theoretic measures (MI, CMI, PID)
- Ablation studies for component importance
- Synaptic turnover and plasticity analysis
- Noise robustness testing

### Performance Optimization
- Mixed precision training (AMP)
- Gradient checkpointing for memory efficiency
- FSDP sharding for billion-parameter models
- CPU offloading for memory-constrained setups

### Datasets and Tasks
- Vision datasets (MNIST, CIFAR-10/100)
- Synthetic datasets for controlled experiments  
- Biological task datasets
- Custom dataset support

### Visualization and Reporting
- Network architecture visualization
- Training dynamics plotting
- Weight and activation analysis
- Comprehensive result reporting
- Weights & Biases integration
