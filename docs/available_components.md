# Available Components in Dendritic Modeling Framework

This document lists all the actual components available in the codebase.

## Networks

### Core Architecture
- `ExcitationInhibitionNetwork` - Main E/I network with dendritic branches
- `ExcitationInhibitionLayer` - Single E/I layer
- `DendriticBranchLayer` - Individual dendritic branch processing
- `DendriNet` - Multi-layer dendritic network

### Sparse Connectivity
- `TopKLinear` - Standard top-K sparse connections
- `StochasticTopKLinear` - Probabilistic top-K selection
- `VarianceTopKLinear` - Variance-based selection
- `DeepstLinear` - DEEPST dynamic sparse connectivity
- `BlockLinear` - Block-structured sparse connections

### Convolutional
- `DendriteConv2d` - Convolutional layer with E/I computation
- `DendriteAttention` - Attention mechanism with dendrites

### Classical Networks (for comparison)
- `MLP` - Standard multi-layer perceptron
- `MatchedParamMLP` - MLP with matched parameters
- `AlexNet` - AlexNet architecture
- `CNNDownsample` - Downsampling CNN
- `CNNUpsample` - Upsampling CNN

### Vision Units
- `create_gabor_unit()` - Gabor filter units
- `create_center_surround_unit()` - Center-surround receptive fields
- `create_motion_unit()` - Motion detection units
- `create_end_stopped_unit()` - End-stopped cells

## Models

- `Classifier` - Classification wrapper
- `Regressor` - Regression wrapper
- `BaseModel` - Base class for custom models

## Training Strategies

- `Trainer` - Standard backpropagation training
- `LayerWiseTrainer` - Train layers sequentially
- `LocalCreditAssignment` - Local learning rules
- `VoltageStabilizationTrainer` - Stabilize dendritic voltages
- `TwoStepTrainer` - Two-phase training with local supervision
- `TwoStepTrainerWithKL` - Two-step with KL divergence

## Analysis Tools

### Core Analysis
- `AblationAnalyzer` - Systematic ablation studies
- `BranchActivationAnalyzer` - Analyze branch activations
- `InformationAnalyzer` - Information-theoretic analysis

### Metrics
- `PerformanceAnalyzer` - Accuracy, AUC, MSE, etc.
- `SingleLayerContributionAnalyzer` - Layer-wise contributions
- `NoisePerturbationAnalyzer` - Robustness to noise

### Tools
- `CorrelationAnalyzer` - Synaptic correlation analysis
- `WeightAnalyzer` - Weight distribution analysis
- `SynapticActivationAnalyzer` - Synaptic activity patterns
- `SynapticTurnoverAnalyzer` - Track synaptic changes

### Utilities
- `AnalysisManager` - Coordinate all analyses

## Datasets

### Synthetic
- `LineDataset` - Line detection
- `CenterSurroundDataset` - Center-surround discrimination
- `OrientationBarsDataset` - Orientation selectivity
- `MultiXORDataset` - Complex XOR tasks
- `InfoShuntingDataset` - Shunting inhibition tasks
- `NoisyLineDataset` - Noisy line detection
- `SparseMNISTDataset` - Sparse MNIST variants
- `SyntheticCorrelatedDataset` - Correlated features

### Data Loading
- `load_mnist_as_datasets()` - Load MNIST
- `load_cifar10_as_datasets()` - Load CIFAR-10
- `get_unified_datasets()` - Unified interface

## Configuration

- `Config` - Main configuration class
- `ModelConfig` - Model architecture config
- `TrainConfig` - Training parameters
- `AnalysisConfig` - Analysis settings
- `load_config()` - Load from YAML

## Activations

- `ParametricTanh` - Learnable tanh activation
- `ParametricLinearSigmoid` - Learnable linear-sigmoid
- `ParametricLinearTanh` - Learnable linear-tanh

## Loss Functions

- Cross-entropy (classification)
- MSE (regression)
- Voltage stabilization loss
- Custom loss functions

## Scripts

Main training scripts are in `src/dendritic_modeling/scripts/`:
- `train_experiments.py` - Main experiment runner
- `train_encoder_network.py` - Encoder pre-training

## NOT Implemented

The following components mentioned in some documentation do NOT exist:
- DendriticRNN / DendriticLSTM
- DendriticTransformer
- DendriticResNet / DendriticUNet
- DendriticActorCritic
- CorticalMicrocircuit
- Medical imaging specific components
- NLP-specific components

These would need to be implemented if required for your use case. 