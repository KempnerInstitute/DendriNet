# Enhanced Training Features Guide

This guide explains the new enhanced training features that support flexible optimization, regularization, and pruning with detailed reporting.

## Features Overview

1. **Flexible Optimizer Selection**: Choose from Adam, SGD, RMSprop, AdamW with full parameter control
2. **L1/L2 Regularization**: Selective regularization by layer, branch, and weight type
3. **Weight Pruning**: Threshold-based or magnitude-based pruning with selective targeting
4. **Detailed Reporting**: Comprehensive statistics on network sparsity and pruning effects
5. **Re-evaluation**: Automatic performance assessment after pruning

## Quick Start

### 1. Basic Usage

```python
from dendritic_modeling.training.strategies import EnhancedTrainer
from dendritic_modeling.config.training import (
    OptimizerConfig, RegularizationConfig, PruningConfig
)

# Configure optimizer
optimizer_config = OptimizerConfig(
    name="adam",
    lr=0.001,
    weight_decay=0.0001
)

# Configure regularization
regularization_config = RegularizationConfig(
    l1_weight=0.001,
    l2_weight=0.0
)

# Configure pruning
pruning_config = PruningConfig(
    enabled=True,
    threshold=0.001,
    re_evaluate_after=True
)

# Create trainer
trainer = EnhancedTrainer(
    optimizer_config=optimizer_config,
    regularization_config=regularization_config,
    pruning_config=pruning_config,
    epochs=100,
    batch_size=128
)

# Train model
results = trainer.train(model, train_data, valid_data)
```

### 2. Using Configuration Files

```python
from omegaconf import OmegaConf
from dendritic_modeling.training.strategies import create_enhanced_trainer_from_config

# Load configuration
config = OmegaConf.load('configs/enhanced_training_example.yaml')

# Create trainer from config
trainer = create_enhanced_trainer_from_config(
    config.train.main,
    epochs=config.train.main.trainer.epochs,
    batch_size=config.train.main.trainer.batch_size
)

# Train model
results = trainer.train(model, train_data, valid_data)
```

## Configuration Options

### Optimizer Configuration

```yaml
optimizer:
  name: "adam"              # "adam", "sgd", "rmsprop", "adamw"
  lr: 0.001                # Learning rate
  weight_decay: 0.0001     # Built-in L2 regularization
  momentum: 0.9            # For SGD/RMSprop
  betas: [0.9, 0.999]      # For Adam/AdamW
  eps: 1e-8                # For Adam/AdamW
```

### Regularization Configuration

```yaml
regularization:
  l1_weight: 0.001         # L1 regularization strength
  l2_weight: 0.0           # L2 regularization strength
  selective:
    layers: [0, 1, 2]      # Apply only to specific layers (empty = all)
    branches: []           # Apply only to specific branches (empty = all)
    weight_types: ["recurrent", "input"]  # Weight types to regularize
```

### Pruning Configuration

```yaml
pruning:
  enabled: true
  threshold: 0.001         # Prune weights with abs(value) < threshold
  selective:
    layers: []             # Layers to prune (empty = all)
    branches: []           # Branches to prune (empty = all)
    weight_types: ["all"]  # Weight types to prune
  re_evaluate_after: true  # Re-evaluate after pruning
  dynamic_pruning: false   # Prune during training
  pruning_frequency: 10    # Epochs between dynamic pruning
```

## Advanced Usage Examples

### Example 1: Heavy Regularization with Selective Pruning

```python
# Apply strong L1 regularization to recurrent weights only
regularization_config = RegularizationConfig(
    l1_weight=0.01,
    selective=SelectiveConfig(
        weight_types=["recurrent"]
    )
)

# Prune only the first two layers aggressively
pruning_config = PruningConfig(
    enabled=True,
    threshold=0.01,
    selective=SelectiveConfig(
        layers=[0, 1]
    )
)
```

### Example 2: Dynamic Pruning During Training

```python
# Enable dynamic pruning every 20 epochs
pruning_config = PruningConfig(
    enabled=True,
    threshold=0.001,
    dynamic_pruning=True,
    pruning_frequency=20,
    re_evaluate_after=True
)

trainer = EnhancedTrainer(
    optimizer_config=optimizer_config,
    pruning_config=pruning_config
)

# The trainer will automatically apply pruning during training
results = trainer.train(model, train_data, valid_data)
```

### Example 3: SGD with Different Learning Rates

```python
# Use SGD with momentum
optimizer_config = OptimizerConfig(
    name="sgd",
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001
)

# Apply different regularization to different layers
regularization_config = RegularizationConfig(
    l1_weight=0.001,
    selective=SelectiveConfig(
        layers=[2, 3, 4],  # Only deeper layers
        weight_types=["input", "output"]
    )
)
```

## Selective Targeting

### Layer Selection
Specify layer indices to target specific layers:
```yaml
selective:
  layers: [0, 1, 2]  # Apply to layers 0, 1, and 2 only
```

### Branch Selection
Target specific dendritic branches:
```yaml
selective:
  branches: ["apical", "basal"]  # Apply to specific branch types
```

### Weight Type Selection
Target specific types of weights:
```yaml
selective:
  weight_types: ["recurrent", "input", "output"]  # Specific weight types
  # Options: "recurrent", "input", "output", "linear", "all"
```

## Reporting and Analysis

### Automatic Reports
The enhanced trainer provides detailed reports:

```
WEIGHT PRUNING REPORT
==================================================
Total parameters before pruning: 1,000,000
Total parameters after pruning:  750,000
Pruned parameters:               250,000
Pruning ratio:                   0.2500

Layer           Before     After      Pruned     Ratio     
------------------------------------------------------------
layer_0         100,000    80,000     20,000     0.2000
layer_1         500,000    375,000    125,000    0.2500
layer_2         400,000    295,000    105,000    0.2625

Weight Type     Before     After      Pruned     Ratio     
------------------------------------------------------------
recurrent       600,000    450,000    150,000    0.2500
input           300,000    225,000    75,000     0.2500
output          100,000    75,000     25,000     0.2500
```

### Performance Comparison
Automatic before/after pruning comparison:

```
PRUNING PERFORMANCE COMPARISON
==================================================
Pre-pruning  - Loss: 0.234567, Accuracy: 92.34%
Post-pruning - Loss: 0.245678, Accuracy: 91.89%
Change       - Loss: +0.011111, Accuracy: -0.45%
```

### Programmatic Access
Access statistics programmatically:

```python
# Get current model sparsity
sparsity_stats = trainer.get_model_sparsity(model)
print(f"Overall sparsity: {sparsity_stats['overall_sparsity']:.4f}")

# Access training results
results = trainer.train(model, train_data, valid_data)
print(f"Pruning ratio: {results['pruning_stats']['pruning_ratio']:.4f}")
print(f"Performance change: {results['pruning_performance_delta']['accuracy_change']:.2f}%")
```

## Integration with Existing Codebase

### Using with Training Factory

```python
from dendritic_modeling.training.factory import get_trainer

# Configure enhanced training
config = {
    'optimizer': {'name': 'adam', 'lr': 0.001},
    'regularization': {'l1_weight': 0.001},
    'pruning': {'enabled': True, 'threshold': 0.001}
}

# Get enhanced trainer through factory
trainer = get_trainer(
    strategy="enhanced",
    optimizer=None,  # Will be created internally
    trainer_config_dict=config
)
```

### Using with OmegaConf

```python
# Load from YAML configuration
config = OmegaConf.load('configs/enhanced_training_example.yaml')

# Extract training configuration
training_config = config.train.main

# Create trainer
trainer = create_enhanced_trainer_from_config(training_config)
```

## Best Practices

### 1. Start Conservative
Begin with small regularization and pruning values:
```yaml
regularization:
  l1_weight: 0.0001  # Start small
pruning:
  threshold: 0.0001  # Conservative pruning
```

### 2. Use Selective Targeting
Apply regularization/pruning selectively to avoid over-constraining:
```yaml
regularization:
  selective:
    weight_types: ["recurrent"]  # Focus on recurrent connections
pruning:
  selective:
    layers: [1, 2]  # Avoid input/output layers initially
```

### 3. Monitor Performance
Always enable re-evaluation and reporting:
```yaml
pruning:
  re_evaluate_after: true
reporting:
  report_non_pruned: true
  save_pruning_stats: true
```

### 4. Experiment with Dynamic Pruning
For long training runs, consider dynamic pruning:
```yaml
pruning:
  dynamic_pruning: true
  pruning_frequency: 50  # Less frequent for stability
```

## Troubleshooting

### Common Issues

1. **Over-regularization**: Model performance drops significantly
   - Solution: Reduce `l1_weight` or `l2_weight`
   - Use selective targeting to focus on specific layers/weights

2. **Over-pruning**: Model becomes too sparse and loses accuracy
   - Solution: Increase `threshold` value
   - Use selective pruning to preserve critical connections

3. **Memory issues**: Large models with detailed reporting
   - Solution: Disable detailed branch reporting for very large models
   ```yaml
   reporting:
     detailed_branch_report: false
   ```

4. **Slow training**: Heavy regularization/pruning overhead
   - Solution: Reduce pruning frequency or disable dynamic pruning
   - Use selective targeting to reduce computation

### Performance Tuning

1. **For sparse networks**: Use higher learning rates
2. **With L1 regularization**: Consider using AdamW optimizer
3. **With pruning**: Monitor gradient flow and adjust clipping
4. **For large models**: Enable mixed precision training

## Examples for Common Use Cases

### Sparse Neural Networks
```yaml
regularization:
  l1_weight: 0.01
pruning:
  enabled: true
  threshold: 0.005
  dynamic_pruning: true
  pruning_frequency: 25
```

### Biological Realism
```yaml
regularization:
  selective:
    weight_types: ["recurrent"]  # Focus on synaptic connections
pruning:
  selective:
    branches: ["apical"]  # Prune specific dendritic branches
```

### Network Compression
```yaml
pruning:
  enabled: true
  threshold: 0.01  # Aggressive pruning
  re_evaluate_after: true
reporting:
  save_pruning_stats: true  # Track compression ratios
```

This enhanced training system provides powerful tools for exploring network sparsity, regularization effects, and optimization strategies while maintaining detailed tracking of all changes and their effects on model performance.