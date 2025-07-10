# Training Strategies Guide

This guide covers all training strategies available in the Dendritic Modeling Framework, from standard backpropagation to biologically-inspired local learning approaches.

## Table of Contents

1. [Overview](#overview)
2. [Standard Training](#standard-training)
3. [Local Learning Strategies](#local-learning-strategies)
4. [Multi-Stage Training](#multi-stage-training)
5. [Specialized Strategies](#specialized-strategies)
6. [Optimizer Configuration](#optimizer-configuration)
7. [Training Parameters](#training-parameters)
8. [Strategy Selection Guide](#strategy-selection-guide)

## Overview

The framework provides multiple training strategies designed for different network architectures and learning objectives:

- **Standard Training**: Traditional backpropagation
- **Local Learning**: Layer-wise and branch-wise training
- **Multi-Stage Training**: Sequential training phases
- **Specialized Training**: Biologically-motivated approaches
- **Custom Training**: User-defined strategies

Each strategy can be configured with specific parameters and combined with various optimization techniques.

## Standard Training

**Standard training** uses traditional backpropagation with gradient-based optimization.

### Configuration

```yaml
train:
  learning_strategy: standard
  
  trainer:
    epochs: 100
    batch_size: 128
    loss_function: categorical_nll
    early_stopping: true
    patience: 10
    grad_clip_value: 5.0
```

### Key Features

- **End-to-end training**: All parameters updated simultaneously
- **Global optimization**: Uses full network gradients
- **Efficient**: Fastest training approach
- **Stable**: Well-established convergence properties

### When to Use

- **Simple networks**: MLPs and basic EINets
- **Standard tasks**: Classification and regression
- **Baseline experiments**: Comparison with other strategies
- **Production**: When training speed is critical

### Mathematical Formulation

For standard backpropagation:

1. **Forward pass**: Compute outputs and loss
2. **Backward pass**: Compute gradients via chain rule
3. **Parameter update**: Apply optimizer step

```
L = loss(f(x; θ), y)
∇θ = ∂L/∂θ
θ ← θ - α∇θ
```

## Local Learning Strategies

Local learning strategies train network components sequentially, enabling better gradient flow and biologically-plausible learning.

### Layer-wise Training

**Layer-wise training** (freeze_layers) trains network layers sequentially from soma to input.

#### Configuration

```yaml
train:
  learning_strategy: freeze_layers
  
  trainer:
    epochs: 100                 # Total epochs
    pretrain_epochs: 20         # Full network pretraining
    epochs_per_layer: 10        # Epochs per layer
    final_tune_epochs: 20       # Final fine-tuning
    reverse_training: false     # Train soma→input (false) or input→soma (true)
```

#### Training Process

1. **Pretraining**: Train full network for initialization
2. **Layer freezing**: Freeze all layers except current
3. **Sequential training**: Train each layer individually
4. **Final tuning**: Train full network with best weights

#### Example

```yaml
# Layer-wise training example
train:
  learning_strategy: freeze_layers
  
  trainer:
    epochs: 80
    pretrain_epochs: 10
    epochs_per_layer: 15
    final_tune_epochs: 25
    reverse_training: false     # Train from soma outward
    
    # Standard training parameters
    batch_size: 128
    loss_function: categorical_nll
    grad_clip_value: 1.0
```

### Branch-wise Training

**Branch-wise training** (freeze_branches) trains dendritic branches sequentially for local credit assignment.

#### Configuration

```yaml
train:
  learning_strategy: freeze_branches
  
  trainer:
    epochs: 100
    pretrain_epochs: 15
    epochs_per_branch: 8
    final_tune_epochs: 25
    reverse_training: false     # Train soma→input (false) or input→soma (true)
```

#### Training Process

1. **Pretraining**: Initialize with full network training
2. **Branch identification**: Find all dendritic branch layers
3. **Sequential training**: Train each branch independently
4. **Final tuning**: Global fine-tuning with best weights

#### Benefits

- **Local credit assignment**: Each branch learns specific features
- **Better gradient flow**: Avoids vanishing gradients
- **Biological plausibility**: Mimics local synaptic plasticity
- **Stable training**: Reduces optimization landscape complexity

## Multi-Stage Training

Multi-stage strategies combine local and global training phases for improved performance.

### Two-Step Training

**Two-step training** combines local pretraining with global fine-tuning.

#### Configuration

```yaml
train:
  learning_strategy: two_step
  
  trainer:
    epochs: 100                 # Global training epochs
    pretrain_epochs: 30         # Local pretraining epochs
```

#### Training Process

**Stage 1: Local Pretraining**
1. Create local supervision heads for each layer
2. Train each layer with local classification loss
3. Learn layer-specific representations

**Stage 2: Global Fine-tuning**
1. Remove local supervision heads
2. Train full network end-to-end
3. Refine global representations

#### Mathematical Details

**Local pretraining loss:**
```
L_local = Σ_i CE(h_i(x), y)
```
where `h_i` are local classification heads.

**Global fine-tuning loss:**
```
L_global = CE(f(x), y)
```
where `f` is the full network.

### Two-Step with KL Regularization

**Two-step KL training** adds regularization during local pretraining.

#### Configuration

```yaml
train:
  learning_strategy: two_step_kl
  
  trainer:
    epochs: 100
    pretrain_epochs: 30
    kl_weight: 0.1              # KL divergence weight
```

#### Loss Function

Combines classification loss with KL divergence regularization:

```
L_total = L_classification + λ * KL(p_local || p_global)
```

- **Classification loss**: Standard cross-entropy
- **KL regularization**: Encourages consistency between local and global predictions
- **λ (kl_weight)**: Balance between the two terms

#### Benefits

- **Consistency**: Local and global representations align
- **Regularization**: Prevents overfitting during pretraining
- **Stability**: Smoother transition between training phases

## Specialized Strategies

### Voltage Stabilization

**Voltage stabilization** includes a biologically-motivated warm-up phase.

#### Configuration

```yaml
train:
  learning_strategy: voltage_stabilization
  
  trainer:
    epochs: 100                 # Main training epochs
    pretrain_epochs: 15         # Voltage stabilization epochs
    target_voltage: 0.5         # Target dendritic voltage
```

#### Training Process

**Phase 1: Voltage Stabilization**
1. Use specialized voltage stabilization loss
2. Target specific dendritic voltage levels
3. Establish stable baseline activity

**Phase 2: Standard Training**
1. Switch to classification/regression loss
2. Train with stabilized voltage dynamics
3. Maintain dendritic stability

#### Voltage Stabilization Loss

```
L_voltage = MSE(V_dendrite, V_target)
```

where `V_dendrite` are dendritic voltages and `V_target` is the target voltage.

#### Benefits

- **Stable dynamics**: Prevents unstable dendritic activity
- **Biological realism**: Mimics dendritic homeostasis
- **Better convergence**: Establishes good initialization

### Specialized Parameter Training

**Train only M reactivation** focuses on specific parameter subsets.

#### Configuration

```yaml
train:
  learning_strategy: specialized
  
  # Custom trainer parameters
  trainer:
    train_only_m_reactivation: true  # Train only slope parameters
    freeze_log_b: true               # Freeze bias parameters
```

#### Use Cases

- **Parameter analysis**: Study specific parameter importance
- **Transfer learning**: Fine-tune specific components
- **Ablation studies**: Isolate parameter contributions

## Optimizer Configuration

### Basic Optimization

```yaml
train:
  optim_dict:
    split_params: false         # Single learning rate
    lr: 0.001                  # Learning rate
```

### Parameter Group Optimization

Different parameter types can use different learning rates:

```yaml
train:
  optim_dict:
    split_params: true
    
    # Component-specific learning rates
    input_net_lr: 0.001        # Input transformation network
    topk_lr: 0.005             # Synaptic weights (TopK layers)
    blocklinear_lr: 0.001      # Branch aggregation weights
    reactivation_lr: 0.01      # Activation parameters
    output_layer_lr: 0.001     # Final classification layer
```

### Parameter Group Benefits

- **Balanced learning**: Different components learn at appropriate rates
- **Stability**: Prevents some parameters from dominating
- **Efficiency**: Faster convergence with proper rates
- **Flexibility**: Adapt to different network components

### Weight Decay Configuration

```yaml
train:
  weight_decay_rate: 0.1        # L2 regularization strength
  
  # Applied to log-space weights in TopK layers
  # Promotes sparse connectivity patterns
```

## Training Parameters

### Basic Training Configuration

```yaml
train:
  trainer:
    # Training duration
    epochs: 100                 # Number of training epochs
    batch_size: 128            # Batch size
    
    # Data handling
    shuffle: true              # Shuffle training data
    
    # Loss and optimization
    loss_function: categorical_nll  # Loss function
    grad_clip_value: 5.0       # Gradient clipping threshold
```

### Early Stopping

```yaml
train:
  trainer:
    early_stopping: true       # Enable early stopping
    patience: 10               # Epochs to wait for improvement
    load_best_state_dict: true # Load best model after training
```

### Logging and Visualization

```yaml
train:
  trainer:
    # Output control
    plot_losses: true          # Plot training curves
    suppress_prints: false     # Console output
    print_every: 10           # Print frequency (epochs)
    
    # Model saving
    save_path: ./checkpoints   # Checkpoint directory
```

### Available Loss Functions

| Loss Function | Use Case | Configuration |
|---------------|----------|---------------|
| `categorical_nll` | Multi-class classification | Default for classification |
| `mse` | Regression | `loss_function: mse` |
| `cross_entropy` | Classification (alternative) | `loss_function: cross_entropy` |
| `bce` | Binary classification | `loss_function: bce` |

## Strategy Selection Guide

### Decision Tree

```
Network Type?
├── MLP
│   └── Use: standard
├── Simple EINet (1-2 layers)
│   └── Use: standard or freeze_layers
└── Complex EINet (3+ layers)
    ├── Fast training needed?
    │   └── Use: standard
    ├── Better performance needed?
    │   └── Use: two_step or freeze_layers
    └── Biological realism needed?
        └── Use: voltage_stabilization or freeze_branches
```

### Performance Comparison

| Strategy | Speed | Performance | Biological Realism | Complexity |
|----------|-------|-------------|-------------------|------------|
| **standard** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐ |
| **freeze_layers** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| **freeze_branches** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **two_step** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **two_step_kl** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **voltage_stabilization** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

### Task-Specific Recommendations

#### Image Classification
```yaml
# MNIST (simple)
learning_strategy: standard

# CIFAR-10 (complex)
learning_strategy: two_step
trainer:
  pretrain_epochs: 25
  epochs: 75
```

#### Synthetic Tasks
```yaml
# XOR, line detection, etc.
learning_strategy: freeze_branches
trainer:
  epochs_per_branch: 12
  final_tune_epochs: 30
```

#### Regression Tasks
```yaml
learning_strategy: standard
trainer:
  loss_function: mse
  epochs: 150
  early_stopping: true
```

#### Research/Analysis
```yaml
learning_strategy: voltage_stabilization
trainer:
  pretrain_epochs: 20
  target_voltage: 0.4
  epochs: 120
```

### Architecture-Specific Guidelines

#### EINet with Standard TopK
```yaml
train:
  learning_strategy: standard    # Simple and effective
  
  optim_dict:
    split_params: true
    topk_lr: 0.005
    reactivation_lr: 0.01
```

#### EINet with Stochastic TopK
```yaml
train:
  learning_strategy: two_step    # Handle exploration better
  
  trainer:
    pretrain_epochs: 30
    epochs: 80
```

#### EINet with DeepST
```yaml
train:
  learning_strategy: freeze_layers  # Stable for rewiring
  
  trainer:
    epochs_per_layer: 15
    final_tune_epochs: 40
```

#### Multi-Layer EINet (3+ layers)
```yaml
train:
  learning_strategy: freeze_branches
  
  trainer:
    pretrain_epochs: 20
    epochs_per_branch: 10
    final_tune_epochs: 30
    reverse_training: false    # Train soma→input
```

## Advanced Training Techniques

### Curriculum Learning

```yaml
# Start with easier examples, progress to harder ones
train:
  learning_strategy: two_step
  
  trainer:
    pretrain_epochs: 20        # Learn basics first
    epochs: 60                # Then full complexity
    
task:
  # Use synthetic tasks for pretraining, real data for fine-tuning
  dataset: line               # Simple patterns first
```

### Transfer Learning

```yaml
train:
  learning_strategy: specialized
  
  # Fine-tune specific components
  trainer:
    freeze_encoder: true       # Keep pre-trained features
    train_only_decoder: true   # Update only output layers
```

### Multi-Task Learning

```yaml
train:
  learning_strategy: freeze_branches
  
  # Different branches can specialize for different tasks
  trainer:
    epochs_per_branch: 15
    task_rotation: true        # Rotate between tasks
```

### Hyperparameter Optimization

#### Learning Rate Scheduling

```yaml
train:
  optim_dict:
    split_params: true
    
    # Start high, decay during training
    topk_lr: 0.01             # Initial rate
    lr_decay: 0.95            # Decay factor
    decay_every: 20           # Decay frequency
```

#### Adaptive Training

```yaml
train:
  trainer:
    # Adjust training based on performance
    adaptive_epochs: true     # Extend if improving
    min_epochs: 50           # Minimum training
    max_epochs: 200          # Maximum training
```

## Troubleshooting Training Issues

### Common Problems

#### Vanishing Gradients
**Symptoms**: Loss plateaus early, poor performance
**Solutions**:
```yaml
train:
  learning_strategy: freeze_branches  # Use local learning
  
  optim_dict:
    reactivation_lr: 0.02            # Higher activation LR
    
model:
  core_network:
    parameters:
      reactivation_strategy: inverse  # Gradient scaling
```

#### Exploding Gradients
**Symptoms**: NaN loss, unstable training
**Solutions**:
```yaml
train:
  trainer:
    grad_clip_value: 1.0             # Strong clipping
    
  optim_dict:
    topk_lr: 0.001                   # Lower learning rates
```

#### Slow Convergence
**Symptoms**: Very slow learning, many epochs needed
**Solutions**:
```yaml
train:
  learning_strategy: two_step        # Better initialization
  
  optim_dict:
    split_params: true
    topk_lr: 0.01                    # Higher synaptic LR
    
model:
  core_network:
    parameters:
      topk_strategy: conductance_dynamic  # Better gradients
```

#### Overfitting
**Symptoms**: Large train/validation gap
**Solutions**:
```yaml
train:
  weight_decay_rate: 0.2             # Stronger regularization
  
  trainer:
    early_stopping: true
    patience: 5                      # Stop early
    
analysis:
  noise_perturbation_analysis:
    enabled: true                    # Test robustness
```

### Debugging Strategies

#### Gradient Analysis
```yaml
model:
  core_network:
    parameters:
      print_hooks: true              # Print gradient stats
```

#### Learning Curve Analysis
```yaml
train:
  trainer:
    plot_losses: true                # Visualize training
    print_every: 5                   # Frequent logging
```

#### Component Analysis
```yaml
train:
  learning_strategy: specialized
  
  # Test individual components
  trainer:
    train_only_m_reactivation: true
```

---

This guide covers all training strategies available in the Dendritic Modeling Framework. Each strategy is designed for specific use cases and network architectures. Choose the strategy that best matches your experimental goals and computational constraints. 