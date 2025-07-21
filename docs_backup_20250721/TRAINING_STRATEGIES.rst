Training Strategies Guide
==================================================

This guide provides comprehensive coverage of all training strategies available in the Dendritic Modeling Framework, including their theoretical foundations, biological motivation, and practical implementation details.

Table of Contents
------------------------------

#. [Standard Training](#standard-training)
#. [Layer-wise Training](#layer-wise-training)
#. [Branch-wise Training (Local Credit Assignment)](#branch-wise-training-local-credit-assignment)
#. [Two-Step Training](#two-step-training)
#. [Voltage Stabilization](#voltage-stabilization)
#. [Specialized Training Strategies](#specialized-training-strategies)
#. [Strategy Selection Guide](#strategy-selection-guide)
#. [Optimization and Hyperparameters](#optimization-and-hyperparameters)

---

Standard Training
------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~

Standard training uses traditional backpropagation with gradient descent, adapted for dendritic architectures with specialized gradient scaling strategies.

Theoretical Foundation
~~~~~~~~~~~~~~~~~~~~

* **Global optimization**: All parameters updated simultaneously
* **End-to-end gradient flow**: Gradients flow from output to input
* **Gradient scaling**: Custom scaling for different dendritic components
* **Regularization**: L2 weight decay on log-transformed weights

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~

`````
θ_{t+1} = θ_t - α ∇L(θ_t) - λ ||log(θ_t)||²
`````

Where:
* ``α`` is the learning rate (component-specific if ``split_params=true``)
* ``λ`` is the weight decay rate
* ``∇L`` includes gradient scaling for dendritic components

Configuration
~~~~~~~~~~~~~~~~~~~~

`````yaml
train:
  learning_strategy: standard
  
  # Optimizer settings
  optim_dict:
    split_params: true                    # Component-specific learning rates
    input_net_lr: 0.001                  # Input network parameters
    topk_lr: 0.005                       # Synaptic weights
    blocklinear_lr: 0.001                # Branch aggregation weights
    reactivation_lr: 0.01                # Activation parameters
    output_layer_lr: 0.001               # Output layer
  
  weight_decay_rate: 0.1                 # L2 regularization strength
  
  trainer:
    epochs: 100
    batch_size: 128
    loss_function: categorical_nll
    early_stopping: true
    patience: 10
    grad_clip_value: 5.0                 # Gradient clipping
`````

Gradient Scaling Strategies
~~~~~~~~~~~~~~~~~~~~

`````yaml
model:
  network:
    parameters:
      # Reactivation gradient scaling
      reactivation_strategy: inverse      # Scale by 1/derivative
      
      # TopK layer gradient scaling
      topk_strategy: conductance_dynamic  # Scale by total conductance
      
      # Branch layer gradient scaling  
      blocklinear_strategy: distal_upweight_by_idx  # Upweight distal branches
`````

Use Cases
~~~~~~~~~~~~~~~~~~~~

* **Standard neural networks**: When biological constraints are not critical
* **Baseline comparisons**: Establishing performance benchmarks
* **Simple architectures**: Single-layer EINets or MLPs
* **Fast prototyping**: Quick experimental iterations

Advantages
~~~~~~~~~~~~~~~~~~~~

* Fast convergence for well-behaved problems
* Stable training dynamics
* Well-understood optimization properties
* Good for parameter-matched comparisons

Disadvantages
~~~~~~~~~~~~~~~~~~~~

* May struggle with deep dendritic hierarchies
* Potential gradient flow issues in complex architectures
* Less biologically plausible than local learning approaches

---

Layer-wise Training
------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~

Layer-wise training (freeze layers strategy) trains network layers sequentially, either from soma to input (standard) or input to soma (reversed).

Biological Motivation
~~~~~~~~~~~~~~~~~~~~

* **Developmental learning**: Mimics how neural circuits develop over time
* **Local plasticity**: Each layer adapts to its immediate input statistics
* **Reduced interference**: Prevents catastrophic forgetting between layers
* **Hierarchical learning**: Natural progression from simple to complex features

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~

For standard layer-wise training (soma to input):

`````
Phase 1: Train only layer 1, freeze layers 2,...,N
Phase 2: Train layers 1-2, freeze layers 3,...,N  
...
Phase N: Train all layers together (fine-tuning)
`````

Configuration
~~~~~~~~~~~~~~~~~~~~

`````yaml
train:
  learning_strategy: freeze_layers
  
  trainer:
    # Pre-training phase
    pretrain_epochs: 10                   # Full network warm-up
    
    # Layer-wise phases
    epochs_per_layer: 10                  # Training epochs per layer
    reverse_training: false               # false: soma→input, true: input→soma
    
    # Fine-tuning phase
    final_tune_epochs: 20                 # End-to-end fine-tuning
    
    # Standard training parameters
    batch_size: 128
    loss_function: categorical_nll
    early_stopping: true
    patience: 10
`````

Training Phases
~~~~~~~~~~~~~~~~~~~~

Phase 1: Pre-training (Optional)
^^^^^^^^^^^^^^^
`````yaml
pretrain_epochs: 10    # Train all layers together briefly
`````
* Initializes all layers with end-to-end signal
* Provides stable starting point
* Can be skipped for some architectures

Phase 2: Layer-wise Training
^^^^^^^^^^^^^^^
`````yaml
epochs_per_layer: 10   # Train each layer individually
`````

**Standard Direction (soma→input):**
#. Train layer closest to output (soma)
#. Add next layer toward input
#. Continue until all layers included

**Reverse Direction (input→soma):**
#. Train layer closest to input
#. Add next layer toward output
#. Continue until all layers included

Phase 3: Fine-tuning
^^^^^^^^^^^^^^^
`````yaml
final_tune_epochs: 20  # End-to-end optimization
`````
* All layers trainable
* Global optimization
* Refines layer interactions

Advanced Options
~~~~~~~~~~~~~~~~~~~~

Gradual Unfreezing
^^^^^^^^^^^^^^^
`````yaml
Custom phase durations
==================================================
phase_epochs: [5, 10, 15, 20]           # Increasing epochs per phase
`````

Learning Rate Scheduling
^^^^^^^^^^^^^^^
`````yaml
optim_dict:
  phase_lr_schedule:
    phase_1: 0.01
    phase_2: 0.005
    phase_3: 0.001                       # Decreasing rates
`````

Use Cases
~~~~~~~~~~~~~~~~~~~~

* **Deep dendritic networks**: Networks with 3+ layers
* **Hierarchical feature learning**: When layer specialization is desired
* **Gradient flow issues**: When standard training shows vanishing gradients
* **Interpretable learning**: Understanding layer-by-layer development

Advantages
~~~~~~~~~~~~~~~~~~~~

* Better gradient flow in deep networks
* More stable training dynamics
* Clear layer specialization
* Reduced overfitting risk

Disadvantages
~~~~~~~~~~~~~~~~~~~~

* Longer total training time
* Complex hyperparameter tuning
* May suboptimal global solutions
* Requires careful phase management

---

Branch-wise Training (Local Credit Assignment)
------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~

Branch-wise training implements local credit assignment by training dendritic branches sequentially, allowing each branch to specialize independently.

Biological Motivation
~~~~~~~~~~~~~~~~~~~~

* **Local synaptic plasticity**: Synapses modify based on local signals
* **Branch independence**: Each dendritic branch can learn different features
* **Spike-timing dependent plasticity**: Local temporal correlations drive learning
* **Compartmentalized learning**: Separate learning in different dendritic compartments

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~

For a network with ``B`` branches:

`````
Phase 1: Train branch 1, freeze branches 2,...,B
Phase 2: Train branches 1-2, freeze branches 3,...,B
...
Phase B: Train all branches together
`````

Each branch optimizes local objectives while maintaining global coherence.

Configuration
~~~~~~~~~~~~~~~~~~~~

`````yaml
train:
  learning_strategy: freeze_branches
  
  trainer:
    # Pre-training phase
    pretrain_epochs: 10                   # Full network initialization
    
    # Branch-wise phases  
    epochs_per_branch: 10                 # Training epochs per branch
    reverse_training: false               # false: proximal→distal, true: distal→proximal
    
    # Fine-tuning phase
    final_tune_epochs: 20                 # Global optimization
    
    # Standard parameters
    batch_size: 128
    loss_function: categorical_nll
`````

Training Phases
~~~~~~~~~~~~~~~~~~~~

Phase 1: Pre-training
^^^^^^^^^^^^^^^
* Brief end-to-end training
* Establishes global signal flow
* Provides branch initialization

Phase 2: Sequential Branch Training
^^^^^^^^^^^^^^^

**Standard Direction (proximal→distal):**
#. Train branches closest to soma
#. Progressively add more distal branches
#. Maintains output signal quality

**Reverse Direction (distal→proximal):**
#. Train most distal branches first
#. Add branches toward soma
#. Builds features from bottom up

Phase 3: Global Fine-tuning
^^^^^^^^^^^^^^^
* All branches trainable
* Optimizes branch interactions
* Refines global performance

Local Credit Assignment Variants
~~~~~~~~~~~~~~~~~~~~

Information-Based Credit Assignment
^^^^^^^^^^^^^^^
`````yaml
trainer:
  credit_assignment_type: information
  branch_info_weight: 0.5               # Weight for information-based objectives
`````

Correlation-Based Assignment
^^^^^^^^^^^^^^^
`````yaml
trainer:
  credit_assignment_type: correlation
  correlation_target: 0.8               # Target correlation between branches
`````

Use Cases
~~~~~~~~~~~~~~~~~~~~

* **Multi-branch dendritic networks**: DendriNet with complex branching
* **Feature specialization**: When branch-specific features are desired  
* **Biological modeling**: Studying local plasticity mechanisms
* **Interpretable representations**: Understanding branch-specific computations

Advantages
~~~~~~~~~~~~~~~~~~~~

* Natural branch specialization
* Biologically realistic learning
* Robust to local optima
* Clear interpretability

Disadvantages
~~~~~~~~~~~~~~~~~~~~

* Computationally expensive
* Complex coordination between branches
* May sacrifice global optimality
* Requires careful branch identification

---

Two-Step Training
------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~

Two-step training combines local pre-training with global fine-tuning, using auxiliary supervision heads for each layer during the pre-training phase.

Theoretical Foundation
~~~~~~~~~~~~~~~~~~~~

* **Local supervision**: Each layer receives direct supervision
* **Hierarchical representations**: Layers learn meaningful features independently
* **Global coordination**: Fine-tuning aligns layer representations
* **Auxiliary heads**: Temporary supervision heads provide local gradients

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~

Phase 1: Local Pre-training
^^^^^^^^^^^^^^^
`````
L_total = Σᵢ λᵢ L_local(hᵢ, y)
`````
Where ``hᵢ`` is the output of auxiliary head at layer ``i``.

Phase 2: Global Fine-tuning
^^^^^^^^^^^^^^^
`````
L_total = L_global(f(x), y)
`````
Where ``f(x)`` is the full network output.

Configuration
~~~~~~~~~~~~~~~~~~~~

`````yaml
train:
  learning_strategy: two_step
  
  trainer:
    # Phase 1: Local pre-training
    pretrain_epochs: 20                   # Local supervision epochs
    
    # Phase 2: Global fine-tuning
    epochs: 100                           # End-to-end training epochs
    
    # Standard parameters
    batch_size: 128
    loss_function: categorical_nll
`````

Two-Step with KL Divergence
~~~~~~~~~~~~~~~~~~~~

Version with KL regularization between local and global representations:

`````yaml
train:
  learning_strategy: two_step_kl
  
  trainer:
    pretrain_epochs: 20
    epochs: 100
    kl_weight: 0.1                        # KL divergence weight
    temperature: 3.0                      # Softmax temperature for KL
`````

KL Regularization Loss
^^^^^^^^^^^^^^^
`````
L_total = L_classification + λ_KL Σᵢ KL(p_local^i || p_global^i)
`````

Auxiliary Head Architecture
~~~~~~~~~~~~~~~~~~~~

The framework automatically creates auxiliary heads for each layer:

`````python
Automatic head creation based on layer output dimensions
==================================================
for layer in network.layers:
    head = Linear(layer.output_dim, num_classes)
    auxiliary_heads.append(head)
`````

Advanced Two-Step Variants
~~~~~~~~~~~~~~~~~~~~

Hierarchical Two-Step
^^^^^^^^^^^^^^^
`````yaml
trainer:
  hierarchical_pretraining: true
  layer_pretrain_schedule:
    layer_1: 10                          # Epochs for first layer
    layer_2: 15                          # More epochs for deeper layers
    layer_3: 20
`````

Progressive Two-Step
^^^^^^^^^^^^^^^
`````yaml
trainer:
  progressive_supervision: true
  supervision_decay: 0.9                 # Gradually reduce local supervision
`````

Use Cases
~~~~~~~~~~~~~~~~~~~~

* **Deep networks**: Networks with multiple processing stages
* **Representation learning**: When interpretable intermediate features are needed
* **Gradient flow issues**: When deep networks struggle to train
* **Transfer learning**: Pre-trained layers can be reused

Advantages
~~~~~~~~~~~~~~~~~~~~

* Excellent gradient flow
* Meaningful intermediate representations
* Robust training dynamics
* Good for deep architectures

Disadvantages
~~~~~~~~~~~~~~~~~~~~

* Additional computational overhead
* More complex implementation
* Auxiliary head tuning required
* May over-regularize representations

---

Voltage Stabilization
------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~

Voltage stabilization implements a biologically-motivated warm-up procedure that stabilizes dendritic voltages before standard training.

Biological Motivation
~~~~~~~~~~~~~~~~~~~~

* **Homeostatic plasticity**: Neurons maintain stable activity levels
* **Voltage regulation**: Dendritic voltages are actively controlled
* **Developmental stability**: Young networks require stabilization
* **Excitation-inhibition balance**: Proper E-I balance is established

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~

Stabilization Loss
^^^^^^^^^^^^^^^
`````
L_stabilization = Σᵢ (V_dendritic^i - V_target)² + L_classification
`````

Where ``V_target`` is the desired dendritic voltage level.

Configuration
~~~~~~~~~~~~~~~~~~~~

`````yaml
train:
  learning_strategy: voltage_stabilization
  
  trainer:
    # Stabilization phase
    pretrain_epochs: 10                   # Voltage stabilization epochs
    target_voltage: 0.5                   # Target dendritic voltage
    
    # Standard training phase  
    epochs: 100                           # Normal training epochs
    
    # Standard parameters
    batch_size: 128
    loss_function: categorical_nll
`````

Stabilization Process
~~~~~~~~~~~~~~~~~~~~

Phase 1: Voltage Stabilization
^^^^^^^^^^^^^^^
#. **Voltage monitoring**: Track dendritic voltages across branches
#. **Stability objective**: Minimize deviation from target voltage
#. **E-I balancing**: Adjust excitation-inhibition ratios
#. **Gradual transition**: Smoothly transition to classification objective

Phase 2: Standard Training
^^^^^^^^^^^^^^^
#. **Remove voltage constraint**: Switch to pure classification objective
#. **Fine-tune dynamics**: Optimize with stable voltage baseline
#. **Maintain stability**: Monitor for voltage drift

Advanced Stabilization Options
~~~~~~~~~~~~~~~~~~~~

Multi-Target Stabilization
^^^^^^^^^^^^^^^
`````yaml
trainer:
  voltage_targets:
    excitatory: 0.6                      # Target for excitatory dendrites
    inhibitory: 0.4                      # Target for inhibitory dendrites
`````

Adaptive Stabilization
^^^^^^^^^^^^^^^
`````yaml
trainer:
  adaptive_voltage: true
  voltage_adaptation_rate: 0.01          # How quickly targets adapt
`````

Layer-Specific Stabilization
^^^^^^^^^^^^^^^
`````yaml
trainer:
  layer_voltage_targets: [0.3, 0.5, 0.7] # Different targets per layer
`````

Use Cases
~~~~~~~~~~~~~~~~~~~~

* **EINet training**: When E-I balance is critical
* **Biological modeling**: Modeling homeostatic mechanisms
* **Unstable dynamics**: When training shows voltage instability
* **Deep dendritic networks**: Multi-layer dendritic architectures

Advantages
~~~~~~~~~~~~~~~~~~~~

* Stable training dynamics
* Biologically realistic
* Prevents pathological states
* Good for E-I networks

Disadvantages
~~~~~~~~~~~~~~~~~~~~

* Additional hyperparameter tuning
* Longer training time
* May constrain learning
* Requires voltage monitoring

---

Specialized Training Strategies
------------------------------

Train Only M Reactivation
~~~~~~~~~~~~~~~~~~~~

Specialized strategy that focuses training on slope parameters while freezing bias terms.

`````yaml
train:
  learning_strategy: train_only_m_reactivation
  
  # Automatically freezes log_b parameters
  # Trains all other parameters normally
`````

**Use Cases:**
* Studying slope vs. bias parameter importance
* Constrained optimization experiments
* Reactivation function analysis

Custom Training Strategies
~~~~~~~~~~~~~~~~~~~~

The framework supports custom training strategies through the factory pattern:

`````python
Custom strategy implementation
==================================================
class CustomTrainer(Trainer):
    def train(self, model, train_data, valid_data):
        # Custom training logic
        pass

Registration
==================================================
def get_trainer(strategy, ...):
    if strategy == "custom":
        return CustomTrainer(...)
`````

---

Strategy Selection Guide
------------------------------

Task-Based Selection
~~~~~~~~~~~~~~~~~~~~

| Task Characteristics | Recommended Strategy | Rationale |
|---------------------|---------------------|-----------|
| **Simple datasets** | Standard | Fast convergence, well-understood |
| **Deep networks (3+ layers)** | Layer-wise or Two-step | Improved gradient flow |
| **Multi-branch architectures** | Branch-wise | Natural branch specialization |
| **EINet with E-I dynamics** | Voltage stabilization | Stable E-I balance |
| **Representation learning** | Two-step | Interpretable intermediate features |
| **Biological modeling** | Branch-wise or Voltage stabilization | More realistic learning |
| **Large datasets** | Standard with split_params | Efficient component-specific learning |

Architecture-Based Selection
~~~~~~~~~~~~~~~~~~~~

| Architecture | Primary Strategy | Alternative | Notes |
|-------------|-----------------|-------------|-------|
| **Single-layer EINet** | Standard | Voltage stabilization | Simple enough for standard training |
| **Multi-layer EINet** | Layer-wise | Two-step | Manage layer interactions |
| **DendriNet** | Branch-wise | Two-step | Leverage branch structure |
| **Hybrid CNN+EINet** | Two-step | Standard | Complex representations need local supervision |
| **MLP** | Standard | - | Standard backprop is optimal |

Performance Considerations
~~~~~~~~~~~~~~~~~~~~

| Factor | Standard | Layer-wise | Branch-wise | Two-step | Voltage Stab. |
|--------|----------|------------|-------------|----------|---------------|
| **Training Speed** | Fast | Medium | Slow | Medium | Medium |
| **Memory Usage** | Low | Low | Medium | High | Low |
| **Convergence Quality** | Good | Very Good | Good | Excellent | Good |
| **Biological Realism** | Low | Medium | High | Medium | High |
| **Interpretability** | Low | Medium | High | High | Medium |

Debugging Strategy Issues
~~~~~~~~~~~~~~~~~~~~

Poor Convergence
^^^^^^^^^^^^^^^
#. Try two-step training for improved initialization
#. Use layer-wise if deep network struggles
#. Check learning rates for each component

Training Instability
^^^^^^^^^^^^^^^
#. Use voltage stabilization for E-I networks
#. Add gradient clipping
#. Reduce learning rates

Overfitting
^^^^^^^^^^^^^^^
#. Add weight decay
#. Use early stopping
#. Try branch-wise training for regularization

---

Optimization and Hyperparameters
------------------------------

Learning Rate Strategies
~~~~~~~~~~~~~~~~~~~~

Component-Specific Learning Rates
^^^^^^^^^^^^^^^
`````yaml
optim_dict:
  split_params: true
  input_net_lr: 0.001      # Usually lower for input processing
  topk_lr: 0.005           # Higher for sparse synaptic weights
  blocklinear_lr: 0.001    # Standard for dense layers
  reactivation_lr: 0.01    # Higher for activation parameters
  output_layer_lr: 0.001   # Standard for final layer
`````

Learning Rate Scheduling
^^^^^^^^^^^^^^^
`````yaml
trainer:
  lr_scheduler:
    type: StepLR
    step_size: 30
    gamma: 0.1               # Reduce LR by 10x every 30 epochs
`````

Weight Decay
~~~~~~~~~~~~~~~~~~~~

Standard Weight Decay
^^^^^^^^^^^^^^^
`````yaml
weight_decay_rate: 0.1     # L2 regularization on log-weights
`````

Component-Specific Decay
^^^^^^^^^^^^^^^
`````yaml
weight_decay:
  topk_weights: 0.1
  branch_weights: 0.05
  activation_params: 0.01
`````

Gradient Management
~~~~~~~~~~~~~~~~~~~~

Gradient Clipping
^^^^^^^^^^^^^^^
`````yaml
trainer:
  grad_clip_value: 5.0     # Clip gradients to prevent explosion
  grad_clip_norm: 10.0     # Alternatively, clip by norm
`````

Gradient Scaling
^^^^^^^^^^^^^^^
`````yaml
parameters:
  reactivation_strategy: inverse        # Scale by 1/derivative
  topk_strategy: conductance_dynamic    # Scale by conductance
  blocklinear_strategy: distal_upweight_by_idx  # Upweight distal
`````

Training Monitoring
~~~~~~~~~~~~~~~~~~~~

Early Stopping
^^^^^^^^^^^^^^^
`````yaml
trainer:
  early_stopping: true
  patience: 10             # Stop if no improvement for 10 epochs
  min_delta: 0.001         # Minimum improvement threshold
`````

Loss Plotting
^^^^^^^^^^^^^^^
`````yaml
trainer:
  plot_losses: true        # Generate loss curves
  save_path: "./plots"     # Save location
`````

Advanced Optimization
~~~~~~~~~~~~~~~~~~~~

Custom Optimizers
^^^^^^^^^^^^^^^
`````yaml
train:
  optimizer_type: CustomWeightDecayOptimizer
  base_optimizer: Adam
  optimizer_kwargs:
    betas: [0.9, 0.999]
    eps: 1e-8
`````

Parameter Groups
^^^^^^^^^^^^^^^
`````python
Automatic parameter grouping
==================================================
param_groups = model.get_param_groups(config.param_groups)
optimizer = torch.optim.Adam(param_groups)
`````

---

Best Practices
------------------------------

Training Strategy Selection
~~~~~~~~~~~~~~~~~~~~

#. **Start simple**: Begin with standard training
#. **Add complexity gradually**: Move to specialized strategies as needed
#. **Match strategy to architecture**: Use biological strategies for biological architectures
#. **Monitor training dynamics**: Watch for convergence issues
#. **Compare strategies**: Try multiple approaches for important experiments

Hyperparameter Tuning
~~~~~~~~~~~~~~~~~~~~

#. **Learning rates**: Start with component-specific rates
#. **Batch sizes**: Adjust based on memory constraints
#. **Weight decay**: Balance regularization vs. performance
#. **Training duration**: Use early stopping to prevent overfitting

Debugging Training
~~~~~~~~~~~~~~~~~~~~

#. **Check gradients**: Monitor gradient magnitudes and directions
#. **Visualize training**: Use loss plots and activation monitoring
#. **Validate implementation**: Compare with simpler baselines
#. **Monitor resources**: Watch memory and compute usage

Experiment Design
~~~~~~~~~~~~~~~~~~~~

#. **Control variables**: Change one strategy aspect at a time
#. **Multiple seeds**: Run experiments with different random seeds
#. **Comprehensive evaluation**: Use multiple metrics and analysis tools
#. **Document choices**: Record strategy selection rationale

---

Summary
------------------------------

The Dendritic Modeling Framework provides a comprehensive suite of training strategies:

* **Standard**: Fast, reliable, good for most tasks
* **Layer-wise**: Suitable for deep networks, hierarchical learning
* **Branch-wise**: Biologically realistic, good for interpretability
* **Two-step**: Excellent for representation learning and deep networks
* **Voltage stabilization**: Essential for stable E-I dynamics

Choose strategies based on your architecture, task complexity, and research goals. Start with standard training and add biological constraints as needed for your specific research questions. 