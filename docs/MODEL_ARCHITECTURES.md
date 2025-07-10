# Model Architectures Guide

This guide provides detailed documentation for all network architectures available in the Dendritic Modeling Framework, from basic MLPs to sophisticated dendritic networks with excitation-inhibition dynamics.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [EINet - Excitation-Inhibition Networks](#einet---excitation-inhibition-networks)
3. [DendriNet - Pure Dendritic Networks](#dendrinet---pure-dendritic-networks)
4. [Classical Networks](#classical-networks)
5. [Input Processing Networks](#input-processing-networks)
6. [Synaptic Mechanisms](#synaptic-mechanisms)
7. [Activation Functions](#activation-functions)
8. [Gradient Scaling Strategies](#gradient-scaling-strategies)

## Architecture Overview

The framework uses a three-component architecture:

```
Input → Encoder Network → Core Network → Decoder Network → Output
```

- **Encoder Network**: Preprocesses raw inputs (Identity, CNN, MLP, AlexNet)
- **Core Network**: Main computational unit (EINet, DendriNet, MLP)
- **Decoder Network**: Post-processes outputs (Identity, MLP)

This modular design allows mixing and matching different components for various tasks.

## EINet - Excitation-Inhibition Networks

**EINet** is the flagship architecture, implementing biologically-inspired excitation-inhibition dynamics with dendritic computation.

### Core Concepts

#### Excitation-Inhibition Dynamics
- **Excitatory (E) neurons**: Positive signal transmission
- **Inhibitory (I) neurons**: Negative signal modulation
- **Shunting inhibition**: Biologically-plausible divisive normalization
- **Subtractive inhibition**: Traditional linear subtraction

#### Dendritic Branching
- **Hierarchical structure**: Multi-level branching from soma to dendrites
- **Branch factors**: Control branching pattern (e.g., [2,2] = 4→2→1)
- **Synaptic integration**: Complex input combination at each branch level

#### Connectivity Patterns
- **E→E connections**: Excitatory-to-excitatory pathways
- **E→I connections**: Excitatory-to-inhibitory pathways
- **I→E connections**: Inhibitory-to-excitatory pathways
- **I→I connections**: Inhibitory-to-inhibitory (usually zero)

### Architecture Parameters

#### Layer Structure
```yaml
excitatory_layer_sizes: [256, 128, 64]     # E neurons per layer
inhibitory_layer_sizes: [128, 64, 32]      # I neurons per layer
excitatory_branch_factors: [2, 2]          # E branching pattern
inhibitory_branch_factors: []              # I branching (usually none)
```

#### Connectivity Configuration
```yaml
ee_synapses_per_branch_per_layer: [200, 100]  # E→E connections
ei_synapses_per_branch_per_layer: [100, 50]   # E→I connections
ie_synapses_per_branch_per_layer: [50, 25]    # I→E connections
ii_synapses_per_branch_per_layer: [0, 0]      # I→I connections
```

**Connection density calculation:**
- Total E inputs to layer L: `excitatory_layer_sizes[L-1] * ee_synapses_per_branch_per_layer[L-1]`
- Effective sparsity: `synapses_per_branch / total_possible_connections`

#### Inhibition Types

**Shunting Inhibition** (Biologically-plausible):
```
V_out = E / (E + I + 1)
```
- Divisive normalization
- Bounded output [0, 1]
- Prevents runaway excitation

**Subtractive Inhibition** (Traditional):
```
V_out = E - I
```
- Linear subtraction
- Unbounded output
- Simpler computation

### Input Processing Modes

#### Mode 0: Raw Input
```yaml
input_mode: 0
# Excitatory pathway: raw input
# Inhibitory pathway: None (zeros)
```

#### Mode 1: Shared Input
```yaml
input_mode: 1
# Both pathways: same input
# Simple and effective for most tasks
```

#### Mode 2: Transformed Input
```yaml
input_mode: 2
input_net_dict:
  type: CNNInputTransform
  # Separate transformation networks for E and I pathways
```

### Advanced Features

#### Somatic Synapses
```yaml
somatic_synapses: true
# Allow direct connections to soma, bypassing dendritic tree
# Useful for rapid, direct signal transmission
```

#### Reactivation Functions
```yaml
reactivate: true
reactivation_type: param_tanh
# Apply learnable activation functions after dendritic integration
```

### Mathematical Formulation

For a single EINet layer with shunting inhibition:

1. **Dendritic Integration**:
   ```
   E_branch = Σ(W_ee * X_e) + Σ(W_ie * X_i)
   I_branch = Σ(W_ei * X_e) + Σ(W_ii * X_i)
   ```

2. **Branch Aggregation**:
   ```
   E_soma = Σ(E_branch_k)
   I_soma = Σ(I_branch_k)
   ```

3. **Shunting Computation**:
   ```
   V_out = E_soma / (E_soma + I_soma + 1)
   ```

4. **Reactivation**:
   ```
   Output = f_activation(V_out)
   ```

## DendriNet - Pure Dendritic Networks

**DendriNet** focuses purely on dendritic computation without explicit E-I separation.

### Key Features

#### Multi-Level Branching
- **Soma**: Final integration point
- **Primary branches**: First level of dendrites
- **Secondary branches**: Higher-order dendritic structures
- **Synaptic inputs**: Distributed across branch levels

#### Synaptic Distribution
```yaml
n_soma: 64                           # Number of output neurons
branch_factors: [2, 2]               # 64 → 128 → 256 synapses
excitatory_synapses_per_branch: 100  # Synapses per branch
inhibitory_synapses_per_branch: 50   # Optional inhibitory input
```

#### Reactivation at Branches
```yaml
reactivate: true
reactivation_type: param_tanh
# Apply activation functions at each branch level
# Enables complex nonlinear integration
```

### Architecture Example

```yaml
model:
  core_network:
    type: DendriNet
    parameters:
      n_soma: 128
      branch_factors: [2, 2]           # 3-level tree: 128→256→512
      excitatory_input_dim: 784
      excitatory_synapses_per_branch: 150
      inhibitory_input_dim: 784        # Optional
      inhibitory_synapses_per_branch: 75
      reactivate: true
      reactivation_type: param_tanh
      use_shunting: true
```

## Classical Networks

### MLP - Multi-Layer Perceptron

Standard feedforward neural network for baseline comparisons.

```yaml
model:
  core_network:
    type: MLP
    parameters:
      input_dim: 784
      hidden_dims: [512, 256, 128]     # Hidden layer sizes
      activation: relu                 # "relu", "sigmoid", "tanh"
      output_dim: 10
```

**Features:**
- Standard weight matrices
- Configurable activation functions
- Batch normalization support
- Dropout (via training config)

### MatchedParamMLP

MLP with parameter count matched to equivalent EINet for fair comparison.

```yaml
model:
  core_network:
    type: MatchedParamMLP
    parameters:
      # Uses EINet architecture parameters
      input_dim: 784
      excitatory_layer_sizes: [256, 128]
      inhibitory_layer_sizes: [128, 64]
      ee_synapses_per_branch_per_layer: [200]
      # ... other EINet parameters
      activation: relu
```

**Parameter Matching Algorithm:**
1. Calculate total parameters in equivalent EINet
2. Design MLP layer sizes to match parameter count
3. Minimize layers while exceeding parameter threshold

## Input Processing Networks

### Identity Network
```yaml
encoder_network:
  type: Identity
  parameters:
    input_dim: 784    # Pass-through with shape validation
```

### CNN Networks

#### CNNDownsample
For spatial input processing with downsampling:

```yaml
encoder_network:
  type: CNNDownsample
  parameters:
    input_shape: [3, 32, 32]        # [channels, height, width]
    hidden_dims: [32, 64, 128]      # Conv channel progression
    kernel_sizes: [5, 3, 3]         # Kernel size per layer
    strides: [1, 2, 2]              # Stride per layer
    activation: relu
    output_dim: 256                 # Final flattened dimension
```

#### CNNUpsample
For generative tasks with upsampling:

```yaml
decoder_network:
  type: CNNUpsample
  parameters:
    input_dim: 256
    hidden_dims: [128, 64, 32]      # Deconv channel progression
    kernel_sizes: [3, 3, 5]         # Kernel size per layer
    strides: [2, 2, 1]              # Stride per layer
    output_shape: [3, 32, 32]       # Target output shape
```

### AlexNet Encoder
Pre-trained AlexNet features:

```yaml
encoder_network:
  type: AlexNet
  parameters:
    output_dim: 1000               # Feature dimension
    freeze_weights: true           # Freeze pre-trained weights
```

### Input Transform Networks

For complex input preprocessing in EINet:

#### CNN Input Transform
```yaml
input_net_dict:
  type: CNNInputTransform
  input_shape: [3, 32, 32]
  
  # Excitatory pathway CNN
  excitatory_hidden_dims: [32, 64]
  excitatory_kernel_sizes: [5, 3]
  excitatory_strides: [1, 2]
  
  # Inhibitory pathway CNN  
  inhibitory_hidden_dims: [16, 32]
  inhibitory_kernel_sizes: [7, 5]
  inhibitory_strides: [1, 2]
```

#### MLP Input Transform
```yaml
input_net_dict:
  type: MLPInputTransform
  hidden_dims: [512, 256]          # Shared hidden layers
  excitatory_dim: 256              # E pathway output
  inhibitory_dim: 128              # I pathway output
```

## Synaptic Mechanisms

### TopK Selection

#### Standard TopK
Deterministic selection of top-K strongest connections:

```yaml
topk_type: standard
topk_init_method: xavier_normal     # Weight initialization
ee_synapses_per_branch_per_layer: [100, 50]
```

**Algorithm:**
1. Initialize all potential connections
2. Select top-K by weight magnitude
3. Mask unused connections to zero

#### Stochastic TopK
Probabilistic selection based on weight ranks:

```yaml
topk_type: stochastic
topk_temperature: 0.3              # Lower = more deterministic
topk_ultrafast: true               # Fast approximation
topk_noise_level: 0.1              # Exploration noise
```

**Selection probability:**
```
P(select) = softmax(weights / temperature)
```

#### Variance TopK
Selection based on input variance and weight magnitude:

```yaml
topk_type: variance
# Selects synapses with high variance × weight product
```

### Dynamic Sparse Training (DeepST)

Evolving connectivity patterns during training:

```yaml
topk_type: deepst
excitatory_target_density: 0.1     # Target sparsity (10% active)
inhibitory_target_density: 0.05    # Different sparsity for I
rewiring_mode: global              # "global", "local", "constant_branch"
rewire_frequency: 10               # Rewire every N epochs
use_noise: true                    # Add exploration noise
sigma: 0.05                        # Noise standard deviation
weight_threshold: 1e-6             # Pruning threshold
```

**Rewiring Modes:**
- **Global**: Can rewire to any neuron in layer
- **Local**: Rewire only within same branch
- **Constant Branch**: Maintain fixed synapses per branch

**Rewiring Algorithm:**
1. **Prune**: Remove connections below threshold
2. **Grow**: Add new random connections
3. **Noise**: Add Gaussian noise for exploration
4. **Normalize**: Maintain target density

## Activation Functions

### Standard Activations
```yaml
reactivation_type: relu             # Standard ReLU
reactivation_type: sigmoid          # Standard sigmoid
reactivation_type: tanh             # Standard tanh
reactivation_type: none             # Identity (no activation)
```

### Parametric Activations

#### Parametric Tanh
Learnable slope and bias parameters:

```yaml
reactivation_type: param_tanh
reactivation_init_m: 1.5           # Initial slope
reactivation_init_b: 0.5           # Initial bias
```

**Function:**
```
f(x) = tanh(m * x + b)
```

#### Parametric Tanh (Slope Only)
Only slope parameter is learnable:

```yaml
reactivation_type: param_tanh_only_m
reactivation_init_m: 1.5
```

#### Linear-Sigmoid Interpolation
Smooth transition between linear and sigmoid:

```yaml
reactivation_type: param_linear_sigmoid
# Trainable: f(x) = α * x + (1-α) * sigmoid(x)

reactivation_type: linear_sigmoid
# Fixed interpolation
```

#### Linear-Tanh Interpolation
Smooth transition between linear and tanh:

```yaml
reactivation_type: param_linear_tanh
# Trainable interpolation

reactivation_type: linear_tanh
# Fixed interpolation
```

#### Transition Functions
Smooth transitions with learnable parameters:

```yaml
reactivation_type: param_linear_tanh_transition
# Complex parametric transition function

reactivation_type: linear_tanh_transition
# Fixed transition function
```

## Gradient Scaling Strategies

Biologically-motivated gradient modifications for stable training.

### Reactivation Scaling
```yaml
reactivation_strategy: inverse
# Scale gradients by 1/derivative
# Helps with vanishing gradients in saturated regions
```

### TopK Scaling
```yaml
topk_strategy: conductance_dynamic
# Scale based on total synaptic conductance
# Balances gradients across different sparsity levels
```

### BlockLinear Scaling
```yaml
blocklinear_strategy: conductance_dynamic    # Conductance-based scaling
blocklinear_strategy: distal_upweight_by_idx # Upweight distal branches
blocklinear_strategy: none                   # No scaling
```

**Conductance Dynamic Scaling:**
- Compensates for different numbers of active synapses
- Normalizes gradients by total conductance
- Maintains balanced learning across branches

**Distal Upweighting:**
- Increases learning rate for branches farther from soma
- Compensates for distance-dependent attenuation
- Biologically-inspired gradient modification

## Architecture Selection Guidelines

### Task-Based Recommendations

#### Image Classification
```yaml
# Small datasets (MNIST)
encoder_network: Identity
core_network: EINet (simple)
decoder_network: Identity

# Large datasets (CIFAR-10)
encoder_network: CNNDownsample
core_network: EINet (multi-layer)
decoder_network: MLP
```

#### Regression Tasks
```yaml
encoder_network: Identity or MLP
core_network: EINet with shunting=false
decoder_network: MLP
loss_function: mse
```

#### Information Processing
```yaml
# For synthetic tasks like XOR, info routing
encoder_network: Identity
core_network: EINet with sophisticated branching
decoder_network: Identity
```

### Performance Considerations

#### Memory Usage
- **EINet**: Higher memory due to sparse connectivity
- **DendriNet**: Moderate memory with branching
- **MLP**: Lowest memory with dense connections

#### Computational Speed
- **Standard TopK**: Fastest synaptic selection
- **Stochastic TopK**: Moderate speed, better exploration
- **DeepST**: Slower due to rewiring overhead

#### Training Stability
- **Shunting inhibition**: More stable (bounded output)
- **Gradient scaling**: Essential for deep dendritic networks
- **Parametric activations**: Better gradient flow

### Best Practices

#### For Beginners
1. Start with simple EINet configuration
2. Use `input_mode: 1` (shared input)
3. Enable `use_shunting: true`
4. Use `topk_type: standard`
5. Enable gradient scaling strategies

#### For Advanced Users
1. Experiment with complex branching patterns
2. Try stochastic or variance TopK selection
3. Use sophisticated input transforms
4. Combine multiple gradient scaling strategies
5. Implement dynamic sparse training for evolving connectivity

#### For Research
1. Implement custom activation functions
2. Design novel synaptic selection mechanisms
3. Explore different E-I balance ratios
4. Investigate biological constraints
5. Develop new gradient scaling approaches

---

This guide covers all the architectural components available in the Dendritic Modeling Framework. For specific configuration examples, see the [Configuration Guide](CONFIGURATION_GUIDE.md) and the example configurations in the `configs/` directory. 