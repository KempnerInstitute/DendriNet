Glossary
========

This glossary defines key terms used throughout the Dendritic Modeling Framework documentation.

.. glossary::

   Dendritic Computation
      A biologically-inspired computational approach where artificial neurons model the sophisticated processing that occurs in biological dendritic trees, including nonlinear integration and compartmentalized processing.

   EINet
   Excitation-Inhibition Network
      A neural network architecture that explicitly separates excitatory and inhibitory pathways, following Dale's principle. The primary architecture of the framework implementing dendritic computation with E/I dynamics.

   Shunting Inhibition
      A divisive form of inhibition implementing the equation V = E/(E+I+ε), providing gain control and dynamic range adaptation. More biologically realistic than subtractive inhibition.

   Dale's Principle
      The biological principle that neurons are either purely excitatory or purely inhibitory, never both. Enforced throughout the framework's network architectures.

   TopK Sparse Connectivity
      A connectivity pattern where each neuron connects to only the K strongest inputs based on weight magnitude. Variants include standard, stochastic, and variance-based selection.

   Branch Factor
      The number of dendritic branches per neuron or the branching pattern in hierarchical dendritic structures (e.g., [4, 2] means 4 branches splitting into 2 sub-branches each).

   DeepST
   Dynamic Sparse Training
      A training method that dynamically rewires connections during training, allowing the network to explore different connectivity patterns and maintain optimal sparsity.

   Synapses per Branch
      The number of synaptic connections allocated to each dendritic branch, controlling the local computational capacity and connectivity density.

   Reactivation Function
      Activation functions applied to dendritic branch outputs before integration at the soma. Examples include parametric tanh, linear-sigmoid interpolation.

   Gradient Scaling
      Techniques to handle vanishing or exploding gradients in deep dendritic networks, including inverse scaling and conductance-based approaches.

   Block Linear
      A memory-efficient sparse connectivity implementation that organizes weights into blocks, reducing memory usage from O(n²) to O(n×block_size).

   Input Mode
      Determines how inputs are distributed to excitatory and inhibitory pathways:
      - Mode 0: Raw input to excitatory only
      - Mode 1: Same input to both pathways  
      - Mode 2: Network-transformed inputs

   Voltage Stabilization
      A training strategy that includes a warm-up phase to stabilize dendritic voltages before main training, particularly useful for shunting inhibition networks.

   FSDP
   Fully Sharded Data Parallel
      A distributed training strategy that shards model parameters, gradients, and optimizer states across multiple GPUs for memory-efficient large-scale training.

   Branch-wise Training
      A biologically-inspired training strategy that trains dendritic branches sequentially, allowing for specialized feature learning and improved interpretability.

   Two-step Training
      A training approach with separate local and global optimization phases, often including unsupervised pretraining followed by supervised fine-tuning.

   Information Theory Analysis
      Analysis tools that compute mutual information (MI), conditional mutual information (CMI), and partial information decomposition (PID) to understand information flow in networks.

   Ablation Study
      Systematic removal or modification of network components to understand their contribution to overall performance and behavior.

   E/I Balance
      The ratio and dynamics between excitatory and inhibitory signals in the network, crucial for stable and biologically realistic computation.

   Morphological Parameters
      Parameters defining the physical structure of dendritic trees, including branch factors, synapse counts, and hierarchical organization.

   Homeostatic Regularization
      Mechanisms that maintain healthy activity levels and prevent runaway excitation or inhibition in dendritic networks.

   Sparse Training
      Training methods that maintain or introduce sparsity in network connections, reducing computational cost and improving biological realism.

   Mixed Precision Training
      Training with both 16-bit and 32-bit floating point numbers to speed up training while maintaining numerical stability.

   Gradient Checkpointing
      A memory optimization technique that trades computation for memory by recomputing intermediate activations during backpropagation.

   Parameter Sweeps
      Systematic exploration of hyperparameter spaces using automated grid search, random search, or more sophisticated optimization methods.

   Biological Realism
      The degree to which artificial neural networks capture the computational principles and constraints of biological neural systems.

   Dendritic Branch Layer
      A neural network layer that implements dendritic branch computation with separate excitatory and inhibitory processing pathways.

   Somatic Integration
      The final integration step where dendritic branch outputs are combined at the cell body (soma) to produce the neuron's output.

   Weight Initialization
      Methods for setting initial network weights, with specialized strategies for excitatory/inhibitory balance and sparse connectivity.

   Learning Strategy
      The overall approach to training, including standard backpropagation, bio-inspired methods, and distributed training strategies.

   Configuration System
      The YAML-based system for defining experiments, models, and training parameters without modifying code.

   Analysis Pipeline
      The comprehensive suite of tools for evaluating network performance, biological plausibility, and information processing capabilities. 