.. _concept_overview:  
  
Concepts Overview
=================

Biological Inspiration and Theoretical Foundation
------------------------------------------------

Dendritic modeling draws inspiration from the complex computational capabilities of biological neurons. In the brain, neurons receive inputs through branching tree-like structures called dendrites, which perform nonlinear operations on incoming signals before they reach the cell body (soma). This dendritic computation is far more sophisticated than the simple weighted summation typically modeled in artificial neural networks.

The dendritic modeling framework implemented in this package aims to capture some of these sophisticated computational properties:

1. **Branch-specific processing**: Dendritic branches can process inputs independently
2. **Nonlinear integration**: Inputs can interact nonlinearly within branches
3. **Excitation-inhibition balance**: The interplay between excitatory and inhibitory signals regulates neuronal activity
4. **Compartmentalized computation**: Different segments of the dendritic tree can perform distinct operations

Network Architecture
-------------------

The initial dendritic model is designed as follows: external input data (considered as excitatory input) is fed into an excitatory/inhibitory classifier (``EINetClassifier``). This classifier model contains a single excitatory/inhibitory network (``ExcitationInhibitionNetwork``), which consists of one or more excitation-inhibition layers (``ExcitationInhibitionLayer``). Each excitation-inhibition layer has two dendritic layers (``Dendrinet``). 

.. figure:: figures/pdf/EINetClassifier_dataflow.pdf
    :align: center
    :width: 100%
    :alt: EINetClassifier dataflow
    :name: EINetClassifier_dataflow
    
    : Dataflow diagram of the EINetClassifier model. The blue arrows represent the flow of excitatory data, and the red arrows represent the flow of inhibitory data. Please note that the first inhibitory dendritic layer is not receiving any inhibitory input.

Within each dendritic layer, there is a specialized feedforward multilayer perceptron.

.. figure:: figures/pdf/Dendrinet_1.pdf
    :align: center
    :width: 100%
    :alt: Dendrinet dataflow 1
    :name: Dendrinet_dataflow_1
    
    : Dataflow diagram of the Dendrinet model. The blue arrows represent the flow of excitatory data, and the red arrows represent the flow of inhibitory data. Each dendritic branch layer (``DendriticBranchLayer``) accepts three types of inputs: excitatory, inhibitory, and data flow from upstream layers through branch factor. In the presented figure, the first dendritic layer branch factor is 2 and the second dendritic layer branch factor is 3. The outermost layer does not have any branch factor.

Inside the ``ExcitationInhibitionLayer``, the Dendrinet is categorized as either excitatory or inhibitory. Consequently, the output of the Dendrinet will be excitatory if it is defined as such, or inhibitory if it is defined as inhibitory. Additionally, note that the input data is fed to all dendritic branch layers. The final layer, known as the somatic layer, will receive direct input (both excitatory and inhibitory) if it is specified when compiling the model (``somatic_synapses = True``). 

The following figure shows the data flow to the second dendritic layer in the Dendrinet model. The inputs are excitatory, inhibitory, and data flow from the upstream layer. The branch factor for the first dendritic layer is 3.

.. figure:: figures/pdf/Dendrinet_2.pdf
    :align: center
    :width: 100%
    :alt: Dendrinet dataflow 2
    :name: Dendrinet_dataflow_2

    : Dataflow diagram of the Dendrinet model for the second layer. The blue arrows represent the flow of excitatory data, and the red arrows represent the flow of inhibitory data. Each dendritic branch layer (``DendriticBranchLayer``) accepts three types of inputs: excitatory, inhibitory, and data flow from upstream layers through branch factor. 

Key Computational Components
---------------------------

### TopKLinear Class

The ``TopKLinear`` class implements a key feature of dendritic computation: selective synaptic activation. In biological dendrites, not all synapses are active for a given input pattern - some are selectively engaged based on various factors. The TopKLinear class models this by:

1. Performing a standard linear transformation of the input: y = Wx + b
2. Selecting only the top k incoming weights for each output neuron
3. Setting all other weights temporarily to zero for the forward pass
4. During backpropagation, gradients are computed for all weights, allowing the network to learn which connections are most important

The mathematical formulation is:

```
y = (W ⊙ M)x + b
```

Where M is a binary mask that retains only the top k elements in each row of W, and ⊙ represents element-wise multiplication.

This selective activation has several computational advantages:
- It enforces sparsity in the network activations
- It allows neurons to specialize in detecting specific input patterns
- It creates nonlinear decision boundaries that better approximate biological neuronal behavior

### BlockLinear Class

The ``BlockLinear`` class handles the aggregation of information from upper layers to lower layers in the dendritic tree. It models how signals flow and combine through the branching structure of dendrites by:

1. Generating a structured mask based on the branch factor
2. Applying this mask to block certain inputs from flowing to certain outputs
3. Creating a compartmentalized information flow that mimics dendritic branching

The mask created by BlockLinear ensures that each output neuron receives input from only a subset of input neurons, determined by the branching structure. This compartmentalization is crucial for the network's ability to perform localized computations within branches before combining results.

### ParametricTanh Activation

The ``ParametricTanh`` class implements a trainable activation function that models the nonlinear response properties of dendritic segments. Unlike standard activation functions with fixed shapes, ParametricTanh has learnable parameters that allow each unit to adapt its activation curve:

```
f(x) = m * tanh(x + b)
```

Where m (multiplier) and b (bias) are learnable parameters. This allows:
- Customized sensitivity for different input ranges
- Adaptive saturation points for each unit
- More flexible nonlinear transformations than standard fixed activations

Inhibitory and Excitatory Interactions
-------------------------------------

The balance between excitation and inhibition is crucial for neural computation. In this model:

1. **Excitatory pathways** (blue arrows) generally promote activity in downstream neurons
2. **Inhibitory pathways** (red arrows) generally suppress or modulate activity

These interactions implement several important computational principles:
- **Shunting inhibition**: Inhibitory inputs can effectively "block" excitatory inputs from propagating
- **Gain control**: Inhibitory signals can modulate the sensitivity of the network to inputs
- **Selective activation**: The combination of excitation and inhibition can create highly selective response patterns

Training and Learning Approaches
-------------------------------

The framework supports several training strategies tailored to dendritic networks:

1. **MLE (Maximum Likelihood Estimation)**: Standard end-to-end training of all network components
2. **Freeze Layers**: Progressive training of layers from input to output
3. **Two-Step Training**: A two-phase approach with pretraining followed by fine-tuning
4. **Sequential Input Network**: Training input transformation networks before training dendritic components

Each approach offers different advantages in terms of training stability, final performance, and ability to learn complex dendritic computations.

Benefits of Dendritic Computation
--------------------------------

This architectural design captures several benefits of dendritic computation:

1. **Enhanced computational capacity**: The branching structure allows more complex functions to be computed compared to standard neural networks
2. **Feature detection in branches**: Individual branches can specialize in detecting specific input patterns
3. **Robustness to noise**: The selective activation mechanism helps filter out irrelevant inputs
4. **Biologically plausible computation**: The model more closely approximates the computation performed by biological neurons

Advanced Computational Features
------------------------------

The dendritic modeling framework includes several advanced features:

1. **Shunting inhibition**: When enabled (``use_shunting=True``), inhibitory inputs can effectively negate excitatory inputs, mimicking biological shunting
2. **Dynamic gradient scaling**: Various gradient strategies (e.g., ``block_conductance_dynamic``) help address the challenges of training deep dendritic networks
3. **Reactivation mechanisms**: Techniques to ensure branches remain active during training, preventing branch "death"

For details on implementing specific computations, please refer to the API documentation of individual classes and methods.






