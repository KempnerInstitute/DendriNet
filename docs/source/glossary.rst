Glossary
========

.. glossary::

   activation function
      A mathematical function that determines the output of a neural network node given a set of inputs. In dendritic networks, specialized activation functions like ParametricTanh model the nonlinear response properties of biological dendrites.

   ablation study
      A technique where components of a neural network are systematically removed or disabled to assess their contribution to the overall performance.

   CIFAR10
      A dataset consisting of 60,000 32x32 color images in 10 classes, commonly used for evaluating computer vision algorithms.

   dendritic branch
      In biological neurons, a branch of the dendritic tree that receives inputs from other neurons. In the framework, a computational unit that processes inputs before sending signals to the soma.

   dendritic computing
      The principle that dendrites perform complex computations (not just signal transmission) before information reaches the cell body of a neuron.

   excitation-inhibition balance
      The dynamic balance between excitatory (positive) and inhibitory (negative) inputs in neural systems, critical for stable network function and information processing.

   excitatory
      Referring to neural connections or inputs that increase activity or membrane potential in the target neuron.

   gradient scaling
      Techniques for adjusting the magnitude of gradients during backpropagation to address challenges like vanishing or exploding gradients, particularly important in networks with complex architectures like dendritic networks.

   inhibitory
      Referring to neural connections or inputs that decrease activity or membrane potential in the target neuron.

   learning strategy
      An algorithm or approach for training neural networks. The framework implements several specialized learning strategies for dendritic networks.

   maximum likelihood estimation (MLE)
      A statistical method for estimating model parameters by maximizing the likelihood function, commonly used as a training objective in neural networks.

   MNIST
      A dataset of handwritten digits commonly used for training and testing image processing systems.

   mutual information
      A measure of the mutual dependence between two variables, used in the framework to quantify how much information dendritic branches contain about input classes.

   ParametricTanh
      A specialized activation function in the framework with learnable parameters that model the nonlinear response properties of biological dendrites.

   shunting inhibition
      A form of neural inhibition where the inhibitory effect is implemented through division rather than subtraction, mimicking conductance-based inhibition in biological neurons.

   soma
      The cell body of a neuron, which in biological systems integrates inputs from dendrites. In the framework, the final integration stage of a dendritic network.

   sparse connectivity
      A property where neurons connect to only a small subset of possible inputs, implemented in the framework through TopKLinear layers.

   TopKLinear
      A specialized linear layer in the framework that maintains connections only to the K strongest inputs for each output unit, implementing sparse connectivity.

   two-step training
      A learning strategy where a simplified version of the network is trained first, followed by training the full complex network.

   weight decay
      A regularization technique that adds a penalty for large weight values to the loss function, helping prevent overfitting.

   weight initialization
      Methods for setting the initial values of neural network weights before training begins, critical for successful training especially in complex architectures.

   block conductance
      A gradient scaling strategy in the framework that dynamically adjusts gradient flow based on the conductance properties of dendritic branches.

   branch factor
      The number of branches a cell divides into at each level of the dendritic tree.

   BlockLinear
      A specialized linear layer in the framework that handles the integration of information from converging branches in the dendritic tree.

   experiment tracking
      The process of logging and organizing information about model training runs, including hyperparameters, metrics, and results.

   input transformation network
      A network component that preprocesses inputs before they reach the dendritic network, used in the framework with input_mode=2.

   reactivation
      A mechanism in the framework that models the active properties of dendrites, implemented through special activation functions.

   synapses per branch
      The number of input connections that each dendritic branch receives, controlling the sparsity of connectivity.

   synthetic dataset
      Artificially generated datasets designed to test specific computational capabilities, such as line detection or XOR problems.