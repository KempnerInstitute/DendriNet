Introduction
============

.. figure:: _static/dendritic_neuron.png
   :align: center
   :width: 80%
   :alt: Dendritic neuron architecture showing excitatory and inhibitory synapses
   :name: dendritic_neuron_figure

   **Dendritic Neuron Architecture**: Biological neurons feature complex dendritic trees with excitatory (red) and inhibitory (blue) synapses that perform sophisticated local computations before integration at the soma.

Purpose and Vision
------------------

The Dendritic Modeling Framework bridges the gap between biological neuroscience and artificial intelligence by implementing sophisticated dendritic computation mechanisms in deep neural networks. While traditional artificial neurons are simple point processors, biological neurons possess elaborate dendritic trees that perform complex, non-linear computations—capabilities that remain largely unexplored in modern AI systems.

Our framework enables researchers to:

* **Model biological dendritic computation** with unprecedented detail and accuracy
* **Explore excitation-inhibition dynamics** that govern neural circuit behavior  
* **Implement sparse connectivity patterns** inspired by cortical wiring principles
* **Develop bio-inspired learning algorithms** that capture dendritic plasticity
* **Scale dendritic models** from single neurons to large-scale neural networks

Biological Motivation
---------------------

Real Neurons vs. Artificial Neurons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Traditional artificial neural networks represent neurons as simple summation units:

.. math::
   y = f\left(\sum_i w_i x_i + b\right)

In contrast, biological neurons feature:

1. **Dendritic Trees**: Branched structures with thousands of synapses
2. **Local Nonlinearities**: Active dendrites perform local computations
3. **Compartmentalization**: Different branches can process information independently
4. **E/I Integration**: Sophisticated balance between excitation and inhibition

Shunting Inhibition
~~~~~~~~~~~~~~~~~~

A key innovation in our framework is shunting inhibition—a divisive form of inhibition ubiquitous in biological circuits:

.. math::
   V_{out} = \frac{E}{E + I + \varepsilon}

Where:
- :math:`E` represents excitatory input
- :math:`I` represents inhibitory input  
- :math:`\varepsilon` prevents division by zero

This mechanism provides:

* **Gain Control**: Automatic adjustment of neural sensitivity
* **Normalization**: Bounded outputs regardless of input magnitude
* **Contrast Enhancement**: Amplification of signal differences
* **Dynamic Range Adaptation**: Context-dependent response scaling

Dale's Principle and E/I Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Following Dale's principle, our networks maintain separate excitatory and inhibitory populations with distinct connectivity patterns:

* **Excitatory neurons** can only make excitatory connections
* **Inhibitory neurons** can only make inhibitory connections  
* **Realistic connectivity** follows biological wiring patterns (EE, EI, IE connections)

Key Innovations
---------------

The framework introduces several novel computational mechanisms:

Sparse Connectivity with TopK Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Standard TopK**: Deterministic selection of strongest connections
* **Stochastic TopK**: Probabilistic selection with temperature control
* **Variance-based**: Selection based on input variance patterns
* **DeepST**: Dynamic sparse training with connection rewiring

Advanced Reactivation Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Parametric activations**: Learnable non-linearities (e.g., parametric tanh)
* **Transition functions**: Smooth interpolation between linear and non-linear regimes
* **Branch-specific**: Different activations for different dendritic branches

Gradient Scaling Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Inverse scaling**: Compensates for vanishing gradients in deep dendrites
* **Conductance-based**: Biologically-motivated gradient flow
* **Distal upweighting**: Strengthened learning in distal dendritic branches

Training Strategies
~~~~~~~~~~~~~~~~~~

* **Standard**: End-to-end backpropagation
* **Layer-wise**: Sequential training of network layers
* **Branch-wise**: Local credit assignment for dendritic branches
* **Two-step**: Local pre-training followed by global fine-tuning
* **Voltage stabilization**: Biologically-motivated warm-up procedures

Target Audience
---------------

Neuroscience Researchers
~~~~~~~~~~~~~~~~~~~~~~~~

* **Computational neuroscientists** modeling dendritic computation
* **Systems neuroscientists** studying E/I balance and circuit dynamics
* **Theorists** exploring information processing in neural circuits

Machine Learning Practitioners  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Researchers** developing bio-inspired architectures
* **Engineers** seeking efficient sparse neural networks
* **Students** learning about biological computation principles

Interdisciplinary Scientists
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Cognitive scientists** modeling brain-like computation
* **Roboticists** implementing bio-inspired control systems
* **Computer scientists** exploring alternative computing paradigms

Framework Capabilities
---------------------

.. list-table:: Core Features
   :header-rows: 1
   :widths: 25 75

   * - Category
     - Capabilities
   * - **Architectures**
     - EINet, DendriNet, MLP variants, Convolutional networks
   * - **Connectivity**
     - TopK sparse, Stochastic selection, Dynamic rewiring (DeepST)
   * - **Training**
     - Multiple strategies, FSDP scaling, Mixed precision
   * - **Analysis**
     - Information theory, Ablation studies, Noise robustness
   * - **Datasets**  
     - Vision, Synthetic, Biological tasks, Custom loaders
   * - **Visualization**
     - Weight plots, Activation analysis, Interactive figures
   * - **Performance**
     - GPU acceleration, Memory optimization, Distributed training

Current Features
----------------

Recent Enhancements
~~~~~~~~~~~~~~~~~~

* **FSDP Support**: Scale to billion-parameter dendritic models across multiple GPUs
* **Unified Sweep Framework**: Comprehensive parameter exploration with automated analysis
* **Professional Visualizations**: High-quality figures with clear presentation
* **Performance Optimizations**: Mixed precision training, gradient checkpointing, memory management
* **Expanded Analysis Tools**: Information-theoretic measures with multiple estimators

Upcoming Features
~~~~~~~~~~~~~~~~

* **Multi-modal Networks**: Integration of vision, language, and other modalities
* **Online Learning**: Continual adaptation mechanisms
* **Neuromorphic Support**: Deployment on specialized hardware
* **Advanced Plasticity**: Hebbian and other bio-inspired learning rules

Getting Started
---------------

Ready to explore dendritic computation? Here's your roadmap:

1. **Installation**: Set up the framework (:doc:`installation`)
2. **Quick Start**: Run your first experiment (:doc:`quickstart`) 
3. **Core Concepts**: Understand the science (:doc:`core_concepts`)
4. **Architecture Guide**: Design custom networks (:doc:`architectures/index`)
5. **Training Manual**: Master different learning strategies (:doc:`training/index`)
6. **Analysis Tools**: Interpret your results (:doc:`analysis/index`)

.. note::
   For the impatient: Jump straight to the :doc:`quickstart` to run a dendritic network in under 5 minutes!

Research Impact
---------------

The framework has enabled breakthrough research in:

* **Dendritic computation mechanisms** and their role in neural processing
* **E/I balance dynamics** in artificial neural circuits  
* **Sparse connectivity patterns** for efficient deep learning
* **Bio-inspired learning algorithms** that capture neural plasticity
* **Information processing capacity** of dendritic neural networks

Contributing to the Field
-------------------------

This framework represents a collaborative effort to advance our understanding of neural computation. Whether you're interested in:

* **Extending the codebase** with custom features
* **Contributing biological insights** to improve realism  
* **Applying the framework** to novel research questions
* **Sharing results** that advance the field

We welcome your participation in this exciting intersection of neuroscience and AI.

.. seealso::
   * :doc:`contributing` - Guidelines for contributing to the framework
   * :doc:`api/index` - Complete API reference
   * :doc:`tutorials/index` - Step-by-step tutorials and examples 