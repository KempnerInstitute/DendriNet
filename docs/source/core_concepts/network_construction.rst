Network Construction
====================

The *Excitation–Inhibition Network* (EINet) factory is located in
:pyfile:`src/dendritic_modeling/networks/architectures/excitation_inhibition/einet.py` and is wrapped by higher-level models such as :pyclass:`dendritic_modeling.models.Classifier`.

Defining a Network
------------------

.. code-block:: python

   from dendritic_modeling.networks import ExcitationInhibitionNetwork
   ei_net = ExcitationInhibitionNetwork(
       input_dim=784,
       layer_sizes=[512, 256, 64],
       excitatory_branch_factors=[5, 3, 1],
       inhibitory_branch_factors=[3, 2, 0],
       ee_synapses_per_branch_per_layer=[10, 6, 2],
       ei_synapses_per_branch_per_layer=[5, 3, 1],
       ei_weight_init="xavier_uniform",
   )

Branch Factors
--------------
``branch_factors`` control dendrite *fan-out*.  Higher factors increase dendritic integration capacity but also parameter count.

Layer API
---------
Each layer instance of :pyclass:`dendritic_modeling.networks.layers.DendriticLinear` contains two populations:

* **Excitatory (E)** – weight matrix ``W_E`` (learned)
* **Inhibitory (I)** – weight matrix ``W_I`` (learned)

Within-layer inhibitory control is implemented as:

.. math::

   h^{(l)}_E \; = \; \sigma\!\left( W_E x^{(l-1)} \right)
   \quad\text{and}\quad
   h^{(l)}_I \; = \; W_I x^{(l-1)}

   x^{(l)} \; = \; \frac{h^{(l)}_E}{h^{(l)}_E + \alpha h^{(l)}_I + \varepsilon}

``alpha`` is a learnable gain for flexibility.

Between-Layer Connectivity
-------------------------
E neurons project to both E and I neurons in the next layer, whereas I neurons project only within-layer (enforcing Dale’s principle).  This pattern is hard-coded in the module :pyfile:`dendritic_modeling/networks/layers/routing.py`.

Templates
~~~~~~~~~
Factory helpers in :pyfile:`dendritic_modeling/networks/architectures/factory.py` instantiate:

* **MLP-EINet** – fully-connected layers with Top-K sparsity
* **Conv-EINet** – convolutional dendrites using :pyclass:`dendritic_modeling.networks.layers.DendriteConv2d`
* **ViT-EINet** – patch tokens processed by dendrite-aware attention

Selecting a template in YAML:

.. code-block:: yaml

   core_network:
     type: ConvEINet
     params:
       backbone: resnet18          # uses torchvision resnet as encoder
       dendrite_layers: 4
       K: 32                       # Top-K per conv layer

Gradient Scaling
----------------
Large ratios in \(\tfrac{E}{E+I}\) can saturate; enabling *gradient scaling* normalises layer-wise gradient norms.  Activate by setting `gradient_scaling: true` in the config or passing the flag to layer constructors. 