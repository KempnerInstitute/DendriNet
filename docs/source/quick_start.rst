Quick Start
===========

Install
-------

.. code-block:: bash

   pip install dendritic-modeling

or clone the repo for development:

.. code-block:: bash

   git clone https://github.com/your-org/dendritic-modeling.git
   cd dendritic-modeling
   pip install -e .

Train Your First Network
------------------------

.. code-block:: python

   import torch
   from dendritic_modeling.models import Classifier
   from dendritic_modeling.networks import ExcitationInhibitionNetwork
   from dendritic_modeling.training.utils import train_model
   from dendritic_modeling.datasets import get_unified_datasets

   # dataset (MNIST 28×28 → flat 784)
   train_ds, val_ds, _ = get_unified_datasets("mnist", data_path="./data")

   # network
   einet = ExcitationInhibitionNetwork(
       input_dim=784,
       layer_sizes=[256, 64, 10],
       excitatory_branch_factors=[4, 2, 1],
       inhibitory_branch_factors=[2, 1, 0],
       ei_synapses_per_branch_per_layer=[5, 3, 1],
   )

   model = Classifier(core_network=einet)

   train_model(model, train_ds, val_ds, epochs=20)

Visualise Dendrite Activations
------------------------------

.. code-block:: python

   from dendritic_modeling.plotting.analysis import plot_dendrinet_activations
   plot_dendrinet_activations(einet, train_ds[:256][0], train_ds[:256][0])

Next Steps
~~~~~~~~~~
• Explore :doc:`core_concepts/index` to understand the biology & implementation.  
• Dive into :doc:`source/user_guide/index` for advanced configuration and experiment tracking. 