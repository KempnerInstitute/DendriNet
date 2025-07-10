Tutorials
=========

This section provides hands-on tutorials for using the Dendritic Modeling Framework.

Basic E/I Network
-----------------

Create a simple excitation-inhibition network with dendritic branches:

.. code-block:: python

   import torch
   from dendritic_modeling.networks import ExcitationInhibitionNetwork
   from dendritic_modeling.models import Classifier
   from dendritic_modeling.training import Trainer
   
   # Create network
   network = ExcitationInhibitionNetwork(
       input_dim=784,
       transfer_params={
           "input_mode": 0,
           "multiplicative": True,
           "shunting_inhibition": True
       },
       excitatory_layer_sizes=[100],
       inhibitory_layer_sizes=[50],
       excitatory_branch_factors=[2, 2],
       inhibitory_branch_factors=[2],
       ee_synapses_per_branch_per_layer=[20],
       ei_synapses_per_branch_per_layer=[10],
       ie_synapses_per_branch_per_layer=[5],
       use_shunting=True
   )
   
   # Wrap in classifier
   model = Classifier(
       encoder_network=None,
       core_network=network,
       decoder_network=None
   )
   
   # Train
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   trainer = Trainer(
       optimizer=optimizer,
       loss_function="cross_entropy",
       epochs=10
   )

Working with Dendritic Branches
-------------------------------

Understanding dendritic branch computation:

.. code-block:: python

   from dendritic_modeling.networks import DendriticBranchLayer
   
   # Create a single dendritic branch layer
   branch_layer = DendriticBranchLayer(
       n_branches=100,           # Number of dendritic branches
       n_excitatory_inputs=784,  # Excitatory input dimension
       n_inhibitory_inputs=784,  # Inhibitory input dimension
       excitatory_k=20,          # Excitatory synapses per branch
       inhibitory_k=10,          # Inhibitory synapses per branch
       use_shunting=True,        # Enable shunting inhibition
       topk_type="standard"      # TopK selection strategy
   )
   
   # Forward pass
   excitatory_input = torch.randn(32, 784)
   inhibitory_input = torch.randn(32, 784)
   prev_dendrite = torch.randn(32, 100)
   
   output = branch_layer(excitatory_input, inhibitory_input, prev_dendrite)

Sparse Connectivity with TopK
-----------------------------

Using different TopK strategies for sparse connectivity:

.. code-block:: python

   from dendritic_modeling.networks import TopKLinear, StochasticTopKLinear
   
   # Standard TopK
   topk_layer = TopKLinear(
       in_features=100,
       out_features=50,
       k=10,  # Top 10 weights per output
       init_method="xavier_normal"
   )
   
   # Stochastic TopK with temperature
   stochastic_layer = StochasticTopKLinear(
       in_features=100,
       out_features=50,
       k=10,
       temperature=0.5,
       ultrafast=False
   )

Vision Models with DendriteConv2d
---------------------------------

Creating convolutional dendritic layers for vision tasks:

.. code-block:: python

   import torch.nn as nn
   from dendritic_modeling.networks.layers import DendriteConv2d
   
   class DendriticCNN(nn.Module):
       def __init__(self):
           super().__init__()
           
           # Dendritic convolutional layer
           self.conv1 = DendriteConv2d(
               in_channels=3,
               out_channels=32,
               kernel_size=5,
               padding=2,
               epsilon=1e-4
           )
           
           self.pool = nn.MaxPool2d(2)
           
           self.conv2 = DendriteConv2d(
               in_channels=32,
               out_channels=64,
               kernel_size=5,
               padding=2
           )
           
       def forward(self, x):
           # DendriteConv2d automatically handles E/I computation
           x = self.conv1(x)
           x = self.pool(x)
           x = self.conv2(x)
           x = self.pool(x)
           return x

Pre-built Vision Units
----------------------

Using factory functions for common vision neurons:

.. code-block:: python

   from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic import (
       create_gabor_unit,
       create_center_surround_unit,
       create_motion_unit
   )
   
   # Gabor filter unit
   gabor = create_gabor_unit(
       orientation=45.0,  # degrees
       frequency=0.1,
       sigma=3.0
   )
   
   # Center-surround unit
   center_surround = create_center_surround_unit(
       center_size=5,
       surround_size=11
   )
   
   # Motion detection unit
   motion = create_motion_unit(
       direction="rightward",
       kernel_size=15
   )

Training Strategies
-------------------

Different training approaches for dendritic networks:

.. code-block:: python

   from dendritic_modeling.training import (
       Trainer,
       LayerWiseTrainer,
       LocalCreditAssignment,
       VoltageStabilizationTrainer
   )
   
   # Standard training
   standard_trainer = Trainer(
       optimizer=optimizer,
       loss_function="cross_entropy",
       epochs=100,
       early_stopping=True,
       patience=10
   )
   
   # Layer-wise training
   layerwise_trainer = LayerWiseTrainer(
       optimizer=optimizer,
       loss_function="cross_entropy",
       pretrain_epochs=10,
       epochs_per_layer=5,
       final_tune_epochs=10
   )
   
   # Voltage stabilization
   voltage_trainer = VoltageStabilizationTrainer(
       optimizer=optimizer,
       loss_function="cross_entropy",
       pretrain_epochs=5,
       target_voltage=0.5
   )

Analysis Tools
--------------

Analyzing trained dendritic networks:

.. code-block:: python

   from dendritic_modeling.analysis import (
       AblationAnalyzer,
       InformationAnalyzer,
       SynapticTurnoverAnalyzer
   )
   
   # Ablation analysis
   ablation = AblationAnalyzer(
       ablation_types=["excitatory", "inhibitory", "branches"]
   )
   results = ablation.analyze(
       model=model,
       test_dataset=test_data,
       device="cuda"
   )
   
   # Information analysis
   info_analyzer = InformationAnalyzer(
       params={"n_bins": 10, "method": "kde"}
   )
   mi_results = info_analyzer.analyze(
       model=model,
       test_dataset=test_data
   )
   
   # Synaptic turnover
   turnover = SynapticTurnoverAnalyzer(model=model)
   turnover_stats = turnover.analyze(
       model=model,
       training=True
   )

Working with Datasets
---------------------

Loading and using built-in datasets:

.. code-block:: python

   from dendritic_modeling.datasets import (
       load_mnist_as_datasets,
       load_cifar10_as_datasets,
       get_unified_datasets
   )
   
   # Load MNIST
   train_ds, val_ds, test_ds = load_mnist_as_datasets(
       data_path="./data",
       train_valid_split=0.8
   )
   
   # Load with unified interface
   datasets = get_unified_datasets(
       dataset_name="mnist",
       data_path="./data",
       train_valid_split=0.8
   )

Configuration Management
------------------------

Using configuration files for experiments:

.. code-block:: python

   from dendritic_modeling.config import Config, load_config
   
   # Load from YAML
   config = load_config("config.yaml")
   
   # Access configuration
   model_config = config.model
   train_config = config.train
   
   # Create model from config
   from dendritic_modeling.networks.architectures.factory import get_architecture
   
   network = get_architecture(
       type=model_config.core_network.type,
       parameters=model_config.core_network.parameters
   )

Next Steps
----------

- Explore the :doc:`api_reference` for detailed documentation
- Read the :doc:`theory` section to understand the biological foundations
- Check out :doc:`examples` for real-world applications 