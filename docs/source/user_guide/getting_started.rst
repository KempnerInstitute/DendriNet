Getting Started
===============

This section provides everything you need to get up and running with the Dendritic Modeling Framework, from installation to running your first experiments.

Installation
-----------

System Requirements
~~~~~~~~~~~~~~~~~~

The framework requires:

- Python 3.8 or higher
- PyTorch 1.9 or higher
- CUDA-capable GPU (optional but recommended)

Basic Installation
~~~~~~~~~~~~~~~~~

Install the framework using pip:

.. code-block:: bash

   pip install dendritic-modeling

Development Installation
~~~~~~~~~~~~~~~~~~~~~~

For development or to access the latest features, install from source:

.. code-block:: bash

   git clone https://github.com/your-org/dendritic-modeling.git
   cd dendritic-modeling
   pip install -e .

This installs the package in development mode, allowing you to modify the source code and see changes immediately.

Optional Dependencies
~~~~~~~~~~~~~~~~~~~

Install additional dependencies for specific features:

.. code-block:: bash

   # For advanced visualization
   pip install "dendritic-modeling[visualization]"
   
   # For experiment tracking
   pip install "dendritic-modeling[wandb]"
   
   # For all optional features
   pip install "dendritic-modeling[all]"

Conda Environment
~~~~~~~~~~~~~~~

We recommend using a conda environment for isolation:

.. code-block:: bash

   conda create -n dendritic python=3.9
   conda activate dendritic
   pip install dendritic-modeling

Verification
~~~~~~~~~~~

Verify your installation:

.. code-block:: python

   import dendritic_modeling
   print(f"Dendritic Modeling Framework v{dendritic_modeling.__version__}")
   
   # Test basic functionality
   from dendritic_modeling.networks import ExcitationInhibitionNetwork
   from dendritic_modeling.config import TransferFunctionParams
   
   params = TransferFunctionParams().asdict()
   network = ExcitationInhibitionNetwork(
       input_dim=10,
       transfer_params=params,
       excitatory_layer_sizes=[5],
       inhibitory_layer_sizes=[3]
   )
   print("Installation successful!")

Quick Start Tutorial
------------------

This tutorial walks you through creating and training your first dendritic network in just a few minutes.

Step 1: Import Required Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   from dendritic_modeling.models import Classifier
   from dendritic_modeling.networks import ExcitationInhibitionNetwork
   from dendritic_modeling.config import TransferFunctionParams
   from dendritic_modeling.datasets import get_unified_datasets
   from dendritic_modeling.training.utils import train_model

Step 2: Create Transfer Function Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Transfer function parameters control how dendritic computation is performed:

.. code-block:: python

   transfer_params = TransferFunctionParams(
       input_mode=0,           # Direct input mode
       multiplicative=True,    # Use multiplicative interactions
       shunting_inhibition=True  # Enable shunting inhibition
   )

Step 3: Build the Network
~~~~~~~~~~~~~~~~~~~~~~~~

Create an excitation-inhibition network with multiple layers:

.. code-block:: python

   ei_network = ExcitationInhibitionNetwork(
       input_dim=784,          # MNIST image size (28x28)
       transfer_params=transfer_params.asdict(),
       excitatory_layer_sizes=[100, 10],      # 100 excitatory neurons, then 10 output
       inhibitory_layer_sizes=[50, 0],        # 50 inhibitory neurons, then none
       excitatory_branch_factors=[5, 1],      # 5 branches per excitatory neuron
       inhibitory_branch_factors=[3, 0],      # 3 branches per inhibitory neuron
       ee_synapses_per_branch_per_layer=[10, 1],  # Excitatory-excitatory connections
       ei_synapses_per_branch_per_layer=[5, 0],   # Excitatory-inhibitory connections
       ie_synapses_per_branch_per_layer=[2, 0],   # Inhibitory-excitatory connections
       ii_synapses_per_branch_per_layer=[],       # No inhibitory-inhibitory connections
       reactivate=True,        # Enable reactivation mechanisms
       somatic_synapses=True   # Include somatic synapses
   )

Step 4: Create the Model
~~~~~~~~~~~~~~~~~~~~~~~

Wrap the network in a classifier model:

.. code-block:: python

   model = Classifier(
       core_network=ei_network,
       encoder_network=nn.Identity(),  # No input transformation
       decoder_network=nn.Identity()   # No output transformation
   )

Step 5: Load Dataset
~~~~~~~~~~~~~~~~~~~

Load the MNIST dataset:

.. code-block:: python

   train_dataset, val_dataset, test_dataset = get_unified_datasets(
       dataset_name="mnist",
       data_path="./data",
       train_valid_split=0.8
   )

Step 6: Train the Model
~~~~~~~~~~~~~~~~~~~~~~

Train the model using the built-in training utilities:

.. code-block:: python

   # Train with default settings
   train_model(
       model=model,
       train_dataset=train_dataset,
       val_dataset=val_dataset,
       epochs=10,
       batch_size=64,
       learning_rate=0.001
   )

Step 7: Evaluate Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate the trained model:

.. code-block:: python

   from dendritic_modeling.training.utils import evaluate_model
   
   test_accuracy = evaluate_model(model, test_dataset)
   print(f"Test accuracy: {test_accuracy:.4f}")

Complete Example
~~~~~~~~~~~~~~~

Here's the complete code for the quick start tutorial:

.. code-block:: python

   import torch
   import torch.nn as nn
   from dendritic_modeling.models import Classifier
   from dendritic_modeling.networks import ExcitationInhibitionNetwork
   from dendritic_modeling.config import TransferFunctionParams
   from dendritic_modeling.datasets import get_unified_datasets
   from dendritic_modeling.training.utils import train_model, evaluate_model
   
   # Create transfer function parameters
   transfer_params = TransferFunctionParams(
       input_mode=0,
       multiplicative=True,
       shunting_inhibition=True
   )
   
   # Build the network
   ei_network = ExcitationInhibitionNetwork(
       input_dim=784,
       transfer_params=transfer_params.asdict(),
       excitatory_layer_sizes=[100, 10],
       inhibitory_layer_sizes=[50, 0],
       excitatory_branch_factors=[5, 1],
       inhibitory_branch_factors=[3, 0],
       ee_synapses_per_branch_per_layer=[10, 1],
       ei_synapses_per_branch_per_layer=[5, 0],
       ie_synapses_per_branch_per_layer=[2, 0],
       ii_synapses_per_branch_per_layer=[],
       reactivate=True,
       somatic_synapses=True
   )
   
   # Create the model
   model = Classifier(
       core_network=ei_network,
       encoder_network=nn.Identity(),
       decoder_network=nn.Identity()
   )
   
   # Load dataset
   train_dataset, val_dataset, test_dataset = get_unified_datasets(
       dataset_name="mnist",
       data_path="./data"
   )
   
   # Train the model
   train_model(
       model=model,
       train_dataset=train_dataset,
       val_dataset=val_dataset,
       epochs=10,
       batch_size=64,
       learning_rate=0.001
   )
   
   # Evaluate performance
   test_accuracy = evaluate_model(model, test_dataset)
   print(f"Test accuracy: {test_accuracy:.4f}")

Basic Usage Patterns
-------------------

This section covers common usage patterns and best practices for working with the framework.

Working with Different Network Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework supports multiple network architectures:

**Standard MLP Networks**

.. code-block:: python

   from dendritic_modeling.networks import MLP
   
   mlp_network = MLP(
       input_dim=784,
       hidden_dims=[128, 64],
       output_dim=10,
       activation='relu',
       dropout=0.2
   )

**Convolutional Networks**

.. code-block:: python

   from dendritic_modeling.networks import ConvNet
   
   conv_network = ConvNet(
       input_channels=1,
       conv_layers=[
           {'out_channels': 32, 'kernel_size': 3, 'stride': 1},
           {'out_channels': 64, 'kernel_size': 3, 'stride': 2}
       ],
       fc_layers=[128, 10]
   )

**Hybrid Networks**

.. code-block:: python

   from dendritic_modeling.networks import HybridNetwork
   
   hybrid_network = HybridNetwork(
       conv_backbone=conv_network,
       dendritic_head=ei_network
   )

Working with Different Model Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework provides specialized models for different tasks:

**Classification Models**

.. code-block:: python

   from dendritic_modeling.models import Classifier
   
   classifier = Classifier(
       core_network=network,
       num_classes=10,
       loss_function='cross_entropy'
   )

**Regression Models**

.. code-block:: python

   from dendritic_modeling.models import Regressor
   
   regressor = Regressor(
       core_network=network,
       output_dim=1,
       loss_function='mse'
   )

Dataset Loading and Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework provides utilities for loading various datasets:

**Built-in Datasets**

.. code-block:: python

   from dendritic_modeling.datasets import get_unified_datasets
   
   # MNIST
   train_ds, val_ds, test_ds = get_unified_datasets("mnist")
   
   # CIFAR-10
   train_ds, val_ds, test_ds = get_unified_datasets("cifar10")
   
   # Custom synthetic datasets
   train_ds, val_ds, test_ds = get_unified_datasets("line_dataset", 
                                                   dataset_params={'n_samples': 1000})

**Custom Datasets**

.. code-block:: python

   import torch.utils.data as data
   
   class CustomDataset(data.Dataset):
       def __init__(self, data, labels):
           self.data = data
           self.labels = labels
           
       def __len__(self):
           return len(self.data)
           
       def __getitem__(self, idx):
           return self.data[idx], self.labels[idx]
   
   # Use with the framework
   custom_dataset = CustomDataset(your_data, your_labels)
   train_model(model, custom_dataset, val_dataset)

Configuration-Based Usage
~~~~~~~~~~~~~~~~~~~~~~~~

For more complex experiments, use YAML configuration files:

Create a configuration file ``config.yaml``:

.. code-block:: yaml

   model:
     task: classification
     network:
       type: EINet
       parameters:
         excitatory_layer_sizes: [100, 10]
         inhibitory_layer_sizes: [50, 0]
         excitatory_branch_factors: [5, 1]
         inhibitory_branch_factors: [3, 0]
   
   train:
     learning_strategy: mle
     lr: 0.001
     epochs: 50
     batch_size: 64
   
   task:
     dataset: mnist
     data_path: "./data"

Load and use the configuration:

.. code-block:: python

   from dendritic_modeling.config import load_config
   from dendritic_modeling.training.experiment_runner import run_experiment
   
   config = load_config("config.yaml")
   results = run_experiment(config)

Common Troubleshooting
--------------------

Installation Issues
~~~~~~~~~~~~~~~~~

**Import Errors**

.. code-block:: python

   # Check if package is correctly installed
   import sys
   print(sys.path)
   
   # Verify specific modules
   try:
       from dendritic_modeling.networks import ExcitationInhibitionNetwork
       print("Networks module imported successfully")
   except ImportError as e:
       print(f"Import error: {e}")

**CUDA Issues**

.. code-block:: python

   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   
   # Force CPU usage if needed
   device = torch.device('cpu')
   model = model.to(device)

Runtime Issues
~~~~~~~~~~~~

**Memory Errors**

- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training

.. code-block:: python

   # Reduce batch size
   train_model(model, train_dataset, val_dataset, batch_size=32)
   
   # Enable gradient accumulation
   train_model(model, train_dataset, val_dataset, 
              accumulate_grad_batches=4)

**Convergence Issues**

- Check learning rate
- Verify data preprocessing
- Monitor gradients

.. code-block:: python

   # Learning rate scheduling
   from dendritic_modeling.training.schedulers import CosineAnnealingLR
   
   scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
   train_model(model, train_dataset, val_dataset, scheduler=scheduler)

Next Steps
---------

Now that you have the framework installed and running, you can:

1. **Explore Architecture**: Learn about different network architectures in :doc:`architecture_and_models`
2. **Advanced Training**: Discover advanced training strategies in :doc:`training_and_configuration`
3. **Analysis Tools**: Learn about analysis and visualization in :doc:`analysis_and_visualization`
4. **API Reference**: Consult the detailed API documentation in :doc:`../api/index`

For more examples and tutorials, see the :doc:`../examples/index` section. 