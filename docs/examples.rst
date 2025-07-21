Examples
========

This section provides practical examples using the Dendritic Modeling Framework.

MNIST Classification with E/I Network
-------------------------------------

A complete example of training an E/I network on MNIST:

.. code-block:: python

   import torch
   from torch.utils.data import DataLoader
   from torchvision import datasets, transforms
   
   from dendritic_modeling.networks import ExcitationInhibitionNetwork
   from dendritic_modeling.models import Classifier
   from dendritic_modeling.training import Trainer
   from dendritic_modeling.config import TransferFunctionParams
   
   # Load MNIST
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,))
   ])
   
   train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
   test_dataset = datasets.MNIST('./data', train=False, transform=transform)
   
   # Create E/I network
   transfer_params = TransferFunctionParams(
       input_mode=0,
       multiplicative=True,
       shunting_inhibition=True
   )
   
   network = ExcitationInhibitionNetwork(
       input_dim=784,
       transfer_params=transfer_params.asdict(),
       excitatory_layer_sizes=[256, 128, 10],
       inhibitory_layer_sizes=[64, 32, 0],
       excitatory_branch_factors=[4, 2, 1],
       inhibitory_branch_factors=[2, 1, 0],
       ee_synapses_per_branch_per_layer=[32, 16, 1],
       ei_synapses_per_branch_per_layer=[16, 8, 0],
       ie_synapses_per_branch_per_layer=[8, 4, 0],
       use_shunting=True,
       topk_type="standard"
   )
   
   # Create classifier
   model = Classifier(
       encoder_network=None,
       core_network=network,
       decoder_network=None
   )
   
   # Training
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
   trainer = Trainer(
       optimizer=optimizer,
       loss_function="cross_entropy",
       epochs=50,
       batch_size=64,
       early_stopping=True,
       patience=10
   )
   
   # Train model
   results = trainer.train(model, train_dataset, test_dataset)
   print(f"Best accuracy: {results['best_epoch']}")

Vision Tasks with DendriteConv2d
--------------------------------

Creating biologically-inspired vision models:

.. code-block:: python

   import torch
   import torch.nn as nn
   from dendritic_modeling.networks.layers import DendriteConv2d
   from dendritic_modeling.networks import MLP
   
   class DendriticVisionNet(nn.Module):
       def __init__(self, num_classes=10):
           super().__init__()
           
           # Dendritic convolutional layers
           self.conv1 = DendriteConv2d(3, 32, kernel_size=5, padding=2)
           self.conv2 = DendriteConv2d(32, 64, kernel_size=5, padding=2)
           self.conv3 = DendriteConv2d(64, 128, kernel_size=3, padding=1)
           
           self.pool = nn.MaxPool2d(2, 2)
           self.dropout = nn.Dropout2d(0.25)
           
           # Classifier head
           self.classifier = MLP(
               input_dim=128 * 4 * 4,
               hidden_dims=[256, 128],
               output_dim=num_classes,
               activation="relu"
           )
           
       def forward(self, x):
           # Conv block 1
           x = self.conv1(x)
           x = torch.relu(x)
           x = self.pool(x)
           
           # Conv block 2
           x = self.conv2(x)
           x = torch.relu(x)
           x = self.pool(x)
           x = self.dropout(x)
           
           # Conv block 3
           x = self.conv3(x)
           x = torch.relu(x)
           x = self.pool(x)
           
           # Flatten and classify
           x = x.view(x.size(0), -1)
           x = self.classifier(x)
           
           return x
   
   # Create model for CIFAR-10
   model = DendriticVisionNet(num_classes=10)
   
   # Example forward pass
   x = torch.randn(1, 3, 32, 32)
   output = model(x)
   print(f"Output shape: {output.shape}")

Sparse Networks with DEEPST
---------------------------

Using dynamic sparse connectivity:

.. code-block:: python

   from dendritic_modeling.networks import (
       ExcitationInhibitionNetwork,
       DeepstLinear
   )
   
   # Create sparse E/I network with DEEPST
   sparse_network = ExcitationInhibitionNetwork(
       input_dim=784,
       excitatory_layer_sizes=[500, 300, 10],
       inhibitory_layer_sizes=[100, 60, 0],
       excitatory_branch_factors=[1, 1, 1],  # No branching for DEEPST
       inhibitory_branch_factors=[1, 1, 0],
       ee_synapses_per_branch_per_layer=[50, 30, 1],
       ei_synapses_per_branch_per_layer=[25, 15, 0],
       ie_synapses_per_branch_per_layer=[10, 6, 0],
       topk_type="deepst",  # Use DEEPST sparse connectivity
       excitatory_target_density=0.1,  # 10% connectivity
       inhibitory_target_density=0.2,  # 20% connectivity
       use_noise=True,
       sigma=0.05,
       rewiring_mode="global",
       rewire_frequency=100
   )
   
   # The network will maintain sparse connectivity during training
   # and periodically rewire connections based on importance

Biological Vision Units
-----------------------

Using pre-built biologically-inspired units:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic import (
       create_gabor_unit,
       create_center_surround_unit,
       create_motion_unit,
       create_end_stopped_unit
   )
   
   # Create different types of vision units
   units = {
       'Gabor (0°)': create_gabor_unit(orientation=0.0),
       'Gabor (45°)': create_gabor_unit(orientation=45.0),
       'Center-Surround': create_center_surround_unit(),
       'Motion (right)': create_motion_unit(direction="rightward"),
       'End-stopped': create_end_stopped_unit(preferred_length=20)
   }
   
   # Test on sample images
   test_image = torch.randn(1, 1, 64, 64)
   
   fig, axes = plt.subplots(2, 3, figsize=(12, 8))
   axes = axes.flatten()
   
   for idx, (name, unit) in enumerate(units.items()):
       response = unit(test_image)
       axes[idx].imshow(response[0, 0].detach().numpy(), cmap='viridis')
       axes[idx].set_title(name)
       axes[idx].axis('off')
   
   plt.tight_layout()
   plt.show()

Analysis and Ablation Studies
-----------------------------

Comprehensive analysis of trained models:

.. code-block:: python

   from dendritic_modeling.analysis import AnalysisManager
   from dendritic_modeling.config import AnalysisConfig
   
   # Configure analysis
   analysis_config = AnalysisConfig(
       performance_analysis={"enabled": True, "accuracy": True},
       information_analysis={"enabled": True, "n_bins": 10},
       ablation_analysis={
           "enabled": True,
           "ablation_types": ["excitatory", "inhibitory", "branches", "shunting"]
       },
       synapse_turnover_analysis={"enabled": True},
       weight_analysis={"enabled": True}
   )
   
   # Create analysis manager
   analyzer = AnalysisManager(
       model=model,
       data={"train": train_dataset, "valid": val_dataset, "test": test_dataset},
       analysis_config=analysis_config,
       save_root="./analysis_results",
       device="cuda"
   )
   
   # Run all analyses
   analyzer.run_analysis(filename="final_model")
   
   # Individual analyses
   from dendritic_modeling.analysis import AblationAnalyzer
   
   ablation = AblationAnalyzer(
       ablation_types=["excitatory", "inhibitory", "branches"]
   )
   
   results = ablation.analyze(
       model=model,
       test_dataset=test_dataset,
       device="cuda",
       save_path="./ablation_results"
   )
   
   # Plot results
   import pandas as pd
   
   df = pd.DataFrame(results)
   df.plot(kind='bar', x='ablation_type', y='accuracy_drop')
   plt.ylabel('Accuracy Drop (%)')
   plt.title('Impact of Ablating Different Components')
   plt.show()

Custom Training Strategies
--------------------------

Implementing specialized training approaches:

.. code-block:: python

   from dendritic_modeling.training import (
       LayerWiseTrainer,
       VoltageStabilizationTrainer,
       TwoStepTrainer
   )
   
   # Layer-wise training
   layerwise_trainer = LayerWiseTrainer(
       optimizer=optimizer,
       loss_function="cross_entropy",
       pretrain_epochs=10,
       epochs_per_layer=5,
       final_tune_epochs=20,
       reverse_training=False  # Train from input to output
   )
   
   # Voltage stabilization for stable dendritic computation
   voltage_trainer = VoltageStabilizationTrainer(
       optimizer=optimizer,
       loss_function="cross_entropy",
       pretrain_epochs=5,
       target_voltage=0.5,
       epochs=50
   )
   
   # Two-step training with local supervision
   twostep_trainer = TwoStepTrainer(
       optimizer=optimizer,
       loss_function="cross_entropy",
       pretrain_epochs=10,
       epochs=40
   )
   
   # Train with different strategies
   results_layerwise = layerwise_trainer.train(model, train_dataset, val_dataset)
   results_voltage = voltage_trainer.train(model, train_dataset, val_dataset)
   results_twostep = twostep_trainer.train(model, train_dataset, val_dataset)

Working with Synthetic Datasets
-------------------------------

Using built-in synthetic datasets for controlled experiments:

.. code-block:: python

   from dendritic_modeling.datasets import (
       LineDataset,
       CenterSurroundDataset,
       OrientationBarsDataset,
       MultiXORDataset
   )
   
   # Line detection task
   line_dataset = LineDataset(
       n_samples=10000,
       img_size=32,
       n_lines=3,
       noise_level=0.1
   )
   
   # Center-surround task
   cs_dataset = CenterSurroundDataset(
       n_samples=5000,
       img_size=28,
       center_radius=5,
       surround_radius=10
   )
   
   # Orientation discrimination
   orientation_dataset = OrientationBarsDataset(
       n_samples=8000,
       img_size=32,
       n_orientations=8,
       bar_length=20
   )
   
   # Complex XOR task
   xor_dataset = MultiXORDataset(
       n_samples=10000,
       n_features=100,
       n_xor_pairs=5,
       noise_level=0.05
   )
   
   # Use with models
   from torch.utils.data import DataLoader
   
   loader = DataLoader(line_dataset, batch_size=32, shuffle=True)
   
   for images, labels in loader:
       print(f"Images shape: {images.shape}")
       print(f"Labels shape: {labels.shape}")
       break

Configuration-Based Experiments
-------------------------------

Running experiments with YAML configurations:

.. code-block:: yaml

   # experiment_config.yaml
   model:
     task: classification
     encoder_network:
       type: identity
       parameters: {}
     core_network:
       type: EINet
       parameters:
         input_dim: 784
         excitatory_layer_sizes: [300, 150, 10]
         inhibitory_layer_sizes: [75, 37, 0]
         excitatory_branch_factors: [4, 2, 1]
         inhibitory_branch_factors: [2, 1, 0]
         ee_synapses_per_branch_per_layer: [30, 15, 1]
         ei_synapses_per_branch_per_layer: [15, 7, 0]
         ie_synapses_per_branch_per_layer: [7, 3, 0]
         topk_type: "standard"
         use_shunting: true
         transfer_params:
           input_mode: 0
           multiplicative: true
           shunting_inhibition: true
     decoder_network:
       type: identity
       parameters: {}
   
   train:
     main:
       learning_strategy: standard
       optimizer: AdamW
       lr: 0.001
       epochs: 100
       batch_size: 128
       early_stopping: true
       patience: 15
   
   task:
     dataset: mnist
     data_path: ./data
   
   analysis:
     performance_analysis:
       enabled: true
       accuracy: true
     ablation_analysis:
       enabled: true
       ablation_types: [excitatory, inhibitory, branches]

.. code-block:: python

   from dendritic_modeling.config import load_config
   from dendritic_modeling.scripts.train_experiments import main
   
   # Load and run experiment
   config = load_config("experiment_config.yaml")
   results = main(config)

Interactive Examples
--------------------

- `Basic Training <../notebooks/basic_training.ipynb>`_

These examples demonstrate the key capabilities of the Dendritic Modeling Framework. For more details, see the :doc:`api_reference`. 