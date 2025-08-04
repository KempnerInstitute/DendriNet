API Reference
=============

This document provides a comprehensive API reference for the Dendritic Modeling Framework, organized by module and functionality.

Core Networks
-------------

ExcitationInhibitionNetwork
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The primary network class implementing dendritic computation with E/I separation.

.. code-block:: python

   from dendritic_modeling.networks import ExcitationInhibitionNetwork
   
   # Create E/I network with dendritic branches
   model = ExcitationInhibitionNetwork(
       input_dim=784,
       excitatory_layer_sizes=[512, 256, 128],
       inhibitory_layer_sizes=[128, 64, 32],
       excitatory_branch_factors=[8, 4, 2],
       inhibitory_branch_factors=[4, 2, 1],
       ee_synapses_per_branch=[64, 32, 16],
       ei_synapses_per_branch=[32, 16, 8],
       ie_synapses_per_branch=[16, 8, 4],
       use_shunting=True
   )

DendriNet
~~~~~~~~~

Multi-layer dendritic network with sequential branch processing.

.. code-block:: python

   from dendritic_modeling.networks import DendriNet
   
   dendrinet = DendriNet(
       n_soma=100,
       branch_factors=[8, 4, 2],
       excitatory_input_dim=784,
       excitatory_synapses_per_branch=64,
       inhibitory_input_dim=200,
       inhibitory_synapses_per_branch=32,
       use_shunting=True
   )

DendriticBranchLayer
~~~~~~~~~~~~~~~~~~~~

Individual layer implementing dendritic branch computation.

.. code-block:: python

   from dendritic_modeling.networks import DendriticBranchLayer
   
   layer = DendriticBranchLayer(
       output_dim=100,
       excitatory_input_dim=784,
       excitatory_synapses_per_branch=64,
       inhibitory_input_dim=200,
       inhibitory_synapses_per_branch=32,
       input_branch_factor=4,
       use_shunting=True
   )

Sparse Connectivity
------------------

TopKLinear
~~~~~~~~~~

Standard deterministic top-K sparse connectivity.

.. code-block:: python

   from dendritic_modeling.networks import TopKLinear
   
   sparse_layer = TopKLinear(
       in_features=512,
       out_features=256,
       K=10  # Each output connects to only 10 inputs
   )

StochasticTopKLinear
~~~~~~~~~~~~~~~~~~~~

Probabilistic top-K selection based on weight rank.

.. code-block:: python

   from dendritic_modeling.networks import StochasticTopKLinear
   
   stochastic_layer = StochasticTopKLinear(
       in_features=512,
       out_features=256,
       K=10,
       temperature=0.5,
       ultrafast=False
   )

VarianceTopKLinear
~~~~~~~~~~~~~~~~~~

Selection based on input activation variance and weight magnitude.

.. code-block:: python

   from dendritic_modeling.networks import VarianceTopKLinear
   
   variance_layer = VarianceTopKLinear(
       in_features=512,
       out_features=256,
       K=10
   )

DeepstLinear
~~~~~~~~~~~~

Dynamic sparse connectivity with adaptive rewiring.

.. code-block:: python

   from dendritic_modeling.networks import DeepstLinear
   
   deepst_layer = DeepstLinear(
       in_features=512,
       out_features=256,
       K=10,
       use_noise=True,
       sigma=0.05,
       rewiring_mode='global'
   )

BlockLinear
~~~~~~~~~~~

Block-structured sparse connections.

.. code-block:: python

   from dendritic_modeling.networks import BlockLinear
   
   block_layer = BlockLinear(
       in_features=512,
       out_features=256,
       block_size_input=64,
       block_size_output=32
   )

Models
------

Classifier
~~~~~~~~~~

Classification wrapper for networks.

.. code-block:: python

   from dendritic_modeling.models import Classifier
   
   classifier = Classifier(
       encoder_network=None,  # Optional encoder
       core_network=model,    # Your E/I network
       decoder_network=None   # Optional decoder
   )
   
   # Training
   outputs = classifier(inputs)
   loss = criterion(outputs, labels)

Regressor
~~~~~~~~~

Regression wrapper for networks.

.. code-block:: python

   from dendritic_modeling.models import Regressor
   
   regressor = Regressor(
       encoder_network=None,
       core_network=model,
       decoder_network=None
   )

Training Strategies
------------------

Trainer
~~~~~~~

Standard backpropagation training.

.. code-block:: python

   from dendritic_modeling.training import Trainer
   
   trainer = Trainer(
       model=classifier,
       optimizer=optimizer,
       criterion=criterion,
       device='cuda'
   )
   
   trainer.train(train_loader, val_loader, epochs=100)

LayerWiseTrainer
~~~~~~~~~~~~~~~~

Train layers sequentially.

.. code-block:: python

   from dendritic_modeling.training import LayerWiseTrainer
   
   layerwise_trainer = LayerWiseTrainer(
       model=model,
       num_epochs_per_layer=10
   )

TwoStepTrainer
~~~~~~~~~~~~~~

Two-phase training with local supervision.

.. code-block:: python

   from dendritic_modeling.training import TwoStepTrainer
   
   two_step_trainer = TwoStepTrainer(
       model=model,
       local_epochs=50,
       global_epochs=50
   )

Analysis Tools
-------------

AblationAnalyzer
~~~~~~~~~~~~~~~~

Systematic ablation studies.

.. code-block:: python

   from dendritic_modeling.analysis import AblationAnalyzer
   
   analyzer = AblationAnalyzer(model)
   results = analyzer.ablate_components(
       test_loader,
       components=['shunting', 'sparsity', 'dendrites']
   )

BranchActivationAnalyzer
~~~~~~~~~~~~~~~~~~~~~~~~

Analyze dendritic branch activations.

.. code-block:: python

   from dendritic_modeling.analysis import BranchActivationAnalyzer
   
   branch_analyzer = BranchActivationAnalyzer()
   activations = branch_analyzer.analyze(model, data_loader)

InformationAnalyzer
~~~~~~~~~~~~~~~~~~~

Information-theoretic analysis.

.. code-block:: python

   from dendritic_modeling.analysis import InformationAnalyzer
   
   info_analyzer = InformationAnalyzer(
       model=model,
       n_bins=30
   )
   
   mutual_info = info_analyzer.compute_mutual_information(
       inputs, outputs
   )

Visualization
-------------

Plotting Functions
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dendritic_modeling.plotting import (
       plot_dendrinet_activations,
       plot_branch_weights,
       plot_branch_gradients,
       plot_weight_distributions,
       plot_activation_maps
   )
   
   # Plot branch activations
   plot_dendrinet_activations(
       dendrinet, 
       exc_input_list, 
       inh_input_list
   )
   
   # Plot weight distributions
   plot_branch_weights(model, layer_idx=0)
   
   # Plot gradient flow
   plot_branch_gradients(model, layer_idx=0)

Configuration
-------------

TransferFunctionParams
~~~~~~~~~~~~~~~~~~~~~~

Configure transfer function behavior.

.. code-block:: python

   from dendritic_modeling.config import TransferFunctionParams
   
   transfer_params = TransferFunctionParams(
       input_mode=0,
       multiplicative=True,
       shunting_inhibition=True,
       use_shunting=True,
       epsilon=1e-6
   )

ModelConfig
~~~~~~~~~~~

Type-safe model configuration.

.. code-block:: python

   from dendritic_modeling.config import ModelConfig
   
   config = ModelConfig(
       model_type='EINetClassifier',
       core_network={
           'type': 'ExcitationInhibitionNetwork',
           'parameters': {
               'input_dim': 784,
               'excitatory_layer_sizes': [512, 256, 10],
               'inhibitory_layer_sizes': [128, 64, 0],
               'use_shunting': True
           }
       }
   )

Utilities
--------

Data Loading
~~~~~~~~~~~~

.. code-block:: python

   from dendritic_modeling.utils import create_dataset, create_dataloader
   
   # Create dataset
   train_dataset = create_dataset(
       name='CIFAR10',
       root='./data',
       train=True,
       transform=transforms
   )
   
   # Create dataloader
   train_loader = create_dataloader(
       dataset=train_dataset,
       batch_size=64,
       shuffle=True,
       num_workers=4
   )

Model Persistence
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dendritic_modeling.utils import save_checkpoint, load_checkpoint
   
   # Save model
   save_checkpoint(
       model=model,
       optimizer=optimizer,
       epoch=50,
       loss=0.05,
       path='checkpoint.pth'
   )
   
   # Load model
   checkpoint = load_checkpoint('checkpoint.pth')
   model.load_state_dict(checkpoint['model_state_dict'])

Experiment Tracking
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dendritic_modeling.utils import ExperimentTracker
   
   tracker = ExperimentTracker(
       experiment_name='dendritic_mnist',
       log_dir='./logs'
   )
   
   # Log metrics
   tracker.log_metrics({
       'loss': 0.05,
       'accuracy': 0.95
   }, step=100)

Vision Neuron Factories
----------------------

Create Gabor Units
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dendritic_modeling.networks import create_gabor_unit
   
   gabor_unit = create_gabor_unit(
       input_channels=3,
       num_orientations=8,
       num_scales=4
   )

Create Center-Surround Units
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dendritic_modeling.networks import create_center_surround_unit
   
   cs_unit = create_center_surround_unit(
       input_channels=3,
       center_size=3,
       surround_size=7
   )

Create Motion Units
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dendritic_modeling.networks import create_motion_unit
   
   motion_unit = create_motion_unit(
       input_channels=3,
       num_directions=8,
       temporal_window=5
   )

Best Practices
-------------

Model Design
~~~~~~~~~~~~

1. **Start Simple**: Begin with fewer dendrites and layers
2. **Gradual Complexity**: Add dendrites as needed
3. **Monitor Sparsity**: Ensure reasonable connectivity levels
4. **Validate Dynamics**: Check activation patterns

Training Tips
~~~~~~~~~~~~~

1. **Learning Rate**: Start with 1e-4 for dendritic networks
2. **Batch Size**: Larger batches (128+) often work better
3. **Regularization**: Use weight decay and dropout carefully
4. **Initialization**: Proper initialization is crucial

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Sparse Operations**: Use sparse implementations when sparsity > 90%
2. **Mixed Precision**: Use automatic mixed precision for speed
3. **Gradient Accumulation**: For memory-limited scenarios
4. **Distributed Training**: Use DataParallel or DistributedDataParallel 