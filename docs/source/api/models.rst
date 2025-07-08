Models API Reference
====================

The models module provides high-level model wrappers that combine networks with task-specific heads and training utilities.

Base Model Classes
------------------

BaseModel
~~~~~~~~~

.. autoclass:: dendritic_modeling.models.BaseModel
   :members:
   :undoc-members:
   :show-inheritance:

   Abstract base class for all models in the framework.

   **Key attributes:**

   - ``core_network``: The underlying network architecture
   - ``input_network``: Optional input transformation network
   - ``task_head``: Task-specific output head (classification, regression, etc.)

   **Key methods:**

   - ``forward(x)``: Forward pass through the model
   - ``configure_optimizers()``: Set up optimizers for training
   - ``save_checkpoint(path)``: Save model state
   - ``load_checkpoint(path)``: Load model state

DendriticModel
~~~~~~~~~~~~~~

.. autoclass:: dendritic_modeling.models.DendriticModel
   :members:
   :undoc-members:
   :show-inheritance:

   General-purpose wrapper for dendritic networks.

   **Example usage:**

   .. code-block:: python

      from dendritic_modeling.models import DendriticModel

      # Create a classification model
      model = DendriticModel(
          network_name="ExcitationInhibitionNetwork",
          network_params={
              "input_dim": 784,
              "n_layers": 3,
              "n_excitatory": [200, 100, 10],
              "n_inhibitory": [50, 25, 0],
              "n_branches": [20, 15, 10]
          },
          task="classification",
          num_classes=10,
          input_network_name="identity"
      )

      # Forward pass
      logits = model(input_batch)
      
      # Get predictions
      predictions = model.predict(input_batch)

Classification Models
---------------------

DendriticClassifier
~~~~~~~~~~~~~~~~~~~

.. autoclass:: dendritic_modeling.models.classification.DendriticClassifier
   :members:
   :undoc-members:
   :show-inheritance:

   Specialized model for classification tasks.

   **Features:**

   - Automatic cross-entropy loss
   - Built-in accuracy metrics
   - Support for multi-class and multi-label classification
   - Optional temperature scaling

   **Example:**

   .. code-block:: python

      classifier = DendriticClassifier(
          network_config={
              "name": "ExcitationInhibitionNetwork",
              "params": {...}
          },
          num_classes=10,
          use_temperature_scaling=True,
          initial_temperature=1.0
      )

      # Training
      loss = classifier.training_step(batch, batch_idx)
      
      # Evaluation
      accuracy = classifier.validation_step(batch, batch_idx)

CNNClassifier
~~~~~~~~~~~~~

.. autoclass:: dendritic_modeling.models.classification.CNNClassifier
   :members:
   :undoc-members:
   :show-inheritance:

   CNN-based classifier with dendritic layers.

ViTClassifier
~~~~~~~~~~~~~

.. autoclass:: dendritic_modeling.models.classification.ViTClassifier
   :members:
   :undoc-members:
   :show-inheritance:

   Vision Transformer classifier with dendritic attention.

Regression Models
-----------------

DendriticRegressor
~~~~~~~~~~~~~~~~~~

.. autoclass:: dendritic_modeling.models.regression.DendriticRegressor
   :members:
   :undoc-members:
   :show-inheritance:

   Model for regression tasks.

   **Features:**

   - MSE and MAE loss options
   - Optional output normalization
   - Support for multi-output regression

   **Example:**

   .. code-block:: python

      regressor = DendriticRegressor(
          network_config={...},
          output_dim=1,
          loss_type="mse",
          normalize_output=True
      )

Baseline Models
---------------

The framework includes baseline models for comparison:

MLPBaseline
~~~~~~~~~~~

.. autoclass:: dendritic_modeling.models.baselines.MLPBaseline
   :members:
   :undoc-members:
   :show-inheritance:

   Standard MLP for baseline comparisons.

CNNBaseline
~~~~~~~~~~~

.. autoclass:: dendritic_modeling.models.baselines.CNNBaseline
   :members:
   :undoc-members:
   :show-inheritance:

   Standard CNN for baseline comparisons.

Model Configuration
-------------------

Models can be configured via YAML:

.. code-block:: yaml

   model:
     name: DendriticClassifier
     params:
       network_config:
         name: ExcitationInhibitionNetwork
         params:
           input_dim: 784
           n_layers: 3
           n_excitatory: [200, 100, 10]
           n_inhibitory: [50, 25, 0]
           n_branches: [20, 15, 10]
       
       num_classes: 10
       use_temperature_scaling: true
       
       optimizer:
         name: Adam
         lr: 0.001
         weight_decay: 0.0001
       
       scheduler:
         name: CosineAnnealingLR
         T_max: 100

Model Factory
-------------

.. autofunction:: dendritic_modeling.models.create_model

Factory function for creating models from configuration:

.. code-block:: python

   from dendritic_modeling.models import create_model
   
   # From config dict
   model = create_model(config_dict)
   
   # From YAML file
   model = create_model.from_yaml("config.yaml")

Model Utilities
---------------

Checkpointing
~~~~~~~~~~~~~

.. automodule:: dendritic_modeling.models.utils.checkpointing
   :members:

Functions for saving and loading model checkpoints:

- ``save_checkpoint(model, path, metadata=None)``
- ``load_checkpoint(model, path, strict=True)``
- ``get_best_checkpoint(checkpoint_dir, metric="val_acc")``

Model Analysis
~~~~~~~~~~~~~~

.. automodule:: dendritic_modeling.models.utils.analysis
   :members:

Tools for analyzing trained models:

- ``get_layer_representations(model, dataloader)``
- ``compute_feature_importance(model, dataset)``
- ``visualize_learned_features(model)``

Integration with PyTorch Lightning
----------------------------------

All models inherit from ``pytorch_lightning.LightningModule``:

.. code-block:: python

   import pytorch_lightning as pl
   
   # Create model
   model = DendriticClassifier(...)
   
   # Create trainer
   trainer = pl.Trainer(
       max_epochs=100,
       gpus=1,
       precision=16,
       callbacks=[...]
   )
   
   # Train
   trainer.fit(model, train_dataloader, val_dataloader)

See Also
--------

- :doc:`networks` - Network architectures
- :doc:`training` - Training utilities
- :doc:`/user_guide/training` - Training guide 