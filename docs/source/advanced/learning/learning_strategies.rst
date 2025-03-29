.. _learning_strategies:

Learning Strategies
===================

Overview
--------

This document describes the learning strategies available in the dendritic modeling framework
for training neural networks with dendritic-inspired architectures.

Available Learning Strategies
-----------------------------

The framework provides several learning strategies tailored to the unique characteristics
of dendritic networks:

1. **Maximum Likelihood Estimation (MLE)**: Standard end-to-end training
2. **Freeze Branches**: Training branches sequentially from soma to distal branches
3. **Freeze Branches Reversed**: Training branches sequentially from distal branches to soma
4. **Freeze Branches with Pretraining**: Initial full-network training followed by branch-wise training from soma to distal
5. **Freeze Branches Reversed with Pretraining**: Initial full-network training followed by branch-wise training from distal to soma
6. **Two-Step Training**: Pre-training followed by main training
7. **Voltage Stabilization**: Initial voltage stabilization phase followed by main training

You can select the learning strategy in your configuration file:

.. code-block:: yaml

    train:
      learning_strategy: "mle"  # Choices: mle, freeze_branches, freeze_branches_reversed, 
                                # freeze_branches_with_pretraining, freeze_branches_reversed_with_pretraining,
                                # two_step, voltage_stabilization

Maximum Likelihood Estimation (MLE)
-----------------------------------

The MLE strategy is the most straightforward approach, applying standard backpropagation
to train the entire network end-to-end.

.. code-block:: python

    # Example of using MLE trainer
    from dendritic_modeling.learning_strategies import TrainerMLE
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = TrainerMLE(optimizer)
    
    results = trainer.train(
        model=model,
        train_data=train_ds,
        valid_data=valid_ds,
        epochs=50,
        batch_size=128,
        grad_clip_value=5
    )

Key features of MLE training:

- Optimizes all parameters simultaneously
- Applies gradient clipping to prevent instability
- Monitors validation performance for early stopping
- Supports both standard and dendritic network models

Freeze Branches Strategy
------------------------

The freeze branches strategy trains the dendritic network one branch at a time, starting from the branches
closest to the soma and progressively moving to more distal branches. After all branches have been trained 
independently, a final fine-tuning phase trains the full network end-to-end for a specified number of epochs.

.. code-block:: python

    # Example of using freeze branches trainer
    from dendritic_modeling.learning_strategies import TrainerMLE
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = TrainerMLE(optimizer)
    
    results = trainer.train_freeze_branches(
        model=model,
        train_data=train_ds,
        valid_data=valid_ds,
        epochs_per_layer=20,
        batch_size=128,
        final_tune_epochs=10  # Final fine-tuning with all branches unfrozen
    )

Key benefits of the freeze branches approach:

- Stabilizes training for complex dendritic networks
- Focuses training on specific dendritic compartments
- Allows fine-tuning all branches together during the final phase
- Better mimics biological learning in dendrites
- Provides a systematic approach to training dendritic trees

Freeze Branches with Pretraining Strategy
-----------------------------------------

The freeze branches with pretraining strategy enhances the standard freeze branches approach by adding an initial
full-network training phase before the branch-by-branch training:

1. Initial full-network training: Train all branches together for pretrain_epochs
2. Branch-by-branch training: Train each branch sequentially from soma to distal branches (proximal to distal)
3. Final fine-tuning: Train the full network end-to-end for fine-tuning

.. code-block:: python

    # Example of using freeze branches with pretraining trainer
    from dendritic_modeling.learning_strategies import TrainerMLE
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = TrainerMLE(optimizer)
    
    results = trainer.train_freeze_branches_with_pretraining(
        model=model,
        train_data=train_ds,
        valid_data=valid_ds,
        pretrain_epochs=10,     # Initial full network training
        epochs_per_layer=20,   # Per-branch training
        batch_size=128,
        final_tune_epochs=10    # Final fine-tuning with all branches unfrozen
    )

Key benefits of this approach:

- Provides a better initialization for branch-specific training
- Helps avoid getting stuck in poor local optima during branch-specific training
- Initial training can establish useful feature representations across the network
- Ensures that representations across branches are coherent
- Combines the benefits of joint and branch-wise training

Freeze Branches Reversed Strategy
---------------------------------

The freeze branches reversed strategy is similar to the freeze branches strategy, but trains the network
in the opposite direction: starting from the most distal branches (furthest from soma) and progressively moving
toward the soma. This approach follows the biological flow of information in dendrites.

.. code-block:: python

    # Example of using freeze branches reversed trainer
    from dendritic_modeling.learning_strategies import TrainerMLE
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = TrainerMLE(optimizer)
    
    results = trainer.train_freeze_branches_reversed(
        model=model,
        train_data=train_ds,
        valid_data=valid_ds,
        epochs_per_layer=20,
        batch_size=128,
        final_tune_epochs=10  # Final fine-tuning with all branches unfrozen
    )

Key benefits of the freeze branches reversed approach:

- Follows the natural flow of information in neural dendrites (from distal to proximal)
- May improve feature extraction in distal branches by training them first
- Potentially more biologically plausible learning sequence
- Training progresses from input-focused branches to output-focused branches
- Includes a final fine-tuning phase for whole-network integration

Freeze Branches Reversed with Pretraining Strategy
-------------------------------------------------

This strategy combines the benefits of initial full-network training with the reversed branch-by-branch training:

1. Initial full-network training: Train all branches together for pretrain_epochs
2. Branch-by-branch training: Train each branch sequentially from distal to proximal (input to soma)
3. Final fine-tuning: Train the full network end-to-end for fine-tuning

.. code-block:: python

    # Example of using freeze branches reversed with pretraining trainer
    from dendritic_modeling.learning_strategies import TrainerMLE
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = TrainerMLE(optimizer)
    
    results = trainer.train_freeze_branches_reversed_with_pretraining(
        model=model,
        train_data=train_ds,
        valid_data=valid_ds,
        pretrain_epochs=10,     # Initial full network training
        epochs_per_layer=20,   # Per-branch training
        batch_size=128,
        final_tune_epochs=10    # Final fine-tuning with all branches unfrozen
    )

Key benefits of this approach:

- Combines initial joint training with biologically-plausible branch sequence
- Follows the natural flow of information in dendrites (distal → proximal)
- Distal branches receive focused training after good initialization
- May be particularly effective for complex dendritic networks
- Reduces risk of catastrophic forgetting during branch-specific training

Two-Step Training
----------------

The two-step training strategy applies a two-phase approach: pretraining followed by main training. 
This can help with difficult optimization landscapes.

.. code-block:: python

    # Example of using two-step trainer
    from dendritic_modeling.learning_strategies import TwoStepTrainer
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = TwoStepTrainer(optimizer)
    
    results = trainer.train(
        model=model,
        train_data=train_ds,
        valid_data=valid_ds,
        pretrain_epochs=10,     # Phase 1: Pretraining
        maintrain_epochs=40,    # Phase 2: Main training
        batch_size=128
    )

Key benefits:

- First phase can establish good initial representations
- Second phase refines the representations for the specific task
- Can help with difficult optimization problems
- Provides a more stable training trajectory

Voltage Stabilization Strategy
----------------------------

The voltage stabilization strategy adds a warm-up phase to stabilize voltages before the main training phase.

.. code-block:: python

    # Example of using voltage stabilization trainer
    from dendritic_modeling.learning_strategies import VoltageStabilizationTrainer
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = VoltageStabilizationTrainer(optimizer)
    
    results = trainer.train(
        model=model,
        train_data=train_ds,
        valid_data=valid_ds,
        epochs=50,
        batch_size=128,
        grad_clip_value=5.0,
        shuffle=True,
        plot_losses=True,
        save_path="./outputs"
    )

Key benefits:

- Initial warm-up phase stabilizes branch voltages
- Targets a specific voltage level to optimize branch response
- Prepares the network for better learning in the main phase
- Can improve stability and convergence

Configuration Parameters for Learning Strategies
-----------------------------------------------

The following configuration parameters are available in the `train` section:

.. code-block:: yaml

    train:
      learning_strategy: "mle"         # Learning strategy to use
      run_name: "experiment_1"         # Name for the experiment run
      seed: 0                          # Random seed for reproducibility 
      train_valid_split: 0.8           # Split ratio for training/validation
      lr: 0.01                         # Learning rate
      weight_decay_rate: 0.1           # Weight decay rate
      epochs: 50                       # Total epochs for training
      batch_size: 128                  # Batch size
      grad_clip_value: 5.0             # Gradient clipping value
      epochs_per_layer: 50            # Epochs per layer for layer-wise training
      final_tune_epochs: 10            # Fine-tuning epochs after layer-wise training
      pretrain_epochs: 50              # Pretraining epochs (for two-step approaches)
      warmup_epochs: 5                 # Warmup epochs for voltage stabilization
      stability_target_voltage: 0.5    # Target voltage for stabilization

These parameters control various aspects of the learning process and can be adjusted to optimize training for your specific task and model architecture.

Factory Function
---------------

The framework provides a factory function to create the appropriate trainer based on the learning strategy:

.. code-block:: python

    from dendritic_modeling.learning_strategies import get_trainer
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = get_trainer(
        strategy="mle",  # learning strategy name
        optimizer=optimizer
    )

The get_trainer function supports all learning strategies discussed in this document. 