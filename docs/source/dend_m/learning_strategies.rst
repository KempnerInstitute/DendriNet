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
2. **Freeze Branches**: Sequential branch-by-branch training within dendritic trees
3. **Local Credit Assignment**: Training with locally-computed gradients
4. **Two-Step Training**: Pre-training an auxiliary network before the full model
5. **Two-Step Training with KL Divergence**: Similar to two-step, but with KL regularization

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
- Supports both probabilistic and deterministic models

.. note::
   The TrainerMLE class also includes a placeholder for SimCLR contrastive learning 
   (train_simclr method), which will be implemented in future versions.

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
        epochs_per_branch=20,
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
        epochs_per_branch=20,   # Per-branch training
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
        epochs_per_branch=20,
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
        epochs_per_branch=20,   # Per-branch training
        batch_size=128,
        final_tune_epochs=10    # Final fine-tuning with all branches unfrozen
    )

Key benefits of this approach:

- Combines initial joint training with biologically-plausible branch sequence
- Follows the natural flow of information in dendrites (distal → proximal)
- Distal branches receive focused training after good initialization
- May be particularly effective for complex dendritic networks
- Reduces risk of catastrophic forgetting during branch-specific training

Local Credit Assignment
-----------------------

The local credit assignment strategy trains each layer with a locally-computed loss,
reducing reliance on backpropagating gradients through the entire network.

.. code-block:: python

    # Example of using local credit assignment
    from dendritic_modeling.learning_strategies import LocalCreditAssignment
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = LocalCreditAssignment(optimizer)
    
    results = trainer.train(
        model=model,
        train_data=train_ds,
        valid_data=valid_ds,
        epochs=50,
        batch_size=128
    )

Key characteristics:

- Computes local targets for each layer
- Applies a weighted combination of local and global losses
- More biologically plausible learning approach
- Well-suited for networks with many layers

Two-Step Training
-----------------

The two-step training strategy first trains a simplified model, then uses this model
to initialize parts of the full dendritic network.

.. code-block:: python

    # Example of using two-step trainer
    from dendritic_modeling.learning_strategies import TwoStepTrainer
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = TwoStepTrainer(optimizer)
    
    results = trainer.train(
        model=model,
        train_data=train_ds,
        valid_data=valid_ds,
        pretrain_epochs=10,
        maintrain_epochs=40,
        batch_size=128
    )

Key aspects of two-step training:

- Trains an auxiliary network (typically an MLP) initially
- Transfers knowledge from the auxiliary to the main network
- Fine-tunes the full dendritic network
- Provides a good initialization for complex architectures

Voltage Stabilization Training
-----------------------------

The voltage stabilization strategy includes a warm-up phase that stabilizes model activations to a target value before starting the regular training.

.. code-block:: python

    # Example of using voltage stabilization trainer
    from dendritic_modeling.learning_strategies import VoltageStabilizationTrainer
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = VoltageStabilizationTrainer(
        optimizer, 
        warmup_epochs=5, 
        stability_target_voltage=0.5
    )
    
    results = trainer.train(
        model=model,
        train_data=train_ds,
        valid_data=valid_ds,
        epochs=50,  # Total epochs including warmup
        batch_size=128,
        grad_clip_value=5
    )

Key aspects of voltage stabilization training:

- Performs an initial warmup phase that minimizes the MSE between model outputs and a target value
- Stabilizes model activations before training begins
- Helps reduce convergence issues in complex models
- Can significantly improve training of dendritic networks with many branches
- Combines warmup with regular training in a two-phase approach

Two-Step Training with KL Divergence
------------------------------------

This strategy extends the two-step approach by using KL divergence to ensure
the dendritic network remains close to the auxiliary network's behavior.

.. code-block:: python

    # Example of using two-step trainer with KL
    from dendritic_modeling.learning_strategies import TwoStepTrainerWithKL
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = TwoStepTrainerWithKL(optimizer)
    
    results = trainer.train(
        model=model,
        train_data=train_ds,
        valid_data=valid_ds,
        pretrain_epochs=10,
        maintrain_epochs=40,
        kl_weight=0.1,
        batch_size=128
    )

Benefits of the KL approach:

- Prevents catastrophic forgetting of learned patterns
- Balances innovation with stability
- Ensures smooth transitions between training phases
- Provides regularization for the dendritic network

Input Network Training Strategies
---------------------------------

When using input transformation networks with the dendritic model, several specialized
training strategies are available:

.. code-block:: python

    # Example configuration for input network training
    input_net_training_config = {
        'training_mode': 'sequential',  # or 'joint'
        'train_input_net': True,
        'input_net_epochs': 20,
        'train_heads_independently': False,
        'inhibitory_head_objective': 'classification',  # or 'diversity' or 'combined'
        'diversity_weight': 0.5,
        'pretrained_options': {
            'source': 'none',  # or 'file' or 'pytorch'
            'weights_path': '',
            'pytorch_model': ''
        }
    }

Sequential vs. Joint Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Input networks can be trained either before the dendritic network (sequential)
or simultaneously with it (joint).

**Sequential Training**:

1. Train the input transformation network first
2. Freeze its weights or allow limited updates
3. Train the dendritic network

.. code-block:: python

    # Sequential training process
    def train_sequential(model, train_ds, valid_ds):
        # 1. Train input network only
        input_net_optimizer = torch.optim.Adam(
            model.input_transform.parameters(), 
            lr=0.001
        )
        
        for epoch in range(input_net_epochs):
            train_input_network(model, train_ds, input_net_optimizer)
            
        # 2. Freeze input network weights (optional)
        for param in model.input_transform.parameters():
            param.requires_grad = False
            
        # 3. Train dendritic network
        dendritic_optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.001
        )
        
        trainer = TrainerMLE(dendritic_optimizer)
        trainer.train(model, train_ds, valid_ds, epochs=50)

**Joint Training**:

1. Train all components simultaneously
2. Optionally use different learning rates for different components

.. code-block:: python

    # Joint training process
    def train_joint(model, train_ds, valid_ds):
        # Use parameter groups with different learning rates
        optimizer = torch.optim.Adam([
            {'params': model.input_transform.parameters(), 'lr': 0.001},
            {'params': model.dendritic_network.parameters(), 'lr': 0.0005}
        ])
        
        trainer = TrainerMLE(optimizer)
        trainer.train(model, train_ds, valid_ds, epochs=50)

Independent Head Training
~~~~~~~~~~~~~~~~~~~~~~~~~

When the input transformation network has separate heads for excitatory and inhibitory
pathways, they can be trained with different objectives.

.. code-block:: python

    # Independent head training
    def train_independent_heads(model, train_ds, valid_ds):
        # Train excitatory head for classification
        exc_optimizer = torch.optim.Adam(
            model.excitatory_head.parameters(),
            lr=0.001
        )
        
        for epoch in range(head_training_epochs):
            train_excitatory_head(model, train_ds, exc_optimizer)
            
        # Train inhibitory head with diversity objective
        inh_optimizer = torch.optim.Adam(
            model.inhibitory_head.parameters(),
            lr=0.001
        )
        
        for epoch in range(head_training_epochs):
            train_inhibitory_head_diversity(model, train_ds, inh_optimizer)
            
        # Fine-tune the entire network
        full_optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        trainer = TrainerMLE(full_optimizer)
        trainer.train(model, train_ds, valid_ds, epochs=20)

Available objectives for inhibitory head:

- **Classification**: Same objective as excitatory head
- **Diversity**: Maximize difference from excitatory representation
- **Combined**: Weighted sum of classification and diversity

Using Pre-trained Input Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pre-trained networks can be used as feature extractors, particularly for image data.

.. code-block:: python

    # Using pre-trained weights
    def setup_pretrained_input_network(model_cfg):
        """Setup input network with pre-trained weights"""
        pretrained_options = model_cfg.input_net_training.pretrained_options
        source = pretrained_options.get('source', 'none')
        
        if source == 'pytorch':
            # Load from torchvision models
            model_name = pretrained_options.get('pytorch_model', 'resnet18')
            input_net = getattr(torchvision.models, model_name)(pretrained=True)
            
            # Remove the classifier layer
            if hasattr(input_net, 'fc'):
                in_features = input_net.fc.in_features
                input_net.fc = nn.Identity()
                
            return input_net, in_features
                
        elif source == 'file':
            # Load from saved weights file
            weights_path = pretrained_options.get('weights_path', '')
            if not weights_path:
                raise ValueError("weights_path must be specified when source='file'")
                
            # Create input network and load weights
            input_net = create_input_network(model_cfg)
            input_net.load_state_dict(torch.load(weights_path))
            
            return input_net
            
        else:
            # Create new input network
            return create_input_network(model_cfg)

Combined Strategies
~~~~~~~~~~~~~~~~~~~

For complex tasks, different learning strategies can be combined:

.. code-block:: python

    # Example of combining strategies
    def train_combined_strategy(model, train_ds, valid_ds):
        # 1. Pre-train input network
        input_net_optimizer = torch.optim.Adam(
            model.input_transform.parameters(), 
            lr=0.001
        )
        
        for epoch in range(input_net_epochs):
            train_input_network(model, train_ds, input_net_optimizer)
            
        # 2. Train dendritic network using freeze branches
        dendritic_optimizer = torch.optim.Adam(model.dendritic_network.parameters(), lr=0.0005)
        trainer = TrainerMLE(dendritic_optimizer)
        
        trainer.train_freeze_branches(
            model, train_ds, valid_ds,
            epochs_per_branch=10,
            final_tune_epochs=5  # Fine-tune branches together after individual training
        )
        
        # 3. Fine-tune everything with MLE
        full_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        trainer = TrainerMLE(full_optimizer)
        
        trainer.train(model, train_ds, valid_ds, epochs=10)

Choosing the Right Strategy
---------------------------

Guidelines for selecting an appropriate learning strategy:

1. **For simple networks**: Start with standard MLE
2. **For deep networks**: Consider freeze branches or two-step approach
3. **For networks with input transformation**: Sequential training often works best
4. **For complex tasks**: Consider combining strategies or using KL regularization
5. **For biological plausibility**: Local credit assignment provides a more biologically-inspired approach

Custom Weight Decay
-------------------

The framework provides a custom weight decay implementation that applies regularization
differently to different network components:

.. code-block:: python

    # Custom weight decay optimizer
    from dendritic_modeling.learning_strategies import CustomWeightDecayOptimizer
    
    base_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = CustomWeightDecayOptimizer(model, base_optimizer, weight_decay=0.1)

Implementation Details
----------------------

All learning strategies are implemented as trainer classes that follow a common interface:

.. code-block:: python

    # Generic trainer structure
    class BaseTrainer:
        def __init__(self, optimizer, suppress_prints=False, print_every=10):
            self.optimizer = optimizer
            self.suppress_prints = suppress_prints
            self.print_every = print_every
            
        def _eval_dataset(self, model, dataset, device, batch_size=256):
            # Common evaluation logic
            ...
            
        def train(self, model, train_data, valid_data, **kwargs):
            # Training loop implementation
            ...
    
    # Select appropriate trainer based on strategy
    def get_trainer(strategy, optimizer, **kwargs):
        if strategy == "mle":
            return TrainerMLE(optimizer, **kwargs)
        elif strategy == "freeze_branches":
            return TrainerMLE(optimizer, **kwargs)  # Uses train_freeze_branches method
        elif strategy == "freeze_branches_reversed":
            return TrainerMLE(optimizer, **kwargs)  # Uses train_freeze_branches_reversed method
        elif strategy == "freeze_branches_with_pretraining":
            return TrainerMLE(optimizer, **kwargs)  # Uses train_freeze_branches_with_pretraining method
        elif strategy == "freeze_branches_reversed_with_pretraining":
            return TrainerMLE(optimizer, **kwargs)  # Uses train_freeze_branches_reversed_with_pretraining method
        elif strategy == "local_credit_assignment":
            return LocalCreditAssignment(optimizer, **kwargs)
        elif strategy == "two_step":
            return TwoStepTrainer(optimizer, **kwargs)
        elif strategy == "two_step_kl":
            return TwoStepTrainerWithKL(optimizer, **kwargs)
        elif strategy == "voltage_stabilization":
            return VoltageStabilizationTrainer(optimizer, **kwargs)
        else:
            logger.info(f"Unrecognized strategy '{strategy}', defaulting to 'mle'.")
            return TrainerMLE(optimizer, **kwargs) 