Training and Configuration
=========================

This comprehensive guide covers training strategies, configuration systems, and experiment management for the Dendritic Modeling Framework.

.. figure:: ../../../_static/training_workflow_diagram.svg
   :align: center
   :width: 90%
   :alt: Training Workflow Overview

   **Training Workflow**: Complete training pipeline from data loading through model evaluation, showing configuration management, training strategies, and experiment tracking.

Training Strategies
------------------

The framework supports multiple training strategies optimized for different network architectures and learning objectives.

Maximum Likelihood Estimation (MLE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standard gradient-based training with backpropagation:

.. code-block:: python

   from dendritic_modeling.training import MLETrainer
   from dendritic_modeling.config import TrainingConfig
   
   # Basic MLE training
   trainer = MLETrainer(
       model=model,
       optimizer='adam',
       learning_rate=0.001,
       weight_decay=1e-4
   )
   
   # Training configuration
   train_config = TrainingConfig(
       learning_strategy='mle',
       epochs=100,
       batch_size=64,
       validation_split=0.2,
       early_stopping=True,
       patience=10
   )
   
   # Train the model
   results = trainer.train(
       train_dataset=train_dataset,
       val_dataset=val_dataset,
       config=train_config
   )

**Key Features:**

* Standard backpropagation through dendritic networks
* Automatic gradient clipping and normalization
* Support for various optimizers (Adam, SGD, RMSprop)
* Learning rate scheduling
* Mixed precision training support

Branch-Wise Training Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specialized training strategies that leverage the dendritic structure:

**Freeze Branches Forward**

Train layers sequentially from input to output:

.. code-block:: python

   from dendritic_modeling.training import BranchWiseTrainer
   
   # Freeze branches forward training
   trainer = BranchWiseTrainer(
       model=model,
       strategy='freeze_branches_forward'
   )
   
   train_config = TrainingConfig(
       learning_strategy='freeze_branches_forward',
       epochs_per_layer=20,
       final_tune_epochs=10,
       layer_wise_lr=[0.001, 0.0005, 0.0001],
       freeze_previous_layers=True
   )
   
   results = trainer.train(train_dataset, val_dataset, train_config)

**Freeze Branches Reversed**

Train layers sequentially from output to input:

.. code-block:: python

   # Freeze branches reversed training
   train_config = TrainingConfig(
       learning_strategy='freeze_branches_reversed',
       epochs_per_layer=25,
       final_tune_epochs=15,
       layer_wise_lr=[0.0001, 0.0005, 0.001],  # Reversed order
       unfreeze_schedule='exponential'
   )
   
   results = trainer.train(train_dataset, val_dataset, train_config)

**Freeze Branches Interleaved**

Alternate training between different layers:

.. code-block:: python

   # Freeze branches interleaved training
   train_config = TrainingConfig(
       learning_strategy='freeze_branches_interleaved',
       epochs_per_cycle=10,
       num_cycles=5,
       interleave_pattern=[0, 2, 1, 2, 0],  # Layer training order
       cycle_lr_decay=0.9
   )
   
   results = trainer.train(train_dataset, val_dataset, train_config)

**Branch-Wise Training Benefits:**

* Improved convergence for deep dendritic networks
* Better gradient flow through complex architectures
* Reduced overfitting through structured regularization
* Biologically-inspired learning patterns

Voltage Stabilization Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Training strategy that maintains stable voltage dynamics:

.. code-block:: python

   from dendritic_modeling.training import VoltageStabilizationTrainer
   
   # Voltage stabilization training
   trainer = VoltageStabilizationTrainer(
       model=model,
       voltage_target=0.5,
       stabilization_weight=0.1
   )
   
   train_config = TrainingConfig(
       learning_strategy='voltage_stabilization',
       epochs=100,
       batch_size=32,
       voltage_monitoring=True,
       voltage_clipping=True,
       voltage_regularization=0.01
   )
   
   results = trainer.train(train_dataset, val_dataset, train_config)

**Key Features:**

* Maintains stable dendritic voltage dynamics
* Prevents voltage saturation and instability
* Biologically-inspired voltage regulation
* Improved network robustness

Curriculum Learning
~~~~~~~~~~~~~~~~~~

Progressive training with increasing difficulty:

.. code-block:: python

   from dendritic_modeling.training import CurriculumTrainer
   from dendritic_modeling.datasets import create_curriculum_dataset
   
   # Create curriculum dataset
   curriculum_dataset = create_curriculum_dataset(
       base_dataset=train_dataset,
       difficulty_levels=5,
       progression_strategy='linear'
   )
   
   # Curriculum training
   trainer = CurriculumTrainer(
       model=model,
       curriculum_strategy='difficulty_based'
   )
   
   train_config = TrainingConfig(
       learning_strategy='curriculum',
       epochs_per_level=20,
       difficulty_progression='adaptive',
       performance_threshold=0.8
   )
   
   results = trainer.train(curriculum_dataset, val_dataset, train_config)

Meta-Learning and Few-Shot Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Training for rapid adaptation to new tasks:

.. code-block:: python

   from dendritic_modeling.training import MetaLearner
   
   # Meta-learning setup
   meta_learner = MetaLearner(
       model=model,
       meta_algorithm='maml',
       inner_steps=5,
       inner_lr=0.01,
       outer_lr=0.001
   )
   
   train_config = TrainingConfig(
       learning_strategy='meta_learning',
       meta_batch_size=16,
       support_shots=5,
       query_shots=15,
       num_tasks_per_batch=8
   )
   
   results = meta_learner.train(meta_dataset, train_config)

Configuration System
-------------------

The framework provides a comprehensive configuration system for reproducible experiments.

YAML Configuration
~~~~~~~~~~~~~~~~~

Complete experiment configuration in YAML:

.. code-block:: yaml

   # experiment_config.yaml
   experiment:
     name: "dendritic_mnist_classification"
     description: "E/I network training on MNIST with branch-wise learning"
     tags: ["mnist", "classification", "dendritic", "branch-wise"]
     
   model:
     task: classification
     core_network:
       type: EINet
       parameters:
         input_dim: 784
         transfer_params:
           input_mode: 0
           multiplicative: true
           shunting_inhibition: true
         excitatory_layer_sizes: [200, 100, 10]
         inhibitory_layer_sizes: [100, 50, 0]
         excitatory_branch_factors: [10, 5, 1]
         inhibitory_branch_factors: [5, 3, 0]
         ee_synapses_per_branch_per_layer: [15, 10, 1]
         ei_synapses_per_branch_per_layer: [8, 5, 0]
         ie_synapses_per_branch_per_layer: [3, 2, 0]
         ii_synapses_per_branch_per_layer: []
         reactivate: true
         somatic_synapses: true
     
     encoder_network:
       type: identity
       
     decoder_network:
       type: identity
   
   training:
     learning_strategy: freeze_branches_reversed
     epochs_per_layer: 25
     final_tune_epochs: 15
     batch_size: 64
     validation_split: 0.2
     
     optimizer:
       type: adam
       lr: 0.001
       weight_decay: 1e-4
       betas: [0.9, 0.999]
       
     scheduler:
       type: cosine_annealing
       T_max: 100
       eta_min: 1e-6
       
     early_stopping:
       patience: 15
       min_delta: 1e-4
       monitor: val_accuracy
       mode: max
       
     regularization:
       dropout: 0.2
       weight_decay: 1e-4
       label_smoothing: 0.1
       
   task:
     dataset: mnist
     data_path: "./data"
     num_classes: 10
     input_shape: [1, 28, 28]
     
     data_augmentation:
       enabled: true
       transforms:
         - type: random_rotation
           degrees: 10
         - type: random_translation
           translate: [0.1, 0.1]
         - type: gaussian_noise
           std: 0.01
   
   evaluation:
     metrics: [accuracy, precision, recall, f1_score]
     test_time_augmentation: false
     ensemble_size: 1
     
   logging:
     level: INFO
     log_dir: "./logs"
     save_model: true
     save_frequency: 10
     
   experiment_tracking:
     enabled: true
     backend: wandb
     project: "dendritic-modeling"
     entity: "your-team"
     tags: ["experiment", "dendritic", "mnist"]

Load and use configuration:

.. code-block:: python

   from dendritic_modeling.config import load_config
   from dendritic_modeling.training import run_experiment
   
   # Load configuration
   config = load_config('experiment_config.yaml')
   
   # Run experiment
   results = run_experiment(config)

Python Configuration
~~~~~~~~~~~~~~~~~~~

Programmatic configuration for dynamic experiments:

.. code-block:: python

   from dendritic_modeling.config import (
       ExperimentConfig,
       ModelConfig,
       TrainingConfig,
       TaskConfig,
       CoreNetworkConfig,
       EINetParameters,
       TransferFunctionParams,
       OptimizerConfig,
       SchedulerConfig
   )
   
   # Create comprehensive configuration
   config = ExperimentConfig(
       name="dendritic_experiment",
       description="Programmatic experiment configuration",
       
       model=ModelConfig(
           task='classification',
           core_network=CoreNetworkConfig(
               type='EINet',
               parameters=EINetParameters(
                   input_dim=784,
                   transfer_params=TransferFunctionParams(
                       input_mode=0,
                       multiplicative=True,
                       shunting_inhibition=True
                   ),
                   excitatory_layer_sizes=[200, 100, 10],
                   inhibitory_layer_sizes=[100, 50, 0],
                   excitatory_branch_factors=[10, 5, 1],
                   inhibitory_branch_factors=[5, 3, 0],
                   ee_synapses_per_branch_per_layer=[15, 10, 1],
                   ei_synapses_per_branch_per_layer=[8, 5, 0],
                   ie_synapses_per_branch_per_layer=[3, 2, 0],
                   ii_synapses_per_branch_per_layer=[],
                   reactivate=True,
                   somatic_synapses=True
               )
           )
       ),
       
       training=TrainingConfig(
           learning_strategy='freeze_branches_reversed',
           epochs_per_layer=25,
           final_tune_epochs=15,
           batch_size=64,
           validation_split=0.2,
           
           optimizer=OptimizerConfig(
               type='adam',
               lr=0.001,
               weight_decay=1e-4
           ),
           
           scheduler=SchedulerConfig(
               type='cosine_annealing',
               T_max=100,
               eta_min=1e-6
           )
       ),
       
       task=TaskConfig(
           dataset='mnist',
           data_path='./data',
           num_classes=10
       )
   )
   
   # Run experiment
   results = run_experiment(config)

Configuration Validation
~~~~~~~~~~~~~~~~~~~~~~~~

Automatic validation of configuration parameters:

.. code-block:: python

   from dendritic_modeling.config import validate_config
   
   # Validate configuration
   is_valid, errors = validate_config(config)
   
   if not is_valid:
       for error in errors:
           print(f"Configuration error: {error}")
   else:
       print("Configuration is valid")
       
   # Auto-fix common issues
   config = auto_fix_config(config)

Advanced Training Techniques
---------------------------

Mixed Precision Training
~~~~~~~~~~~~~~~~~~~~~~~

Accelerate training with mixed precision:

.. code-block:: python

   from dendritic_modeling.training import MixedPrecisionTrainer
   
   # Mixed precision configuration
   trainer = MixedPrecisionTrainer(
       model=model,
       use_amp=True,
       loss_scale='dynamic',
       opt_level='O1'
   )
   
   train_config = TrainingConfig(
       learning_strategy='mle',
       epochs=100,
       batch_size=128,  # Larger batch size with mixed precision
       mixed_precision=True,
       gradient_clipping=1.0
   )
   
   results = trainer.train(train_dataset, val_dataset, train_config)

Gradient Accumulation
~~~~~~~~~~~~~~~~~~~~

Handle large effective batch sizes:

.. code-block:: python

   # Gradient accumulation setup
   train_config = TrainingConfig(
       learning_strategy='mle',
       epochs=100,
       batch_size=32,
       accumulate_grad_batches=4,  # Effective batch size = 128
       gradient_clipping=1.0
   )
   
   results = trainer.train(train_dataset, val_dataset, train_config)

Distributed Training
~~~~~~~~~~~~~~~~~~~

Scale training across multiple GPUs:

.. code-block:: python

   from dendritic_modeling.training import DistributedTrainer
   
   # Distributed training setup
   trainer = DistributedTrainer(
       model=model,
       backend='nccl',
       world_size=4,
       rank=0
   )
   
   train_config = TrainingConfig(
       learning_strategy='mle',
       epochs=100,
       batch_size=64,  # Per-GPU batch size
       distributed=True,
       sync_batchnorm=True
   )
   
   results = trainer.train(train_dataset, val_dataset, train_config)

Hyperparameter Optimization
---------------------------

Automated hyperparameter tuning:

Optuna Integration
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import optuna
   from dendritic_modeling.training import OptunaOptimizer
   
   def objective(trial):
       # Suggest hyperparameters
       lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
       batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
       excitatory_size = trial.suggest_int('excitatory_size', 50, 300)
       
       # Create configuration
       config = TrainingConfig(
           learning_strategy='mle',
           epochs=50,
           batch_size=batch_size,
           optimizer=OptimizerConfig(
               type='adam',
               lr=lr
           )
       )
       
       # Update model configuration
       config.model.core_network.parameters.excitatory_layer_sizes = [excitatory_size, 10]
       
       # Train and evaluate
       results = run_experiment(config)
       return results['val_accuracy']
   
   # Run optimization
   study = optuna.create_study(direction='maximize')
   study.optimize(objective, n_trials=100)
   
   best_params = study.best_params
   print(f"Best parameters: {best_params}")

Weights & Biases Sweeps
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # wandb_sweep.yaml
   program: train_experiment.py
   method: bayes
   metric:
     name: val_accuracy
     goal: maximize
   
   parameters:
     learning_rate:
       distribution: log_uniform
       min: 1e-5
       max: 1e-2
     batch_size:
       values: [32, 64, 128]
     excitatory_layer_sizes:
       values: [[100, 10], [200, 10], [300, 10]]
     learning_strategy:
       values: ['mle', 'freeze_branches_forward', 'freeze_branches_reversed']

Run sweep:

.. code-block:: bash

   wandb sweep wandb_sweep.yaml
   wandb agent <sweep_id>

Experiment Management
--------------------

Experiment Tracking
~~~~~~~~~~~~~~~~~~

Integration with experiment tracking platforms:

.. code-block:: python

   from dendritic_modeling.training import ExperimentTracker
   
   # Initialize tracker
   tracker = ExperimentTracker(
       backend='wandb',
       project='dendritic-modeling',
       entity='your-team',
       name='experiment_1'
   )
   
   # Log configuration
   tracker.log_config(config)
   
   # Log metrics during training
   for epoch in range(epochs):
       # Training step
       train_loss, train_acc = train_step(model, train_loader)
       val_loss, val_acc = validate_step(model, val_loader)
       
       # Log metrics
       tracker.log_metrics({
           'epoch': epoch,
           'train_loss': train_loss,
           'train_accuracy': train_acc,
           'val_loss': val_loss,
           'val_accuracy': val_acc
       })
       
       # Log model artifacts
       if epoch % 10 == 0:
           tracker.log_model(model, f'model_epoch_{epoch}')
   
   # Log final results
   tracker.log_results(results)
   tracker.finish()

Model Checkpointing
~~~~~~~~~~~~~~~~~~

Automatic model saving and loading:

.. code-block:: python

   from dendritic_modeling.training import ModelCheckpointer
   
   # Checkpoint configuration
   checkpointer = ModelCheckpointer(
       save_dir='./checkpoints',
       save_frequency=10,
       save_best_only=True,
       monitor='val_accuracy',
       mode='max'
   )
   
   # Training with checkpointing
   trainer = MLETrainer(
       model=model,
       checkpointer=checkpointer
   )
   
   results = trainer.train(train_dataset, val_dataset, train_config)
   
   # Load best checkpoint
   best_model = checkpointer.load_best_model()

Result Analysis
~~~~~~~~~~~~~~

Comprehensive analysis of training results:

.. code-block:: python

   from dendritic_modeling.analysis import TrainingAnalyzer
   
   # Analyze training results
   analyzer = TrainingAnalyzer(results)
   
   # Generate training curves
   analyzer.plot_training_curves()
   
   # Analyze convergence
   convergence_analysis = analyzer.analyze_convergence()
   
   # Compare different runs
   comparison = analyzer.compare_runs([results1, results2, results3])
   
   # Generate report
   report = analyzer.generate_report()

Debugging and Troubleshooting
-----------------------------

Training Diagnostics
~~~~~~~~~~~~~~~~~~~

Tools for diagnosing training issues:

.. code-block:: python

   from dendritic_modeling.training import TrainingDiagnostics
   
   # Initialize diagnostics
   diagnostics = TrainingDiagnostics(model)
   
   # Check gradient flow
   grad_analysis = diagnostics.analyze_gradients(train_loader)
   
   # Monitor activations
   activation_stats = diagnostics.monitor_activations(train_loader)
   
   # Check for common issues
   issues = diagnostics.check_common_issues()
   
   for issue in issues:
       print(f"Issue detected: {issue['type']}")
       print(f"Description: {issue['description']}")
       print(f"Suggested fix: {issue['fix']}")

Performance Profiling
~~~~~~~~~~~~~~~~~~~~

Optimize training performance:

.. code-block:: python

   from dendritic_modeling.training import PerformanceProfiler
   
   # Profile training step
   profiler = PerformanceProfiler()
   
   with profiler.profile():
       for batch in train_loader:
           loss = train_step(model, batch)
   
   # Analyze results
   profile_results = profiler.analyze()
   
   # Identify bottlenecks
   bottlenecks = profile_results['bottlenecks']
   
   # Generate optimization suggestions
   suggestions = profiler.get_optimization_suggestions()

Common Training Issues
~~~~~~~~~~~~~~~~~~~~

**Gradient Explosion/Vanishing**

.. code-block:: python

   # Gradient clipping
   train_config = TrainingConfig(
       learning_strategy='mle',
       gradient_clipping=1.0,
       gradient_clip_algorithm='norm'
   )
   
   # Gradient normalization
   train_config.gradient_normalization = True

**Slow Convergence**

.. code-block:: python

   # Learning rate scheduling
   train_config = TrainingConfig(
       learning_strategy='mle',
       scheduler=SchedulerConfig(
           type='reduce_on_plateau',
           factor=0.5,
           patience=5,
           min_lr=1e-6
       )
   )
   
   # Warmup
   train_config.warmup_epochs = 5
   train_config.warmup_factor = 0.1

**Memory Issues**

.. code-block:: python

   # Memory optimization
   train_config = TrainingConfig(
       learning_strategy='mle',
       batch_size=32,  # Reduce batch size
       accumulate_grad_batches=4,  # Maintain effective batch size
       gradient_checkpointing=True,  # Trade compute for memory
       mixed_precision=True  # Reduce memory usage
   )

Best Practices
-------------

Configuration Management
~~~~~~~~~~~~~~~~~~~~~~~

1. **Version Control**: Store configurations in version control
2. **Environment Variables**: Use environment variables for paths and secrets
3. **Configuration Templates**: Create reusable configuration templates
4. **Validation**: Always validate configurations before training
5. **Documentation**: Document configuration parameters and their effects

Training Workflow
~~~~~~~~~~~~~~~~

1. **Data Validation**: Verify data quality and preprocessing
2. **Model Validation**: Test model architecture on small datasets
3. **Hyperparameter Tuning**: Systematic hyperparameter optimization
4. **Monitoring**: Comprehensive logging and monitoring
5. **Reproducibility**: Ensure experiments are reproducible

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~

1. **Profiling**: Regular performance profiling
2. **Batch Size**: Optimize batch size for hardware
3. **Mixed Precision**: Use mixed precision when possible
4. **Gradient Accumulation**: Handle memory constraints
5. **Distributed Training**: Scale to multiple GPUs when needed

Next Steps
---------

Now that you understand training and configuration:

1. **Analysis Tools**: Learn about analysis and visualization in :doc:`analysis_and_visualization`
2. **API Reference**: See detailed API documentation in :doc:`../api/index`
3. **Examples**: Try practical training examples in :doc:`../examples/index`
4. **Advanced Topics**: Explore advanced techniques in :doc:`../advanced/index` 