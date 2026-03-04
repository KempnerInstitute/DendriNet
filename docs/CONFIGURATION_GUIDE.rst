Configuration Guide
===================

The primary configuration reference is ``configs/config_exp.yaml``.  That file
lists **every parameter** accepted by the framework together with its default
value and valid options.  This page summarises the structure and highlights
the most important settings; for the full parameter list, consult the YAML
file directly or the :doc:`config_reference` page.

Quick Start
-----------

.. code-block:: bash

   # Standard training
   python src/dendritic_modeling/scripts/training/train_experiments.py configs/config_exp.yaml

   # With overrides
   python ... configs/config_exp.yaml --output_dir results/my_run --run_name test1

Templates:

- ``configs/config_exp.yaml`` -- single-GPU training (complete reference)
- ``configs/config_exp_fsdp.yaml`` -- FSDP multi-GPU training
- ``configs/EI_sweeps_example.yaml`` -- parameter sweeps

Top-Level Structure
-------------------

.. code-block:: yaml

   experiment:   # seed, AMP, profiling, checkpointing
   data:         # dataset, processing, dataset-specific params
   model:        # encoder, core network, decoder
   training:     # encoder + main training, strategies, regularization, pruning
   analysis:     # 16 analysis modules
   wandb:        # Weights & Biases logging
   outputs:      # run_name, results_dir
   fsdp:         # distributed training

Core Network Types
------------------

Set ``model.core.type`` to one of:

+---------------------------+-----------------------------------------------+
| Type string               | Description                                   |
+===========================+===============================================+
| ``dendritic_shunting``    | Shunting (divisive) inhibition                |
+---------------------------+-----------------------------------------------+
| ``dendritic_additive``    | Additive inhibition                           |
+---------------------------+-----------------------------------------------+
| ``dendritic_mlp``         | Dendritic morphology, no E/I sign constraint  |
+---------------------------+-----------------------------------------------+
| ``point_mlp``             | Point-neuron MLP baseline (matched params)    |
+---------------------------+-----------------------------------------------+
| ``EINet``                 | Legacy; uses ``morphology.use_shunting``       |
+---------------------------+-----------------------------------------------+
| ``flat_shunting``         | Single-branch shunting variant                |
+---------------------------+-----------------------------------------------+
| ``flat_additive``         | Single-branch additive variant                |
+---------------------------+-----------------------------------------------+
| ``ei_unified``            | Recurrent E-I network                         |
+---------------------------+-----------------------------------------------+
| ``gru``, ``lstm``         | Baseline RNNs                                 |
+---------------------------+-----------------------------------------------+
| ``mlp``, ``sparse_mlp``   | Standard / sparse MLP                         |
+---------------------------+-----------------------------------------------+

See :doc:`network_architectures` for architecture details.

Training Strategies
-------------------

Set ``training.main.strategy`` to one of:

``standard``, ``fa``, ``dfa``, ``shunting_fa``, ``shunting_dfa``,
``local_ca``, ``freeze_layers``, ``freeze_branches``, ``freeze_branch_kl``,
``voltage_stabilization``, ``homeostatic_control``,
``multi_stage``, ``recurrent``, ``fsdp``, ``fsdp_standard``, ``enhanced``

Each strategy has its own sub-configuration under
``training.main.strategies.<name>``.  See :doc:`training_strategies` for details.

Loss Functions
--------------

Set ``training.main.common.loss_function`` (or ``training.encoder.common.loss_function``) to:

``mse``, ``log_mse``, ``ce``, ``bce``, ``cat_nll``,
``vae_elbo``, ``neg_cos_sim``, ``log_adj_neg_cos_sim``

Datasets
--------

Set ``data.dataset_name`` to one of:

- **Standard**: ``mnist``, ``fashion_mnist``, ``cifar10``, ``imagenet``,
  ``mnist_modulo10``, ``random_flip_mnist``
- **Synthetic**: ``correlated_gaussian``, ``nonlinear_interaction``,
  ``linear_separable``, ``nonlinear_separable``, ``block_correlated``,
  ``balanced_information``, ``poisson_generator``, ``multixor``,
  ``orientation_bars``, ``orthonet``
- **Sequence**: ``sequential_mnist``, ``copy_task``, ``adding_problem``,
  ``variable_delay_dms``, ``multi_frequency``, ``hierarchical_temporal``,
  ``lorenz``, ``stringer_v1``

Dataset-specific parameters go under ``data.dataset_params.<name>``.
See :doc:`datasets` for details.

Analysis Modules
----------------

Each module lives under ``analysis.<module_name>`` with ``enabled`` and
``training`` flags:

.. code-block:: yaml

   analysis:
     performance_analysis:
       enabled: true
       training: true    # run every epoch
     information_analysis:
       enabled: true
       training: false   # run after training only

Available modules: ``performance_analysis``, ``information_analysis``,
``correlation_analysis``, ``ablation_analysis``, ``synapse_turnover_analysis``,
``noise_perturbation_analysis``, ``weight_analysis``,
``single_layer_contribution``, ``synaptic_activation_analysis``,
``branch_activation_analysis``, ``compartment_statistics_analysis``,
``compartment_snr_analysis``, ``multiplicative_gain_analysis``,
``receptive_field_analysis``, ``adversarial_robustness_analysis``,
``routing_analysis``.

See :doc:`analysis` for per-module parameter details.

Recurrent Networks
------------------

For recurrent E-I networks use ``model.core.type: ei_unified`` with the
``unified_ei`` dict:

.. code-block:: yaml

   model:
     core:
       type: ei_unified
       unified_ei:
         output_mode: last          # last, all, mean
         layers:
           - recurrent: true
             dt: 1.0
             excitatory:
               n_neurons: 64
               branch_factors: [3, 3]
               ff_excitatory_synapses: 12
               ff_inhibitory_synapses: 6
               rec_excitatory_synapses: 8
               tau_base: 50.0
             inhibitory:
               n_neurons: 16
               branch_factors: [1]

   training:
     main:
       strategy: recurrent

For baseline RNNs use ``model.core.type: gru`` (or ``lstm``, ``vanilla_rnn``).
See :doc:`network_architectures` for full details.

Multi-Stage Training
--------------------

.. code-block:: yaml

   training:
     main:
       strategy: multi_stage
       strategies:
         multi_stage:
           analyze_between_stages: true
           stages:
             - learning_strategy: standard
               epochs: 50
               trainer_config:
                 loss_function: ce
             - learning_strategy: local_ca
               epochs: 50

See :doc:`training_strategies` for per-stage options.

FSDP / Distributed
-------------------

See :doc:`FSDP_TROUBLESHOOTING` for distributed training setup.

Full Parameter Reference
------------------------

The complete annotated configuration with every parameter and its valid
options is in ``configs/config_exp.yaml`` and reproduced in
:doc:`config_reference`.
