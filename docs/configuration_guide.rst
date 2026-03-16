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
   python src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/config_exp.yaml --output_dir results/my_run --run_name test1

Templates:

- ``configs/config_exp.yaml`` -- single-GPU training (complete reference)
- ``configs/config_exp_fsdp.yaml`` -- FSDP multi-GPU training
- ``configs/sweep_config.yaml`` -- parameter sweeps

Top-Level Structure
-------------------

.. code-block:: yaml

   experiment:   # seed, profiling, checkpointing
   data:         # dataset, processing, dataset-specific params
   model:        # encoder, core network, decoder, pretrained_replacement
   training:     # encoder + main training, strategies, regularization, pruning, AMP
   analysis:     # 16 analysis modules
   wandb:        # Weights & Biases logging
   outputs:      # run_name, results_dir
   distributed:  # mode (none/ddp/fsdp), FSDP tuning knobs

Core Network Types
------------------

Set ``model.core.type`` to one of:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Type string
     - Description
   * - ``dendritic_shunting``
     - Shunting (divisive) inhibition
   * - ``dendritic_additive``
     - Additive (subtractive) inhibition
   * - ``dendritic_mlp``
     - Dendritic morphology, no E/I sign constraint
   * - ``point_mlp``
     - Point-neuron MLP baseline (matched params)
   * - ``einet``
     - Legacy generic E-I type; uses ``morphology.use_shunting``
   * - ``flat_shunting``
     - Single-branch shunting variant
   * - ``flat_additive``
     - Single-branch additive variant
   * - ``ei_unified``
     - Unified E-I, supports feedforward and recurrent
   * - ``gru``, ``lstm``
     - Baseline RNNs
   * - ``mlp``
     - MLP baseline

See :doc:`network_architectures` for architecture details.

Training Strategies
-------------------

Set ``training.main.strategy`` to one of:

``standard``, ``fa``, ``dfa``, ``shunting_fa``, ``shunting_dfa``,
``local_ca``, ``freeze_layers``, ``freeze_branches``, ``freeze_branch_kl``,
``voltage_stabilization``, ``homeostatic_control``,
``multi_stage``, ``recurrent``

Strategy-specific configuration usually lives under
``training.main.strategies.<name>``. Local credit assignment uses the direct
path ``training.main.learning_strategy_config``. See
:doc:`training_strategies` for details.

Loss Functions
--------------

Set ``training.main.common.loss_function`` (or ``training.encoder.common.loss_function``) to:

``mse``, ``log_mse``, ``ce``, ``bce``, ``cat_nll``,
``vae_elbo``, ``neg_cos_sim``, ``log_adj_neg_cos_sim``

Datasets
--------

Set ``data.dataset_name`` to one of:

- **Standard**: ``mnist``, ``fashion_mnist``, ``cifar10``, ``imagenet``,
  ``mnist_modulo10``
- **Synthetic**: ``correlated_gaussian``, ``nonlinear_interaction``,
  ``linear_separable``, ``nonlinear_separable``, ``block_correlated``,
  ``balanced_information``, ``poisson_generator``, ``multixor``,
  ``orientation_bars``, ``orthonet``
- **Sequence**: ``sequential_mnist``, ``copy_task``, ``adding_problem``,
  ``variable_delay_dms``, ``multi_frequency``, ``hierarchical_temporal``,
  ``context_dependent_decision``, ``switching_context_decision``,
  ``timescale_generalization``, ``lorenz``, ``stringer_v1``

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

Pretrained Backbone Replacement
--------------------------------

Replace one or more layers of a pretrained torchvision backbone (e.g. AlexNet,
VGG) with an EINet core.  The backbone is loaded once and split into three
segments:

- **prefix** (encoder): all layers before the replacement range
- **core**: the dendritic EI network that replaces the removed layers
- **suffix** (decoder): all layers after the replacement range

When ``model.pretrained_replacement.enabled`` is ``true``, the legacy
``model.encoder`` and ``model.decoder`` sections are ignored.

.. code-block:: yaml

   model:
     pretrained_replacement:
       enabled: true
       backbone: "alexnet"                    # torchvision model name
       weights: "IMAGENET1K_V1"               # pretrained weights (null = random init)
       input_shape: [3, 224, 224]

       # Layers to replace (inclusive range).
       # Use list_backbone_layers("alexnet") to see all available names.
       replace:
         start: "classifier.4"                # fc7 Linear
         end: "classifier.5"                  # fc7 ReLU

       # Drop layers entirely (e.g. Dropout that is not useful with EI nets).
       omit_layers: ["classifier.0", "classifier.3"]

       # Freeze/train policy for encoder and decoder segments.
       trainability:
         encoder:
           mode: "frozen"                     # freeze conv backbone + fc6
         decoder:
           mode: "trainable"                  # train final classifier

     core:
       type: "einet"
       architecture:
         excitatory_layer_sizes: [4096]       # must match suffix input dim
         # ... rest of EINet config ...

Learning rates for the prefix and suffix are controlled in the training config::

   training.main.common.param_groups.encoder_lr    # prefix LR (0.0 = frozen)
   training.main.common.param_groups.decoder_lr    # suffix LR

To discover valid layer names for a backbone, use the ``list_backbone_layers``
utility:

.. code-block:: python

   from dendritic_modeling.networks.architectures.classical.pretrained import (
       list_backbone_layers,
   )
   print(list_backbone_layers("alexnet"))
   # ['features.0', 'features.1', ..., 'avgpool', 'classifier.0', ..., 'classifier.6']

**Scope**: The current splitter linearizes ``nn.Sequential`` containers.  This
works for AlexNet, VGG, and coarse splits of ResNet (e.g. at ``avgpool`` or
``fc``).  Replacing layers *inside* residual blocks requires an FX-graph
backend (planned).

Example configs:

- ``configs/vision/alexnet_k1_einet.yaml`` — replace fc7 only (K=1)
- ``configs/vision/alexnet_k2_einet.yaml`` — replace fc6 + fc7 (K=2)
- ``configs/vision/alexnet_k1_einet_smoke.yaml`` — 1-epoch smoke test

Recurrent Networks
------------------

Any E-I type (``dendritic_shunting``, ``dendritic_additive``, ``einet``)
becomes recurrent when ``connectivity.rec_*`` synapse counts are nonzero.
Recurrence is controlled solely by these counts; no separate ``enabled``
flag is needed:

.. code-block:: yaml

   model:
     core:
       type: dendritic_shunting       # same type as feedforward
       architecture:
         excitatory_layer_sizes: [64]
         excitatory_branch_factors: [3, 3]
       connectivity:
         ee_synapses_per_branch_per_layer: [12]
         ie_synapses_per_branch_per_layer: [6]
         # Recurrent synapse counts (nonzero = recurrent)
         rec_ee_synapses_per_branch: [8]   # E->E recurrent
         rec_ie_synapses_per_branch: [4]   # I->E recurrent
         rec_ei_synapses_per_branch: [8]   # E->I recurrent
         rec_ii_synapses_per_branch: [4]   # I->I recurrent
       transfer:
         input_mode: 1
         inhibitory_mode: "first"     # none, first, all
       # morphology, sparsity, reactivation as usual...

       recurrent_ei:                  # temporal params only
         output_mode: last            # last, all, mean
         use_transfer: true           # default true (set false to bypass TransferLayer)
         dt: 1.0
         tau_base: 50.0               # scalar or per-layer list
         tau_ratio: 3.0               # scalar or per-layer list

   training:
     main:
       strategy: recurrent

For full per-layer, per-population control use ``core.type: ei_unified``
with the ``unified_ei`` config instead. For baseline RNNs use
``model.core.type: gru`` (or ``lstm``, ``vanilla_rnn``).
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

Distributed Training
--------------------

Set ``distributed.mode`` to choose the runtime:

.. code-block:: yaml

   distributed:
     mode: ddp            # none | ddp | fsdp
     fsdp:                # FSDP-specific knobs (only used when mode=fsdp)
       sharding_strategy: "FULL_SHARD"
       mixed_precision: true
       cpu_offload: false

- ``none`` — single-GPU, plain ``python`` (default)
- ``ddp`` — DistributedDataParallel via ``torchrun``.  Best when the model
  fits on one GPU and you want data-parallel speedup.  Supports multi-node.
- ``fsdp`` — FullyShardedDataParallel via ``torchrun``.  Use when the model
  is too large for a single GPU.  Supports multi-node.

Node and GPU counts are controlled by the SLURM allocation (``sbatch --nodes``,
``--gpus-per-node``) or by the launcher CLI argument, **not** by the YAML
config.  The training config only selects the *mode*; the scheduler decides
the resources.

FSDP-specific knobs (sharding strategy, mixed precision, CPU offload, etc.)
live under ``distributed.fsdp``.

Launch scripts
~~~~~~~~~~~~~~

The unified launcher reads ``distributed.mode`` from the YAML:

.. code-block:: bash

   # Auto-selects entrypoint based on distributed.mode
   sbatch scripts/train.sh configs/my_config.yaml

   # Multi-node (override nodes at submit time)
   sbatch --nodes=2 scripts/train.sh configs/my_config.yaml

   # Override GPUs per node
   sbatch scripts/train.sh configs/my_config.yaml 2

Dedicated launchers are also available:

.. code-block:: bash

   sbatch scripts/train_ddp.sh configs/my_config.yaml      # DDP
   sbatch scripts/train_fsdp.sh configs/my_config.yaml     # FSDP

See :doc:`fsdp_troubleshooting` for FSDP-specific troubleshooting.

Full Parameter Reference
------------------------

The complete annotated configuration with every parameter and its valid
options is in ``configs/config_exp.yaml`` and reproduced in
:doc:`config_reference`.
