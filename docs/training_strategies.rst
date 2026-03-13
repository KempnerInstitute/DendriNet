Training Strategies
===================

The framework uses a dual-trainer structure: an optional encoder trainer
(for the input encoder network) followed by the main trainer (for the core
network and decoder). The main trainer strategy is set via
``training.main.strategy`` in the YAML config.

.. contents:: On this page
   :local:
   :depth: 2

Available Strategies
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Strategy
     - Description
   * - ``standard``
     - Backpropagation with optional regularization and pruning
   * - ``fa``
     - Feedback Alignment with fixed random feedback matrices
   * - ``dfa``
     - Direct Feedback Alignment -- output error projected to each layer
   * - ``shunting_fa``
     - FA with conductance-based modulation
   * - ``shunting_dfa``
     - DFA with conductance-based modulation
   * - ``local_ca``
     - Local credit assignment (3/4/5-factor Hebbian rules)
   * - ``freeze_layers``
     - Progressive layer unfreezing
   * - ``freeze_branches``
     - Progressive branch unfreezing
   * - ``freeze_branch_kl``
     - Branch freezing with KL divergence loss
   * - ``voltage_stabilization``
     - Pretrain to stabilize dendritic voltages
   * - ``homeostatic_control``
     - Homeostatic regulation of soma activity
   * - ``recurrent``
     - Sequence training for RNN architectures
   * - ``multi_stage``
     - Chain multiple strategies in sequence
   * - ``fsdp``
     - Fully Sharded Data Parallel (multi-GPU/node)

Standard Training
-----------------

Basic backpropagation. Supports gradient clipping, early stopping, mixed
precision, NaN detection, and adaptive clipping.

.. code-block:: yaml

   training:
     main:
       strategy: standard
       common:
         epochs: 100
         batch_size: 256
         loss_function: ce
         grad_clip_value: 5.0
         early_stopping: true
         patience: 20

Key ``common`` parameters:

``epochs`` (int, default: 1000), ``batch_size`` (int, default: 512),
``loss_function`` (str, default: ``"ce"``), ``grad_clip_value`` (float,
default: 2.0), ``early_stopping`` (bool, default: True), ``patience``
(int, default: 500), ``use_amp`` (bool, default: False),
``enable_nan_checking`` (bool, default: False),
``enable_adaptive_clipping`` (bool, default: False).

Feedback Alignment
------------------

Implements FA (Lillicrap et al., 2016) and DFA (Nokland, 2016). Uses fixed
random feedback matrices instead of weight transposes for error propagation.

- **FA** (``fa``): errors backpropagate layer-by-layer through random B
  matrices.
- **DFA** (``dfa``): each hidden layer receives the output error directly.

``shunting_fa`` and ``shunting_dfa`` extend these with conductance-based
modulation: the error at each layer is modulated by the local total
conductance and driving force, ``error * g_tot * (E_rev - V)``.

.. code-block:: yaml

   training:
     main:
       strategy: dfa

Local Credit Assignment
-----------------------

Biologically-plausible local learning using Hebbian-style updates with
locally available signals (pre/post-synaptic activity) and a broadcast soma
error.

Rule Variants
~~~~~~~~~~~~~

``rule_variant: "3f"`` -- three-factor rule
   ``pre * post * error``. The post-synaptic factor depends on the dynamics
   mode: in conductance mode it uses driving force ``(E_rev - V)`` scaled by
   ``1/g_tot``; in additive mode it uses ``(V - theta)`` with optional gain
   control.

``rule_variant: "4f"`` -- four-factor rule
   Adds a morphology correlation factor ``rho`` that captures the correlation
   between branch-level and soma-level activity. Estimated via EMA
   (``rho_estimator: "ema"``) or data augmentation (``"augment"``).

``rule_variant: "5f"`` -- five-factor rule
   Adds an information proxy ``phi`` that estimates class-conditional
   information content. Modes: ``"conditional"`` (conditional variance proxy)
   or ``"variance"`` (raw variance).

Error Broadcast
~~~~~~~~~~~~~~~

``error_broadcast_mode``
   ``"scalar"`` -- single scalar error for the whole network.
   ``"per_soma"`` -- per-neuron error at the soma level.
   ``"local_mismatch"`` -- locally computed mismatch signal.

``broadcast_bandwidth``
   ``"full"`` -- transmit full error values.
   ``"sign_only"`` -- transmit only the sign of the error.
   ``"quantized"`` -- quantize to ``broadcast_bits`` bits.
   ``"sparse_topk"`` -- keep only top ``broadcast_topk_fraction`` of values.

HSIC Auxiliary Loss
~~~~~~~~~~~~~~~~~~~

Optional Hilbert-Schmidt Independence Criterion loss to encourage decorrelated
representations. Enabled via ``hsic.enabled: true``. Kernel options:
``"linear"``, ``"rbf"``, ``"polynomial"``.

.. code-block:: yaml

   training:
     main:
       strategy: local_ca
       learning_strategy_config:
         rule_variant: "4f"
         error_broadcast_mode: per_soma
         three_factor:
           dynamics_mode: auto
           use_conductance_scaling: true
           use_driving_force: true
         four_factor:
           rho_mode: pearson
           rho_estimator: ema
           ema_alpha: 0.05
         hsic:
           enabled: true
           weight: 0.01

Morphology-Aware Extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extensions that use dendritic tree topology explicitly:

``use_path_propagation``
   Approximate path-integrated attenuation along dendritic branches.

``use_dendritic_normalization``
   Normalize update by total conductance at each branch.

``use_branch_type_rules``
   Differentiate apical vs basal branches with separate scaling
   (``apical_branch_scale``, ``basal_branch_scale``).

Recurrent Training
------------------

For sequence data (``[B, T, D]`` inputs). Handles temporal loss computation
for both ``output_mode: last`` and ``output_mode: all``.

``grad_clip`` (float, default: 1.0), ``lr_schedule`` (``"none"`` or
``"cosine"``), ``lr_warmup_epochs`` (int, default: 0).

.. code-block:: yaml

   training:
     main:
       strategy: recurrent
       common:
         epochs: 100
         batch_size: 64
         loss_function: ce

Multi-Stage Training
--------------------

Chains multiple strategies in sequence. Each stage can have its own strategy,
epoch count, optimizer state, and config overrides.

.. code-block:: yaml

   training:
     main:
       strategy: multi_stage
       multi_stage_config:
         analyze_between_stages: true
         use_best_from_each_stage: true
         stages:
           - learning_strategy: standard
             epochs: 50
             trainer_config:
               loss_function: ce
               early_stopping: true
           - learning_strategy: local_ca
             epochs: 50
             local_rule_config:
               rule_variant: "4f"

FSDP Distributed Training
--------------------------

For large models across multiple GPUs or nodes. See
:doc:`FSDP_TROUBLESHOOTING` for detailed guidance.

.. code-block:: yaml

   fsdp:
     enabled: true
     sharding_strategy: "FULL_SHARD"
     mixed_precision: true
     cpu_offload: false
     gradient_checkpointing: false

Parameter Groups
----------------

Per-component learning rates, set via ``training.main.common.param_groups``:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``lr``
     - 0.01
     - Base learning rate
   * - ``split_params``
     - True
     - Use separate LRs per component type
   * - ``topk_lr``
     - 0.001
     - Sparse synaptic layer LR
   * - ``blocklinear_lr``
     - 0.0001
     - BlockLinear (branch aggregation) LR
   * - ``reactivation_lr``
     - 0.0001
     - Reactivation nonlinearity LR
   * - ``decoder_lr``
     - 0.001
     - Decoder network LR

Optimizer
---------

Set via ``training.main.optimizer``: ``"adam"`` (default), ``"adamw"``,
``"sgd"``, ``"rmsprop"``. Standard parameters: ``lr``, ``weight_decay``,
``momentum`` (SGD), ``betas`` (Adam).

Loss Functions
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Name
     - Alias
     - Description
   * - ``ce``
     - ``cross_entropy``
     - Cross-entropy (classification)
   * - ``mse``
     - ``mean_squared_error``
     - Mean squared error (regression)
   * - ``log_mse``
     - ``log_mean_squared_error``
     - Log of MSE
   * - ``bce``
     - ``binary_cross_entropy``
     - Binary cross-entropy
   * - ``cat_nll``
     - ``categorical_negative_log_likelihood``
     - Categorical NLL
   * - ``vae_elbo``
     - ``variational_autoencoder_elbo``
     - VAE ELBO loss
   * - ``neg_cos_sim``
     - ``negative_cosine_similarity``
     - Negative cosine similarity
   * - ``log_adj_neg_cos_sim``
     -
     - Log-adjusted negative cosine similarity

Regularization and Pruning
--------------------------

Regularization (``training.main.regularization``):

- ``l1_weight``, ``l2_weight``: L1/L2 penalty weights
- ``selective_layers``, ``selective_branches``, ``selective_weight_types``:
  restrict to specific components
- ``ei_ratio_weight``: enforce E/I weight ratio constraint

Pruning (``training.main.pruning``):

- ``enabled``: bool
- ``threshold``: magnitude threshold for pruning
- ``frequency``: prune every N epochs
- ``selective_layers``, ``selective_branches``: restrict scope
