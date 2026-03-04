Available Components
====================

Quick reference of all implemented components and their configuration strings.

Core Network Types
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Type string
     - Description
   * - ``dendritic_shunting``
     - E-I network with shunting (divisive) inhibition
   * - ``dendritic_additive``
     - E-I network with additive (subtractive) inhibition
   * - ``flat_shunting``
     - Flattened dendrites (single level) with shunting
   * - ``flat_additive``
     - Flattened dendrites with additive inhibition
   * - ``dendritic_mlp``
     - E-I with signed weights (identity transform)
   * - ``einet``
     - Generic E-I; reads use_shunting from config
   * - ``ei_unified``
     - Unified E-I (feedforward + recurrent capable)
   * - ``gru``
     - Gated Recurrent Unit
   * - ``lstm``
     - Long Short-Term Memory
   * - ``vanilla_rnn``
     - Standard RNN
   * - ``mlp``
     - Multi-layer perceptron

Encoder / Decoder Types
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Type
     - Description
   * - ``Identity``
     - Pass-through (no transformation)
   * - ``MLP``
     - Multi-layer perceptron encoder/decoder
   * - ``CNN``
     - Convolutional encoder (encoder only)

TopK Variants
-------------

Set via ``model.core.sparsity.type``:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Type
     - Description
   * - ``standard``
     - Deterministic top-K mask
   * - ``stochastic``
     - Soft top-K with temperature
   * - ``variance``
     - Variance-aware selection
   * - ``deepst``
     - Dynamic rewiring with density targets

Weight Transforms
-----------------

Set via ``model.core.morphology.weight_transform``:

``exp`` -- always positive (default for shunting).
``softplus`` -- always positive, smoother gradient.
``relu`` -- non-negative, can be exactly zero.
``identity`` -- signed weights (for dendritic_mlp).

Activation Types
----------------

Set via ``model.core.reactivation.type``:

``param_tanh``, ``param_tanh_only_m``, ``param_linear_sigmoid``,
``param_linear_tanh``, ``param_linear_tanh_transition``, ``relu``,
``sigmoid``, ``tanh``, ``leaky_relu``, ``elu``, ``selu``, ``softplus``.

Training Strategies
-------------------

Set via ``training.main.strategy``:

``standard``, ``fa``, ``dfa``, ``shunting_fa``, ``shunting_dfa``,
``local_ca``, ``freeze_layers``, ``freeze_branches``, ``freeze_branch_kl``,
``voltage_stabilization``, ``homeostatic_control``, ``recurrent``,
``multi_stage``, ``fsdp``.

Loss Functions
--------------

Set via ``training.main.common.loss_function``:

``ce``, ``mse``, ``log_mse``, ``bce``, ``cat_nll``, ``vae_elbo``,
``neg_cos_sim``, ``log_adj_neg_cos_sim``.

Datasets
--------

Set via ``data.dataset_name``:

**Standard**: ``mnist``, ``fashion_mnist``, ``cifar10``, ``mnist_modulo10``.

**Synthetic**: ``correlated_gaussian``, ``linear_separable``,
``nonlinear_separable``, ``nonlinear_interaction``.

**Sequence**: ``sequential_mnist``, ``copy_task``, ``adding_problem``,
``variable_delay_dms``, ``multi_frequency``, ``lorenz_sequence``,
``stringer_v1``.

Analysis Modules
----------------

Each enabled under ``analysis.<module_name>.enabled``:

``performance_analysis``, ``information_analysis``, ``ablation_analysis``,
``correlation_analysis``, ``weight_analysis``, ``synapse_turnover_analysis``,
``noise_perturbation_analysis``, ``compartment_statistics_analysis``,
``compartment_snr_analysis``, ``multiplicative_gain_analysis``,
``routing_analysis``, ``receptive_field_analysis``,
``adversarial_robustness_analysis``, ``single_layer_contribution_analysis``.

Initialization Methods
----------------------

Set via ``model.core.sparsity.init_method``:

``xavier_normal``, ``xavier_uniform``, ``kaiming_normal``,
``kaiming_uniform``, ``orthogonal``, ``normal``, ``uniform``, ``eye``.

BlockLinear Initialization
--------------------------

Set via ``model.core.morphology.dbl_init_method``:

``analytical_expectation``, ``ei_equivalence``, ``naive``, ``default``.
