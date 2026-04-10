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
   * - ``flat_mlp``
     - Flattened dendrites with signed weights (MLP-like dendritic baseline)
   * - ``einet``
     - Generic E-I; reads use_shunting from config
   * - ``ei_unified``
     - Unified E-I (feedforward + recurrent capable)
   * - ``point_mlp``
     - Plain point-neuron MLP baseline
   * - ``total_param_mlp``
     - MLP baseline matched by total parameter count
   * - ``active_param_mlp``
     - MLP baseline matched by active parameter count
   * - ``ss_mlp``
     - Sparse structured MLP baseline
   * - ``ss_mlp_flat``
     - Flattened sparse structured MLP baseline
   * - ``hierarchical_dendritic_tensor_map``
     - Pretrained-replacement baseline: keep the pretrained prefix/suffix around a newly initialized dendritic core, flatten the prefix tensor, run an EINet-style hierarchical dendritic core, then optionally project to a flat suffix input
   * - ``hierarchical_dendritic_conv``
     - Spatial dendritic convolutional core over local patches; preserves the prefix/suffix spatial tensor contract for convolutional replacement
   * - ``gru``
     - Gated Recurrent Unit
   * - ``lstm``
     - Long Short-Term Memory
   * - ``vanilla_rnn``
     - Standard RNN
   * - ``mlp``
     - Multi-layer perceptron
   * - ``alexnet``
     - AlexNet-style classical CNN backbone
   * - ``cnn``
     - Small classical CNN
   * - ``cnn_down``
     - Downsampling CNN decoder/encoder helper
   * - ``cnn_up``
     - Upsampling CNN decoder helper
   * - ``identity``
     - Identity core / pass-through

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

Set via ``model.core.sparsity.type``. Canonical values are:

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

``topk`` is retained as a backward-compatible alias for ``standard``.

Weight Transforms
-----------------

Set via ``model.core.morphology.weight_transform``:

``exp`` -- always positive, sharp exponential parameterization (default recurrent paper setting).
``softplus`` -- always positive, smoother gradient.
``relu`` -- non-negative, can be exactly zero.
``identity`` -- signed weights (for dendritic_mlp).

For biologically constrained recurrent runs, pair a positive transform
(``exp`` or ``softplus``) with ``allow_self_recurrence: false`` in the
population config to remove same-population autaptic feedback.

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
``multi_stage``.

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
``variable_delay_dms``, ``multi_frequency``, ``hierarchical_temporal``,
``context_dependent_decision``, ``switching_context_decision``,
``timescale_generalization``, ``switching_lds``, ``lorenz``,
``stringer_v1``.

Analysis Modules
----------------

Each enabled under ``analysis.<module_name>.enabled``:

Preferred short names in YAML are ``performance``, ``information``,
``ablation``, and ``correlation``. The longer ``*_analysis`` names are still
accepted for backward compatibility.

Available modules: ``performance``, ``information``, ``ablation``,
``correlation``, ``weight``, ``synapse_turnover``, ``noise_perturbation``,
``compartment_statistics``, ``compartment_snr``, ``multiplicative_gain``,
``routing``, ``receptive_field``, ``adversarial_robustness``,
``single_layer_contribution``, ``synaptic_activation``, ``branch_activation``.

Information Analysis Components
-------------------------------

Set via ``analysis.information.params.selection.components``.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Component
     - Implemented metrics and notes
   * - ``basic``
     - ``I(E; C)``, ``I(I; C)``, ``I(Vb; C)``, ``I(Vout; C)``,
       ``I(E, I; C)``, and ``I(Vinf; C)`` when ``include_vinf`` is enabled
   * - ``pairwise``
     - ``I(E; I)``, ``I(E; Vout)``, ``I(I; Vout)``, ``I(E; Vb)``,
       ``I(I; Vb)``, ``I(Vb; Vout)``, plus analogous ``Vinf`` pairwise terms
       when enabled
   * - ``conditional``
     - ``I(E; I | C)``, ``I(E; Vout | C)``, ``I(I; Vout | C)``,
       ``I(E; Vb | C)``, ``I(I; Vb | C)``, ``I(Vb; Vout | C)``, and
       ``Vinf`` variants when enabled
   * - ``ablation_aligned``
     - Adds ablation-mirroring CMI terms including ``I(Vout; C | E)``,
       ``I(Vout; C | I)``, ``I(Vout; C | Vb)``, ``I(Vout; C | E, I)``,
       ``I(E; C | I, Vb)``, ``I(I; C | E, Vb)``, ``I(E, I; C | Vb)``, and
       no-``Vb`` fallbacks such as ``I(E; C | I)`` and ``I(I; C | E)``
   * - ``upstream_unique_cmi``
     - Adds ``I(Vb; C | E, I)`` when ``Vb`` exists
   * - ``gaussian_fisher``
     - Fisher or Gaussian proxy metrics such as ``I(Vout; C)_fisher`` and
       ``I(Vb; C | E, I)_fisher``
   * - ``layer_proxies``
     - Layer-level derived summaries such as ``_sum``, ``_union``, and
       ``topK_sum`` built from single-branch metrics
   * - ``pid``
     - Partial information decomposition. Present in config, but currently
       reserved or disabled

For ``Vout`` specifically, the analyzer currently emits ``I(Vout; C)``,
``I(Vout; C | E)``, ``I(Vout; C | I)``, and ``I(Vout; C | E, I)``.

Initialization Methods
----------------------

Set via ``model.core.sparsity.init_method``:

``xavier_normal``, ``xavier_uniform``, ``kaiming_normal``,
``kaiming_uniform``, ``orthogonal``, ``normal``, ``uniform``, ``eye``.

BlockLinear Initialization
--------------------------

Set via ``model.core.morphology.dbl_init_method``:

``analytical_expectation``, ``ei_equivalence``, ``naive``, ``default``.
