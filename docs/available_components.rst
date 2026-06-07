Available Components
====================

Quick reference of all implemented components and their configuration strings.

Extension Registries
--------------------

Most built-in components below are still selected by YAML strings. For
project-local extensions, use the registry APIs so new components can be added
without editing central dispatch chains:

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Family
     - Registry API
     - Selection path
   * - Architecture / core
     - ``dendritic_modeling.networks.architectures.register_architecture``
     - ``model.core.type`` or direct ``get_architecture(...)`` calls
   * - Optimizer
     - ``dendritic_modeling.training.optimizers.register_optimizer``
     - ``training.main.optimizer.name``
   * - Loss
     - ``dendritic_modeling.training.loss.register_loss_function``
     - trainer ``loss_function`` / configured loss name
   * - Dataset
     - ``dendritic_modeling.datasets.register_dataset``
     - ``data.task.dataset``
   * - Analyzer metadata
     - ``dendritic_modeling.analysis.register_analyzer``
     - metadata/discovery only; execution remains in ``AnalysisManager``

Built-in architecture names are reserved and cannot be shadowed by extension
registrations. This keeps existing experiments reproducible while allowing new
components to be introduced cleanly.

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
   * - ``population_graph``
     - Named recurrent dendritic populations for cell-type experiments
   * - ``cell_type_graph``, ``cell_types``
     - Aliases for ``population_graph``
   * - ``multi_population_ei``
     - Multi-population feedforward/recurrent dendritic layers with named
       populations, per-population morphology, and validated directional
       same-step connections
   * - ``rnn_dendritic_shunting``, ``rnn_dendritic_additive``
     - Unified recurrent-EI aliases with type-implied shunting/additive defaults
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
   * - ``indexed``
     - Fixed-index sparse synapses that store only ``out_features x K``
       weights plus integer presynaptic indices
   * - ``indexed_dynamic``
     - Candidate-pool sparse TopK; stores ``out_features x C`` candidate
       weights and dynamically selects ``K`` active candidates per forward pass

``topk`` is retained as a backward-compatible alias for ``standard``.

Weight Transforms
-----------------

Set via ``model.core.morphology.weight_transform``:

``softplus`` -- always positive, smoother gradient (default structured EI / recurrent setting).
``exp`` -- always positive, sharp exponential parameterization.
``relu`` -- non-negative, can be exactly zero.
``identity`` -- signed weights (for dendritic_mlp).

For biologically constrained recurrent runs, pair a positive transform
(``exp`` or ``softplus``) with ``allow_self_recurrence: false`` in the
population config to remove same-population autaptic feedback.

Recurrent Extensions
--------------------

.. list-table::
   :header-rows: 1
   :widths: 24 32 44

   * - Component
     - Config keys
     - Notes
   * - Named cell types
     - ``model.core.population_graph.cell_types``
     - Each entry defines ``name``, ``polarity``, ``n_neurons``, and
       ``branch_factors``.
   * - Source-target cell-type masks
     - ``model.core.population_graph.connections``
     - Optional ``source``, ``target``, ``enabled``, ``probability`` /
       ``p_connect``, and ``seed``.
   * - Structured TopK masks
     - ``model.core.connectivity.structured``
     - Supports ``fixed_indegree``, ``bernoulli``, and ``distance`` pathway
       masks for feedforward and recurrent E/I pathways.
   * - Synapse types
     - ``synapse_types.enabled``, ``excitatory``, ``inhibitory``
     - Each type can define ``name``, ``fraction``, ``tau`` or
       ``tau_rise``/``tau_decay``, ``magnesium_block``, ``stp``, and
       ``reversal_potential``.
   * - Spiking soma
     - ``dynamics_mode: spike`` or ``model.core.dynamics.mode: spike``
     - LIF soma with ``spike_readout`` set to ``spikes``, ``rate``, or
       ``membrane``.
   * - Dendritic spike / plateau events
     - ``dendritic_spikes_enabled`` or
       ``model.core.dynamics.dendritic_spikes.enabled``
     - Optional local branch threshold/plateau events at distal, proximal, or
       all non-soma dendritic levels; propagation can be ``additive``,
       ``replace``, or ``gated``.
   * - Soma-to-dendrite feedback
     - ``soma_feedback_enabled`` or ``model.core.soma_feedback.enabled``
     - Optional previous-timestep soma output/voltage/readout projected back
       into dendritic levels as additive, shunting, or gated feedback.
   * - Transformer FFN replacement
     - ``model.transformer_replacement`` plus ``model.core.type: dendritic_ffn``
     - Hugging Face model patcher for replacing decoder-block MLP/FFN modules
       with token-wise ``DendriticFFNReplacement`` modules; use
       ``sparsity.type: indexed_dynamic`` for low-parameter dynamic TopK.
   * - Transformer replacement distillation
     - ``training.transformer_replacement``
     - Layerwise block-output distillation for dendritic FFN replacements using
       ``synthetic_mlp``, ``hidden_cache``, or ``hf_text`` teacher sources.
   * - Recurrent analysis introspection
     - ``analysis.utils.recurrent_introspection`` and
       ``analysis.spike_train``
     - Shared helper for legacy E/I and named-population analysis tools, plus
       LIF raster/rate/ISI/Fano summaries.
   * - Recurrent protocols
     - ``networks.RecurrentCore``, ``networks.SteppableRecurrentCore``
     - Small structural interfaces for trainer-facing sequence cores and
       one-step recurrent dendritic populations.

Multi-Population Layers
-----------------------

Set ``model.core.type: multi_population_ei`` to define layers with more than
one named cell population. Each population owns its own ``PopulationConfig``
payload, so morphology and dynamics can differ by population.

The current two-population E/I feedforward layer is the simple special case:

.. code-block:: yaml

   model:
     core:
       type: multi_population_ei
       multi_population:
         layers:
           - name: layer0
             readout_population: e
             population_defaults:
               ff_excitatory_synapses: 4
               ff_inhibitory_synapses: 2
               rec_excitatory_synapses: 0
               rec_inhibitory_synapses: 0
               reactivate: true
             populations:
               - name: i
                 polarity: inhibitory
                 n_neurons: 16
                 branch_factors: [1]
               - name: e
                 polarity: excitatory
                 n_neurons: 64
                 branch_factors: [3, 2]
             connections:
               - source: input
                 target: i
               - source: input
                 target: e
               - source: i
                 target: e

For legacy-style independent transfer streams, enable the transfer layer and
use explicit external source names. ``source: input`` remains the compatibility
stream; ``source: input_e`` and ``source: input_i`` refer to the excitatory and
inhibitory tensors returned by ``TransferLayer`` on the first layer:

.. code-block:: yaml

   model:
     core:
       type: multi_population_ei
       multi_population:
         use_transfer: true
         transfer_params:
           input_mode: 0
           independent_pathways: true
           excitatory_dim: 392
           inhibitory_dim: 392
         layers:
           - name: layer0
             readout_population: e
             population_defaults:
               ff_excitatory_synapses: 4
               ff_inhibitory_synapses: 2
               rec_excitatory_synapses: 0
               rec_inhibitory_synapses: 0
             populations:
               - name: i
                 polarity: inhibitory
                 n_neurons: 16
                 branch_factors: [1]
               - name: e
                 polarity: excitatory
                 n_neurons: 64
                 branch_factors: [3, 2]
             connections:
               - source: input_e
                 target: i
               - source: input_e
                 target: e
               - source: input_i
                 target: e
               - source: i
                 target: e

Same-step population-to-population connections must be directional and
acyclic. For example, ``i -> e`` is allowed, but ``e -> i`` plus ``i -> e`` in
the same layer and same timestep raises a config error. Bidirectional
population interactions are supported with ``timing: delayed`` on recurrent
layers, where they use previous-timestep outputs instead of creating an
algebraic loop.

For multiple layers, each layer defines its own population graph.
``source: input`` remains the compatibility shorthand for the previous layer's
readout stream. Prefer explicit qualified sources when a later layer should
receive specific earlier populations:

.. code-block:: yaml

   connections:
     - source: layer0.e0
       target: i1
     - source: layer0.e0
       target: e1
     - source: layer0.i0
       target: e1
     - source: i1
       target: e1

Same-step qualified sources must come from earlier layers. Delayed qualified
sources are allowed on recurrent target layers and use previous-timestep
population outputs.

If ``input_projection_dims`` is combined with ``use_transfer: true``, the
projection applies to ``input``/``input_e``. ``input_i`` keeps the inhibitory
dimension produced by ``TransferLayer``.

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

**Standard**: ``mnist``, ``fashion_mnist``, ``cifar10``, ``imagenet``,
``mnist_modulo10``, ``cifar10_modulo10``,
``double_cifar10_contextual_mod10``, ``double_mnist_contextual_mod10``.

**Synthetic**: ``correlated_gaussian``, ``linear_separable``,
``nonlinear_separable``, ``nonlinear_interaction``, ``block_correlated``,
``balanced_information``, ``contextual_stream_gain_shift``,
``poisson_generator``, ``multixor``, ``orientation_bars``, ``orthonet``.

**Sequence**: ``sequential_mnist``, ``copy_task``, ``adding_problem``,
``variable_delay_dms``, ``multi_frequency``, ``hierarchical_temporal``,
``context_dependent_decision``, ``context_dependent_decision_hard``,
``context_dependent_decision_positive``,
``context_dependent_decision_hard_positive``,
``switching_context_decision``, ``switching_context_decision_positive``,
``timescale_generalization``, ``timescale_generalization_positive``,
``odr_generalization``, ``odr_generalization_positive``,
``romo_delay_comparison``, ``romo_delay_comparison_positive``,
``gain_modulated_contextual_comparison``,
``gain_modulated_contextual_comparison_positive``, ``switching_lds``,
``lorenz``, ``lorenz_sequence``, ``multi_sine_forecast``, ``stringer_v1``.

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
``single_layer_contribution``, ``synaptic_activation``, ``branch_activation``,
``reactivation_dynamics``, ``inhibitory_specialization``,
``jacobian_spectrum``, ``dendritic_timetraces``, ``local_rule_components``,
``spike_train``.

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
