Network Architectures
=====================

The framework implements a three-part model: **encoder** (input processing),
**core network** (computation), and **decoder** (output mapping). This page
documents all core network types and their configuration.

.. contents:: On this page
   :local:
   :depth: 2

Architecture Pipeline
---------------------

Every model follows the encoder-core-decoder pattern:

1. **Encoder**: processes raw input (Identity, MLP, or CNN)
2. **Core network**: main computational unit (documented below)
3. **Decoder**: maps to output space (Identity or MLP)

The core network type is set via ``model.core.type`` in the YAML config file.

Core Network Types
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Type string
     - Description
     - Aliases
   * - ``dendritic_shunting``
     - E-I network with shunting (divisive) inhibition
     -
   * - ``dendritic_additive``
     - E-I network with additive (subtractive) inhibition
     -
   * - ``flat_shunting``
     - Flattened dendrites with shunting inhibition
     -
   * - ``flat_additive``
     - Flattened dendrites with additive inhibition
     -
   * - ``einet``
     - Generic E-I; reads ``use_shunting`` from morphology config
     -
   * - ``dendritic_mlp``
     - E-I with signed weights, ``synapse_mode="mlp"``
     -
   * - ``ei_unified``
     - Unified E-I, supports feedforward and recurrent
     - ``unified_ei``, ``ei_net``, ``unified_einet``, ``rnn_dendritic_shunting``, ``rnn_dendritic_additive``, ``rnn_flat_shunting``, ``rnn_flat_additive``
   * - ``gru``
     - Gated Recurrent Unit
     -
   * - ``lstm``
     - Long Short-Term Memory
     -
   * - ``vanilla_rnn``
     - Standard RNN
     - ``rnn``
   * - ``mlp``
     - Multi-layer perceptron baseline
     - ``point_mlp``

ExcitationInhibitionNetwork
---------------------------

The primary architecture. Each network consists of a TransferLayer followed by
one or more ExcitationInhibitionLayer instances. Each layer pairs an excitatory
DendriNet population with an inhibitory population (DendriNet or MLP).

**Forward pass**: input passes through the TransferLayer, producing separate
excitatory and inhibitory signals. At each layer, the inhibitory population
processes the excitatory input first; its output then modulates the excitatory
population via shunting or additive inhibition. The final layer's excitatory
output is the network output.

TransferLayer
~~~~~~~~~~~~~

Controls how input is routed to the E and I pathways.

``input_mode``
   0 -- both pathways receive the full input.
   1 -- E gets full input; I gets an independent pathway (if ``independent_pathways=True``).

``independent_pathways``
   If ``True``, input dimensions are split between E and I.
   If ``False``, input is duplicated to both.

``excitatory_dim``, ``inhibitory_dim``
   Explicit dimension split. Defaults to ``input_dim / 2`` each.

``output_activation``
   Optional nonlinearity on the split outputs: ``"sigmoid"``, ``"relu"``,
   ``"tanh"``, ``"softplus"``, or ``None``.

   For shunting E-I models with positive weight transforms, this is also the
   recommended place to clamp normalized raw inputs back to a nonnegative
   range, e.g. ``output_activation: "relu"``.

Config path: ``model.core.transfer``.

.. code-block:: yaml

   model:
     core:
       transfer:
         input_mode: 1
         independent_pathways: false
         output_activation: null

E-I Network Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   model:
     core:
       type: dendritic_shunting
       architecture:
         excitatory_layer_sizes: [100, 50]
         inhibitory_layer_sizes: [20, 10]
         excitatory_branch_factors: [3, 3, 3, 3]
         inhibitory_branch_factors: [1]
         inhibitory_network_type: "dendritic"
       connectivity:
         ee_synapses_per_branch_per_layer: [40]
         ei_synapses_per_branch_per_layer: [0]
         ie_synapses_per_branch_per_layer: [10]
         ii_synapses_per_branch_per_layer: [0]
       morphology:
         somatic_synapses: false
         use_shunting: true
         weight_transform: "softplus"
         dbl_init_method: "analytical_expectation"
       sparsity:
         init_method: "xavier_normal"
         type: "standard"
         gamma: 1.0
       reactivation:
         enabled: true
         type: "param_tanh"
         init_m: 1.5
         init_b: 0.5

DendriNet
---------

A single-population dendritic network with hierarchical branching. Used as the
building block within each E-I layer.

``branch_factors`` defines the dendritic tree. For example, ``[3, 2]`` creates
two levels of branching: 3 branches at the first level, each splitting into 2,
giving 6 distal branches per soma neuron.

The architecture is built from distal to soma:

- **Level 0** (most distal): receives synaptic input from external sources
- **Intermediate levels**: aggregate child branches via BlockLinear
- **Soma level**: final aggregation, produces the layer output

If ``somatic_synapses`` is ``True``, the soma level also receives direct
synaptic input in addition to branch aggregation.

Key parameters:

``n_soma``
   Number of output neurons.

``branch_factors``
   List of integers defining the branching tree structure.

``excitatory_input_dim``, ``excitatory_synapses_per_branch``
   Excitatory input specification per branch.

``inhibitory_input_dim``, ``inhibitory_synapses_per_branch``
   Inhibitory input specification per branch.

``somatic_synapses``
   Whether the soma level receives direct synaptic input.

``use_shunting``
   ``True`` for shunting (divisive) inhibition; ``False`` for additive.

DendriticBranchLayer
--------------------

The core computation unit. Each branch layer contains synaptic input layers,
optional branch aggregation, and a post-voltage nonlinearity.

Shunting Mode
~~~~~~~~~~~~~

When ``use_shunting=True``, the voltage is computed as a normalized ratio:

.. code-block:: text

   numerator   = E + E_rec + V_branch
   denominator = 1 + E + E_rec + g_branch + I + I_rec

   V = numerator / (denominator + epsilon)
   output = reactivation(V)

Where ``E`` is the excitatory synaptic current (TopKLinear), ``I`` is
inhibitory, ``V_branch`` is aggregated child-branch input (BlockLinear),
``g_branch`` is the branch conductance, and ``E_rec``, ``I_rec`` are recurrent
inputs (used only in recurrent mode). The current implementation uses these raw
terms directly in the denominator. Epsilon defaults to ``1e-8``.

Additive Mode
~~~~~~~~~~~~~

When ``use_shunting=False``:

.. code-block:: text

   V = E + E_rec + V_branch - I - I_rec
   output = reactivation(V)

Optional additive normalization: ``V = (V - mean) / (std + epsilon)``.

Components
~~~~~~~~~~

``branch_excitation``
   TopKLinear -- excitatory synaptic input.

``branch_inhibition``
   TopKLinear -- inhibitory synaptic input (optional).

``branch_recurrent``
   TopKLinear -- recurrent excitatory input (recurrent mode only).

``branch_rec_inhibition``
   TopKLinear -- recurrent inhibitory input (recurrent mode only).

``branches_to_output``
   BlockLinear or EfficientBlockLinear -- aggregates child branches.

``reactivation``
   Parametric nonlinearity applied to the voltage output.

TopKLinear
----------

Sparse synaptic layer. For each output neuron, only the ``K`` largest-weight
connections are active; all others are masked to zero.

``K``
   Number of active synapses per output neuron.

``weight_transform``
   Transformation applied to raw parameters before masking.
   ``"exp"`` (default) -- always positive.
   ``"softplus"`` -- always positive, smoother gradient.
   ``"relu"`` -- non-negative, can be exactly zero.
   ``"identity"`` -- signed weights (used by ``dendritic_mlp``).

``init_method``
   Weight initialization: ``"xavier_normal"``, ``"xavier_uniform"``,
   ``"kaiming_normal"``, ``"kaiming_uniform"``, ``"orthogonal"``,
   ``"normal"``, ``"uniform"``, ``"eye"``.

``noise_level``
   Gaussian noise added during top-K selection (default: 0.0).

``weight_norm_order``
   Optional L-p normalization per output neuron.

``gamma``
   Post-normalization scaling factor (default: 1.0).

Config path: ``model.core.sparsity``.

TopK Variants
~~~~~~~~~~~~~

Set via ``model.core.sparsity.type``:

``standard``
   Deterministic top-K mask (default).

``stochastic``
   Soft top-K with temperature for differentiable selection.
   Config: ``sparsity.temperature``.

``variance``
   Variance-aware synapse selection.

``deepst``
   Dynamic rewiring with density targets. Additional parameters under
   ``sparsity.deepst``: ``excitatory_target_density``,
   ``inhibitory_target_density``, ``sigma``, ``rewiring_mode``
   (``"global"``/``"local"``), ``rewire_frequency``.

BlockLinear
-----------

Block-diagonal linear layer for aggregating child-branch activations. Each
output neuron receives exactly ``block_size`` inputs from consecutive positions,
where ``block_size = in_features / out_features``.

``EfficientBlockLinear`` is a memory-efficient variant that uses reshape
operations instead of a sparse matrix. Same API and behavior.

Config path: ``model.core.blocklinear``.

Activation Functions
--------------------

Post-voltage nonlinearities applied at each branch layer. Set via
``model.core.reactivation.type``.

``param_tanh`` (default)
   ``(tanh(m * (V - b)) + 1) / 2`` with learnable slope ``m`` and midpoint
   ``b``. Output range is [0, 1]. Initialized with ``init_m`` and ``init_b``.

``param_tanh_only_m``
   Same as ``param_tanh`` but ``b`` is fixed (not trainable).

``param_linear_sigmoid``, ``param_linear_tanh``, ``param_linear_tanh_transition``
   Blended activations combining a linear region with a saturating function.

``relu``, ``sigmoid``, ``tanh``, ``leaky_relu``, ``elu``, ``selu``, ``softplus``
   Standard PyTorch activations (no learnable parameters).

.. code-block:: yaml

   model:
     core:
       reactivation:
         enabled: true
         type: "param_tanh"
         init_m: 1.5
         init_b: 0.5

Reactivation Gate Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The reactivation gate ``(tanh(m * (V - b)) + 1) / 2`` has two learnable
parameters: the slope ``m`` and the midpoint ``b``. How these are initialized
determines the gate's operating point at the start of training.

Reactivation policies can initialize the gate either from analytical branch
statistics or from sampled branch-voltage statistics. The data-driven
policies place the gate using the observed voltage distribution ``V``, which
keeps the same interface usable across shunting and additive morphologies.

``init_policy`` selects the initialization strategy:

``quantile`` **(backprop starting point; validate with init diagnostics)**
   Post-init calibration using quantile-occupancy. After the model is built,
   a few training batches are run through it. For each reactivation layer,
   the 10th and 90th percentiles of the pre-reactivation voltage ``V`` are
   measured, and ``(m, b)`` are set to pin those quantiles to the [0.1, 0.9]
   gate-output interval:

   - ``b = (q10 + q90) / 2``
   - ``m = atanh(0.8) / ((q90 - q10) / 2)``

   This is morphology-invariant by construction: it places the gate's active
   region around the observed voltage distribution regardless of whether the
   network uses shunting (V bounded in [0, 1]) or additive (V unbounded)
   morphology. All data-driven modes use iterative calibration
   (``n_iterations=3``) because updating upstream layers reshapes downstream
   voltage distributions.

   The outer tails are allowed to saturate on purpose. ``quantile`` is not
   trying to use the entire gate range uniformly; it is trying to place the
   bulk of the observed voltage mass into the gate's sensitive region while
   keeping the tails as margin.

   Quantile-based calibration now includes safeguards for ill-conditioned
   upper-layer fits. If the observed quantile span is too small or the fitted
   slope would have to exceed the configured ``calibration_max_m``, the layer
   falls back to its pre-calibration gate state instead of forcing a saturated
   occupancy fit.

   The fitted ``(m, b)`` are layer-specific by design. Different dendritic
   depths see different voltage statistics, so forcing one shared gate slope
   across depth is usually a worse calibration. For depth analysis, compare
   normalized gate coordinates or occupancy statistics rather than raw
   voltages across layers.

   Pair with a small ``reactivation_lr`` (``1e-4`` or ``1e-5``) to keep gate
   slope drift under control. For data-driven init policies, check the
   automatically written ``init_gate_stats.json`` artifact in the run
   directory before treating a run as healthy.

``occupancy_quantile`` **(recommended for local learning)**
   Generalized quantile-occupancy calibration with configurable quantile and
   target pairs. For local learning, use ``q_low=0.05, q_high=0.95`` mapping
   to gate outputs ``r_low=0.2, r_high=0.8``. This gives the gate a wider
   active region than ``quantile``, which gives the gate room to self-organize
   under noisier local credit signals. This is the recommended local-learning
   default for both shunting and additive.

``empirical``
   Robust median/MAD calibration from sampled voltages.

``analytical`` (default)
   Analytical initialization derived from the weight-init statistics by the
   chosen ``dbl_init_method``. No data is required. Uses ``init_m`` and
   ``init_b`` as starting values.

``fixed``
   Uses the YAML-supplied ``init_m`` and ``init_b`` verbatim, ignoring all
   calibration.

Strategy-Specific Recommendations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recommended default choices are:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Regime
     - Default
     - Settings
   * - **Backprop**
     - ``quantile``
     - ``reactivation_lr = 1e-4`` or ``1e-5``; standard optimizer recipe
   * - **Local CA**
     - ``occupancy_quantile``
     - ``q_low=0.05, q_high=0.95, r_low=0.2, r_high=0.8``; ``reactivation_lr = 1e-3``

No morphology-specific defaults are needed.

Init diagnostics
^^^^^^^^^^^^^^^^

When a model contains learnable reactivation gates, the training script writes
``init_gate_stats.json`` into the run directory immediately after model build.
This summarizes the initial ``m`` and ``b`` distributions per reactivation
module and flags suspicious patterns such as:

- slopes near the calibration clip
- repeated identical values across a full layer combined with saturation
- gate centers pushed to the voltage-range edge

This diagnostic is especially useful when using ``quantile`` or
``occupancy_quantile`` policies on new datasets or new recurrent settings.

**Standard backprop** (both morphologies):

.. code-block:: yaml

   model:
     core:
       reactivation:
         init_policy: "quantile"
   training:
     main:
       common:
         param_groups:
           reactivation_lr: 0.0001

**Local learning** (both morphologies):

.. code-block:: yaml

   model:
     core:
       reactivation:
         init_policy: "occupancy_quantile"
         occupancy_quantile_low: 0.05
         occupancy_quantile_high: 0.95
         occupancy_target_low: 0.2
         occupancy_target_high: 0.8
   training:
     main:
       common:
         param_groups:
           reactivation_lr: 0.001

Gate-occupancy warmup in a ``multi_stage`` pipeline is available when a staged
training setup is desired, but it is separate from the default local-learning
configuration shown above.

Recommended defaults
^^^^^^^^^^^^^^^^^^^^

- ``quantile`` for backprop, with a small ``reactivation_lr``
- ``occupancy_quantile`` for local CA, using q05/q95 -> 0.2/0.8

These defaults apply across both shunting and additive settings.

Training-Time Gate Maintenance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Initialization and training-time gate updates are separate controls.

For the paper-facing backprop settings, the recommended default is:

- use ``init_policy: "quantile"`` once at the start of training
- keep ``reactivation_update_mode: "backprop"``
- leave ``recalibrate_reactivation_every: 0``

This uses quantile only to place the gate initially. After that, ``(m, b)``
continue to learn through the optimizer like the rest of the model.

For local learning, the recommended default is:

- use ``init_policy: "occupancy_quantile"``
- keep Local CA updates enabled for ``(m, b)``
- do not enable periodic quantile refresh unless the experiment is explicitly
  about comparing local gate updates against quantile-maintained gates

Optional periodic quantile maintenance is available for controlled ablations
and diagnostics. The relevant trainer fields live under
``training.<strategy>.common``:

``reactivation_update_mode``
   ``"backprop"`` keeps optimizer or local-rule learning of ``(m, b)``.
   ``"quantile"`` freezes those direct updates and refreshes ``(m, b)``
   from the observed voltage distribution during training.

``recalibrate_reactivation_every``
   Refresh period in epochs. ``0`` disables training-time refresh.

``reactivation_recalibration_mode``
   Calibration rule used for periodic refresh. ``"from_init_policy"``
   reuses the configured initialization rule; ``"occupancy_quantile"``
   is the most common explicit choice for quantile-maintained gates.

``reactivation_recalibration_start_epoch``
   First epoch at which periodic refresh is allowed. Setting this to ``2``
   or later preserves the chosen initialization through the first training
   epoch.

``reactivation_recalibration_num_batches``
   Number of training batches used to estimate the live voltage distribution
   for each refresh.

``reactivation_recalibration_ema_alpha``
   Blend factor for periodic refresh. ``1.0`` means hard overwrite with the
   newly calibrated ``(m, b)``. Values in ``(0, 1)`` apply an EMA-style
   update toward the calibrated target.

In short: quantile initialization is the default paper recipe for backprop,
while periodic quantile maintenance is a supported experimental option for
studying simplified gate learning.

Recurrent E-I Network
---------------------

Any E-I architecture (``dendritic_shunting``, ``dendritic_additive``,
``einet``) becomes recurrent when ``connectivity.rec_*`` synapse counts are
nonzero.  Recurrence is controlled **solely** by these synapse counts;
any layer with at least one nonzero ``rec_*`` value becomes recurrent.
The existing ``architecture``, ``connectivity``, ``morphology``,
``sparsity``, ``reactivation``, and ``blocklinear`` sections are reused;
a small ``recurrent_ei`` block provides only the temporal parameters.
The factory translates this to an ``EINetwork`` internally.

Alternatively, use ``core.type: ei_unified`` with the ``unified_ei`` config
for full per-layer, per-population control.

Making an E-I core recurrent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   model:
     core:
       type: dendritic_shunting          # same type as feedforward
       architecture:
         excitatory_layer_sizes: [64]
         excitatory_branch_factors: [3, 3]
       connectivity:
         ee_synapses_per_branch_per_layer: [12]
         ie_synapses_per_branch_per_layer: [6]
         # Nonzero rec counts trigger recurrent mode
         rec_ee_synapses_per_branch: [8]   # E->E recurrent
         rec_ie_synapses_per_branch: [4]   # I->E recurrent
         rec_ei_synapses_per_branch: [8]   # E->I recurrent
         rec_ii_synapses_per_branch: [4]   # I->I recurrent
       transfer:
         input_mode: 1
         inhibitory_mode: "first"        # none, first, all
       # morphology, sparsity, reactivation, blocklinear as usual...

       recurrent_ei:                     # temporal params only
         output_mode: last               # "last", "all", "mean"
         use_transfer: true              # default true (set false to bypass TransferLayer)
         dt: 1.0                         # timestep (scalar or per-layer list)
         tau_base: 50.0                  # base time constant (scalar or per-layer list)
         tau_ratio: 3.0                  # geometric ratio between levels (scalar or per-layer list)

   training:
     main:
       strategy: recurrent

For recurrent ``einet``-style configs, ``use_transfer`` defaults to ``true``
for legacy parity, so a TransferLayer is built for both ``input_mode=1`` and
``input_mode=0``.
Set ``recurrent_ei.use_transfer: false`` to disable it explicitly.

Recurrent connectivity parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These go in the ``connectivity:`` section alongside the feedforward synapse
counts, as per-layer lists (repeat-last semantics):

``rec_ee_synapses_per_branch``
   E->E recurrent synapses per branch (applied to E population).

``rec_ie_synapses_per_branch``
   I->E recurrent synapses per branch (applied to E population).

``rec_ei_synapses_per_branch``
   E->I recurrent synapses per branch (applied to I population, defaults to ``rec_ee``).

``rec_ii_synapses_per_branch``
   I->I recurrent synapses per branch (applied to I population, defaults to ``rec_ie``).

A layer is recurrent if any of its rec synapse counts are > 0.

Temporal parameters (``recurrent_ei``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``output_mode``
   ``"last"``, ``"mean"``, ``"all"`` (default: ``"last"``).

``use_transfer``
   Whether to build a TransferLayer before recurrent E-I layers.
   Default: ``true`` for recurrent ``einet`` path (set ``false`` to bypass).

``dt``
   Integration timestep. Scalar or per-layer list (default: 1.0).

``tau_base``, ``tau_ratio``
   Base time constant and geometric ratio between dendritic levels.
   Scalar or per-layer list. Distal levels get longer time constants:
   ``tau[i] = tau_base * tau_ratio^(n_levels - 1 - i)``.

``level_taus``
   Explicit per-level time constants. If set, overrides ``tau_base``/``tau_ratio``.

Other parameters
~~~~~~~~~~~~~~~~

``transfer.inhibitory_mode``
   Controls which layers receive the transfer layer's inhibitory stream
   when ``input_mode=1``.  ``"none"`` - no layer receives it.
   ``"first"`` (default) - only the first layer.  ``"all"`` - every layer.
   This is strictly enforced: layers not selected receive ``None`` as
   inhibitory input.

``architecture.input_projection_dims``
   Optional MLP dims before the E-I layers.

``implementation.store_routing``
   Record per-level routing info for analysis.

Using ``ei_unified`` directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For full per-layer, per-population control, use ``core.type: ei_unified``
with the ``unified_ei`` config. Each layer is an ``EILayerConfig`` containing
``excitatory`` and ``inhibitory`` ``PopulationConfig`` objects.

The recurrent unified-EI path reuses the same ``DendriticBranchLayer`` and
``initialize.py`` logic as the feedforward EI stack. The main difference is
that recurrent populations wrap those branch layers with per-level temporal
traces and optional recurrent synapse banks.

PopulationConfig fields
^^^^^^^^^^^^^^^^^^^^^^^

Synapse counts:
   ``ff_excitatory_synapses``, ``ff_inhibitory_synapses`` (feedforward),
   ``rec_excitatory_synapses``, ``rec_inhibitory_synapses`` (recurrent, 0 for FF).

Dendritic structure:
   ``n_neurons``, ``branch_factors``.

TopK sparsity:
   ``topk_init_method``, ``topk_noise_level``, ``topk_type``
   (``"standard"``, ``"stochastic"``, ``"variance"``, ``"deepst"``),
   ``topk_weight_norm_order``, ``topk_gamma``, ``topk_temperature``,
   ``topk_ultrafast``, ``topk_strategy`` (``"none"``, ``"conductance_dynamic"``).

Dendritic computation:
   ``weight_transform`` (``"exp"``, ``"relu"``, ``"softplus"``, ``"identity"``),
   ``use_shunting``, ``use_additive_normalization``, ``allow_self_recurrence``,
   ``somatic_synapses``, ``dbl_init_method``, ``efficient_blocklinear``,
   ``blocklinear_strategy``.

Reactivation:
   ``reactivate``, ``reactivation_type`` (``"param_tanh"``, ``"tanh"``,
   ``"sigmoid"``), ``reactivation_init_m``, ``reactivation_init_b``,
   ``reactivation_strategy``.

Temporal:
   ``tau_base``, ``tau_ratio``, ``level_taus``, ``epsilon``.

Recurrent structure:
   ``allow_self_recurrence`` controls whether same-population recurrent synapses
   may read from the previous-timestep soma/output state of the same neuron.
   When set to ``False``, self-feedback is masked for E->E and I->I recurrent
   synapses while cross-population recurrence remains unchanged. This flag is
   only meaningful when the corresponding recurrent synapse count is nonzero.

Implementation:
   ``print_hooks``, ``adaptive_initialization``.

EILayerConfig fields
^^^^^^^^^^^^^^^^^^^^

``excitatory``
   PopulationConfig for the excitatory population.

``inhibitory``
   PopulationConfig for the inhibitory population (``null`` to omit).

``recurrent``
   ``True`` to enable recurrent dynamics; ``False`` for feedforward.

``dt``
   Integration timestep (default: 1.0).

``direct_ff_inhibitory_to_excitatory``
   If ``True``, excitatory population receives external inhibitory input
   directly (e.g., from transfer layer) even without a local inhibitory
   population.

EINetworkConfig fields
^^^^^^^^^^^^^^^^^^^^^^

``layers``
   List of EILayerConfig instances.

``use_transfer``
   Use TransferLayer for input routing (default: ``False``).

``transfer_params``
   TransferLayer parameters (``input_mode``, ``independent_pathways``, etc.).

``input_projection_dims``
   Optional MLP dims before the E-I stack.

``output_mode``
   ``"last"`` -- final timestep ``[B, C]``.
   ``"mean"`` -- temporal average ``[B, C]``.
   ``"all"`` -- all timesteps ``[B, T, C]``.

``store_routing``
   Record routing info for analysis.

**Recurrent dynamics**: at each timestep, raw synaptic currents are computed
per dendritic level, then integrated with leaky dynamics:
``trace' = decay * trace + (1 - decay) * raw``, where
``decay = exp(-dt / tau)``. The shunting (or additive) computation is applied
to the integrated traces. Five trace types are maintained per level:
feedforward E, feedforward I, recurrent E, recurrent I, and branch aggregation.

.. code-block:: yaml

   model:
     core:
       type: ei_unified
       unified_ei:
         output_mode: last
         use_transfer: false
         layers:
           - recurrent: true
             dt: 1.0
             excitatory:
               n_neurons: 64
               branch_factors: [3, 3]
               ff_excitatory_synapses: 12
               ff_inhibitory_synapses: 6
               rec_excitatory_synapses: 8
               rec_inhibitory_synapses: 4
               allow_self_recurrence: false
               tau_base: 50.0
               tau_ratio: 3.0
             inhibitory:
               n_neurons: 16
               branch_factors: [3, 3]
               ff_excitatory_synapses: 8
               ff_inhibitory_synapses: 4
               rec_excitatory_synapses: 4
               rec_inhibitory_synapses: 2
               allow_self_recurrence: false

Baseline RNN Cores
------------------

Standard recurrent architectures for comparison: ``gru``, ``lstm``,
``vanilla_rnn``.

.. code-block:: yaml

   model:
     core:
       type: gru
       baseline_rnn:
         hidden_dim: 128
         num_layers: 1
         dropout: 0.0
         output_mode: last
         input_projection_dims: []

MLP Baseline
------------

Standard multi-layer perceptron. Variants include ``PointMatchedParamMLP``,
``MatchedTotalParamMLP``, ``MatchedActiveParamMLP``, and
``SparseStructuredMLP`` for controlled parameter-count comparisons.

.. code-block:: yaml

   model:
     core:
       type: mlp
       parameters:
         hidden_dims: [128, 64, 32]
         activation: relu

Configuration Reference
-----------------------

ArchitectureConfig
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 15 35

   * - Parameter
     - Type
     - Default
     - Description
   * - ``excitatory_layer_sizes``
     - list[int]
     - ``[20]``
     - Neurons per excitatory layer
   * - ``inhibitory_layer_sizes``
     - list[int]
     - ``[]``
     - Neurons per inhibitory layer
   * - ``excitatory_branch_factors``
     - list[int]
     - ``[3,3,3,3]``
     - Branching factors for excitatory dendrites
   * - ``inhibitory_branch_factors``
     - list[int]
     - ``[1]``
     - Branching factors for inhibitory dendrites
   * - ``inhibitory_network_type``
     - str
     - ``"dendritic"``
     - ``"dendritic"`` or ``"mlp"``
   * - ``input_projection_dims``
     - list[int]
     - ``[]``
     - MLP dims before first E-I layer (empty = none)

ConnectivityConfig
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 30

   * - Parameter
     - Type
     - Default
     - Description
   * - ``ee_synapses_per_branch_per_layer``
     - list[int]
     - ``[40]``
     - Excitatory-to-excitatory synapses per branch
   * - ``ei_synapses_per_branch_per_layer``
     - list[int]
     - ``[0]``
     - Excitatory-to-inhibitory synapses per branch
   * - ``ie_synapses_per_branch_per_layer``
     - list[int]
     - ``[10]``
     - Inhibitory-to-excitatory synapses per branch
   * - ``ii_synapses_per_branch_per_layer``
     - list[int]
     - ``[0]``
     - Inhibitory-to-inhibitory synapses per branch
   * - ``rec_ee_synapses_per_branch``
     - list[int]
     - ``[0]``
     - E->E recurrent synapses per branch (nonzero = recurrent)
   * - ``rec_ie_synapses_per_branch``
     - list[int]
     - ``[0]``
     - I->E recurrent synapses per branch
   * - ``rec_ei_synapses_per_branch``
     - list[int]
     - ``[0]``
     - E->I recurrent synapses (defaults to ``rec_ee``)
   * - ``rec_ii_synapses_per_branch``
     - list[int]
     - ``[0]``
     - I->I recurrent synapses (defaults to ``rec_ie``)

MorphologyConfig
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 15 35

   * - Parameter
     - Type
     - Default
     - Description
   * - ``somatic_synapses``
     - bool
     - ``False``
     - Direct synaptic input at soma level
   * - ``use_shunting``
     - bool
     - ``True``
     - Shunting (divisive) vs additive inhibition
   * - ``use_additive_normalization``
     - bool
     - ``False``
     - Normalize voltage in additive mode
   * - ``weight_transform``
     - str
     - ``"softplus"``
     - Weight transform: exp, softplus, relu, identity
   * - ``dbl_init_method``
     - str
     - ``"analytical_expectation"``
     - BlockLinear initialization method

SparsityConfig
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 40

   * - Parameter
     - Type
     - Default
     - Description
   * - ``init_method``
     - str
     - ``"xavier_normal"``
     - TopK weight initialization
   * - ``noise_level``
     - float
     - ``0.0``
     - Noise during top-K selection
   * - ``type``
     - str
     - ``"standard"``
     - standard, stochastic, variance, deepst
   * - ``weight_norm_order``
     - int/None
     - ``None``
     - L-p normalization order
   * - ``gamma``
     - float
     - ``1.0``
     - Post-normalization scale
   * - ``temperature``
     - float
     - ``0.0``
     - Stochastic TopK temperature

ReactivationConfig
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 40

   * - Parameter
     - Type
     - Default
     - Description
   * - ``enabled``
     - bool
     - ``True``
     - Apply reactivation nonlinearity
   * - ``type``
     - str
     - ``"param_tanh"``
     - Activation type
   * - ``init_m``
     - float
     - ``1.5``
     - Initial slope parameter
   * - ``init_b``
     - float
     - ``0.5``
     - Initial midpoint parameter

TransferFunctionConfig
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 40

   * - Parameter
     - Type
     - Default
     - Description
   * - ``input_mode``
     - int
     - ``1``
     - 0: both get full input; 1: split
   * - ``independent_pathways``
     - bool
     - ``False``
     - Split input dims between E and I
   * - ``excitatory_dim``
     - int/None
     - ``None``
     - Explicit E pathway dimension
   * - ``inhibitory_dim``
     - int/None
     - ``None``
     - Explicit I pathway dimension
   * - ``output_activation``
     - str/None
     - ``None``
     - Post-split activation
   * - ``inhibitory_mode``
     - str
     - ``"first"``
     - Recurrent I routing: ``"none"``, ``"first"``, ``"all"``

ImplementationConfig
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 15 15 35

   * - Parameter
     - Type
     - Default
     - Description
   * - ``adaptive_initialization``
     - bool
     - ``False``
     - Network-aware weight scaling
   * - ``print_hooks``
     - bool
     - ``False``
     - Print gradient statistics
   * - ``store_routing``
     - bool
     - ``False``
     - Store per-timestep routing info for analysis
   * - ``compile_forward``
     - bool
     - ``True``
     - Opt-in/out ``torch.compile`` on the E-I core forward path; enabled by
       default for parity with the historical BP path and can be disabled
       explicitly for debugging or platform-specific reliability issues
