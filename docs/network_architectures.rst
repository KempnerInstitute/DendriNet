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
     - ``einet_sh``
   * - ``dendritic_additive``
     - E-I network with additive (subtractive) inhibition
     - ``einet_ns``
   * - ``flat_shunting``
     - Flattened dendrites with shunting inhibition
     - ``einet_sh_flat``
   * - ``flat_additive``
     - Flattened dendrites with additive inhibition
     - ``einet_ns_flat``
   * - ``einet``
     - Generic E-I; reads ``use_shunting`` from morphology config
     -
   * - ``dendritic_mlp``
     - E-I with signed weights, ``synapse_mode="mlp"``
     -
   * - ``ei_unified``
     - Unified E-I, supports feedforward and recurrent
     - ``unified_ei``, ``ei_net``, ``unified_einet``
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
   denominator = 1 + ReLU(E) + ReLU(E_rec) + g_branch + ReLU(I) + ReLU(I_rec)

   V = numerator / (denominator + epsilon)
   output = reactivation(V)

Where ``E`` is the excitatory synaptic current (TopKLinear), ``I`` is
inhibitory, ``V_branch`` is aggregated child-branch input (BlockLinear),
``g_branch`` is the branch conductance, and ``E_rec``, ``I_rec`` are recurrent
inputs (used only in recurrent mode). Epsilon defaults to ``1e-8``.

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

Recurrent E-I Network
---------------------

The ``ei_unified`` type supports both feedforward and recurrent operation
through the same architecture. Each layer is an EILayer containing excitatory
and (optionally) inhibitory StatefulDendriNet populations.

PopulationConfig
~~~~~~~~~~~~~~~~

Per-population parameters:

``n_neurons``
   Number of neurons (default: 64).

``branch_factors``
   Dendritic tree structure (default: ``[1]``).

``ff_excitatory_synapses``, ``ff_inhibitory_synapses``
   Feedforward synapse counts per branch.

``rec_excitatory_synapses``, ``rec_inhibitory_synapses``
   Recurrent synapse counts per branch (set to 0 for feedforward-only layers).

``tau_base``
   Base time constant (default: 50.0).

``tau_ratio``
   Geometric ratio between dendritic levels (default: 3.0). Distal levels
   receive longer time constants; soma receives the shortest.

``level_taus``
   Explicit per-level time constants. If empty, auto-generated from
   ``tau_base`` and ``tau_ratio`` as:
   ``tau[i] = tau_base * tau_ratio^(n_levels - 1 - i)``.

EILayerConfig
~~~~~~~~~~~~~

``excitatory``
   PopulationConfig for the excitatory population.

``inhibitory``
   PopulationConfig for the inhibitory population (set to ``null`` to omit).

``recurrent``
   ``True`` to enable recurrent dynamics; ``False`` for feedforward.

``dt``
   Integration timestep (default: 1.0).

EINetworkConfig
~~~~~~~~~~~~~~~

``layers``
   List of EILayerConfig instances (one per network layer).

``input_dim``
   Input dimension.

``output_mode``
   ``"last"`` -- output at final timestep, shape ``[B, C]``.
   ``"mean"`` -- temporal average, shape ``[B, C]``.
   ``"all"`` -- output at every timestep, shape ``[B, T, C]``.

``store_routing``
   If ``True``, record which branches are active per input (for routing
   analysis).

**Recurrent dynamics**: at each timestep, raw synaptic currents are computed
per dendritic level, then integrated with leaky dynamics:
``trace' = decay * trace + (1 - decay) * raw``, where
``decay = exp(-dt / tau)``. The shunting (or additive) computation is applied
to the integrated traces.

.. code-block:: yaml

   model:
     core:
       type: ei_unified
       unified_ei:
         output_mode: last
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
               tau_base: 50.0
               tau_ratio: 3.0
             inhibitory:
               n_neurons: 16
               branch_factors: [1]
               ff_excitatory_synapses: 8
               ff_inhibitory_synapses: 4

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
         bidirectional: false
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
