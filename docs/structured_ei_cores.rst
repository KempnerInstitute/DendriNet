.. _structured-ei-cores:

Structured E/I cores
====================

A structured E/I core is a stack of layers, each with one excitatory and one
optional inhibitory population of dendritic neurons, built from the typed
sections of ``model.core``: ``architecture``, ``connectivity``, ``transfer``,
``morphology``, ``sparsity``, ``reactivation``, ``blocklinear`` and
``implementation``. One set of options applies to every population, and the
core type selects how excitation and inhibition combine. Use a structured core
for E/I stacks of this shape; use a :doc:`population network
<population_networks>` for named populations, per-population options or
explicit connections. This page also covers the unified E/I core
(``unified_ei``), which takes one option block per population, the spatial
cores built from the same sections, and a reference of every section field.

Two names recur on this page:

* **structured E/I core**: the ``einet``, ``dendritic_*`` and ``flat_*`` core
  types. The feedforward network is ``ConfigurableEINetwork`` (a subclass of
  ``ExcitationInhibitionNetwork``); the recurrent network is ``EINetwork``.
* **unified E/I core**: the ``ei_unified`` type (also spelled ``unified_ei``)
  and the ``rnn_*`` aliases, configured under ``model.core.unified_ei`` and
  built as ``EINetwork``.

Core types and integration mode
-------------------------------

The structured core types differ in how a compartment combines its
excitatory and inhibitory input (see :ref:`integration-modes` for the
equations), in the sign of their synaptic weights and in the shape of the
dendritic tree. :ref:`core-types` lists every ``model.core.type``.

.. list-table::
   :header-rows: 1
   :widths: 30 26 26 18

   * - ``model.core.type``
     - Integration
     - Synaptic weights
     - Recurrent form
   * - ``einet`` (the default)
     - set by ``morphology.use_shunting`` and
       ``morphology.use_additive_normalization``
     - positive (``morphology.weight_transform``)
     - yes
   * - ``dendritic_shunting``
     - shunting (divisive)
     - positive
     - yes
   * - ``dendritic_additive``
     - raw additive
     - positive
     - yes
   * - ``dendritic_normalized_additive``
     - additive, normalized across the layer's outputs
     - positive
     - yes
   * - ``flat_shunting``, ``flat_additive``, ``flat_normalized_additive``
     - as the ``dendritic_*`` type of the same name, on a flattened tree
     - positive
     - no
   * - ``dendritic_signed``, ``flat_signed``
     - additive
     - signed (``identity`` transform), separate E and I synapse banks
     - no
   * - ``dendritic_mlp``, ``flat_mlp``
     - additive
     - signed (``identity`` transform), one bank that reads the excitatory and
       inhibitory inputs together
     - no

A ``flat_*`` type flattens the excitatory tree into one dendritic level that
holds all of its compartments: ``excitatory_branch_factors: [3, 3]`` (3 + 9
compartments) becomes ``[12]``. The inhibitory tree is not changed.

Which setting decides
~~~~~~~~~~~~~~~~~~~~~

Each family has one source of truth for the integration mode:

* ``dendritic_*`` and ``flat_*`` types (except ``*_signed``) fix
  ``morphology.use_shunting``; the additive types also fix
  ``morphology.use_additive_normalization``. A written value that contradicts
  the type raises when the config loads. A saved ``config.json``, which lists
  every field, loads with a warning and the type's value.
* ``*_signed`` and ``*_mlp`` types always build ``identity`` weights with
  additive integration: they fix ``morphology.weight_transform: identity``
  and ``morphology.use_shunting: false``, and a written value that
  contradicts them raises when the config loads (a saved ``config.json``
  loads with a warning and the type's values).
* ``einet``, the spatial types and ``dendritic_alexnet`` read
  ``morphology.use_shunting`` and ``morphology.use_additive_normalization``.
* The ``rnn_*`` aliases of the unified E/I core set ``use_shunting`` and
  ``use_additive_normalization`` in every population block, and the
  ``rnn_flat_*`` aliases also set ``branch_factors: [1]``. A block value that
  contradicts the alias is replaced, with a warning.
* ``ei_unified`` and ``population_network`` read the fields of each population
  block.

When ``reactivation.init_policy`` is omitted, the additive structured types and
``einet`` with ``use_shunting: false`` initialize their gates with
``occupancy_quantile``; every other core uses ``analytical``
(:ref:`gate-initialization`).

ExcitationInhibitionNetwork
---------------------------

The feedforward structured core first passes its input through a
``TransferLayer``, which produces an excitatory and an inhibitory input
stream. Each layer then runs its inhibitory population before its excitatory
population:

.. code-block:: text

   (x_E, x_I) = TransferLayer(x)
   h_I[l] = inhibitory_cells[l](h_E[l-1], h_I[l-1])     # h_E[0] = x_E, h_I[0] = x_I
   h_E[l] = excitatory_cells[l](h_E[l-1], h_I[l])
   output = h_E[last]

A layer without inhibitory cells passes its inhibitory input on unchanged
(:ref:`inhibitory-wiring` says which cells may read it). The output size is
the last entry of ``architecture.excitatory_layer_sizes``. The
excitatory cells are a ``DendriNet`` with ``excitatory_branch_factors``
(:ref:`dendritic-tree`). The inhibitory cells are a ``DendriNet`` with
``inhibitory_branch_factors`` or, with ``architecture.inhibitory_network_type:
mlp``, a dense MLP with sigmoid outputs that reads the layer's excitatory
input (``mlp_inhibitory_network_params``; feedforward cores only).

A shunting network with a nonnegative weight transform (``exp``, ``softplus``
or ``relu``) checks the transferred input on every forward pass and raises
when it is negative, because negative conductance drive can make the shunting
denominator vanish. Keep the input
nonnegative (for example ``data.processing.normalize: false`` for images),
set ``transfer.output_activation: relu``, or use an additive type.

.. _transfer-layer:

TransferLayer
~~~~~~~~~~~~~

``TransferLayer`` turns the core input into the excitatory and the inhibitory
input stream. With ``independent_pathways: false`` both streams carry the
whole input. With ``true`` the input channels are split between the streams,
so each channel is either excitatory or inhibitory. Its options are in
``model.core.transfer``:

.. list-table::
   :header-rows: 1
   :widths: 24 14 14 48

   * - Field
     - Type
     - Default
     - Meaning
   * - ``independent_pathways``
     - bool
     - ``false``
     - Split the input channels between the streams instead of copying the
       input into both.
   * - ``excitatory_dim``, ``inhibitory_dim``
     - int or null
     - ``null``
     - Stream sizes of the split. An unset size is the rest of the input; with
       both unset the input is halved. Setting them without
       ``independent_pathways: true`` raises.
   * - ``split_strategy``
     - str
     - ``random``
     - ``random`` assigns channels with a seeded permutation; ``ordered`` gives
       the first ``excitatory_dim`` channels to the excitatory stream.
   * - ``split_seed``
     - int or null
     - ``null``
     - Seed of the ``random`` split. Training fills an unset seed from the
       experiment's topology seed (:ref:`random-seeds`).
   * - ``output_activation``
     - str or null
     - ``null``
     - Applied to both streams: ``relu``, ``sigmoid``, ``softplus``,
       ``tanh`` or ``none``. The nonnegative weight transforms (``exp``,
       ``softplus``, ``relu``) reject ``tanh``, whose output is signed.
   * - ``input_transform``
     - str or null
     - ``null``
     - Only ``identity`` is accepted. Signed-input adapters are selected
       elsewhere (:ref:`spatial-cores`).

``TransferLayer`` also accepts ``input_normalization`` (a mapping with
``mode``: ``none``, ``global_l1`` or ``group_l1``; ``group_ranges`` for
``group_l1``; ``epsilon``), which divides the input by the L1 norm of all
channels or of each declared channel range. ``model.core.transfer`` does not
have this field; it is set through ``model.core.unified_ei.transfer_params``
or ``model.core.population_network.transfer_params``.

.. _inhibitory-wiring:

.. _structured-inhibitory-wiring:

Inhibitory wiring
~~~~~~~~~~~~~~~~~

Feedforward and recurrent structured cores wire their inhibitory synapses by
one rule, determined by ``architecture.inhibitory_layer_sizes`` and the
synapse counts in ``model.core.connectivity``:

* The first layer's feedforward inhibitory input is the transfer layer's
  inhibitory stream; a later layer's is the previous layer's inhibitory cells.
* Inhibitory cells read the layer's excitatory input on their ``ei`` synapses
  and its feedforward inhibitory input on their ``ii`` synapses.
* Excitatory cells read their own layer's inhibitory cells on their ``ie``
  synapses. When the first layer has no inhibitory cells, its excitatory
  cells read the transferred inhibitory stream directly.

``inhibitory_layer_sizes`` is either empty (no layer has inhibitory cells) or
gives one size per layer, ``0`` for a layer without inhibitory cells. Synapse
lists give one count per layer and
repeat their last entry for later layers. A synapse count without a source
raises: ``ie`` synapses in a later layer that has no inhibitory cells, ``ii``
synapses in a layer that receives no inhibitory input, and inhibitory synapse
counts for cells a layer does not have. Inhibitory cells without any input
synapse (``ei``, ``ii``, ``rec_ei`` and ``rec_ii`` all 0) raise as well.

The input therefore reaches inhibition in one of two ways:

* **through inhibitory cells**: declare them (``inhibitory_layer_sizes: [20]``
  with ``ei`` synapses); the input drives them and they inhibit the excitatory
  cells;
* **directly**: declare none (``inhibitory_layer_sizes: []``); the input's
  inhibitory stream drives the first layer's ``ie`` synapses. In a network of
  several layers, write ``ie_synapses_per_branch_per_layer: [8, 0]``, because
  later layers have no inhibitory source.

A :doc:`population network <population_networks>` expresses the same wirings
with connections from ``input_e`` and ``input_i``
(:ref:`population-connections`). The retired ``model.core.transfer`` keys
``input_mode``, ``inhibitory_mode``, ``allow_direct_inhibitory_stream``,
``allow_direct_inhibitory_input``, ``require_explicit_direct_inhibitory_stream``,
``input_mode1_build_inhibitory_population`` and
``build_inhibitory_population_for_input_mode1`` load with a warning when they
describe the wiring above and raise when they selected a different network
(for example ``input_mode: 1`` with declared inhibitory cells).

``input_mode: 1`` was the default of feedforward cores: a feedforward config
written for an earlier version that declares inhibitory cells without setting
``input_mode`` ran without them, while the current library builds them.
``dendritic-upgrade-config`` (:ref:`config-upgrade`) rewrites such configs so
that they describe the network they ran.

Feedforward example
~~~~~~~~~~~~~~~~~~~

A two-layer shunting core with an inhibitory population in each layer, for a
flattened 28 x 28 input. ``dendritic-train`` sets
``model.encoder.params.input_dim`` from the dataset (the number of elements of
one sample); the examples on this page write it so that they also build
without a dataset.

.. code-block:: yaml

   model:
     encoder:
       type: identity
       params: {input_dim: 784}
     core:
       type: dendritic_shunting
       architecture:
         excitatory_layer_sizes: [100, 50]
         inhibitory_layer_sizes: [20, 10]
         excitatory_branch_factors: [3, 3]
         inhibitory_branch_factors: [1]
       connectivity:
         ee_synapses_per_branch_per_layer: [40]   # input or previous E onto E
         ei_synapses_per_branch_per_layer: [10]   # input or previous E onto I
         ie_synapses_per_branch_per_layer: [10]   # this layer's I onto E
         ii_synapses_per_branch_per_layer: [0]
       morphology:
         weight_transform: softplus
         dbl_init_method: analytical_expectation
       sparsity:
         type: indexed_rewire
         indexed:
           rewire_frequency: 100
           rewire_quantile: 0.05
       reactivation:
         type: param_tanh
     decoder:
       type: MLP
       params: {hidden_dims: [], output_dim: 10}

.. _recurrent-structured-cores:

Recurrent structured cores
--------------------------

The structured types ``einet``, ``dendritic_shunting``, ``dendritic_additive``
and ``dendritic_normalized_additive`` build the recurrent ``EINetwork``
instead of the feedforward network when any of these is set:

* a positive ``connectivity.rec_*`` synapse count in any layer;
* a stateful feature: ``model.core.dynamics.mode: spike``
  (:ref:`spiking-soma`), ``model.core.dendritic_spikes.enabled``
  (:ref:`dendritic-spikes`), ``model.core.soma_feedback.enabled``
  (:ref:`soma-feedback`) or ``model.core.synapse_types.enabled``
  (:ref:`synapse-types`);
* ``model.core.recurrent_ei.enabled: true``. A layer becomes recurrent only
  through one of the above or a true ``recurrent_ei.recurrent_layers`` entry;
  when no layer does, the build raises.

The ``flat_*``, ``*_signed`` and ``*_mlp`` types are feedforward only and raise
when one of these is set. A recurrent core raises for the feedforward-only
options ``reactivation.soma_type`` (and non-default ``soma_init_m`` or
``soma_init_b``), ``architecture.inhibitory_network_type: mlp`` and
``implementation.compile_forward: true``. A feedforward core raises for the
recurrent-only options ``architecture.input_projection_dims``,
``implementation.store_routing: true``, ``morphology.allow_self_recurrence:
false`` and any ``recurrent_ei`` setting other than ``enabled: false``.

The recurrent network reuses every structured section: each layer gets one
excitatory and one inhibitory population (the latter when
``inhibitory_layer_sizes`` declares cells) with the options of
``morphology``, ``sparsity``, ``reactivation``, ``blocklinear`` and
``implementation``. It runs the transfer layer, the optional
``input_projection_dims`` (dense ReLU layers applied to the excitatory
stream) and the layers once per timestep, and trains with
``training.main.strategy: recurrent`` (:ref:`training-strategies`).

Recurrent synapses
~~~~~~~~~~~~~~~~~~

The recurrent synapse counts are per-layer lists in
``model.core.connectivity`` (one value per layer, last value repeated, default
``[0]``). Within a timestep the inhibitory population updates first; the
excitatory population then reads this step's inhibitory output on its ``ie``
synapses. Recurrent synapses read the previous timestep's outputs of the same
layer:

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Field
     - Target population
     - Source (previous timestep)
   * - ``rec_ee_synapses_per_branch``
     - excitatory
     - the layer's excitatory output
   * - ``rec_ie_synapses_per_branch``
     - excitatory
     - the layer's inhibitory output
   * - ``rec_ei_synapses_per_branch``
     - inhibitory
     - the layer's excitatory output
   * - ``rec_ii_synapses_per_branch``
     - inhibitory
     - the layer's inhibitory output

``morphology.allow_self_recurrence: false`` removes each neuron's synapse onto
itself from the ``rec_ee`` and ``rec_ii`` banks and keeps all other recurrent
contacts.

Temporal settings (``recurrent_ei``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``model.core.recurrent_ei`` holds the temporal settings of a recurrent
structured core. Every dendritic level keeps leaky traces of its feedforward
excitatory, feedforward inhibitory, recurrent excitatory, recurrent inhibitory
and child-branch input:

.. math::

   \mathrm{trace}_t = d\,\mathrm{trace}_{t-1} + (1 - d)\,\mathrm{raw}_t,
   \qquad d = e^{-\Delta t / \tau}

and integrates the traces with the layer's shunting or additive rule. Level
time constants are geometric unless ``level_taus`` is set:
:math:`\tau_i = \tau_\mathrm{base} \cdot \tau_\mathrm{ratio}^{\,n - 1 - i}`
for level :math:`i` of :math:`n`, counted from the most distal level, so
distal levels are slower (:ref:`temporal-constants`).

.. list-table::
   :header-rows: 1
   :widths: 22 18 12 48

   * - Field
     - Type
     - Default
     - Meaning
   * - ``enabled``
     - bool
     - ``false``
     - Build the recurrent network although no ``rec_*`` count or stateful
       feature asks for it; combine with ``recurrent_layers``.
   * - ``recurrent_layers``
     - bool or list of bool
     - unset
     - Which layers keep state (last entry repeats). Unset, a layer is
       recurrent when it has recurrent synapses or a stateful feature. A
       ``true`` layer without recurrent synapses integrates its input with the
       time constants below; ``false`` for a layer with recurrent synapses
       raises.
   * - ``dt``
     - float or list
     - ``1.0``
     - Integration step, scalar or one value per layer.
   * - ``tau_base``
     - float or list
     - ``50.0``
     - Time constant of the soma level, scalar or per layer.
   * - ``tau_ratio``
     - float or list
     - ``3.0``
     - Ratio between the time constants of adjacent levels, scalar or per
       layer.
   * - ``level_taus``
     - list of float
     - ``[]``
     - Explicit time constant of every level, distal to soma; replaces
       ``tau_base`` and ``tau_ratio``. Its length must equal the number of
       levels.
   * - ``learnable_tau``
     - bool
     - ``false``
     - Train the time constants (stored as log values).
   * - ``epsilon``
     - float
     - ``1e-8``
     - Constant added to the shunting denominator.
   * - ``output_mode``
     - str
     - ``last``
     - ``last``: final timestep ``[B, C]``; ``mean``: average over timesteps
       ``[B, C]``; ``all``: every timestep ``[B, T, C]``.
   * - ``use_transfer``
     - bool
     - ``true``
     - Build the transfer layer. With ``false`` the first layer has no
       transferred inhibitory stream, so its ``ii`` count, and without
       inhibitory cells its ``ie`` count, must be 0.

Other keys under ``recurrent_ei`` raise. Transfer options are read only from
``model.core.transfer``; the spellings ``recurrent_ei.transfer``,
``recurrent_ei.transfer_params`` and ``recurrent_ei.recurrent_per_layer`` raise
with the replacement location.

Recurrent example
~~~~~~~~~~~~~~~~~

A one-layer recurrent shunting core with recurrent E and I synapses:

.. code-block:: yaml

   model:
     encoder:
       type: identity
       params: {input_dim: 32}
     core:
       type: dendritic_shunting
       architecture:
         excitatory_layer_sizes: [64]
         inhibitory_layer_sizes: [16]
         excitatory_branch_factors: [3, 3]
         inhibitory_branch_factors: [1]
       connectivity:
         ee_synapses_per_branch_per_layer: [12]
         ei_synapses_per_branch_per_layer: [8]
         ie_synapses_per_branch_per_layer: [6]
         ii_synapses_per_branch_per_layer: [0]
         rec_ee_synapses_per_branch: [8]
         rec_ie_synapses_per_branch: [4]
         rec_ei_synapses_per_branch: [8]
         rec_ii_synapses_per_branch: [4]
       transfer:
         output_activation: relu
       recurrent_ei:
         output_mode: last
         dt: 1.0
         tau_base: 50.0
         tau_ratio: 3.0
     decoder:
       type: MLP
       params: {hidden_dims: [], output_dim: 10}
   training:
     main:
       strategy: recurrent

The core expects sequences ``[B, T, 32]``. For gated recurrent baselines use
``model.core.type: gru``, ``lstm`` or ``vanilla_rnn`` (:ref:`baseline-models`).

.. _unified-ei:

Unified E/I core
----------------

``model.core.type: ei_unified`` (or ``unified_ei``) builds an ``EINetwork``
from ``model.core.unified_ei``, which gives every layer its own excitatory and
inhibitory population block. Use it when the populations of a layer, or the
layers of a stack, need different options, such as different time constants,
synapse counts or morphologies. The section spelled ``model.core.ei_unified``
loads with a deprecation warning, and so do the type names ``ei_net`` and
``unified_einet``. A ``unified_ei`` core reads only ``model.core.unified_ei``;
other non-default ``model.core`` sections raise.

``model.core.unified_ei`` holds the fields of ``EINetworkConfig``:

.. list-table::
   :header-rows: 1
   :widths: 24 16 12 48

   * - Field
     - Type
     - Default
     - Meaning
   * - ``layers``
     - list
     - required
     - One mapping per layer with the ``EILayerConfig`` fields below.
   * - ``use_transfer``
     - bool
     - ``false``
     - Pass the input through a ``TransferLayer``.
   * - ``transfer_params``
     - mapping
     - ``{}``
     - ``TransferLayer`` options (:ref:`transfer-layer`, including
       ``input_normalization``); unknown keys raise.
   * - ``input_projection_dims``
     - list of int
     - ``[]``
     - Dense ReLU layers applied to the excitatory input stream before the
       first layer.
   * - ``output_mode``
     - str
     - ``last``
     - Recurrent networks: ``last``, ``mean`` or ``all``, as in
       :ref:`recurrent-structured-cores`.
   * - ``store_routing``
     - bool
     - ``false``
     - Record each layer's per-level routing at every timestep for analysis
       (recurrent networks).
   * - ``input_dim``
     - int
     - from the encoder
     - Set from the encoder output size; a different written value raises.

Each entry of ``layers`` holds the fields of ``EILayerConfig``:

.. list-table::
   :header-rows: 1
   :widths: 30 16 12 42

   * - Field
     - Type
     - Default
     - Meaning
   * - ``excitatory``
     - mapping
     - required
     - Population block of the excitatory population.
   * - ``inhibitory``
     - mapping or null
     - required
     - Population block of the inhibitory population, or ``null`` for a layer
       without one. An omitted ``inhibitory`` raises.
   * - ``recurrent``
     - bool
     - ``false``
     - Keep state across timesteps and build recurrent synapses. A
       non-recurrent layer raises when a population sets recurrent synapse
       counts or a stateful feature (spiking soma, ``synapse_types``,
       dendritic spikes or soma feedback).
   * - ``dt``
     - float
     - ``1.0``
     - Integration step of the layer.
   * - ``direct_ff_inhibitory_to_excitatory``
     - bool
     - ``false``
     - With ``use_transfer: true``, the layer's feedforward inhibitory input
       is the transferred inhibitory stream: the inhibitory population reads
       it instead of the previous layer's inhibitory output, and without an
       inhibitory population the excitatory population reads it directly.
       Otherwise excitatory cells receive feedforward inhibition only from
       their own layer's inhibitory population.

``excitatory_input_dim`` and ``inhibitory_input_dim`` are also
``EILayerConfig`` fields, but the network sets them from the previous layer
when it builds, replacing written values.

The wiring within a layer is the one of the recurrent structured core: the
inhibitory population updates first and reads the excitatory input on its
``ff_excitatory_synapses``, the feedforward inhibitory input on its
``ff_inhibitory_synapses``, and the previous timestep's E and I outputs on
its ``rec_excitatory_synapses`` and ``rec_inhibitory_synapses``. The
excitatory population reads the excitatory input, this step's inhibitory
output on its ``ff_inhibitory_synapses``, and the previous E and I outputs on
its recurrent synapses.

Population blocks
~~~~~~~~~~~~~~~~~

``excitatory`` and ``inhibitory`` are ``PopulationConfig`` mappings, the same
options a population network takes in ``population_defaults`` and
``population`` blocks; :ref:`population-options` lists every field. In
``unified_ei`` the morphology is part of the block: ``n_neurons`` (default 64)
and ``branch_factors`` (default ``[1]``) are written there. Unknown keys raise
when the model is built. ``allow_self_recurrence: false`` removes each
neuron's synapse onto itself from its population's recurrent bank.

Some defaults of a population block differ from the structured sections:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Option
     - Structured section default
     - Population block default
   * - weight transform
     - ``morphology.weight_transform: softplus``
     - ``weight_transform: exp``
   * - somatic synapses
     - ``morphology.somatic_synapses: false``
     - ``somatic_synapses: true``
   * - initial gate slope
     - ``reactivation.init_m: 1.5``
     - ``reactivation_init_m: 1.0``

Write these options explicitly to use the same parameterization in both
forms.

Recurrent aliases
~~~~~~~~~~~~~~~~~

``rnn_dendritic_shunting``, ``rnn_dendritic_additive``,
``rnn_dendritic_normalized_additive``, ``rnn_flat_shunting``,
``rnn_flat_additive`` and ``rnn_flat_normalized_additive`` read
``model.core.unified_ei`` like ``ei_unified`` and set ``use_shunting`` and
``use_additive_normalization`` in every population block; the ``rnn_flat_*``
aliases also set ``branch_factors: [1]``. They do not change the gate
initialization: set ``reactivation_init_policy`` in the blocks of an additive
alias (:ref:`gate-initialization`).

Unified core example
~~~~~~~~~~~~~~~~~~~~

A one-layer recurrent unified core:

.. code-block:: yaml

   model:
     encoder:
       type: identity
       params: {input_dim: 32}
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
               allow_self_recurrence: false
               tau_base: 50.0
               tau_ratio: 3.0
             inhibitory:
               n_neurons: 16
               branch_factors: [3, 3]
               ff_excitatory_synapses: 8
               ff_inhibitory_synapses: 0
               rec_excitatory_synapses: 4
               rec_inhibitory_synapses: 2
               allow_self_recurrence: false
     decoder:
       type: MLP
       params: {hidden_dims: [], output_dim: 10}
   training:
     main:
       strategy: recurrent

Without a transfer layer the first layer's inhibitory population has no
feedforward inhibitory input, so the example gives it
``ff_inhibitory_synapses: 0``. A layer's feedforward inhibitory input is the
transferred stream (first layer with ``use_transfer: true``, or a layer with
``direct_ff_inhibitory_to_excitatory: true``) or the previous layer's
inhibitory population; its excitatory population reads the layer's own
inhibitory population, or that input directly when the layer has none and
sets ``direct_ff_inhibitory_to_excitatory``. A written
``ff_inhibitory_synapses`` or ``rec_inhibitory_synapses`` (or its
``*_by_level`` list) above 0 without such a source raises when the model is
built. An omitted ``ff_inhibitory_synapses`` is the population default ``4``,
which builds no synapses where there is no source.

.. _spatial-cores:

Embedded and spatial cores
--------------------------

These cores apply a structured E/I network to image-shaped input or place it
inside a larger network. They read the same structured sections as
``einet``, including ``morphology.use_shunting``, and are feedforward only:
recurrent synapse counts, stateful features and ``recurrent_ei`` settings
raise.

Input and output adapters
~~~~~~~~~~~~~~~~~~~~~~~~~

A dendritic core inside another network follows the contract
``input adapter -> dendritic core -> output adapter``. The adapters are in
``dendritic_modeling.networks.architectures.replacement``:

* **Input adapters** map signed activity into the nonnegative range of
  positive-conductance cores: ``identity`` (the input is already
  nonnegative), ``relu`` (keeps the positive part) and ``signed_split``
  (``[relu(x), relu(-x)]``, which doubles the number of features or
  channels). They are selected by ``model.core.population_network.input_transform``,
  ``model.core.spatial.input_transform`` and
  ``model.core.vision.input_transform``.
* **Output adapters** rescale each channel without mixing channels:
  ``identity`` (the default); ``learned_global`` and ``learned_per_channel``
  (a learned positive gain); ``learned_global_threshold_relu`` and
  ``learned_per_channel_threshold_relu`` (``relu(gain * x - threshold)`` with
  learned positive gain and threshold); ``signed_per_channel_affine`` (a
  learned signed gain and offset per channel, starting at the identity). They
  are selected by ``model.core.spatial.output_adapter_mode`` and
  ``model.core.vision.output_adapter.mode``.

Tensor-map cores
~~~~~~~~~~~~~~~~

``hierarchical_dendritic_tensor_map`` (aliases ``dendritic_tensor_map`` and
``tensor_map``) flattens an input of shape ``(B, C, H, W)``, or takes a flat
input ``(B, D)``, and runs the structured E/I network on it. Its encoder output
size is the flattened size, for example ``input_dim: 3072`` for a
``[3, 32, 32]`` image. The output is the last excitatory layer, so an ``MLP``
decoder reads it as usual. Built directly with
``get_architecture(..., suffix_input_dim=n)``, the core adds a linear
projection to ``n`` outputs when ``n`` differs from its excitatory output;
``initialize_model`` passes no ``suffix_input_dim``.

.. code-block:: yaml

   model:
     encoder:
       type: identity
       params: {input_dim: 3072}   # 3 x 32 x 32, flattened
     core:
       type: hierarchical_dendritic_tensor_map
       architecture:
         excitatory_layer_sizes: [64]
         inhibitory_layer_sizes: [16]
         excitatory_branch_factors: [3, 3]
         inhibitory_branch_factors: [1]
       connectivity:
         ee_synapses_per_branch_per_layer: [40]
         ei_synapses_per_branch_per_layer: [40]
         ie_synapses_per_branch_per_layer: [8]
         ii_synapses_per_branch_per_layer: [0]
       morphology:
         use_shunting: true
     decoder:
       type: MLP
       params: {hidden_dims: [], output_dim: 10}

To restrict the feature-map pathways by image position, declare
``connectivity.structured.pathways.ee`` and ``.ei`` with the method
``spatial_morphology`` and leave ``ie`` undeclared, because the inhibitory
cells' outputs have no spatial layout (:ref:`structured-connectivity`).

Convolutional cores
~~~~~~~~~~~~~~~~~~~

``hierarchical_dendritic_conv`` (alias ``spatial_dendritic_conv``) extracts
local patches of an input ``(B, C_in, H, W)`` and runs one shared structured
E/I network on every patch, giving a feature map ``(B, C_out, H_out, W_out)``
with ``C_out`` the last entry of ``excitatory_layer_sizes``. At each output
position, excitatory soma ``c`` is output channel ``c``; the same soma/channel
mapping and the same parameters are used at every position. The inhibitory
population acts within each position and is not part of the output, so two
convolutional cores can be stacked with ordinary pooling or normalization
between them. ``spatial_output_contract()`` returns this mapping as a
dictionary.

The local geometry and the adapters are set in ``model.core.spatial``:

.. list-table::
   :header-rows: 1
   :widths: 28 14 12 46

   * - Field
     - Type
     - Default
     - Meaning
   * - ``kernel_size``
     - int
     - ``3``
     - Patch size.
   * - ``stride``
     - int
     - ``1``
     - Patch stride.
   * - ``padding``
     - int
     - ``0``
     - Zero padding added before the patches are taken.
   * - ``input_transform``
     - str
     - ``identity``
     - Input adapter: ``identity``, ``relu`` or ``signed_split`` (doubles
       ``C_in``).
   * - ``input_scale``
     - float
     - ``1.0``
     - Positive factor applied to the adapted input, to bring large activations
       into the range the conductance initialization assumes.
   * - ``output_adapter_mode``
     - str
     - ``identity``
     - Output adapter (see above).
   * - ``output_initial_scale``
     - float
     - ``1.0``
     - Initial gain of a learned output adapter.
   * - ``output_initial_threshold``
     - float
     - ``0.25``
     - Initial threshold of a threshold output adapter.

Only the convolutional cores read ``model.core.spatial``, and the sparse
point control below reads only ``kernel_size``, ``stride`` and ``padding``.
For every other built-in core type a ``spatial`` option that differs from its
default raises when the configuration loads, and so do the adapter options of
the sparse point control; a saved ``config.json`` loads with a warning
instead. Core types that plugins register are not checked.

As the ``model.core`` of a classifier, a convolutional core reads the images
themselves: use the ``identity`` encoder and load the dataset as images
(``data.processing.flatten: false``), so that ``dendritic-train`` sets
``model.encoder.params.input_shape`` to ``[C, H, W]``. The core receives ``C``
channels, and its feature maps are flattened, channel first, for the decoder,
which therefore reads ``C_out * H_out * W_out`` values
(``FlattenedSpatialCore`` in ``dendritic_modeling.networks.architectures.vision``
wraps the core). Images of another size raise. A flattened dataset or another
encoder raises when the model is built. To use the core inside another
network, build it directly with the number of input channels as
``input_dim``:

.. code-block:: python

   import torch

   from dendritic_modeling.config import config_from_mapping
   from dendritic_modeling.networks.architectures.factory import get_architecture

   core = config_from_mapping(
       {
           "model": {
               "core": {
                   "type": "hierarchical_dendritic_conv",
                   "architecture": {
                       "excitatory_layer_sizes": [16],
                       "inhibitory_layer_sizes": [4],
                       "excitatory_branch_factors": [3],
                       "inhibitory_branch_factors": [1],
                   },
                   "connectivity": {
                       "ee_synapses_per_branch_per_layer": [9],
                       "ei_synapses_per_branch_per_layer": [9],
                       "ie_synapses_per_branch_per_layer": [2],
                   },
                   "spatial": {"kernel_size": 3, "padding": 1},
               }
           }
       }
   ).model.core
   conv = get_architecture(core.type, core, input_dim=3)  # input_dim: input channels
   features = conv(torch.rand(2, 3, 32, 32))  # shape (2, 16, 32, 32)

``sparse_active_matched_point_conv`` (alias
``spatial_sparse_active_matched_point``) is a control for the convolutional
core: one fixed-support sparse affine layer with ReLU, applied to every patch
and taking ``kernel_size``, ``stride`` and ``padding`` from
``model.core.spatial`` (it has no input or output adapter, so the other
``spatial`` options raise). Like the convolutional core, it returns a feature
map, takes the number of input channels as ``input_dim`` and is flattened
for the decoder of a classifier in the same way. Its number of
trained scalars equals the active parameter count of the E/I network that the
same sections describe, or
``model.core.target_active_parameters`` when set; building raises when that
budget needs more inputs per output than a patch has. It requires
``sparsity.indexed.seed``, which training fills from the experiment's
topology seed (:ref:`random-seeds`). :ref:`baseline-models` describes the
other matched controls.

Dendritic AlexNet
~~~~~~~~~~~~~~~~~

``dendritic_alexnet`` builds a complete classifier with the AlexNet layout:
five convolutional cores, max pooling, adaptive average pooling and three
feedforward structured E/I blocks, the last of which has one excitatory soma
per class. Every block uses the structured sections for its dendritic options
and takes its population sizes from ``model.core.vision``; writing
``architecture.excitatory_layer_sizes`` or ``inhibitory_layer_sizes`` raises.
Set ``data.processing.flatten: false`` so that images keep their
``(B, C, H, W)`` shape; the network takes the number of image channels from
them (an explicit ``vision.input_channels`` must match). Use the ``identity``
decoder so the class somas are the logits;
with an identity decoder the classifier learns a positive scale per class
(:ref:`output-scale`). The example below is a reduced network for 32 x 32
images.

.. list-table::
   :header-rows: 1
   :widths: 30 22 48

   * - ``model.core.vision`` field
     - Default
     - Meaning
   * - ``channels``
     - ``[64, 192, 384, 256, 256]``
     - Output channels of the five convolutional cores.
   * - ``kernels``, ``strides``, ``paddings``
     - ``[11, 5, 3, 3, 3]``, ``[4, 1, 1, 1, 1]``, ``[2, 2, 1, 1, 1]``
     - Patch geometry of the five convolutional cores.
   * - ``pool_after``
     - ``[0, 1, 4]``
     - Convolutional cores (0-4) followed by max pooling.
   * - ``pool_kernel``, ``pool_stride``
     - ``3``, ``2``
     - Max-pooling window and stride.
   * - ``adaptive_pool_shape``
     - ``[6, 6]``
     - Output size of the adaptive average pooling before the classifier.
   * - ``classifier_widths``
     - ``[4096, 4096]``
     - Widths of the two hidden feedforward blocks.
   * - ``num_classes``
     - ``1000``
     - Somas of the final block (at least 2).
   * - ``inhibitory_ratio``
     - ``0.2``
     - Inhibitory cells per block as a fraction of its excitatory cells
       (at least one; ``0`` builds none).
   * - ``dropout``
     - ``0.5``
     - Dropout before each hidden feedforward block.
   * - ``input_transform``
     - ``signed_split``
     - Input adapter of the image (``identity``, ``relu`` or
       ``signed_split``).
   * - ``input_channels``
     - the image channels
     - Number of image channels. A classifier built from a configuration
       takes it from ``model.encoder.params.input_shape``, which
       ``dendritic-train`` sets from the data; a value written here must
       match. A direct ``get_architecture`` call without it uses
       ``input_dim``.
   * - ``seed``
     - ``null``
     - Base seed of the blocks' parameter initialization (block ``i`` uses
       ``seed + i``); unset, ``model.core.initialization_seed`` is used, and
       without either the global random stream. Block ``i`` samples its
       synapses from ``sparsity.indexed.seed + i``.
   * - ``output_adapter``
     - ``{mode: identity}``
     - Output adapter after each convolutional core and hidden block:
       ``mode``, ``spatial_initial_scale`` and ``classifier_initial_scale``
       (one value, or five and two values), and
       ``spatial_initial_threshold`` and ``classifier_initial_threshold``
       (default ``0.25``).
   * - ``output_organization``
     - ``{mode: one_to_one}``
     - Somas per output channel or unit: ``spatial_somas_per_channel`` and
       ``classifier_somas_per_unit`` (one value, or five and two values),
       reduced by ``mode``: ``one_to_one`` (one soma each), ``fixed_mean``
       or ``learned_positive`` (a learned positive weighting within each
       group).

.. code-block:: yaml

   data:
     dataset_name: cifar10
     processing: {flatten: false}
   model:
     encoder:
       type: identity
     core:
       type: dendritic_alexnet
       architecture:
         excitatory_branch_factors: [2]
         inhibitory_branch_factors: [1]
       connectivity:
         ee_synapses_per_branch_per_layer: [8]
         ei_synapses_per_branch_per_layer: [8]
         ie_synapses_per_branch_per_layer: [2]
         ii_synapses_per_branch_per_layer: [0]
       transfer:
         output_activation: relu
       vision:
         input_channels: 3
         channels: [8, 16, 16, 16, 8]
         kernels: [3, 3, 3, 3, 3]
         strides: [1, 1, 1, 1, 1]
         paddings: [1, 1, 1, 1, 1]
         pool_kernel: 2
         pool_stride: 2
         adaptive_pool_shape: [2, 2]
         classifier_widths: [32, 32]
         num_classes: 10
         output_adapter: {mode: learned_per_channel}
     decoder:
       type: identity

.. _structured-ei-reference:

Configuration reference
-----------------------

The tables list every field of the structured sections with its type and
default. ``model.core.transfer`` is described in :ref:`transfer-layer`,
``model.core.recurrent_ei`` in :ref:`recurrent-structured-cores` and
``model.core.spatial`` in :ref:`spatial-cores`. A population network sets the
same options per population (:ref:`population-options`). The common fields
``model.core.type``, ``initialization_seed`` and ``biological_neuron`` apply
to every core (:ref:`random-seeds`, :ref:`biological-constraints`).

``model.core.architecture``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 12 16 42

   * - Field
     - Type
     - Default
     - Meaning
   * - ``excitatory_layer_sizes``
     - list of int
     - ``[20]``
     - Excitatory cells per layer; the last entry is the core output size.
   * - ``inhibitory_layer_sizes``
     - list of int
     - ``[]``
     - Inhibitory cells per layer: empty, or one positive size per layer
       (:ref:`inhibitory-wiring`).
   * - ``excitatory_branch_factors``
     - list of int
     - ``[3, 3, 3, 3]``
     - Dendritic tree of the excitatory cells, shared by all layers
       (:ref:`dendritic-tree`).
   * - ``inhibitory_branch_factors``
     - list of int
     - ``[1]``
     - Dendritic tree of the inhibitory cells.
   * - ``input_projection_dims``
     - list of int
     - ``[]``
     - Recurrent cores only: dense ReLU layers applied to the excitatory input
       before the first layer.
   * - ``inhibitory_network_type``
     - str
     - ``dendritic``
     - ``dendritic`` or ``mlp`` (feedforward cores only).
   * - ``mlp_inhibitory_network_params``
     - mapping
     - ``{hidden_dims: [200], activation: relu}``
     - Hidden layers and activation of ``mlp`` inhibitory cells; other keys
       raise.

``model.core.connectivity``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Counts are synapses per dendritic branch, one value per layer, with the last
value repeated for later layers. Write at most one value per layer: longer
lists raise when the config loads, for every core type built from these
sections (the structured E/I types, the spatial cores and the controls
matched to a structured E/I network).

.. list-table::
   :header-rows: 1
   :widths: 34 12 12 42

   * - Field
     - Type
     - Default
     - Meaning
   * - ``ee_synapses_per_branch_per_layer``
     - list of int
     - ``[40]``
     - Excitatory input (transferred stream or previous layer) onto
       excitatory cells.
   * - ``ei_synapses_per_branch_per_layer``
     - list of int
     - ``[0]``
     - Excitatory input onto inhibitory cells.
   * - ``ie_synapses_per_branch_per_layer``
     - list of int
     - ``[10]``
     - Inhibitory input (own layer's inhibitory cells, or the transferred
       stream) onto excitatory cells.
   * - ``ii_synapses_per_branch_per_layer``
     - list of int
     - ``[0]``
     - Feedforward inhibitory input onto inhibitory cells.
   * - ``rec_ee_synapses_per_branch``, ``rec_ie_synapses_per_branch``,
       ``rec_ei_synapses_per_branch``, ``rec_ii_synapses_per_branch``
     - list of int
     - ``[0]``
     - Recurrent synapses (:ref:`recurrent-structured-cores`).
   * - ``structured``
     - mapping
     - disabled
     - Fixed topology masks per pathway (:ref:`structured-connectivity`).

``model.core.morphology``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 12 18 40

   * - Field
     - Type
     - Default
     - Meaning
   * - ``somatic_synapses``
     - bool
     - ``false``
     - Give the soma level its own synapses. A cell without dendritic levels
       (``branch_factors: []``) has no other input and needs ``true``.
   * - ``use_shunting``
     - bool
     - ``true``
     - Shunting (``true``) or additive integration; fixed by most core types
       (see above).
   * - ``use_additive_normalization``
     - bool
     - ``false``
     - Normalize each sample's raw additive voltage across the layer's
       outputs (zero mean, unit standard deviation); additive layers only.
   * - ``leak_conductance``
     - float
     - ``1.0``
     - Leak term of the shunting denominator (:ref:`conductance-options`).
   * - ``learnable_leak``
     - bool
     - ``false``
     - Learn the leak of every compartment (shunting only).
   * - ``signal_rounding``
     - str
     - ``none``
     - ``none``, ``float16`` or ``bfloat16`` rounding of signals and
       conductances (:ref:`conductance-options`).
   * - ``additive_mode``
     - str
     - ``raw``
     - Additive voltage rule: ``raw``, ``conductance_normalized`` or
       ``tangent_matched`` (:ref:`integration-modes`); shunting layers need
       ``raw``.
   * - ``additive_tangent_n0``, ``additive_tangent_t0``
     - float or null
     - ``null``
     - Anchors of ``tangent_matched`` (:ref:`integration-modes`).
   * - ``additive_tangent_n0_by_depth``, ``additive_tangent_t0_by_depth``
     - list of float
     - ``[]``
     - Anchors of ``tangent_matched`` per depth, soma first.
   * - ``allow_self_recurrence``
     - bool
     - ``true``
     - Recurrent cores: ``false`` removes each neuron's synapse onto itself
       from the ``rec_ee`` and ``rec_ii`` banks.
   * - ``weight_transform``
     - str
     - ``softplus``
     - Map from stored parameters to synaptic weights
       (:ref:`weight-transforms`).
   * - ``dbl_init_method``
     - str
     - ``analytical_expectation``
     - Initializer of the branch-layer synaptic weights and child couplings
       (:ref:`synapse-initialization`).

``model.core.sparsity``
~~~~~~~~~~~~~~~~~~~~~~~

:ref:`synapse-variants` describes each sparse synapse type and its options.

.. list-table::
   :header-rows: 1
   :widths: 24 12 16 48

   * - Field
     - Type
     - Default
     - Meaning
   * - ``type``
     - str
     - ``standard``
     - Sparse synapse type (:ref:`synapse-variants`).
   * - ``init_method``
     - str
     - ``xavier_normal``
     - Constructor initializer of the sparse weights. ``dbl_init_method``
       overwrites them, so this changes only the discarded draw (and, without
       ``initialization_seed``, later draws of the global random stream).
       ``indexed_rewire`` regrows a contact with the weight of the contact it
       replaces.
   * - ``noise_level``
     - float
     - ``0.0``
     - Noise added to the selection scores during training; ``indexed``,
       ``indexed_rewire``, ``indexed_release`` and the DeepST types reject a
       nonzero value.
   * - ``weight_norm_order``
     - int or null
     - ``null``
     - L-p normalization of each output's selected weights; ``stochastic``,
       ``variance`` and the DeepST types reject it.
   * - ``gamma``
     - float
     - ``1.0``
     - Norm of the normalized inhibitory weights (excitatory weights are
       normalized to 1); used with ``weight_norm_order``.
   * - ``temperature``
     - float
     - ``0.5``
     - Stochastic selection: noise scale (``ultrafast: true``) or sampling
       temperature (``ultrafast: false``, which rejects 0).
   * - ``ultrafast``
     - bool
     - ``true``
     - Stochastic selection: noisy Top-K (``true``) or rank-probabilistic
       sampling (``false``).
   * - ``gradient_scaling``
     - str
     - ``none``
     - ``none`` or ``conductance_dynamic``: in shunting layers, scale the
       synaptic weight gradients by the compartment's mean total conductance.
   * - ``variance_score``
     - str
     - ``absolute``
     - ``variance`` type only: rank inputs by their variance (``absolute``)
       or by ``Var[x] / E[x^2]`` (``relative``).
   * - ``indexed``
     - mapping
     - see below
     - Options of the indexed types and of ``indexed_release``.
   * - ``deepst``
     - mapping
     - see below
     - Options of ``deepst`` and ``credit_deepst``.
   * - ``dense_to_sparse``
     - mapping
     - see below
     - Options of ``dense_to_sparse``.

The nested blocks accept only their own keys; other keys raise when the
config loads. Their keys and defaults are listed with the synapse types:
``indexed`` in :ref:`indexed-options` (the ``release_*`` keys of
``indexed_release`` in :ref:`conductance-indexed-release`), ``deepst`` in
:ref:`synapse-deepst` and ``dense_to_sparse`` in
:ref:`synapse-dense-to-sparse`.

Training fills an unset ``indexed.seed`` from the experiment's topology seed.

``model.core.reactivation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The reactivation is the nonlinearity (gate) applied to every compartment's
voltage (:ref:`activation-types`, :ref:`gate-initialization`).

.. list-table::
   :header-rows: 1
   :widths: 30 12 14 44

   * - Field
     - Type
     - Default
     - Meaning
   * - ``enabled``
     - bool
     - ``true``
     - Apply the gate; ``false`` passes each compartment's voltage on.
   * - ``type``
     - str
     - ``param_tanh``
     - Gate function (:ref:`activation-types`).
   * - ``dendritic_activation``
     - str or null
     - ``null``
     - Alternative spelling of ``type``; ``none`` or ``identity`` sets
       ``enabled: false``. A value that contradicts ``type`` or ``enabled``
       raises.
   * - ``soma_type``
     - str or null
     - ``null``
     - Feedforward cores: a different gate for the soma stage of each
       excitatory ``DendriNet``; inhibitory cells keep ``type``.
   * - ``soma_init_m``, ``soma_init_b``
     - float
     - ``1.0``, ``0.0``
     - Initial slope and midpoint of the ``soma_type`` gate.
   * - ``init_policy``
     - str
     - ``analytical``
     - How the gate slope and midpoint start (:ref:`gate-initialization`).
   * - ``init_m``, ``init_b``
     - float
     - ``1.5``, ``0.5``
     - Start values used by ``init_policy: fixed``; other policies set them,
       and a written value warns.
   * - ``occupancy_quantile_low``, ``occupancy_quantile_high``
     - float or null
     - ``null``
     - Voltage quantiles matched by ``occupancy_quantile`` (0.1 and 0.9 when
       null).
   * - ``occupancy_target_low``, ``occupancy_target_high``
     - float or null
     - ``null``
     - Gate outputs those quantiles map to (0.1 and 0.9 when null).
   * - ``calibration_min_quantile_width``
     - float
     - ``0.001``
     - Smallest voltage quantile span used to fit the slope.
   * - ``calibration_max_m``
     - float
     - ``50.0``
     - Largest slope a calibration sets.
   * - ``calibration_revert_on_invalid``
     - bool
     - ``true``
     - ``true``: a calibration whose quantile span is below
       ``calibration_min_quantile_width``, or whose slope is invalid or above
       ``calibration_max_m``, keeps the layer's previous gate. ``false``:
       the span is raised to the floor and the slope clipped.
   * - ``sigma_aware_k``
     - float
     - ``0.25``
     - ``empirical`` policy: factor from the robust voltage spread to the
       slope.
   * - ``gradient_scaling``
     - str
     - ``none``
     - ``none`` or ``straight_through`` (shunting layers, training mode: the
       gate passes its output gradient to its input).
   * - ``memory_efficient``
     - bool
     - ``false``
     - ``param_tanh`` and ``param_tanh_only_m``: a backward pass that keeps
       fewer tensors, with the same outputs.

``model.core.blocklinear``
~~~~~~~~~~~~~~~~~~~~~~~~~~

``BlockLinear`` is the coupling from child branches to their parent
compartment.

.. list-table::
   :header-rows: 1
   :widths: 24 12 12 52

   * - Field
     - Type
     - Default
     - Meaning
   * - ``gradient_scaling``
     - str
     - ``none``
     - ``none``; ``conductance_dynamic`` (a modified backward rule, below:
       shunting layers scale by the compartment's total conductance,
       additive layers divide each child's gradient by its coupling weight);
       or ``distal_upweight_by_idx`` (multiply the coupling gradients by the
       branch level's depth from the soma plus one).
   * - ``efficient``
     - bool
     - ``false``
     - Use ``EfficientBlockLinear``, which gives the same output without
       building the dense block-diagonal weight matrix.

.. _blocklinear-gradient-scaling:

``gradient_scaling: conductance_dynamic`` replaces exact backpropagation
through ``BlockLinear`` by a modified backward rule in training mode. The
forward pass is unchanged, and gradients in evaluation mode are exact. Let
:math:`w_{ij}` be the coupling of child :math:`j` to compartment :math:`i`
and :math:`\bar g_i` the batch mean of the compartment's shunting
denominator (leak plus all conductances):

- In a shunting layer the gradient of :math:`w_{ij}` is multiplied by
  :math:`\bar g_i`, and the gradient passed to child :math:`j` by
  :math:`\bar g_i / \max(w_{ij}, \epsilon)`.
- An additive layer has no such denominator. The gradient of :math:`w_{ij}`
  is left unchanged, and the gradient passed to child :math:`j` is divided by
  :math:`w_{ij}` (by :math:`|w_{ij}| + \epsilon` with ``identity`` weights),
  so each child receives its parent's gradient as if its coupling were 1.
  ``relu`` weights raise, because their couplings can be zero; very small
  ``exp`` or ``softplus`` couplings give correspondingly large gradients
  (infinite below about :math:`10^{-38}` in float32).

:math:`\epsilon = 10^{-8}` (the population field ``epsilon``). A layer run
once per forward pass applies the scales with backward hooks on
``BlockLinear``; a population stepped through time registers them on the
tensors of each step, so every step uses its own scales.

``model.core.implementation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 32 12 22 34

   * - Field
     - Type
     - Default
     - Meaning
   * - ``adaptive_initialization``
     - bool
     - ``true``
     - ``analytical_expectation`` initialization: lower the expected
       conductances when their estimated total exceeds
       ``adaptive_target_conductance``.
   * - ``adaptive_initialization_policy``
     - str
     - ``preserve_shunting_center``
     - ``preserve_shunting_center`` keeps a shunting layer's initial voltage
       operating point while lowering its conductance; ``legacy_scale``
       multiplies all expected conductances by one factor.
   * - ``adaptive_target_conductance``
     - float
     - ``5.0``
     - Target total conductance of the adaptive initialization.
   * - ``initial_child_conductance``
     - float
     - ``1.0``
     - Initial child-to-parent coupling; other values need
       ``dbl_init_method`` ``mechanism_neutral`` or ``fan_in_lognormal``
       (:ref:`conductance-options`).
   * - ``initial_conductance_scale``
     - float
     - ``1.0``
     - Factor on the initialized conductances; other values need
       ``fan_in_lognormal``.
   * - ``initial_child_conductance_log_sd``
     - float
     - ``0.0``
     - Log-normal spread of the child couplings; positive values need
       ``fan_in_lognormal``.
   * - ``print_hooks``
     - bool
     - ``false``
     - Log gradient statistics in the gradient-scaling hooks.
   * - ``store_routing``
     - bool
     - ``false``
     - Recurrent cores: record each layer's per-level routing at every
       timestep for analysis.
   * - ``compile_forward``
     - bool
     - ``false``
     - Feedforward cores: compile the forward pass with ``torch.compile``.
       Shunting cores with ``blocklinear.gradient_scaling:
       conductance_dynamic`` reject it, because compilation fails on those
       gradient hooks at the first training step.
