.. _baseline-models:

Baselines and comparison models
===============================

DendriNet includes conventional networks to compare with dendritic cores:
point-neuron controls matched in size to a dendritic reference, dense and
sparse affine MLPs, and recurrent baselines. They are core types, selected
with ``model.core.type`` like the dendritic cores (:ref:`core-types` lists
every type), and they train with the same training strategies. The second
half of the page describes Python utilities for comparisons at a fixed
parameter budget and a few numerical helpers.

.. _matched-controls:

Controls matched to a dendritic reference
-----------------------------------------

These core types build a point-neuron network whose size follows a dendritic
reference core. The reference is described by the same sections a dendritic
core uses: the structured E/I sections ``model.core.architecture`` and
``model.core.connectivity`` (with ``morphology``, ``sparsity`` and the other
structured sections; see :doc:`structured_ei_cores`), or, where the table
says so, a ``model.core.population_network`` block. The builder constructs the
reference only to count it; the control contains no dendritic operations.

.. list-table::
   :header-rows: 1
   :widths: 24 50 26

   * - ``model.core.type``
     - Network
     - Reference
   * - ``point_mlp``
     - One hidden layer of point neurons, ``Linear -> activation -> Linear``,
       with the smallest hidden width whose parameter count reaches the
       reference's effective parameter count: its active synaptic contacts
       after Top-K selection plus its dendritic couplings and gate
       parameters. The output width is the reference's output width.
     - Structured E/I sections or ``population_network``
   * - ``total_param_mlp``
     - With a structured reference: a dense MLP with as many linear layers as
       the reference has dendritic branch layers in total (E and I trees of
       every layer), equal hidden widths, a sigmoid after every layer and at
       least as many weights as the reference has trainable parameters. With
       a ``population_network`` reference: the one-hidden-layer ReLU network
       of ``point_mlp``, matched to the reference's trainable parameter
       count.
     - Structured E/I sections or ``population_network``
   * - ``active_param_mlp``
     - As ``total_param_mlp``, matched to the reference's active synapses,
       dendritic couplings and gate parameters instead of its total parameter
       count. With a ``population_network`` reference it is the same network
       as ``point_mlp``.
     - Structured E/I sections or ``population_network``
   * - ``ss_mlp``
     - A sparse structured MLP: the last layer of the reference rebuilt as
       signed Top-K synapses on the same branch structure, with signed branch
       couplings and ``param_tanh`` gates, without E/I sign constraints or
       shunting.
     - Structured E/I sections
   * - ``ss_mlp_flat``
     - ``ss_mlp`` built from the flattened reference (all branch compartments
       attached directly to the soma).
     - Structured E/I sections
   * - ``sparse_active_matched_point``
     - One fixed-index sparse affine layer followed by ReLU. Every output is
       active; the weights spread as evenly as possible over the outputs so
       that weights plus biases equal the reference's effective parameter
       count. Needs ``model.core.sparsity.indexed.seed``, which fixes the
       support.
     - Structured E/I sections
   * - ``direct_active_matched_point``
     - One dense ``Linear -> ReLU`` bottleneck whose width matches the
       reference's effective parameter count (within 1 %), zero-padded to
       the reference's output width. The width must fit in that output width.
     - Structured E/I sections
   * - ``sparse_active_matched_point_conv`` (alias
       ``spatial_sparse_active_matched_point``)
     - ``sparse_active_matched_point`` applied to every local patch of a
       ``(batch, channels, height, width)`` input, with the patch geometry of
       ``model.core.spatial`` (``kernel_size``, ``stride``, ``padding``).
     - Structured E/I sections and ``model.core.spatial``

``model.core.target_active_parameters`` replaces the count taken from the
reference for ``sparse_active_matched_point``. Matching parameter counts does
not equalize compute or activation memory.

A ``point_mlp`` control for the population network of
``configs/examples/population_network_feedforward.yaml`` repeats the
population block under ``type: point_mlp``:

.. code-block:: yaml

   model:
     core:
       type: point_mlp
       population_network:
         layers:
           - name: layer0
             readout_population: e
             population_defaults:
               ff_excitatory_synapses: 30
               ff_inhibitory_synapses: 15
             populations:
               - {name: i, polarity: inhibitory, n_neurons: 32, branch_factors: [1]}
               - {name: e, polarity: excitatory, n_neurons: 128, branch_factors: [4, 4]}
             connections:
               - {source: input, target: i}
               - {source: input, target: e}
               - {source: i, target: e}

A fixed-support control for a structured E/I core gives the reference
sections and the support seed:

.. code-block:: yaml

   model:
     core:
       type: sparse_active_matched_point
       architecture:
         excitatory_layer_sizes: [64]
         inhibitory_layer_sizes: [16]
         excitatory_branch_factors: [2, 2]
         inhibitory_branch_factors: [1]
       connectivity:
         ee_synapses_per_branch_per_layer: [16]
         ei_synapses_per_branch_per_layer: [8]
         ie_synapses_per_branch_per_layer: [4]
         ii_synapses_per_branch_per_layer: [0]
       sparsity:
         indexed:
           seed: 11

.. _affine-mlp:

Dense and sparse affine MLPs
----------------------------

``model.core.type: affine_mlp`` builds a conventional multilayer perceptron
with signed weights, in one of three modes, from
``model.core.affine_mlp``:

``dense``
   Dense ``nn.Linear`` layers.
``fixed``
   Sparse layers with a fixed random support of ``fanin`` inputs per unit,
   stored as indices and weights (``IndexedSparseLinear``). Each layer's
   support comes from its entry in ``connectivity_seeds``.
``topk``
   Learned Top-K layers (``TopKLinear``): every unit uses its ``fanin``
   weights of largest magnitude in each forward pass.

.. code-block:: yaml

   model:
     encoder:
       type: identity
       params:
         input_dim: 784
     core:
       type: affine_mlp
       affine_mlp:
         dimensions: [784, 256, 128]
         mode: fixed
         fanin: [64, 32]
         connectivity_seeds: [101, 102]
         dropout: 0.1
     decoder:
       type: MLP
       params:
         hidden_dims: []
         output_dim: 10

The options of ``model.core.affine_mlp``:

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Option
     - Default
     - Meaning
   * - ``dimensions``
     - (required)
     - Widths from input to output, at least two. The first must equal the
       encoder output size.
   * - ``mode``
     - ``dense``
     - ``dense``, ``fixed`` or ``topk``.
   * - ``fanin``
     - ``[]``
     - Inputs per unit for each layer (``fixed`` and ``topk``); not allowed
       in ``dense`` mode.
   * - ``connectivity_seeds``
     - ``[]``
     - One support seed per layer, required by ``fixed`` and rejected by the
       other modes.
   * - ``initialization``
     - ``kaiming_normal``
     - ``kaiming_normal``, or ``kaiming_uniform_xavier_readout`` (``dense``
       only: Kaiming-uniform hidden layers and a Xavier-uniform last layer).
   * - ``activation``
     - ``relu``
     - ``relu`` or ``leaky_relu``.
   * - ``negative_slope``
     - ``0.1``
     - Slope of ``leaky_relu`` for negative inputs.
   * - ``activate_output``
     - ``true``
     - Apply the activation (and dropout) after the last layer too. Set it to
       ``false`` when the last layer produces classifier logits.
   * - ``dropout``
     - ``null``
     - Dropout probability after each activation. ``null`` adds no dropout
       modules; ``0.0`` adds modules that do nothing, which keeps the module
       layout (and checkpoint keys) of a run with dropout.
   * - ``output_weight_scale``
     - ``1.0``
     - Factor applied to the last layer's initial weights.
   * - ``projection_backend``, ``workspace_mb``, ``index_dtype``
     - ``eager``, ``128.0``, ``int32``
     - Execution settings of ``fixed`` layers (see :ref:`indexed-backends`);
       the other modes do not read them.

Initialization: ``dense`` mode constructs all layers, then draws every weight
with Kaiming-normal initialization (the ReLU gain, also for ``leaky_relu``)
and sets the biases to zero. ``fixed`` layers draw normal weights with
standard deviation ``sqrt(2 / fanin)``. ``topk`` layers draw standard-normal
weights and rescale each unit's weights so that its initially selected weights
have squared norm 2. All modes then multiply the last layer's weights by
``output_weight_scale``. The weights come from the global PyTorch random
stream, which the builder does not reseed; ``connectivity_seeds`` fix only the
support of ``fixed`` layers.

The built model is a flat ``nn.Sequential`` (numeric checkpoint keys) that
also forwards the weight decay, weight boosting and rewiring steps of the
optimizer to its sparse layers. The same model can be built from Python:

.. code-block:: python

   from dendritic_modeling.config import AffineMLPConfig
   from dendritic_modeling.networks.architectures.factory import get_architecture

   config = AffineMLPConfig(
       dimensions=[128, 64, 10],
       mode="fixed",
       fanin=[16, 8],
       connectivity_seeds=[101, 10101],
       activate_output=False,
   )
   model = get_architecture("affine_mlp", config, input_dim=128)
   resolved = model.affine_mlp_config  # the configuration as a dictionary

``dendritic_modeling.networks.architectures.classical.mlp.build_affine_mlp(config)``
returns the same model. To count the parameters of Top-K models,
``dendritic_modeling.analysis.utils.model_resources.topk_parameter_inventory(model)``
returns the trainable parameters, the Top-K candidate weights, the parameters
retained after Top-K selection and the buffer bytes; it counts parameters and
does not estimate memory or speed. ``topk`` layers can be converted to fixed
indexed layers for inference; see :ref:`compact-checkpoints`.

.. _baseline-recurrent-cores:

Recurrent baselines
-------------------

``gru``, ``lstm``, ``vanilla_rnn`` (alias ``rnn``)
   Standard PyTorch recurrent networks configured by
   ``model.core.baseline_rnn``: ``hidden_dim`` (default 64), ``num_layers``,
   ``dropout``, ``output_mode`` (``last``, ``mean`` or ``all`` time steps),
   ``input_projection_dims`` (linear and ReLU layers before the recurrence),
   ``use_layernorm`` (layer normalization of the recurrence input and
   output), ``init_method`` (``default`` or ``orthogonal`` recurrent
   weights), ``lstm_forget_bias`` and the GRU update-gate biases
   ``gru_fast_update_bias``, ``gru_slow_update_bias`` and
   ``gru_slow_fraction``.

``heterogeneous_leak_ctrnn`` (alias ``heterogeneous_ctrnn``)
   A continuous-time point-neuron E/I network with sparse indexed contacts
   and leaky channels of several time constants per output neuron,
   configured by ``model.core.heterogeneous_leak_ctrnn``. ``tau_values``
   lists the time constants, ``owner_tau_patterns`` assigns them to the
   channels of each output neuron, and the ``*_contacts`` fields set the
   numbers of input, recurrent and readout contacts. The class
   ``HeterogeneousLeakCTRNNConfig`` in
   ``dendritic_modeling.networks.architectures.recurrent`` checks that these
   counts agree.

``legendre_memory``
   The fixed linear memory of a Legendre delay network: each memory channel
   holds ``memory_order`` Legendre coefficients of its input over a sliding
   window of length ``theta`` (time step ``dt``). Sparse learned contacts
   write the input into the channels and read the coefficients out. It is
   configured by ``model.core.legendre_memory``; the state transition is
   fixed, not trained.

The keys of these sections are checked when the model is built, and a
misspelled one raises an error that names the section and suggests the
closest key. ``input_dim`` is set from the encoder output and the cell of
``baseline_rnn`` from the core type: a written ``input_dim`` that differs
from the encoder output, or a ``cell_type`` other than the core type's cell,
raises.

Recurrent cores read sequences and train with
``training.main.strategy: recurrent`` (see :ref:`training-strategies`). A GRU
baseline:

.. code-block:: yaml

   model:
     core:
       type: gru
       baseline_rnn:
         hidden_dim: 128
         num_layers: 1
         output_mode: last

A small heterogeneous-leak network, with two time constants per output neuron:

.. code-block:: yaml

   model:
     core:
       type: heterogeneous_leak_ctrnn
       heterogeneous_leak_ctrnn:
         input_projection_dims: [3]
         n_excitatory_outputs: 2
         n_inhibitory_outputs: 1
         channels_per_output: 4
         tau_values: [9.0, 3.0]
         owner_tau_patterns:
           - {owner_count: 3, tau_counts: [2, 2], excitatory_owner_count: 2}
         input_contacts: 30
         recurrent_excitatory_contacts: 18
         recurrent_inhibitory_contacts: 9
         readout_contacts: 4
         topology_seed: 711
         initialization_seed: 919

A Legendre memory with two memory channels of order 4 for sequences of three
input features. Each contact count lies between the larger and the product of
the two widths it connects: ``input_contacts`` between the input features and
the memory channels, ``readout_contacts`` between the
``n_memory_channels * memory_order`` coefficients and ``output_dim``.
``dendritic-train`` sets ``model.encoder.params.input_dim`` from the data:

.. code-block:: yaml

   model:
     encoder:
       type: identity
       params:
         input_dim: 3
     core:
       type: legendre_memory
       legendre_memory:
         output_dim: 2
         memory_order: 4
         n_memory_channels: 2
         theta: 8.0
         input_contacts: 5
         readout_contacts: 9

The recurrent E/I cores themselves are described in
:ref:`recurrent-structured-cores`, :ref:`unified-ei` and
:doc:`population_networks`.

MLP building blocks
-------------------

``mlp`` (the decoder MLP) and ``effectivematchedparammlp`` are building
blocks that take flat parameters, not core types: ``model.core.type`` set to
one of them raises an error that names the alternatives. For a plain MLP core
use ``affine_mlp``; for a matched control use ``point_mlp``.
:ref:`core-types` lists the other names of this kind.

Shared projections and native shunting banks
--------------------------------------------

``shared_projection_shunting`` composes existing library operators: one
normalized positive indexed projection per group, a flat native ``DendriNet``
bank for each projected scalar, signed ``BlockLinear`` pooling, and an affine
output. It implements ``v[j] = sum_i theta[j,i] x[i]`` followed by
``h[j] = sum_k a[j,k] v[j]/(v[j]+z[j,k])`` and ``y = C h + b``.
The linear group pool is external to the native compartments; this is not a
tree whose soma applies an additional conductance division or activation.

.. code-block:: python

   from dendritic_modeling.config import SharedProjectionShuntingConfig
   from dendritic_modeling.networks.architectures.excitation_inhibition.shared_projection import (
       build_shared_projection_shunting,
   )

   config = SharedProjectionShuntingConfig(
       input_dim=6, supports=[[0, 1], [3, 4]],
       nodes=[0.001, 0.01, 0.1, 1.0], dtype="float64",
       train_projection=False, train_conductance=False, train_output=False,
   )
   model = build_shared_projection_shunting(config)
   report = model.resource_report()

The same payload can be passed to ``get_architecture`` or written under
``model.core.shared_projection_shunting`` with core type
``shared_projection_shunting``. Unknown configuration keys, unused core
sections, invalid supports, and non-standard trainer strategies raise.
Inputs must be finite and nonnegative. Supports have equal contact counts;
overlap is supported but violates the disjoint-support assumption of some
theory applications. The eager backend and ordinary backpropagation are used.
CPU float32/float64 equations and gradients are tested; accelerated backends,
local learning, recurrent use and export formats are not qualified here.

Construction preserves the caller's Torch RNG. Projection weights start
uniform, branch pooling weights start at ``1/W``, output weights at ``1/q``,
and the bias at zero. Native compartments have unit leak and no extra gate.
Their positive conductance is initialized to ``(1+epsilon)/z`` so the native
denominator, including epsilon, realizes the declared transition node. The
conductance parameters are logarithmic, using the existing bounded exponential
transform. This parameterization differs from historical softplus scalar runs;
it is not a silent migration of those results. Numerical underflow or transform
saturation can limit extreme learned scales.

``features(x)`` returns the actual native features in group-major order.
``effective_nodes()`` measures current transition scales. With ``q`` groups,
``s`` contacts, ``W`` nodes and ``o`` outputs, stored scalar slots number
``q*s + 2*q*W + q*o + o``, including frozen parameters. Freezing output weights
leaves the intercept and group pooling trainable for linear readout fitting.
For one output this is ``q*(s+2*W+1)+1``. Support buffers and all native buffers
are reported separately; the scalar specialization does not automatically
remove its stored projection and outer weights. Arithmetic counts describe
sparse MACs and shunting divisions, excluding weight transforms, normalization,
checks and training overhead; they are not measured FLOPs or latency.

Paired E/I routing controls
---------------------------

``core.type: routed_ei`` builds a reduced rate-model control from
``core.routed_ei``. Each source has an excitatory and an inhibitory channel;
one assignment sends both channels to the same branch. Positive E and I
weights are normalized separately within each branch. The branches use
``networks.functional.shunting_voltage`` or ``additive_voltage``, followed by
a fixed affine rescaling and a learned affine decoder. The decoder has signed
weights: this control is not a biological population network or a multilevel
dendritic tree.

This composition reuses the library's softplus weight transform, row
normalization, current integration, architecture factory and standard trainer.
It differs from ``pathway_router``, which applies signed projections, and from
Top-K synapses, which select contacts within a branch rather than assign each
source to a destination. Both routers share the same soft-assignment utility.

``configs/examples/routed_ei.yaml`` runs with ``dendritic-train`` and reuses
the existing ``branch_local_gain_load`` dataset. Validation uses the training
gain distribution; only the test set has the larger gain fluctuations. This
binary example is not a replay of a paper's multiclass routing task. Other
datasets must supply finite nonnegative concatenated inputs ``[E, I]`` of
width ``2 * features``. Direct callers can supply the streams separately:

.. code-block:: python

   from dendritic_modeling.config import RoutedEIConfig
   from dendritic_modeling.networks.architectures.excitation_inhibition.routed_ei import (
       build_routed_ei,
   )

   config = RoutedEIConfig(features=8, branches=3, output_dim=2)
   model = build_routed_ei(config)
   logits = model(excitation, inhibition)  # Each input: [batch, 8].

``routing: learned`` uses a softmax over branch destinations. With
``hard_routing: true`` (the default), training uses one-hot assignments with
the softmax surrogate derivative; evaluation uses exact one-hot assignments.
With ``hard_routing: false``, both use soft assignments. ``routing: fixed``
requires a ``fixed_assignment`` list containing one branch index per feature.
Empty branches have zero drive; no implicit balancing or reassignment occurs.
``inhibitory_support: global_mean`` is a shunting-only control that replaces
each branch's inhibitory drive by the unweighted mean of the input I channels.

Shunting defaults to positive unit leak and epsilon ``1e-8``. Raw addition is
``E - I``; nondefault leak/epsilon values and global-mean support are rejected
because this comparator does not use them. The optional output transformation is
``branch_scale * (voltage - branch_center) / branch_divisor``; its default is
the identity. No gate, across-feature normalization or learned output scale is
implicit in this core. Set the outer model's ``output_scale_mode: fixed``
and use identity encoder/decoder modules to retain that contract.

An explicit ``branch_activation: tanh`` applies
``tanh(softplus(activation_slope_raw) * preactivation + activation_offset)``
before the affine readout. The initial raw slope defaults to ``0.5413249``
(approximately unit softplus slope) and the offset to zero. These per-branch
values are fixed buffers unless ``trainable_activation: true`` is requested.
This signed gate is not the neuronal ``ParametricTanh``: that gate has a
unit-range output, an exponential slope and a voltage midpoint. Substituting
one for the other changes optimization even when an operating-point response
matches. Only the routed backpropagation control supports these new fields.

The default ``branch_activation: none`` registers no gate state and draws no
random numbers. Nondefault gate parameters with this setting raise rather
than being ignored. ``branch_response(E, I)`` explicitly returns currents,
preactivation, activation and the derivative of the gate with respect to its
preactivation. Ordinary forward calls do not calculate this diagnostic.
``activate_branches(x)`` exposes the same configured gate to analysis-only
interventions without reimplementing its arithmetic. The original resolved
configuration and a strict state-dict round trip recover fixed or learned
gates, including their initial parameterization.

``weight_normalization_floor: null`` reuses the existing robust L1 row
normalization. An explicit positive floor selects division by
``max(sum(weights), floor)``. These conventions differ for tiny rows: replaying
an older clamped-sum model requires its original floor, not a silent switch
to the robust default. The fixed-assignment and global-mean controls retain
the unused routing or I-weight parameters and initialization draws; those
parameters receive no gradients. Store the resolved configuration alongside
the state dict, including any scheduled temperature changes.

Only ordinary backpropagation with the standard trainer is qualified.
LocalCA, feedback alignment and recurrent configurations raise during loading.
The affine readout prevents a ``biological_neuron: true`` declaration.
No production paper pin is changed merely by selecting this architecture.

Routing regularization is opt-in under
``training.main.regularization``. ``routing_entropy_weight`` penalizes
mean source-assignment entropy; ``routing_balance_weight`` penalizes mean
squared deviation from equal branch mass. The regularization manager applies
these terms to modules exposing ``routing_statistics()``, currently the paired
E/I and pathway routers. Positive weights request minimization of those
statistics; fixed routing contributes zero. Both weights default to zero,
which performs no routing discovery or computation in existing runs. Requesting
a routing penalty on a model with no routing statistics raises.

Parameter-budget model families
-------------------------------

``dendritic_modeling.networks.benchmarks.parameter_budget`` builds whole
models (input encoding, hidden layers and a counted affine readout) for
comparisons at a fixed parameter budget, from Python. ``FAMILIES`` lists the
families:

``dense``
   Dense hidden layers.
``sparse``
   Fixed-support sparse hidden layers (``contacts_e + contacts_i`` inputs per
   unit).
``dendritic_signed``, ``dendritic_additive``, ``dendritic_shunting``
   ``ConfigurableEINetwork`` trees with signed weights, or positive E/I
   conductances combined additively or by shunting.
``grouped_gated``
   A conventional control without dendritic dynamics: sparse gated subunits
   (as many per unit as the matching dendritic morphology has sites), group
   RMS normalization and a learned sum.

.. code-block:: python

   from dendritic_modeling.networks.benchmarks.parameter_budget import (
       build_model,
       match_parameter_budget,
       model_report,
   )

   spec = {
       "family": "dendritic_shunting",
       "input_dim": 784,
       "output_dim": 10,
       "branch_factors": [2, 2],
       "contacts_e": 8,
       "contacts_i": 2,
   }
   model = build_model({**spec, "width": 64})
   report = model_report(model)  # counted parameters, storage, topology hash
   matched_spec, matched_report = match_parameter_budget(spec, 100_000)
   matched_model = build_model(matched_spec)

``build_model`` seeds its own random stream (``seed`` for parameters,
``topology_seed``, default ``seed``, for the fixed support) and leaves the
caller's generator untouched. Unknown spec keys raise. Further fields:
``width`` (exposed hidden units), ``network_depth``, ``activation`` (``relu``,
``softplus``, ``sigmoid``, or the learned gates ``param_relu`` and
``param_tanh``, which need ``activation_init: {gain, threshold}``),
``input_encoding`` (``signed_split``, ``nonnegative`` or ``raw``; positive
E/I families reject ``raw``), ``initialization`` (``fan_in``, the default, or
``legacy_xavier`` for Xavier-normal initialization), ``clip_contacts`` and
``projection_backend``. An optional ``readout`` selects a signed dense,
``low_rank`` (``rank``) or ``indexed`` (``topk``) output allocation.

``match_parameter_budget`` searches the integer ``width`` whose whole-model
parameter count is within ``tolerance`` (default 2 %) of the target and
raises when no width qualifies; it never pads with inert parameters. Matching
parameters does not equalize compute or activation memory.

Signed readouts
---------------

``dendritic_modeling.networks.utils.readouts.make_signed_readout(input_dim,
output_dim, *, rank=None, topk=None, bias=False, init_std=0.02, seed=None,
projection_backend="eager")`` returns a dense ``nn.Linear`` (no budget), two
dense low-rank factors (``rank``), or an ``IndexedSparseLinear`` with fixed
support of ``topk`` inputs per output. ``rank`` and ``topk`` are mutually
exclusive, and indexed readouts take no bias.

Readout initialization
----------------------

``dendritic_modeling.training.readout_initialization.initialize_readout(model,
training_inputs, spec)`` optionally rescales an existing ``nn.Linear``
``model.readout`` before training:

``{"mode": "preserve"}``
   Leaves the model unchanged.
``{"mode": "train_batch_rms", "target_rms": r, "examples": n}``
   Multiplies the readout weights and bias by one factor so the logits of the
   first ``n`` training inputs have root-mean-square ``r``. Only training
   inputs are used, never labels or validation data; the hidden state is
   verified unchanged by hash, and any failure restores the original state.

It returns a record of the calibration (initial and achieved RMS, scale,
examples used, hashes) to store with the run: it changes the optimization
starting point, not the model class or its parameter count.

Numerical utilities
-------------------

Streaming covariance
~~~~~~~~~~~~~~~~~~~~

``dendritic_modeling.analysis.utils.streaming_covariance.StreamingCovariance(n_features)``
merges the means and centered second moments of successive
``(observations, features)`` batches in float64 without retaining
observations. ``covariance`` returns the sample covariance (``n - 1``
normalization) once two observations have been seen.

Passive branch and soma equilibrium
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.passive_equilibrium.passive_star_soma_voltage(excitation,
membrane_conductance, axial_conductance, soma_leak=1.0)`` returns the
steady-state soma voltage of independent passive stars: branches with input
current ``E``, membrane load ``h > 0`` and axial coupling ``g >= 0`` around a
soma with leak to reversal zero. Each branch settles at

.. math::

   V_b = \frac{E + g\,s}{h + g},

where :math:`s` is the soma voltage, and the final dimension indexes the
branches. The expression keeps a positive denominator, so it stays accurate at
large :math:`g`, and it is differentiable.
