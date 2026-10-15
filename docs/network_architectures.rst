.. _model-overview:

Model architecture
==================

Every DendriNet model has three parts: an encoder that prepares the input, a
core network that does the computation, and a decoder that maps the core
output to the task output. This page describes the three parts, the model
wrapper that is built around them for each task, the families of core
networks, the ``model.core.biological_neuron`` declaration and the output scale
of classifiers. Each core family has its own page, linked from
:ref:`model-core-families`.

.. contents:: On this page
   :local:
   :depth: 1

Encoder, core and decoder
-------------------------

The ``model`` section of a configuration names the three parts:

.. list-table::
   :header-rows: 1
   :widths: 14 26 60

   * - Part
     - Option (default)
     - Types
   * - Encoder
     - ``model.encoder.type`` (``identity``)
     - ``identity`` (alias ``none``) passes the input through.
       ``pathway_router`` (alias ``router``) routes groups of input features
       into separate pathways (``router_mode: fixed`` or ``learned``).
       The autoencoder types ``mlpautoencoder``,
       ``mlpvariationalautoencoder``, ``cnnautoencoder`` and
       ``cnnvariationalautoencoder`` build an autoencoder whose encoder half
       feeds the core, after reconstruction pretraining
       (:ref:`encoder-pretraining`).
   * - Core
     - ``model.core.type`` (``einet``)
     - A dendritic E/I network or a baseline network; see
       :ref:`model-core-families`.
   * - Decoder
     - ``model.decoder.type`` (``MLP``)
     - ``MLP`` (also written ``mlp``) with ``params.hidden_dims`` (default
       ``[16]``) and ``params.output_dim`` (default ``10``);
       ``hidden_dims: []`` gives a linear readout. ``identity`` (alias
       ``none``) passes the core output through.

:ref:`encoder-decoder-types` lists the parameters each encoder and decoder
type reads. The sizes of the three parts are connected automatically:

- The training command sets ``model.encoder.params.input_dim`` and
  ``input_shape`` from the first training sample: ``input_dim`` is the number
  of input elements for a feedforward core and the number of features per
  time step for a recurrent core. Values written in the configuration are
  replaced.
- The core is built for the output size of the encoder.
- The decoder's ``params.input_dim`` is set to the core output size. An
  identity decoder has no parameters; a ``params.output_dim`` written for it
  must equal the core output size.
- For classification, the number of outputs must cover the training labels:
  a label beyond the last output raises, and outputs that no label uses only
  produce a warning.

A complete ``model`` section with a population-network core and a linear
readout:

.. code-block:: yaml

   model:
     task: classification
     encoder:
       type: identity
     core:
       type: population_network
       population_network:
         layers:
           - name: hidden
             readout_population: pyr
             population_defaults:
               ff_excitatory_synapses: 16
               ff_inhibitory_synapses: 4
             populations:
               - {name: pyr, polarity: excitatory, n_neurons: 32, branch_factors: [2, 2]}
               - {name: pv, polarity: inhibitory, n_neurons: 8, branch_factors: [1]}
             connections:
               - {source: input, target: pyr}
               - {source: input, target: pv}
               - {source: pv, target: pyr}
     decoder:
       type: MLP
       params:
         hidden_dims: []      # linear readout
         output_dim: 10

Tasks and model wrappers
------------------------

``model.task`` is ``classification`` (the default) or ``regression``. The
model wrapper, from ``dendritic_modeling.models``, depends on the task and on
whether the core is recurrent:

.. list-table::
   :header-rows: 1
   :widths: 16 48 36

   * - Core
     - ``classification``
     - ``regression``
   * - Feedforward
     - ``Classifier``: returns the logits, multiplied by the output scale
       (:ref:`output-scale`).
     - ``Regressor``: returns the decoder output.
   * - Recurrent
     - ``RecurrentClassifier``: returns ``[batch, classes]`` logits, or
       ``[batch, time, classes]`` when the core returns every time step
       (``output_mode: all``).
     - ``Regressor``

Recurrent cores read input sequences shaped ``[batch, time, features]``; an
encoder other than ``identity`` is applied at every time step. A core is
recurrent when:

- its type is a recurrent baseline: ``gru``, ``lstm``, ``vanilla_rnn`` (alias
  ``rnn``), ``heterogeneous_leak_ctrnn`` (alias ``heterogeneous_ctrnn``) or
  ``legendre_memory`` (:ref:`baseline-models`);
- it is a population network with a recurrent layer, a delayed connection, a
  population with temporal state (spiking soma, receptor synapse types,
  dendritic spikes or soma feedback) or ``output_mode: mean`` or ``all``
  (:doc:`population_networks`, :doc:`temporal_dynamics`);
- it is a unified E/I core with a recurrent layer or recurrent synapses
  (:ref:`unified-ei`);
- it is a structured E/I core with recurrent synapse counts or a stateful
  feature (:ref:`recurrent-structured-cores`).

The training loss is ``training.main.common.loss_function`` (default ``ce``,
cross-entropy); see :ref:`loss-functions`.

.. _initial-checkpoint:

Starting from saved weights
---------------------------

``model.initial_checkpoint`` names a checkpoint file whose weights replace the
initial weights of the built model before training starts:

.. code-block:: yaml

   model:
     initial_checkpoint: results/previous_run/final_model.pt
     initial_checkpoint_sha256: null   # optional: the file's SHA-256, checked before loading
     initial_checkpoint_strict: true   # missing or unexpected keys raise

The file is read with PyTorch's weights-only loader. It may hold the model's
state dictionary itself or a mapping with that dictionary under
``model_state_dict``, ``state_dict`` or ``model``, such as the checkpoints that
training writes; a ``module.`` prefix from DDP is removed. The model must be
built from a configuration that matches the saved weights. With
``initial_checkpoint_strict: false`` missing and unexpected keys are recorded
instead of raising. The run writes ``initial_checkpoint_provenance.json`` with
the resolved path, the file's SHA-256 and the keys that did not match
(:ref:`run-outputs`). Only the weights are loaded: the optimizer state and the
epoch counter start fresh (:ref:`resuming-training` continues a run instead).

.. _model-core-families:

Core families
-------------

The dendritic cores are built from the same parts: dendritic neurons
(:doc:`dendritic_neurons`) that receive sparse excitatory and inhibitory
synapses (:doc:`synapses`), optionally with receptor kinetics, spiking somas,
dendritic plateaus and soma-to-dendrite feedback (:doc:`temporal_dynamics`).
They differ in how populations of these neurons are declared and wired:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Family
     - Description
   * - Population network
     - ``population_network``: layers of named excitatory and inhibitory
       populations with their own morphology and options, connected by
       directed same-step or delayed connections. Built as
       ``PopulationNetwork``. See :doc:`population_networks`.
   * - Structured E/I core
     - ``einet`` and its aliases (``dendritic_shunting``,
       ``dendritic_additive``, ``flat_shunting``, ...): stacked layers of one
       excitatory and one optional inhibitory population, configured by the
       ``model.core`` sections ``architecture``, ``connectivity``,
       ``transfer``, ``morphology``, ``sparsity``, ``reactivation``,
       ``blocklinear`` and ``implementation``. Built as
       ``ConfigurableEINetwork``, or as the recurrent ``EINetwork``. See
       :ref:`structured-ei-cores`.
   * - Unified E/I core
     - ``unified_ei`` (aliases ``ei_unified`` and the ``rnn_*`` types): layers
       of one excitatory and one inhibitory population given as population
       settings in ``model.core.unified_ei``. Built as ``EINetwork``. See
       :ref:`unified-ei`.
   * - Spatial cores
     - Dendritic cores applied to image feature maps, as a flattened map or as
       a shared core on local patches, and an AlexNet-shaped dendritic network.
       See :ref:`spatial-cores`.
   * - Baselines and controls
     - Point-neuron MLPs, dense or sparse affine MLPs, parameter-matched
       controls and recurrent baselines. See :ref:`baseline-models`.

:ref:`core-types` lists every ``model.core.type`` with its aliases.

A population network reads only ``model.core.population_network``, besides
``type``, ``initialization_seed`` and ``biological_neuron``; the same holds for
``unified_ei`` and the recurrent baselines with their own sections. Writing
another ``model.core`` section with a non-default value for such a core, for
example ``model.core.morphology``, raises when the configuration loads.
Population networks and unified E/I cores set their neuron and synapse options
as population fields (for population networks in ``population_defaults`` or a
population's ``population`` block), where structured E/I cores use the
``model.core`` sections. The tables in :doc:`dendritic_neurons` and :doc:`synapses` give both
spellings of each option.

The figure shows a feedforward structured E/I core inside the model.

.. figure:: _static/architecture/dendritic-ei-stack.svg
   :alt: Encoder, transfer layer and stacked excitatory and inhibitory dendritic populations of a structured E/I core.
   :width: 100%

   A feedforward structured E/I core. ``TransferLayer`` turns the encoder
   output into an excitatory stream ``x_E`` and an inhibitory stream ``x_I``.
   In each ``ExcitationInhibitionLayer`` the inhibitory cells run first, on
   both streams; the excitatory cells then read the excitatory stream and the
   output of the inhibitory cells (or ``x_I`` itself when the layer has no
   inhibitory cells). The core output is the output of the last excitatory
   population.

.. _biological-constraints:

Biological constraints
----------------------

``model.core.biological_neuron: true`` declares that a core is a biologically
constrained E/I network. The declaration is checked on the built model, so
type aliases, defaults and per-population settings are all taken into account,
and building fails with a list of every broken rule. With configuration format
4, this is the default for standard dendritic E/I cores. The explicitly signed
and MLP aliases, and other model families, keep their conventional contracts.
Set ``biological_neuron: false`` explicitly for non-biological exploration;
this disables the constraint check without changing the integration equations.
The declaration is an operational constraint set for a rate or spike model,
not a claim of complete biological realism. A core satisfies it when:

* **Synaptic weights are nonnegative.** Every synapse bank uses a nonnegative
  weight transform (``exp``, ``softplus`` or ``relu``; see
  :ref:`weight-transforms`), and the core has no layer with unconstrained
  weights: no signed input projection, no
  ``inhibitory_network_type: mlp`` and no soma feedback. The ``*_signed`` and
  ``*_mlp`` core types use signed weights and cannot satisfy the set.
* **Excitation and inhibition stay separate (Dale's law).** No synapse bank
  merges excitatory and inhibitory inputs, and every source acts with one sign.
  An inhibitory pathway may read the network input only when the transfer layer
  splits it into separate excitatory and inhibitory channels
  (``independent_pathways: true``), or explicitly declares distinct external
  afferents (``input_source_identity: paired_afferents``). The latter supplies
  the same feature-dependent rates to two separate afferent populations, one
  excitatory and one inhibitory. Dale's law concerns source identity, not
  statistical independence. This assumption does not model the upstream
  inhibitory circuit and must be reported; there are twice as many external
  source identities as input features, with no added trainable parameters.
  The default ``input_source_identity: shared`` makes no such declaration:
  duplicated channels cannot both excite and inhibit. Feedforward inhibition through inhibitory cells that
  read the excitatory stream needs no split. In a population network, each
  connection must use the pathway of its source's polarity.
* **The input is nonnegative.** The core checks or rectifies its input before the first
  synapses: ``transfer.output_activation`` is ``relu``, ``softplus`` or
  ``sigmoid``; or a population network sets ``input_transform: relu`` or
  ``signed_split``, or uses a transfer layer with one of these output
  activations. For already nonnegative data, use ``output_activation:
  nonnegative`` in the transfer layer or ``input_transform: nonnegative`` in
  a population network. This checks finite, nonnegative values on every batch
  and returns the input unchanged, including its derivative at zero. ReLU also
  preserves nonnegative values, but changes the derivative at zero. Runtime
  checks synchronize the device; they are not free. For signed features,
  ``signed_split`` preserves both polarities in separate nonnegative channels
  and doubles the input width. Neither encoding may be introduced silently
  into a historical experiment.
* **Populations emit nonnegative activity.** Each population outputs spikes, a
  spike rate or the output of its soma gate (:ref:`activation-types`).
  ``relu``, ``softplus``, ``sigmoid``, ``param_relu``, ``param_tanh`` and
  ``param_tanh_only_m`` are nonnegative for every voltage. ``none``
  (``reactivate: false``), ``endpoint_tanh``, ``linear_sigmoid``,
  ``linear_tanh`` and their ``param_`` forms are nonnegative only for a
  nonnegative voltage; they qualify when every level of the population
  integrates by shunting with such gates and no synapse type has a negative
  reversal potential, because shunting then keeps every voltage nonnegative.
  A spiking soma that reads out its membrane potential does not qualify.

The declaration validates a specified circuit; it does not choose an input
encoding or create inhibitory afferents. A bare ``ModelConfig()`` therefore
needs these choices before it can build. The reference configurations
``configs/config_exp.yaml`` and ``configs/examples/population_network_feedforward.yaml``
show biological cores with nonnegative image input and modeled inhibitory cells.

Only dendritic E/I cores can declare ``true``: structured E/I, unified E/I and
population-network types. Other core types accept ``false`` and reject
``true`` when the configuration loads. The check lives in
``dendritic_modeling.networks.architectures.biological_constraints``;
``biological_violations(module)`` returns the rules that a built module breaks.
A training run with a dendritic E/I core records the declaration and that list
in the ``biological_neuron`` entry of ``operator_card.json``. The check and the
record concern the neuronal core. A task encoder or signed classification
decoder outside it is not certified; report these components separately.
The constraint set concerns the forward neuronal model, not the biological
plausibility of a training algorithm or its signed error signals.
The operator card also records the external-afferent declaration.

Population networks can use
``input_projection_weight_transform: softplus`` with ``input_projection_dims``
to build a positive input projection followed by ReLU. Its equation is
``y = relu(softplus(W_raw) x + b)``: ``x`` must be nonnegative, the weights are
positive, and ``b`` is an intrinsic offset, not a synaptic input. Weights start
uniformly between zero and inverse fan-in, and offsets start at zero. This
option changes the model and initialization; signed projection checkpoints
cannot be relabeled as positive projections. The historical ``identity``
option still builds ordinary signed ``Linear`` layers and requires an explicit
non-biological declaration. The positive projection is supported through the
population-network builder with standard backpropagation.
Analytic local updates reject trainable positive projections; a frozen projection
can instead supply fixed input features to those rules.

Before using an older configuration, run ``dendritic-upgrade-config``. It writes
``biological_neuron: false`` where an earlier format omitted the declaration,
preserving the historical validation policy and computation. Adoption is a
separate step: identify external sources, establish the input domain, then
enable the check and compare states, outputs and gradients. If that requires
changing an encoding, synaptic signs or circuitry, retrain and retain the old
result and its source pin. The flag alone does not convert a trained network.

.. _output-scale:

Classifier output scale
-----------------------

A feedforward classifier multiplies its decoder output by
``model.fixed_output_scale`` (default ``1.0``, finite and positive) whatever
the decoder type. With an identity decoder it can also learn a positive scale,
selected by ``model.output_scale_mode``:

- ``per_class``: one learned scale per output; the default for an identity
  decoder;
- ``global``: one learned scale for all outputs;
- ``fixed``: no learned scale.

The learned scale is stored as a logarithm (``log_output_scale`` or
``log_global_output_scale``) and trained with ``decoder_lr``. Other decoders
learn no scale, so writing ``per_class`` or ``global`` for them raises when the
model is built. Recurrent classifiers and regression models apply no output
scale: they accept only ``output_scale_mode: fixed`` (or no value) with
``fixed_output_scale: 1.0``, and raise otherwise.

The retired key ``model.learned_output_scale`` still loads with a warning:
``false`` becomes ``output_scale_mode: fixed``, ``true`` keeps the default, and
a value that contradicts a written ``output_scale_mode`` raises.
