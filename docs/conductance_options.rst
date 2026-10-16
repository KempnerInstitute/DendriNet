.. _conductance-options:

Conductance options
===================

These options treat the parameters of a shunting dendritic layer as
conductances: a learnable compartment leak, an activation normalized to the
voltage range, stochastic synaptic release with cellwise rewiring
(``indexed_release``), a fan-in-scaled log-normal initialization and
reduced-precision signal rounding. They are opt-in fields of the standard
builders, and their defaults leave a model unchanged. Use them in feedforward
rate networks trained with backpropagation; the end of the page describes the
shipped context-cued EMNIST examples and the connectivity analysis for these
models.

.. _conductance-field-map:

Where the options are set
-------------------------

In a population network the options are fields of
``model.core.population_network.layers[].population_defaults``, and a
population's ``population`` block overrides them (:ref:`population-options`).
A unified E/I core takes the same names in its population blocks
(:ref:`unified-ei`). A structured E/I core sets them in its sections
(:ref:`structured-ei-cores`):

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * - Option
     - Population field
     - Structured E/I path (under ``model.core``)
   * - Shunting integration
     - ``use_shunting: true``
     - ``type: dendritic_shunting``, or ``einet`` with
       ``morphology.use_shunting: true``
   * - Leak
     - ``leak_conductance`` (default ``1.0``)
     - ``morphology.leak_conductance``
   * - Learnable leak
     - ``learnable_leak`` (default ``false``)
     - ``morphology.learnable_leak``
   * - Endpoint-normalized gate
     - ``reactivate: true``, ``reactivation_type: endpoint_tanh``
     - ``reactivation.enabled: true``, ``reactivation.type: endpoint_tanh``
   * - Gate start values
     - ``reactivation_init_policy: fixed``, ``reactivation_init_m``,
       ``reactivation_init_b``
     - ``reactivation.init_policy: fixed``, ``reactivation.init_m``,
       ``reactivation.init_b``
   * - Positive conductance weights
     - ``weight_transform: exp`` or ``softplus``
     - ``morphology.weight_transform: exp`` or ``softplus``
   * - Stochastic release
     - ``topk_type: indexed_release``
     - ``sparsity.type: indexed_release``
   * - Release options
     - ``release_p_min``, ``release_p_max``, ``release_weight_scale``,
       ``release_impact_decay``, ``release_impact_metric``,
       ``release_max_conductance``, ``release_track_source_activity``
     - the same names under ``sparsity.indexed``
   * - Fan-in-scaled log-normal initialization
     - ``dbl_init_method: fan_in_lognormal``
     - ``morphology.dbl_init_method: fan_in_lognormal``
   * - Mean child-to-parent coupling
     - ``initial_child_conductance`` (default ``1.0``)
     - ``implementation.initial_child_conductance``
   * - Spread of the child couplings
     - ``initial_child_conductance_log_sd`` (default ``0.0``)
     - ``implementation.initial_child_conductance_log_sd``
   * - Initial conductance multiplier
     - ``initial_conductance_scale`` (default ``1.0``)
     - ``implementation.initial_conductance_scale``
   * - Signal rounding
     - ``signal_rounding`` (default ``none``)
     - ``morphology.signal_rounding``
   * - Shunting denominator constant
     - ``epsilon`` (default ``1e-8``)
     - fixed at ``1e-8`` in feedforward cores; ``recurrent_ei.epsilon`` in
       recurrent cores

The population defaults of one layer of the shipped example
``configs/examples/population_network_contextual_remapping.yaml``, reduced to
these options:

.. code-block:: yaml

   population_defaults:
     use_shunting: true
     weight_transform: exp
     leak_conductance: 0.03
     learnable_leak: true
     reactivate: true
     reactivation_type: endpoint_tanh
     reactivation_init_policy: fixed
     reactivation_init_m: 3.0
     reactivation_init_b: 1.5
     dbl_init_method: fan_in_lognormal
     initial_child_conductance: 0.3
     topk_type: indexed_release
     release_p_min: 0.5
     release_p_max: 1.0

The same options in a structured E/I core:

.. code-block:: yaml

   model:
     encoder:
       type: identity
       params: {input_dim: 784}
     core:
       type: dendritic_shunting
       transfer:
         output_activation: nonnegative
       architecture:
         excitatory_layer_sizes: [64]
         inhibitory_layer_sizes: [16]
         excitatory_branch_factors: [3, 2]
         inhibitory_branch_factors: [3, 2]
       connectivity:
         ee_synapses_per_branch_per_layer: [40]
         ei_synapses_per_branch_per_layer: [40]
         ie_synapses_per_branch_per_layer: [4]
         ii_synapses_per_branch_per_layer: [0]
       morphology:
         somatic_synapses: true
         weight_transform: exp
         leak_conductance: 0.03
         learnable_leak: true
         dbl_init_method: fan_in_lognormal
       sparsity:
         type: indexed_release
         indexed:
           release_p_min: 0.5
           release_p_max: 1.0
       reactivation:
         type: endpoint_tanh
         init_policy: fixed
         init_m: 3.0
         init_b: 1.5
       implementation:
         initial_child_conductance: 0.3
     decoder:
       type: MLP
       params: {hidden_dims: [], output_dim: 10}

.. _conductance-requirements:

Requirements
------------

Each option raises when it is combined with a setting it does not support,
instead of being ignored:

* **Shunting integration.** A leak other than 1, a learnable leak,
  ``endpoint_tanh`` and signal rounding need ``use_shunting: true``; the leak
  is a term of the shunting denominator, and the gate expects voltages in
  ``[0, 1]``.
* **Feedforward rate populations.** Configurable leak, ``endpoint_tanh``,
  ``indexed_release`` and signal rounding raise in populations with recurrent
  synapses, a spiking soma, ``synapse_types``, dendritic spikes, soma feedback
  or a ``cross_level_mode`` other than ``hierarchical``.
* **Backpropagation.** The analytic local rules (:ref:`local-credit-assignment`)
  reject ``indexed_release``, a configurable leak and ``endpoint_tanh``.
  ``endpoint_tanh`` also rejects ``reactivation_gradient_scaling:
  straight_through``.
* **Gate initialization.** ``endpoint_tanh`` and ``fan_in_lognormal`` need
  ``reactivation_init_policy: fixed`` (``reactivation.init_policy: fixed``),
  which takes the gate's start values from ``reactivation_init_m`` and
  ``reactivation_init_b``. Data-driven calibration would write a voltage
  midpoint, which is not the offset of ``endpoint_tanh``.
* **Release synapses.** ``indexed_release`` needs ``exp`` or ``softplus``
  weights without ``weight_norm_order``, the ``eager`` projection backend,
  persistent indices and no cached transformed weights.
* **Initialization.** ``initial_conductance_scale`` other than 1 and a
  positive ``initial_child_conductance_log_sd`` need ``fan_in_lognormal``;
  ``initial_child_conductance`` other than 1 needs ``fan_in_lognormal`` or
  ``mechanism_neutral`` with a positive weight transform.
* **Signal rounding.** ``signal_rounding`` other than ``none`` needs
  ``indexed_release`` synapses.
* **Unit-leak analyses.** The analysis-only branch-rule overrides and the
  calibration of ``tangent_matched`` anchors from a shunting reference raise
  for a learnable leak or a leak other than 1 (:ref:`integration-modes`).

Learnable leak
--------------

A shunting compartment's voltage is

.. math::

   V = \frac{E + C}{g_L + E + I + G + \epsilon}

where :math:`E` and :math:`I` are the excitatory and inhibitory synaptic
drives, :math:`C` the weighted input from child branches, :math:`G` the sum
of the child couplings, :math:`g_L` the leak and :math:`\epsilon` a small
constant (:ref:`integration-modes`). ``leak_conductance`` sets :math:`g_L`
(1 by default). With ``learnable_leak: true`` every compartment learns its
own leak, stored as one log-leak parameter per output of each branch level
and starting at ``leak_conductance``. Changing the leak alone moves the
operating point of every compartment; :ref:`conductance-units` describes how
to rescale all conductances together.

Endpoint-normalized gate
------------------------

``reactivation_type: endpoint_tanh`` applies

.. math::

   f(V) = \frac{\tanh(mV - b) - \tanh(-b)}{\tanh(m - b) - \tanh(-b)}

with a trainable slope :math:`m > 0` (stored as its logarithm) and offset
:math:`b` per compartment. It maps :math:`f(0) = 0` and :math:`f(1) = 1` for any
:math:`m, b`, and its midpoint is the voltage :math:`b/m`. The function is
evaluated through an exponential identity that avoids subtracting saturated
``tanh`` values, in float32 for half-precision inputs.
``reactivation_init_m`` and ``reactivation_init_b`` (``reactivation.init_m``
and ``init_b``) give the starting :math:`m` and :math:`b`; both remain
trainable. ``param_tanh`` and ``param_relu`` are other parametric gates
(:ref:`activation-types`), and ``reactivate: false`` removes the gate.

.. _conductance-indexed-release:

Stochastic release
------------------

``topk_type: indexed_release`` (``sparsity.type: indexed_release``) stores K
contacts per output, their source indices and conductances, without dense
candidates. Each contact of conductance :math:`w` transmits with probability

.. math::

   p(w) = p_\mathrm{min} + (p_\mathrm{max} - p_\mathrm{min})
          \frac{w}{w + 0.25\, s}

where :math:`s` is ``release_weight_scale``. In training every contact and
example draws an independent Bernoulli release; evaluation transmits the
mean :math:`w\,p(w)`. Outputs are not rescaled by the release probability.
The Bernoulli draw is not differentiated; the evaluation path is
differentiable through :math:`p(w)`.

.. list-table::
   :header-rows: 1
   :widths: 30 14 56

   * - Field
     - Default
     - Meaning
   * - ``release_p_min``, ``release_p_max``
     - ``0.5``, ``1.0``
     - Bounds of the release probability; ``0 <= p_min <= p_max <= 1``.
   * - ``release_weight_scale``
     - ``null``
     - :math:`s` in final conductance units. ``null`` selects :math:`1/K`; with
       ``fan_in_lognormal`` it selects ``initial_conductance_scale`` divided by
       the compartment's total contact count.
   * - ``release_impact_decay``
     - ``0.99``
     - Decay of the exponential average of each contact's impact, in ``[0, 1)``.
   * - ``release_impact_metric``
     - ``drive``
     - Impact statistic used by rewiring: ``drive`` (absolute realized
       drive), ``active_drive`` (absolute drive averaged only over examples
       whose source is nonzero, so rarely active sources such as one-hot cues
       are not marked low-impact for being rare), ``weight`` (the conductance)
       or ``saliency`` (``|w * dL/dw|`` of the last backward pass). ``weight``
       and ``saliency`` are updated after every optimizer step, the others in
       training forward passes.
   * - ``release_max_conductance``
     - ``null``
     - Cap on every effective conductance, including the one used by the
       release law; capped contacts pass no gradient. It cannot be combined
       with ``saliency``.
   * - ``release_track_source_activity``
     - ``false``
     - Keep a running average of each source's absolute activity (buffer
       ``source_activity``), required by ``regrowth_sources: active``.

``weight_transform: softplus`` makes conductance growth linear in the stored
parameter at large conductance, where ``exp`` grows exponentially. Other
sparse synapse types do not read the release fields. With
``release_p_min: 1`` and ``release_p_max: 1`` every contact transmits in
training and evaluation, which separates the effect of release from that of
the connectivity.

.. _conductance-impact-rewiring:

Cellwise impact rewiring
~~~~~~~~~~~~~~~~~~~~~~~~

``training.main.common.impact_rewiring`` replaces low-impact
``indexed_release`` contacts during training; an empty mapping (the default)
disables it. The contacts of one population and sign (excitatory or
inhibitory) at all dendritic levels form one group. Every ``frequency``
optimizer steps, a contact is eligible when its impact is below
``threshold`` times its cell's mean impact and it is at least
``maturity_steps`` optimizer steps old. Each cell replaces at most
``floor(n_eligible * eligible_fraction)`` eligible contacts, and at most
``floor(K * segment_fraction)`` per dendritic segment, so a cell with fewer
than ``1 / eligible_fraction`` eligible contacts replaces none at that step.
A segment is the K contacts of one compartment in one bank. A new contact
takes a source that its compartment does not use yet and that the connection
mask allows; its impact, age and optimizer moments are reset.

.. list-table::
   :header-rows: 1
   :widths: 26 14 60

   * - Field
     - Default
     - Meaning
   * - ``frequency``
     - ``100``
     - Optimizer steps between replacement events.
   * - ``maturity_steps``
     - ``100``
     - Minimum age, in optimizer steps, of a replaceable contact.
   * - ``threshold``
     - ``0.1``
     - Eligibility: impact below this fraction of the cell's mean impact.
   * - ``eligible_fraction``
     - ``0.1``
     - Largest fraction of a cell's eligible contacts replaced per event.
   * - ``segment_fraction``
     - ``0.25``
     - Largest fraction of a segment's K contacts replaced per event.
   * - ``freeze_last_epochs``
     - ``0``
     - Stop replacing contacts for the final epochs; weights keep training.
   * - ``regrowth_sources``
     - ``uniform``
     - ``uniform`` samples any legal unused source; ``active`` only sources
       whose tracked activity exceeds ``min_source_activity`` (requires
       ``release_track_source_activity``). A compartment without such sources
       keeps its contacts.
   * - ``min_source_activity``
     - ``0.0``
     - Activity threshold of ``active`` regrowth.
   * - ``regrowth_weight``
     - ``cell_median``
     - Conductance of a new contact: the median of the cell's conductances of
       that sign, or ``initial``, a draw from the bank's recorded
       ``fan_in_lognormal`` law (other initializers raise).

The trainer derives the last replacement step from ``epochs`` and
``freeze_last_epochs`` and seeds the replacement sampling from the
experiment's topology seed; writing ``until_step`` or ``seed`` raises.
Impact rewiring needs a population network with ``indexed_release``
synapses, the ``standard`` or ``presentation_stream`` strategy and a single
process. Checkpoints hold the model state, not the controller's step count or
random generator, so a run started from a checkpoint restarts the
replacement schedule (:ref:`training-checkpoints`). Code that builds the
controller itself (``CellwiseImpactRewiring`` in
``dendritic_modeling.networks.architectures.excitation_inhibition.synapse.impact_rewiring``)
can save and restore it with its ``state_dict`` and ``load_state_dict``.

Comparing sparse synapse types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The conductance, leak and gate options compose with these sparse synapse
types (:ref:`synapse-variants` describes all of them):

.. list-table::
   :header-rows: 1
   :widths: 18 41 41

   * - Type
     - Stored parameters and selection
     - Transmission and maintenance
   * - ``indexed_release``
     - K stored contacts per output with their source indices.
     - Bernoulli release in training, mean transmission in evaluation;
       optional cellwise impact rewiring.
   * - ``indexed``
     - K stored contacts with fixed source indices.
     - Deterministic; contacts never change source.
   * - ``indexed_rewire``
     - K stored contacts; each bank periodically replaces its
       lowest-weight contacts.
     - Deterministic; replacement by weight, separate from impact rewiring.
   * - ``standard``
     - A dense candidate matrix; every forward pass selects the K largest
       transformed weights per output.
     - Deterministic; only selected weights receive gradients; maintenance by
       ``synapse_decay_rate`` and ``weight_boosting``
       (:ref:`training-sparse-maintenance`).
   * - ``variance``
     - Dense candidates; selects the K largest values of transformed weight
       times an input score.
     - As ``standard``; the input statistics update in training mode only and
       are saved in checkpoints.

A Top-K bank stores ``in_features`` candidates per output but transmits
through K of them, while the indexed types store only their K contacts, so
stored and active parameter counts differ for Top-K. Top-K starts from the
upper tail of its candidate initialization, so equal fan-in and initializer
settings do not give the same initial function as uniformly sampled indexed
contacts. ``stochastic`` Top-K samples its mask by weight rank; it does not
model release.

``variance`` tracks the mean and second moment of every input with an
exponential moving average (momentum 0.9). With ``variance_score: absolute``
(the default) the input score is the variance, which depends on how each
input is scaled: a one-hot context cue of amplitude 1 has a much smaller
variance than pixels under a wide gain range, so its contacts are rarely
selected. ``topk_variance_score: relative`` (``sparsity.variance_score``)
uses the scale-free :math:`\mathrm{Var}[x] / E[x^2]` in ``[0, 1]``, which is
0 for an input that has been constantly 0; before any statistics exist the
selection is by weight alone. Other synapse types reject a non-default score.

.. _conductance-initialization:

Fan-in-scaled initialization
----------------------------

``dbl_init_method: fan_in_lognormal`` draws every synaptic conductance of a
compartment from one log-normal law with mean ``1 / n`` and log standard
deviation 0.5, where ``n`` is the compartment's total number of contacts over
its excitatory, inhibitory and recurrent banks. Excitatory and inhibitory
contacts use the same law, so it does not balance their drives. Child
couplings start at ``initial_child_conductance``. ``softplus`` weights draw
the same conductances from the same random stream as ``exp`` weights and
store their inverse-softplus values. The law is recorded on each bank for
``regrowth_weight: initial``. Gates without a slope and midpoint
(``tanh``, ``relu``, ``sigmoid``) have nothing to initialize;
``reactivation_init_m`` and ``reactivation_init_b`` apply to gates that have
them.

* ``initial_conductance_scale`` multiplies the initialized synaptic and
  coupling conductances and the automatic release scale. An explicit
  ``release_weight_scale`` is in final conductance units and is not
  multiplied, and the leak and ``epsilon`` stay as written.
* ``initial_child_conductance_log_sd`` draws the child couplings from a
  mean-preserving log-normal law with mean ``initial_child_conductance``
  times ``initial_conductance_scale``. Zero keeps constant couplings and
  draws no random numbers. With an initialization seed the draws use the
  keyed stream ``fan_in_lognormal.child_coupling`` (:ref:`random-seeds`).

The defaults (scale 1, spread 0) leave outputs, state-dict keys and random
streams unchanged.

Signal rounding
---------------

``signal_rounding: float16`` or ``bfloat16`` emulates reduced-precision
arithmetic in a layer. The layer rounds its presynaptic signals, synaptic and
coupling conductances, leak, mean-transmission probabilities, summed synaptic
drives, voltages and outputs to that format; the terms inside each sum, the
parameters and the optimizer state keep float32. Rounding is a cast to the
format and back, so a gradient through a rounded value is rounded too, as in
half-precision autocast without loss scaling. A conductance above the float16
maximum (65504) becomes infinite; ``release_max_conductance`` can bound it.
Rounding emulates the precision of a reduced-precision implementation while
the computation itself stays in float32; it is not mixed-precision training.

.. _conductance-units:

Conductance units
-----------------

The voltage above is unchanged when every conductance and the constants of
its denominator are multiplied by the same positive factor. To compare
conductance units, multiply ``leak_conductance``,
``initial_conductance_scale`` and ``epsilon`` by the same factor and keep the
automatic release scale. The initial voltages, release probabilities and
gate outputs then stay the same, and so do the gradients with respect to the
log-parameterized conductances, up to floating-point roundoff. Training can
still diverge later, for example through a learned leak or contact
replacement. Feedforward structured cores fix ``epsilon`` at ``1e-8``, which
is negligible at these scales.

.. _contextual-remapping-examples:

Context-cued EMNIST examples
----------------------------

Three shipped configurations train on the ``contextual_remapping`` dataset:
EMNIST Balanced images whose label is shifted by one of ``num_contexts``
contexts, with a one-hot context cue appended to the image and per-image gain
and noise corruption (:ref:`dataset-registered`). The inputs have
784 + ``num_contexts`` features (816 by default). The module
``dendritic_modeling.datasets.contextual_remapping`` also provides the batch
operations for other code: ``contextual_remapping_batch`` (corruption, cue and
label shift), ``brightness_dependent_noise`` (Gaussian noise with one standard
deviation per image, proportional to the square root of its mean gained
intensity; only negative values are clipped) and ``load_emnist_balanced``.

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Config in ``configs/examples/``
     - Model
   * - ``population_network_contextual_remapping.yaml``
     - Two layers, each with one excitatory and one inhibitory population;
       ``indexed_release`` contacts, ``endpoint_tanh`` gates, learnable leak,
       ``fan_in_lognormal`` initialization and impact rewiring that stops for
       the final 20 epochs.
   * - ``population_network_target_levels.yaml``
     - Two inhibitory populations per layer: ``p`` inhibits the excitatory
       somas and primary branches, ``t`` the distal branches
       (``target_levels``, :ref:`population-connections`); variance Top-K
       with ``topk_variance_score: relative`` and active-weight maintenance.
   * - ``contextual_remapping_affine_mlp.yaml``
     - A dense MLP baseline (``affine_mlp``, :ref:`baseline-models`) with the
       context cue scaled by 8.

.. code-block:: bash

   dendritic-train configs/examples/population_network_contextual_remapping.yaml \
     --data_dir <root>

``<root>/raw`` is the torchvision EMNIST directory; set
``data.dataset_params.contextual_remapping.download: true`` to download it
there. The examples set ``generator_device: cuda``, which draws the fixed
validation and test corruptions on a GPU and raises without one; ``cpu``
draws them on the CPU, which gives different evaluation sets.

The dataset requires ``training.main.strategy: presentation_stream``
(:ref:`training-presentation-stream`): every epoch samples
``presentations_per_epoch`` fresh presentations, evaluates the noisy and the
clean validation sets, appends a row to ``metrics.jsonl`` and selects the
best epoch by noisy-validation accuracy. This strategy rejects data-driven
gate initialization and periodic gate recalibration, because the training
set's items are clean context-0 images rather than the sampled presentations;
the examples use ``reactivation_init_policy: fixed``.

.. _conductance-source-tuning:

Connectivity and tuning analysis
--------------------------------

``analysis.source_tuning_support`` asks whether the sources that a
population's contacts select are tuned like their target somas (the analyzer
and its outputs are described in :ref:`analyzer-catalogue`). It applies to
feedforward population networks with ``indexed_release``, ``indexed``,
``standard`` or ``variance`` synapses, including same-step interneuron
pathways and branched morphologies, and runs the model's forward pass in
evaluation mode without changing weights, indices or buffers. Each record
reports three weightings of the sources: uniform over the eligible sources,
by contact count, and by effective transmission coefficient. For release
synapses the coefficient is the mean transmission :math:`w\,p(w)`, while
contact counts count stored contacts; with signal rounding the coefficients
use the rounded conductances and probabilities, and the rounding of inputs
and outputs remains part of the forward pass. Variance Top-K masks use the
training statistics saved in the checkpoint. Rows record
``selection_policy``, ``transmission_mode`` and ``signal_rounding``.

``configs/examples/conductance_source_tuning_analysis.yaml`` is an analysis
profile for a trained checkpoint of a contextual example:

.. code-block:: bash

   dendritic-analyze-checkpoint \
     --config <run-dir>/config.json \
     --checkpoint <checkpoint.pt> \
     --analysis-config configs/examples/conductance_source_tuning_analysis.yaml \
     --output-dir <analysis-dir> --device cpu

Preferred classes are selected on ``reference_split`` and source responses
measured on an independent ``evaluation_split``
(``require_independent_splits``). With a contextual task the labels that
define tuning are the shifted outputs by default; other labels must be
provided by the analysis dataset and selected with ``label_batch_index``.
An input of pixels followed by context cues is not a 28 x 28 image, so the
profile leaves ``image_shape`` unset and keeps the native source coordinates.
Connectivity alone does not show functional tuning or causal importance:
report excitatory and inhibitory populations separately, compare
independently trained seeds, and use perturbation analyses for causal
questions.
