Synapses and connectivity
=========================

Every compartment of a dendritic neuron (:doc:`dendritic_neurons`) reads its
inputs through sparse synapse banks: one bank per input pathway, in which each
compartment keeps at most ``K`` active contacts. This page describes the
default Top-K synapses, the other sparse synapse types, the weight transforms
that keep synaptic weights positive, the initializers of the synaptic weights
and child couplings, the ``BlockLinear`` coupling between a branch and its
children, and structured connectivity, which restricts the inputs a
compartment may contact.

.. contents:: On this page
   :local:
   :depth: 1

.. _synapse-options:

Configuration keys
------------------

Structured E/I cores read the synapse options from the ``model.core``
sections ``connectivity``, ``sparsity``, ``morphology``, ``blocklinear`` and
``implementation``. Population networks read them as population fields, in
``model.core.population_network.layers[].population_defaults`` or a
population's ``population`` block (unified E/I cores use the same fields in
``model.core.unified_ei.layers[].excitatory`` and ``.inhibitory``):

.. list-table::
   :header-rows: 1
   :widths: 40 34 26

   * - Structured E/I core (``model.core.``)
     - Population field
     - Default
   * - ``connectivity.ee_synapses_per_branch_per_layer`` (and ``ie``, ``ei``,
       ``ii``, ``rec_*_synapses_per_branch``)
     - ``ff_excitatory_synapses``, ``ff_inhibitory_synapses``,
       ``rec_excitatory_synapses``, ``rec_inhibitory_synapses``
     - ``[40]``, ``[10]``, ``[0]``, ``[0]``, ``[0]``; ``8``, ``4``, ``0``, ``0``
   * - ``sparsity.type``
     - ``topk_type``
     - ``standard``
   * - ``sparsity.noise_level``, ``weight_norm_order``, ``gamma``,
       ``temperature``, ``ultrafast``, ``variance_score``,
       ``gradient_scaling``, ``init_method``
     - the same names with the prefix ``topk_``
     - ``0.0``, unset, ``1.0``, ``0.5``, ``true``, ``absolute``, ``none``,
       ``xavier_normal``
   * - ``sparsity.dense_to_sparse.<key>``
     - ``dense_to_sparse_<key>``
     - see :ref:`synapse-dense-to-sparse`
   * - ``sparsity.deepst.<key>``
     - ``<key>`` without a prefix (``rewiring_mode``, ``sigma``, ...)
     - see :ref:`synapse-deepst`
   * - ``sparsity.indexed.seed``
     - ``indexed_seed``
     - filled by training
   * - ``sparsity.indexed.<key>`` (``candidate_size``, ``selection``,
       ``rewire_*`` and the runtime options)
     - ``indexed_<key>``
     - see :ref:`synapse-indexed`
   * - ``sparsity.indexed.release_<key>``
     - ``release_<key>``
     - see :ref:`conductance-options`
   * - ``morphology.weight_transform``
     - ``weight_transform``
     - ``softplus`` (structured), ``exp`` (population)
   * - ``morphology.dbl_init_method``
     - ``dbl_init_method``
     - ``analytical_expectation``
   * - ``implementation.adaptive_initialization``,
       ``adaptive_initialization_policy``, ``adaptive_target_conductance``,
       ``initial_child_conductance``, ``initial_conductance_scale``,
       ``initial_child_conductance_log_sd``
     - the same names
     - ``true``, ``preserve_shunting_center``, ``5.0``, ``1.0``, ``1.0``,
       ``0.0``
   * - ``blocklinear.efficient``, ``blocklinear.gradient_scaling``
     - ``efficient_blocklinear``, ``blocklinear_gradient_scaling``
     - ``false``, ``none``
   * - ``connectivity.structured``
     - ``structured_connectivity``
     - disabled

In a structured E/I core the four feedforward counts are named by source and
target: ``ee`` (excitatory input onto excitatory cells), ``ie`` (inhibitory
input onto excitatory cells), ``ei`` (excitatory input onto inhibitory cells)
and ``ii`` (inhibitory input onto inhibitory cells), with one entry per network
layer. Every bank of a core or population uses the same synapse type. When a
configuration leaves them unset, training fills ``sparsity.indexed.seed`` and
the population ``indexed_seed`` from ``experiment.topology_seed``
(:ref:`random-seeds`).

.. _topk-linear:

Top-K synapses
--------------

``TopKLinear``, the ``standard`` synapse type, stores a dense trainable matrix
of raw parameters :math:`P` (``pre_w``) with one row per compartment and one
column per input. On every forward pass it computes the weights
:math:`W = \phi(P)` with the weight transform :math:`\phi`
(:ref:`weight-transforms`), keeps the ``K`` largest raw parameters of each row
(largest magnitudes for signed ``identity`` weights) and returns
:math:`y = x\,(M \odot W)^\top` for the selection mask :math:`M`. Options:

``K``
   Active synapses per compartment: the synapse count of the pathway. A count
   larger than the number of inputs is reduced to the number of inputs. A
   compartment has fewer active synapses when a connectivity mask allows
   fewer inputs.

``noise_level`` (default ``0.0``)
   Standard deviation of Gaussian noise added to the raw parameters before the
   selection, in training mode only. Evaluation always uses the deterministic
   selection.

``weight_norm_order`` (default unset)
   When set to ``p``, every row of the selected weights is divided by its
   L\ :sub:`p` norm, also when that norm lies below machine epsilon. Such
   ``exp`` and ``softplus`` rows are normalized in log space from their raw
   parameters, so they keep finite gradients when the weights become subnormal
   or underflow; ``exp`` rows depend exactly on parameter differences, and
   ``softplus`` rows use the raw parameter as the log weight, which changes
   each weight by less than half of machine epsilon (relative) before the
   usual floating-point rounding. Rows without selected weights stay zero.
   Orders ``p <= 0``, which are not norms, divide every row by the larger of
   its "norm" and machine epsilon, as before.

``gamma`` (default ``1.0``)
   Multiplies the normalized rows of the inhibitory banks (excitatory banks
   use ``1``); it has an effect only together with ``weight_norm_order``.

Because the mask is applied to a dense matrix, memory and compute grow with
the number of inputs, not with ``K``; the indexed types
(:ref:`synapse-indexed`) store only the active contacts. Unselected synapses
receive no gradient. During training a maintenance step multiplies the active
weights by ``1 - lr * training.main.common.synapse_decay_rate`` before every
optimizer step (``lr`` is the learning rate of their parameter group;
``synapse_decay_rate`` defaults to ``0.01``), and ``weight_boosting: true``
multiplies the unselected candidates by ``1 + lr * synapse_decay_rate``, so
the selected set changes when active weights fall below unselected ones.
:doc:`training_strategies` describes this maintenance step for every synapse
type. The FSDP entry point does not run it and rejects the rewiring types
(:ref:`distributed-training`).

``TopKLinear(in_features, out_features, K, init_method="xavier_normal",
noise_level=0.0, weight_transform="exp", weight_norm_order=None, gamma=1.0,
connection_mask=None)`` is exported from ``dendritic_modeling.networks``;
``weight_mask()`` returns the current selection and ``pruned_weight()`` the
selected weights.

.. _synapse-variants:

Sparse synapse types
--------------------

``model.core.sparsity.type`` (population field ``topk_type``) selects the
sparse layer of every synapse bank. The name is not checked when the
configuration loads; an unknown name raises when the model is built, listing
the registered types.

.. list-table::
   :header-rows: 1
   :widths: 19 23 33 25

   * - Type (aliases)
     - Stores per bank
     - Selection and changes during training
     - Options read
   * - ``standard`` (``topk``)
     - dense raw parameters ``[out, in]``
     - Top-K of the raw parameters, recomputed every forward pass
     - ``noise_level``, ``weight_norm_order``, ``gamma``
   * - ``stochastic``
     - as ``standard``
     - sampled selection in training, deterministic Top-K in evaluation
     - ``temperature``, ``ultrafast``, ``noise_level``
   * - ``variance``
     - as ``standard``, plus running input moments ``[in]``
     - Top-K of the transformed weight times an input statistic
     - ``variance_score``, ``noise_level``
   * - ``dense_to_sparse`` (``annealed_topk``, ``dense-to-sparse``,
       ``dense_to_sparse_topk``, ``annealed-topk``)
     - as ``standard``, plus a schedule step and a frozen mask
     - the number of active synapses anneals from a dense start to ``K``
     - ``sparsity.dense_to_sparse.*``, ``noise_level``,
       ``weight_norm_order``, ``gamma``
   * - ``deepst``
     - dense raw parameters and a Boolean mask ``[out, in]``
     - fixed budget; weak contacts are pruned and regrown at random
     - ``sparsity.deepst.*`` (except the ``credit_*`` keys)
   * - ``credit_deepst``
     - as ``deepst``, plus credit traces
     - swaps weak contacts for candidates that the ``local_ca`` gradient
       predicts to be better
     - ``sparsity.deepst.*``; requires ``rewiring_mode: constant_branch``
   * - ``indexed``
     - weights and input indices ``[out, K]``
     - fixed contacts; only the weights train
     - ``weight_norm_order``, ``gamma``, seed, runtime options
   * - ``indexed_dynamic`` (``indexed-dynamic``, ``indexed_dynamic_topk``)
     - weights and indices of ``C`` candidates ``[out, C]``
     - Top-K (or sampled) selection among the fixed candidates
     - ``candidate_size``, ``selection``, ``temperature``, ``ultrafast``,
       ``noise_level``, ``weight_norm_order``, ``gamma``, seed, runtime
       options
   * - ``indexed_rewire`` (``indexed-rewire``, ``indexed_rewire_topk``)
     - as ``indexed``, plus a rewiring step and seed
     - periodically moves the weakest contacts of each row to new inputs
     - ``rewire_frequency``, ``rewire_quantile``, ``rewire_until_step``, as
       ``indexed``
   * - ``indexed_release``
     - as ``indexed``, plus a contact-impact average and contact ages
       ``[out, K]``
     - fixed contacts with probabilistic release in training; contacts change
       through the impact rewiring controller
     - ``release_*``, as ``indexed``; see :ref:`conductance-options`

All types read ``K`` (except ``deepst`` without ``constant_branch``, below),
the weight transform and the connectivity masks of :ref:`structured-connectivity`.
Building raises when an option is set that the type does not apply:
``noise_level`` for ``indexed``, ``indexed_rewire``, ``indexed_release``,
``deepst`` and ``credit_deepst``; ``weight_norm_order`` or a ``gamma`` other
than ``1`` for ``stochastic``, ``variance``, ``deepst`` and ``credit_deepst``;
``variance_score: relative`` for any type but ``variance``; and a
``deepst.init_method`` other than the default or ``null``. The other
type-specific options (``temperature``, ``ultrafast``, the ``indexed``,
``dense_to_sparse`` and ``deepst`` blocks and the release options) are not
checked: a type that does not read them ignores them.
``register_sparse_layer`` adds a type (:ref:`extension-registries`), and
``get_available_sparse_types()`` lists the registered names (both in
``dendritic_modeling.networks.architectures.excitation_inhibition.synapse``).

Stochastic Top-K
~~~~~~~~~~~~~~~~

``stochastic`` samples the active set in training mode. With
``ultrafast: true`` (the default) it adds Gaussian noise with standard
deviation ``max(0.01, temperature) * (1 + noise_level)`` to the raw parameters
and keeps the ``K`` largest. With ``ultrafast: false`` it ranks the transformed
weights and keeps each input with probability
:math:`\sigma((K - r) / (K\,\tau))` for rank :math:`r` and temperature
:math:`\tau`; a row with fewer than ``K/2`` sampled contacts adds its ``K``
strongest inputs, a row with more than ``2K`` keeps its ``K`` strongest, and an
empty row keeps its strongest input. A ``temperature`` of ``0`` with
``ultrafast: false`` raises, because it would select exactly the standard
Top-K. Evaluation uses the deterministic Top-K.

Variance Top-K
~~~~~~~~~~~~~~

``variance`` tracks the first and second moments of every input with an
exponential moving average (momentum ``0.9``) during training forward passes
and ranks contacts by the transformed weight (its magnitude for ``identity``
weights) times an input score. ``variance_score: absolute`` (default) uses the
input variance; ``relative`` uses the scale-free :math:`\mathrm{Var}[x] /
E[x^2]`, so inputs coded on a small scale are not at a disadvantage. In
evaluation the saved moments are used and not updated.

.. _synapse-dense-to-sparse:

Dense-to-sparse pruning
~~~~~~~~~~~~~~~~~~~~~~~

``dense_to_sparse`` starts with many active synapses, so all allowed contacts
receive gradients early, and anneals the active count down to ``K``. The
options under ``model.core.sparsity.dense_to_sparse`` (population fields with
the prefix ``dense_to_sparse_``) are:

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Option
     - Default
     - Meaning
   * - ``initial_density``
     - ``1.0``
     - Initial active fraction of the inputs, in ``(0, 1]``; the initial
       count is ``ceil(initial_density * in_features)``.
   * - ``initial_k``
     - unset
     - Initial active count; replaces ``initial_density`` when set. It must
       lie between ``K`` and the number of inputs.
   * - ``start_step``, ``end_step``
     - ``0``, ``1000``
     - Schedule steps between which the count decreases.
   * - ``update_interval``
     - ``1``
     - The count changes only every this many steps.
   * - ``schedule``
     - ``cubic``
     - ``linear``, ``cubic`` (stays dense longer, then commits), ``cosine``
       or ``step`` (drops to ``K`` at ``end_step``).
   * - ``prune_metric``
     - ``weight``
     - Ranking score: ``weight`` or ``transformed_weight`` (the transformed
       weight), ``magnitude`` (its absolute value), ``pre_w`` or ``raw`` (the
       raw parameter).
   * - ``advance_on_forward``
     - ``true``
     - Count steps per training forward pass (one per sequence in recurrent
       networks); ``false`` counts optimizer steps.
   * - ``freeze_on_end``
     - ``false``
     - Freeze the final mask at ``end_step``.

Forward passes that run inside
``dendritic_modeling.networks.utils.sparse_maintenance.frozen_sparse_state()``,
such as auxiliary analysis passes, do not advance the schedule. The block
``sparsity.annealed_topk`` is a deprecated spelling of
``sparsity.dense_to_sparse`` and loads with a warning.

.. code-block:: yaml

   model:
     core:
       type: dendritic_shunting
       transfer:
         output_activation: nonnegative
         input_source_identity: paired_afferents  # Distinct E/I sources, matched tuning.
       architecture:
         excitatory_layer_sizes: [32]
         excitatory_branch_factors: [2, 2]
       sparsity:
         type: dense_to_sparse
         dense_to_sparse:
           initial_density: 1.0
           start_step: 0
           end_step: 4000
           update_interval: 50
           schedule: cubic
           freeze_on_end: true

.. _synapse-deepst:

DeepST rewiring
~~~~~~~~~~~~~~~

``deepst`` keeps a Boolean mask of active contacts over a dense parameter
matrix. Every ``rewire_frequency`` optimizer steps (the count is saved with the
layer) it prunes the active contacts whose weight plus Gaussian noise of
standard deviation ``sigma`` (no noise with ``use_noise: false``) is at most
``weight_threshold``, and regrows as many contacts at random allowed inputs,
starting at the threshold weight. The noise enters only this test. The options
under ``model.core.sparsity.deepst`` (population fields without a prefix) are:

.. list-table::
   :header-rows: 1
   :widths: 34 18 48

   * - Option
     - Default
     - Meaning
   * - ``rewiring_mode``
     - ``global``
     - ``global``: regrow anywhere in the bank; ``local``: on the branch that
       lost the contact; ``constant_branch``: keep exactly ``K`` contacts on
       every branch.
   * - ``excitatory_target_density``, ``inhibitory_target_density``
     - ``0.1``, ``0.1``
     - Active fraction of the allowed contacts of the excitatory and
       inhibitory banks (``global`` and ``local`` modes).
   * - ``sigma``, ``use_noise``
     - ``0.05``, ``true``
     - Noise of the prune test.
   * - ``weight_threshold``
     - ``1e-6``
     - Prune threshold and weight of regrown contacts.
   * - ``rewire_frequency``
     - ``1``
     - Optimizer steps between rewiring events.
   * - ``freeze_excitatory_connectivity``, ``freeze_inhibitory_connectivity``
     - ``false``
     - Keep the contacts of that bank fixed.
   * - ``init_method``
     - ``xavier_normal``
     - Initial draw of the raw parameters; ``null`` draws the initial and
       the regrown parameters uniformly from a small range instead.

With ``global`` and ``local`` rewiring the synapse counts of the pathways are
not used: each bank keeps ``target_density`` of its allowed contacts active,
spread unevenly over the branches. Small inputs can then give a bank less than
one contact per branch on average, which the default branch initializer
rejects. ``constant_branch`` keeps each pathway's configured synapse count
active per branch instead. ``deepst.synapses_per_branch`` (population
``synapses_per_branch``) sets that budget only for a ``DeepstLinear`` built
without a count; dendritic branch layers always pass the pathway's count, so
building a model raises when it is set with another synapse type, with
``global`` or ``local`` rewiring, or to a value other than the count of a
``constant_branch`` bank. The topology draws from ``sparsity.indexed.seed``
(population ``indexed_seed``); data-parallel training requires it.

``credit_deepst`` requires ``rewiring_mode: constant_branch`` and the
``local_ca`` strategy (:ref:`local-credit-assignment`), which records the
local gradient of every contact. It keeps an exponential trace of that
gradient (``credit_trace_decay``, default ``0.95``) at the active contacts and
at a small pool of inactive candidates per branch
(``credit_candidate_pool_size``, ``8``). Moving the weight :math:`w` of an
active contact to a candidate is predicted to improve the loss by
:math:`w\,(\bar g_{active} - \bar g_{candidate})`; each rewiring event applies
the best swaps that exceed ``credit_swap_margin`` (``0.0``) for up to
``credit_turnover_fraction`` (``0.1``) of each branch's contacts, chosen among
its weakest ``credit_weak_active_pool_fraction`` (``0.5``). Swaps start after
``credit_warmup_steps`` (``5``) recorded steps and need
``credit_min_observations`` (``5``) per candidate. ``credit_selection`` can be
``credit`` (default), ``random`` or ``shuffled_credit`` (controls), and
``credit_force_turnover: true`` applies the swaps even without a predicted
gain. A layer that records no credit raises once the warm-up has passed, and
multi-process training rejects ``credit_deepst``.

.. _synapse-indexed:

Indexed synapses
~~~~~~~~~~~~~~~~

The indexed types store only the contacts of each compartment, as a weight
matrix and an integer index matrix of the same shape, so memory grows with
``K`` instead of the number of inputs. Their input indices are drawn without
repetition from the allowed inputs of each row, from the topology seed.

``indexed``
   Fixed contacts: inputs outside the drawn indices can never become active.
   Spatial and manifest connectivity rules (:ref:`structured-connectivity`)
   supply the indices directly and require this type.

``indexed_dynamic``
   Stores ``C`` candidate contacts per row (``indexed.candidate_size``,
   population ``indexed_candidate_size``) and selects ``K`` of them on every
   forward pass, so inactive candidates can become active without a dense
   matrix. Unset, ``C`` is ``min(in_features, 4 K)`` for each bank, reduced to
   the smallest number of allowed inputs of any row when a mask is present. A
   written value applies to every bank and must not exceed the smallest bank
   input (a small inhibitory population, for example). ``indexed.selection``
   is ``standard`` (Top-K of the raw parameters, with ``noise_level`` noise in
   training) or ``stochastic`` (alias ``rank_probabilistic``: sampling as in
   :ref:`stochastic Top-K <synapse-variants>`, controlled by ``temperature``
   and ``ultrafast``).

``indexed_rewire``
   Fixed number of contacts with turnover: every ``indexed.rewire_frequency``
   optimizer steps (default ``100``), until ``rewire_until_step`` (default
   unset, no end), the ``ceil(rewire_quantile * K)`` weakest contacts of each
   row (``rewire_quantile`` default ``0.05``) move to new random allowed
   inputs. The new contact keeps the weight of the slot it replaces, and its
   gradient is cleared. The rewiring stream is seeded and saved with the
   layer. It requires ``persistent_indices: true``.

``indexed_release``
   Stored contacts that each transmit with a probability that grows with the
   weight, in training; see :ref:`conductance-options`.

The runtime options of the indexed types (``output_chunk_size``,
``index_dtype``, ``workspace_mb``, ``cache_transformed_weights``,
``recompute_backward``, ``projection_backend``, ``persistent_indices``,
``init_mode``) choose the projection kernel and how indices are stored; they
are described with the backends in :ref:`indexed-backends`. With
``persistent_indices: false`` the checkpoint does not store the indices, so
the layer needs an explicit seed to regenerate them.

A structured core with candidate-pool synapses, and one with rewired indexed
synapses:

.. code-block:: yaml

   model:
     core:
       type: dendritic_shunting
       transfer:
         output_activation: nonnegative
         input_source_identity: paired_afferents  # Distinct E/I sources, matched tuning.
       architecture:
         excitatory_layer_sizes: [64]
         inhibitory_layer_sizes: []
         excitatory_branch_factors: [2, 2]
       connectivity:
         ee_synapses_per_branch_per_layer: [16]
         ie_synapses_per_branch_per_layer: [4]
       sparsity:
         type: indexed_dynamic
         temperature: 0.5
         indexed:
           candidate_size: 64
           selection: stochastic

.. code-block:: yaml

   model:
     core:
       type: dendritic_shunting
       transfer:
         output_activation: nonnegative
         input_source_identity: paired_afferents  # Distinct E/I sources, matched tuning.
       architecture:
         excitatory_layer_sizes: [64]
         excitatory_branch_factors: [2, 2]
       sparsity:
         type: indexed_rewire
         indexed:
           rewire_frequency: 100
           rewire_quantile: 0.05
           rewire_until_step: 10000

The population fields are ``topk_type``, ``indexed_rewire_frequency``,
``indexed_rewire_quantile`` and ``indexed_rewire_until_step``:

.. code-block:: yaml

   population_defaults:
     topk_type: indexed_rewire
     indexed_rewire_frequency: 100
     indexed_rewire_quantile: 0.05
     indexed_rewire_until_step: 10000

.. _weight-transforms:

Weight transforms
-----------------

Synaptic weights and child couplings are stored as raw parameters and passed
through a weight transform :math:`\phi`, selected by
``model.core.morphology.weight_transform`` (structured E/I cores, default
``softplus``) or the population field ``weight_transform`` (default ``exp``).
The Python constructors (``DendriNet``, ``DendriticBranchLayer``,
``TopKLinear``, ``BlockLinear``) default to ``exp``. One transform applies to
every synapse bank and coupling of a core or population.

``exp``
   :math:`\phi(p) = e^p`: strictly positive weights.

``softplus``
   :math:`\phi(p) = \log(1 + e^p)`: strictly positive weights that grow only
   linearly in the raw parameter.

``relu``
   :math:`\phi(p) = \max(p, 0)`: nonnegative weights that can be exactly zero.

``identity``
   :math:`\phi(p) = p`: signed weights. The ``*_signed`` and ``*_mlp``
   structured core types use it. Top-K selection ranks magnitudes, and the
   branch layers are initialized with Xavier-normal draws whatever the
   ``dbl_init_method``.

The positive transforms make every synapse act with one sign, as excitatory
and inhibitory conductances. Shunting integration with ``identity`` weights logs
a warning, because the denominator of the shunting voltage can then reach zero
or turn negative. ``biological_neuron: true`` requires a nonnegative transform
(:ref:`biological-constraints`), and ``indexed_release`` accepts only ``exp``
and ``softplus``.

.. _synapse-initialization:

Synapse initialization
----------------------

Two options set the initial synaptic weights. ``model.core.sparsity.init_method``
(population ``topk_init_method``) names the PyTorch initializer that a sparse
layer applies to its raw parameters when it is constructed: ``xavier_normal``
(default), ``xavier_uniform``, ``kaiming_normal``, ``kaiming_uniform``,
``orthogonal``, ``normal``, ``uniform`` or ``eye``. A dendritic branch layer
then reinitializes every synapse bank and child coupling with
``model.core.morphology.dbl_init_method`` (population ``dbl_init_method``), so
in models built from a configuration ``init_method`` does not change the
initial weights; it changes only how much of the global random stream the
discarded draw uses, which matters without ``initialization_seed``. It does
not set the weights of rewired contacts either: ``indexed_rewire`` gives a
regrown contact the weight of the contact it replaces (only an
``IndexedRewireLinear`` built in code with ``rewire_init_policy="fresh"``
draws new weights with ``init_method``). (``deepst`` draws its parameters with
``deepst.init_method``.)

``dbl_init_method`` initializes the feedforward and recurrent banks, the
``BlockLinear`` couplings and the analytical gate prior
(:ref:`gate-initialization`) of every branch layer:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - ``dbl_init_method``
     - Initialization
   * - ``analytical_expectation`` (default)
     - Normal raw parameters whose means are chosen so that the expected
       weight of the ``K`` selected contacts gives equal expected excitatory
       and inhibitory conductances. Shunting child couplings follow the rule
       below; additive couplings start at ``1``. For deep trees,
       ``implementation.adaptive_initialization`` (default ``true``) lowers
       the expected conductances towards ``adaptive_target_conductance``
       (default ``5.0``): the default policy ``preserve_shunting_center``
       keeps the shunting voltage centre ``E / (E + I + L)``, while
       ``legacy_scale`` (and every additive layer) scales all conductances
       by a factor set by the contact counts and the number of terminal
       branches.
   * - ``ei_equivalence``
     - Zero-mean normal raw parameters; the spread of the weaker of the
       excitatory and inhibitory banks is increased until their expected
       conductances match. Each shunting child coupling starts at
       ``E + I + L``.
   * - ``naive``
     - Zero-mean normal raw parameters with standard deviation
       :math:`\sqrt{2 / (n_{in} + 1)}`; each shunting child coupling starts
       at ``E + I + L``, and the gate prior sits at the expected voltage.
   * - ``default``
     - The draws of ``naive``, with the fixed gate prior :math:`m = 1.5,
       b = 0.5` (shunting) or :math:`m = 0.1, b = 0` (additive).
   * - ``mechanism_neutral``
     - Zero-mean normal draws and child couplings of
       ``initial_child_conductance`` (default ``1``) whatever the
       integration rule, so a shunting and an additive model with the same
       seed start identical (:ref:`comparing-integration-rules`).
   * - ``fan_in_lognormal``
     - Log-normal conductances with mean ``1 / (total contacts of the
       compartment)`` and log standard deviation ``0.5``, times
       ``initial_conductance_scale``; child couplings of mean
       ``initial_child_conductance``, log-normal with
       ``initial_child_conductance_log_sd`` when that is positive. It
       requires ``exp`` or ``softplus`` weights and ``init_policy: fixed``.

``initial_conductance_scale`` other than ``1`` and a positive
``initial_child_conductance_log_sd`` require ``fan_in_lognormal``;
``initial_child_conductance`` other than ``1`` requires ``fan_in_lognormal``
or ``mechanism_neutral``. The adaptive options apply only to
``analytical_expectation``. :doc:`conductance_options` uses these options for
conductance-based models.

The shunting child-coupling rule of ``analytical_expectation``: with expected
excitatory and inhibitory conductances :math:`E` and :math:`I`, leak
:math:`L` and children at output ``0.5``, a total coupling :math:`G` holds the
compartment at :math:`(E + 0.5\,G) / (E + I + L + G)`. :math:`G` places this
voltage at ``0.49`` when :math:`E / (E + I + L) < 0.48` and otherwise at the
midpoint between :math:`E / (E + I + L)` and ``0.5``; when :math:`E \ge I + L`
(for example without inhibition), no coupling brings the voltage below
``0.5`` and :math:`G = E + I + L` gives the midpoint. Each of the children
receives :math:`G` divided by their number, and the gate prior is centred on
the same voltage.

.. _blocklinear:

BlockLinear
-----------

``BlockLinear`` adds the outputs of the child branches to their parent. With
``in_features`` child compartments and ``out_features`` parents, each parent
reads ``block_size = in_features / out_features`` consecutive children and
weights them with its own couplings:

.. math::

   C_j = \sum_{k=0}^{b-1} g_{jk}\, u_{j b + k}, \qquad
   G_j = \sum_{k=0}^{b-1} g_{jk}

where :math:`b` is ``block_size``, :math:`u` the child outputs and
:math:`g = \phi(\cdot)` the transformed raw couplings ``log_weight`` (shape
``[out_features, block_size]``), with the weight transform of the layer. :math:`C` is the child
current and ``sum_conductances()`` returns :math:`G`, which the shunting rule
adds to the denominator (:ref:`integration-modes`).

``BlockLinear`` builds the dense ``[out_features, in_features]``
block-diagonal matrix on every forward pass. ``EfficientBlockLinear`` gives
the same output with ``O(out_features * block_size)`` memory by reshaping the
input to ``[batch, out_features, block_size]``. Select it with
``model.core.blocklinear.efficient: true`` (population field
``efficient_blocklinear``). ``BlockLinear`` is exported from
``dendritic_modeling.networks``; ``EfficientBlockLinear`` lives in
``dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear``.
The couplings train with ``training.main.common.param_groups.blocklinear_lr``;
``blocklinear.gradient_scaling`` (``none``, ``conductance_dynamic`` or
``distal_upweight_by_idx``; :ref:`BlockLinear gradient scaling
<blocklinear-gradient-scaling>`) rescales their gradients.

.. _structured-connectivity:

Structured connectivity
-----------------------

A structured connectivity rule restricts which inputs each compartment may
contact. The synapse count stays the budget per compartment; the rule defines
the allowed set, and Top-K selection, sampling, variance ranking, DeepST and
indexed rewiring operate only inside it. Write the rule under
``model.core.connectivity.structured`` (structured E/I cores) or as the
population field ``structured_connectivity`` (in ``population_defaults`` or a
population's ``population`` block).

.. code-block:: yaml

   model:
     core:
       type: dendritic_shunting
       transfer:
         output_activation: nonnegative
       architecture:
         excitatory_layer_sizes: [64]
         inhibitory_layer_sizes: [16]
         excitatory_branch_factors: [3, 3]
       connectivity:
         ee_synapses_per_branch_per_layer: [12]
         ie_synapses_per_branch_per_layer: [6]
         ei_synapses_per_branch_per_layer: [8]
         structured:
           enabled: true
           seed: 123
           pathways:
             ee:                 # excitatory input onto excitatory cells
               method: fixed_indegree
               indegree: 24
             ie:                 # inhibitory cells onto excitatory cells
               method: bernoulli
               probability: 0.5

A block has ``enabled`` (default ``false``), an optional ``seed``, root keys
that act as defaults for every pathway, and a ``pathways`` mapping with one
rule per pathway; a pathway entry overrides the root keys. Pathway names refer
to the banks of the target compartment:

- structured E/I cores: ``ee``, ``ie``, ``ei`` and ``ii`` (named as the
  synapse counts) and ``rec_ee``, ``rec_ie``, ``rec_ei``, ``rec_ii`` for the
  recurrent banks;
- population networks: ``ee`` (all same-step excitatory sources of the
  population), ``ie`` (same-step inhibitory sources), ``rec_ee`` and
  ``rec_ie`` (delayed connections), whatever the polarity of the population.

Aliases such as ``e_to_e``, ``exc_to_exc``, ``i_to_e``, ``recurrent_ee`` and
``rec_e_to_e`` are accepted. Pathway names, rule keys and ``spatial`` keys are
checked when the configuration loads (for population networks, when the model
is built), and a misspelled one raises with a suggestion. An enabled block must
define a rule at the root or in a pathway (a mask parameter, a ``spatial``
block or a ``manifest_path``), or set ``method: dense`` on a pathway;
otherwise it raises, because every pathway would stay dense.

Four rule keys have a second spelling: ``p_connect`` (``probability``),
``in_degree`` (``indegree``), ``kernel`` (``distance_kernel``) and ``sigma``
(``distance_sigma``). The root and every pathway entry are read with the
first names, so a pathway entry overrides a root value whichever spelling
either uses. A rule that writes both spellings of one key raises, and so does
a ``pathways`` mapping that names one pathway twice (``ee`` and ``e_to_e``).
Before 2026-10-13 a pathway's second spelling lost to the root key, which a
``model.core.connectivity.structured`` block always has: ``kernel`` and
``sigma`` written in a pathway were ignored, and ``p_connect`` and
``in_degree`` raised.

Methods
~~~~~~~

``method`` (default ``bernoulli``) selects the rule:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``method`` (aliases)
     - Allowed inputs
   * - ``bernoulli`` (``probability``, ``probabilistic``)
     - Each input independently with ``probability`` (alias ``p_connect``).
       A row that draws no input gets one.
   * - ``fixed_indegree`` (``fixed_in_degree``, ``indegree``)
     - Exactly ``indegree`` (alias ``in_degree``) inputs per compartment. A
       count smaller than ``K`` leaves a Top-K or DeepST compartment with
       fewer than ``K`` synapses; the indexed types raise.
   * - ``distance`` (``distance_kernel``)
     - Each input with probability ``base_probability * k(d)`` for the
       distance :math:`d` between ``pre_positions`` (one coordinate list per
       input) and ``post_positions`` (one per compartment), with
       ``distance_kernel`` ``gaussian`` (:math:`e^{-d^2 / 2\sigma^2}`),
       ``exponential`` (:math:`e^{-d/\sigma}`) or ``none``, and
       :math:`\sigma` = ``distance_sigma`` (default ``1.0``).
   * - ``level_feature_blocks`` (``level_blocks``)
     - The input ranges ``[start, stop]`` listed in ``ranges_by_level`` for
       the level, one entry (a range or a list of ranges) per level from the
       most distal level to the soma; the last entry repeats.
   * - ``inventory_feature_blocks`` (``inventory_blocks``)
     - A fixed inventory of branch types per neuron: ``inventory_counts``
       branches per type, each reading its ``feature_ranges`` entry, spread
       over the dendritic levels (``tier_groups`` assigns types to levels;
       ``owner_count`` is the number of neurons). Population networks only,
       without somatic synapses.
   * - ``spatial_morphology`` (``morphology_spatial``)
     - Fixed indices from image regions assigned to the branches
       (:ref:`structured-connectivity-spatial`).
   * - ``index_manifest`` (``manifest``)
     - Fixed indices read from the file ``manifest_path``.
   * - ``dense`` (``none``, ``all``)
     - Every input; keeps a pathway dense when the root defines a rule.

``probability`` and ``indegree`` accept a list with one value per network
layer (``layers[i]`` in a population network); the last value repeats. Each
mask draws from its own stream, derived from ``seed``, the pathway, the layer
(in a population network, the population's position in its layer and, after
the first layer, the layer's index), the level and the mask shape, so a seeded
mask depends neither on the global random stream nor on the order in which
the layers are built.

Training fills an unset (or null) root ``seed`` from
``experiment.topology_seed``: that of ``model.core.connectivity.structured``,
of a layer's ``population_defaults.structured_connectivity`` (its populations
inherit it; a population's own rule is filled when the layer has no default
rule or the population writes ``seed: null``) and of a unified E/I
population's rule (offset by its layer after the first, since unified layers
of equal shape would otherwise draw equal masks from one seed).
``config.json`` records the filled seeds, so a fixed
topology seed fixes the masks, whatever the global random stream. A model
built outside a training run from a rule without a ``seed`` draws its masks
from the global random stream, and its ``spatial_morphology`` contacts use the
base seed ``0``; a pathway entry that writes ``seed: null`` keeps the global
stream in training too.

Before config_version 3 (:ref:`config-version`) training did not seed
population or unified E/I rules, so their masks drew from the global random
stream. Such a rule can write ``seed: global_rng`` to keep that: its random
masks then always draw from the global stream, and training keeps the value.
``spatial_morphology`` needs an integer seed and rejects it, and so does
``model.core.connectivity.structured``, which training has always seeded.
``dendritic-upgrade-config`` writes ``seed: global_rng`` (``seed: 0`` for a
spatial rule) into version-2 files whose rules relied on this.

A ``distance`` key that the pathway entry omits takes the root value, and the
default where the root has none: ``distance_kernel: gaussian`` and
``distance_sigma: 1.0``, in ``model.core.connectivity.structured`` (whose
typed root always has them) as in a population's ``structured_connectivity``.
Before config_version 3 the typed root's default kernel was ``none``, which
gives every input the probability ``base_probability`` whatever the
positions; ``dendritic-upgrade-config`` writes ``distance_kernel: none`` into
version-2 files whose distance rules relied on it.

An omitted ``base_probability`` takes the rule's ``probability`` (its own or
the root's), and ``1.0`` when a population rule has neither. The typed root
always has ``probability``, null unless written, so a
``model.core.connectivity.structured`` distance rule with neither key raises
when the configuration loads; a null or non-numeric ``base_probability``
raises as well.

Write ``distance_kernel``, ``distance_sigma`` and ``base_probability``
explicitly. The same positions are used at every level that has synapses on
the pathway, so ``post_positions`` must have as many entries as each of those
levels has compartments (one level, in practice).

In a population network, connection masks (from a ``probability`` below
``1``, ``target_levels`` or ``allow_self_recurrence: false``) and the
population's structured rule both restrict a pathway: it may use only the
contacts both allow, and a compartment left without any allowed input raises.
A rule that places exact contacts (``spatial_morphology`` or an index
manifest) cannot also follow a connection mask, so the combination raises;
see :ref:`population-connections`.

.. _structured-connectivity-spatial:

Spatial morphology
~~~~~~~~~~~~~~~~~~

``method: spatial_morphology`` maps the tree of every neuron onto an image of
shape ``spatial.input_shape`` ``[C, H, W]`` and gives each branch exactly ``K``
contacts drawn from the image region it owns. It produces compact fixed
indices without a dense mask and requires ``sparsity.type: indexed``
(population ``topk_type: indexed``). Every neuron has its own weights and
indices; only the geometry is shared, so this is not a convolution. The input
is flattened in ``C, H, W`` order, and ``input_shape`` must multiply to the
input size of the pathway.

``spatial.region_mode`` selects the regions:

``partition`` (default)
   The image is divided recursively along ``spatial.split_axes``, one axis
   (``height``/``y`` or ``width``/``x``) per branch factor, listed from the
   soma towards the distal branches. Regions are disjoint.

``overlapping_grid``
   Each parent's region is divided on the most nearly square grid of its
   children; a window smaller than ``spatial.minimum_window_shape``
   ``[h, w]`` is enlarged around its centre, so windows can overlap.
   ``split_axes`` is not used.

``random_windows``
   The window sizes of ``overlapping_grid``, each placed independently and
   uniformly over the image. This keeps the number of contacts and the window
   sizes of every level but removes the alignment between parent and child
   windows. ``spatial.region_seed`` sets the placement separately from the
   contact draws.

The excitatory and inhibitory banks of a neuron share the regions and draw
their contacts independently. The indices are generated once, when the model
is built, and do not change during training; adaptive rewiring is not
available because it could move a contact out of its region. Fixed indexed
synapses also give compact local-learning updates, so ``local_ca`` does not
build the dense ``[branches, pixels]`` tensor. The flattened-map spatial core
types (``tensor_map`` and its aliases, :ref:`spatial-cores`) accept the same
rule in their ``model.core.connectivity.structured`` section.

A root-level rule applies to every pathway, including those whose sources are
other cells rather than the image. Write it at the root only when all pathways
read the image, for example a structured E/I core without inhibitory cells,
where the excitatory cells read the inhibitory stream directly:

.. code-block:: yaml

   model:
     core:
       type: dendritic_shunting
       transfer:
         output_activation: nonnegative
         input_source_identity: paired_afferents  # Distinct E/I sources, matched tuning.
       architecture:
         excitatory_layer_sizes: [32]
         inhibitory_layer_sizes: []
         excitatory_branch_factors: [2, 2]
       connectivity:
         ee_synapses_per_branch_per_layer: [16]
         ie_synapses_per_branch_per_layer: [4]
         structured:
           enabled: true
           method: spatial_morphology
           seed: 42
           spatial:
             input_shape: [1, 28, 28]
             split_axes: [height, width]   # soma towards the distal branches
       sparsity:
         type: indexed

Otherwise put the rule under the pathways that read the image. In a population
network the ``ee`` pathway of a population collects all its same-step
excitatory sources, so the rule below suits populations whose excitatory input
is the image only; the inhibitory pathway from ``pv`` stays unstructured.
Recurrent pathways reject spatial rules, because population states have no
image geometry.

.. code-block:: yaml

   model:
     core:
       type: population_network
       population_network:
         input_transform: nonnegative
         layers:
           - name: hidden
             readout_population: pyr
             population_defaults:
               ff_excitatory_synapses: 16
               ff_inhibitory_synapses: 4
               topk_type: indexed
               structured_connectivity:
                 enabled: true
                 seed: 42
                 pathways:
                   ee:
                     method: spatial_morphology
                     spatial:
                       input_shape: [1, 28, 28]
                       region_mode: random_windows
                       minimum_window_shape: [4, 4]
                       region_seed: 7
             populations:
               - {name: pyr, polarity: excitatory, n_neurons: 32, branch_factors: [2, 2]}
               - {name: pv, polarity: inhibitory, n_neurons: 8, branch_factors: [1]}
             connections:
               - {source: input, target: pyr}
               - {source: input, target: pv}
               - {source: pv, target: pyr}

Index manifests
~~~~~~~~~~~~~~~

``method: index_manifest`` reads the exact contacts of every level from a file
written with ``torch.save``, for supports computed outside the model (for
example from the weights of a trained network). The file has
``format_version: 1`` and an ``entries`` mapping from
``"<pathway>/layer<layer_idx>/level<level_idx>"`` (level ``0`` is the most
distal; in a population network ``layer_idx`` is the position of the
population within its layer, and layers after the first use
``"<pathway>/network_layer<i>/layer<layer_idx>/level<level_idx>"`` for
``layers[i]``) to
``{"indices": LongTensor[rows, K], "in_features": int}``.
``manifest_entry_key`` builds these keys.
``save_index_manifest`` in
``dendritic_modeling.networks.architectures.excitation_inhibition.synapse.index_manifest``
writes a validated file. Loading fails when an entry is missing or does not
match the geometry of the layer. Like ``spatial_morphology``, this method
requires ``sparsity.type: indexed``.
