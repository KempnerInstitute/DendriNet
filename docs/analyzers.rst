Analyzer reference
==================

One section per analyzer, grouped by what it measures. Every section is
written under ``analysis.<analyzer>`` with ``enabled``, ``training`` and
``params``; :doc:`analysis` explains when analyzers run, how
``analysis.runtime`` controls their data iteration, and where they write. The
:ref:`analyzer-catalogue` summarizes all of them. Each section below gives the
model the analyzer needs, the split it reads, its main options with their
defaults, an example, and its output files (``<name>`` is ``final``,
``epoch<N>`` or a stage name). The example configurations under
``configs/examples/`` named in a section are complete analysis profiles for
:doc:`checkpoint_analysis`.

.. contents:: On this page
   :local:
   :depth: 2

Performance and robustness
--------------------------

.. _analyzer-performance:

``performance``
~~~~~~~~~~~~~~~

Scores the model on whole splits: ``accuracy``, ``auc`` and
``categorical_loglikelihood`` (on by default) and ``mse`` and
``cosine_similarity`` (off by default). For a regression model the
classification scores are replaced by ``mse`` and ``cosine_similarity``.
``training_splits`` (per-epoch runs) and ``final_splits`` (after training)
choose the splits, from ``train``, ``validation`` and ``test`` (default: all
three); the older spelling ``valid`` loads with a warning. ``performance``
ignores ``analysis.runtime.<phase>.max_samples``.

.. code-block:: yaml

   analysis:
     performance:
       enabled: true
       training: true
       training_splits: ["validation"]
       final_splits: ["train", "validation", "test"]
       params:
         accuracy: true
         auc: false
         categorical_loglikelihood: true

Output: ``performance/<name>.json`` after training (``{metric: {train, valid,
test}}``) and ``performance/epochs/epoch<N>.json`` per epoch. When per-epoch
results exist, training also draws ``performance_evolution_<metric>.png``.

.. _analyzer-noise-perturbation:

``noise_perturbation``
~~~~~~~~~~~~~~~~~~~~~~

Scores the model on test inputs with added noise. For every magnitude ``m`` in
``uniform_magnitudes`` (noise uniform in ``[-m, m]``) and every standard
deviation in ``gaussian_sdevs``, ``n_samples`` noisy copies of the test inputs
are drawn (seeded by ``seed``), clipped to ``clamp_range`` and scored together.

Options (defaults): ``uniform_noise: true``, ``uniform_magnitudes: [0.1,
0.2]``, ``gaussian_noise: true``, ``gaussian_sdevs: [0.1, 0.2]``,
``n_samples: 10``, ``clamp_range: [0.0, 1.0]`` (``null`` disables the clip, for
inputs outside ``[0, 1]``), and the score switches ``accuracy``, ``auc``,
``categorical_loglikelihood``, ``mse`` and ``cosine_similarity``, all
``true``. The regression scores are NaN for classifiers; switch them off.

.. code-block:: yaml

   analysis:
     noise_perturbation:
       enabled: true
       params:
         uniform_magnitudes: [0.1, 0.2, 0.4]
         gaussian_sdevs: [0.1, 0.2, 0.4]
         n_samples: 10
         mse: false
         cosine_similarity: false

Output: ``noise_perturbation_analysis/<name>`` (JSON: ``uniform`` and
``gaussian``, then the level, then the score; the log-likelihood key is
``categorical_ll``). ``dendritic-collect-perturbation-response`` combines it
with ``performance`` and ``multiplicative_gain`` (see
:doc:`checkpoint_analysis`).

.. _analyzer-multiplicative-gain:

``multiplicative_gain``
~~~~~~~~~~~~~~~~~~~~~~~

Scores the model on test inputs multiplied by a sweep of gains: ``n_steps``
values from ``gain_min`` to ``gain_max``, spaced linearly or, with
``logspace: true``, logarithmically (a log sweep with ``gain_min: 0`` runs from
``1/gain_max`` to ``gain_max``). With ``uniform_gain: true`` every input is
multiplied by the gain; with ``false`` every input element gets its own gain,
uniform in ``[1, gain]``. ``input_preprocess`` (``none``, ``relu``,
``softplus``, ``shift_min``, ``global_shift_min``) is applied before the gain,
and ``noise_model: poisson`` replaces each scaled input by a Poisson count with
that mean (floored at ``poisson_eps``). All gain levels draw their random gains
and noise from ``noise_seed``, so levels differ only in the gain. Poisson
generator datasets apply the gain to their rates and keep their own noise; the
preprocessing and noise options must then keep their defaults.

At least one score must be selected (all are ``false`` by default):
``accuracy``, ``auc`` and ``categorical_loglikelihood`` for classifiers,
``mse`` and ``cosine_similarity`` for regression models. Scores that do not
fit the model are dropped with a warning.

.. code-block:: yaml

   analysis:
     multiplicative_gain:
       enabled: true
       params:
         gain_min: 0.0
         gain_max: 10.0
         n_steps: 11
         accuracy: true
         categorical_loglikelihood: true

Output: ``multiplicative_gain_analysis/<name>.json`` with ``gain_factors``,
``raw_scores`` per gain and the mean scores inside (``ID``), below
(``OOD_lower``) and above (``OOD_upper``) ``in_distribution_gain_range``, the
gain range of the evaluated dataset (``[1, 1]`` unless it is a Poisson
generator with a gain). Per-epoch runs write ``epochs/epoch<N>.json``; after
training the analyzer draws ``<score>_vs_gain_final.png`` and, with per-epoch
results, an animation over epochs.

.. _analyzer-gain-load-perturbation:

``gain_load_perturbation``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Final-only. Scores a classifier on test inputs under a factorial of two
perturbations: a mean-one log-normal gain shared by all input coordinates of a
trial (log standard deviations ``gain_log_sds``) and an input load
(``load_levels``). ``load_mode: gaussian_input`` adds independent zero-mean
Gaussian noise with standard deviation ``load_level`` to every coordinate;
``load_mode: positive_inhibitory`` adds a positive log-normal background with
mean ``load_level`` (log standard deviation ``load_log_sd``) to the
coordinates from ``inhibitory_start`` on, for inputs ordered ``[E | I]``. Every
cell has ``draws_per_condition`` technical draws seeded by ``base_draw_seed``;
write it explicitly so that checkpoints from runs with different seeds share
the same random numbers (an unset seed is derived from each run's
``experiment.probe_seed``). ``clamp_min`` (default
``0``) and ``clamp_max`` clip every cell, the unperturbed one included;
clipping rectifies Gaussian load into a positive-mean perturbation, so each
record reports the fractions of values the clamp moved, and a warning is
logged when clean inputs already fall outside the clamp. Set
``clamp_min: null`` for zero-mean Gaussian load.

Further options: ``max_samples``, the scores ``accuracy`` and
``categorical_loglikelihood`` (both ``true``), ``record_input_statistics``
(mean, standard deviation, zero fraction, coefficient of variation and range
of every cell) and ``record_branch_local_sensitivity`` with its
``branch_sensitivity_*`` options, which evaluates the
:ref:`branch_local_sensitivity <analyzer-branch-local-sensitivity>` estimator
on the draws listed in ``branch_sensitivity_draws``.

.. code-block:: yaml

   analysis:
     gain_load_perturbation:
       enabled: true
       params:
         gain_log_sds: [0.0, 0.2, 0.5, 0.8]
         load_levels: [0.0, 0.1, 0.3, 0.5]
         load_mode: gaussian_input
         draws_per_condition: 5
         clamp_min: null

Output: ``gain_load_perturbation/<name>.json``. Collect checkpoint runs with
``dendritic-collect-gain-load-response``.

.. _analyzer-adversarial-robustness:

``adversarial_robustness``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Accuracy on test inputs perturbed by the fast gradient sign method (FGSM) for
every ``epsilon`` in ``epsilons`` (default ``[0.0, 0.05, 0.1, 0.15, 0.2,
0.3]``); perturbed inputs are clipped to ``[0, 1]``. The batch size is the
phase's ``analysis.runtime`` batch size; ``params.batch_size`` has no effect
in training or checkpoint runs.

.. code-block:: yaml

   analysis:
     adversarial_robustness:
       enabled: true
       params:
         epsilons: [0.0, 0.05, 0.1]

Output: ``adversarial_robustness_analysis/adversarial_robustness_<name>.json``
(``eps_<epsilon>`` to scores).

.. _analyzer-single-layer-contribution:

``single_layer_contribution``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Requires a structured E/I core (an ``ExcitationInhibitionNetwork``; other
cores are skipped with a warning). For each dendritic layer, scores the test
split with the excitatory and inhibitory synapses of all other layers
lesioned: with both of the layer's banks (``both_contribution``), only its
excitatory bank (``excitation_contribution``) or only its inhibitory bank
(``inhibition_contribution``). The model is restored afterwards. The score
switches are those of ``noise_perturbation``, all ``true``; switch off ``mse``
and ``cosine_similarity`` for classifiers.

.. code-block:: yaml

   analysis:
     single_layer_contribution:
       enabled: true
       params:
         mse: false
         cosine_similarity: false

Output: ``single_layer_contribution/<name>`` (JSON) and one figure per score.

Information
-----------

.. _analyzer-information:

``information``
~~~~~~~~~~~~~~~

Information-theoretic measures of the signals of every dendritic branch: the
excitatory current ``E``, the inhibitory current ``I``, the input from child
branches ``Vb``, the branch output ``Vout`` and, with
``compute.include_vinf`` (default ``true``), the soma activation ``Vinf``,
about the class ``C`` and about each other. It requires E/I dendritic branch
layers (a structured E/I core or a ``population_network``); other models are
skipped with a warning. It reads the test split, or the validation split with
``compute.analysis_split: validation``, and processes the selected examples
in one forward pass.

``params.selection.components`` chooses the families of terms. When it is
empty, ``basic``, ``pairwise``, ``conditional``, ``ablation_aligned`` and
``layer_proxies`` are computed.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Component
     - Terms
   * - ``basic``
     - ``I(E; C)``, ``I(I; C)``, ``I(Vb; C)``, ``I(Vout; C)``, ``I(E, I; C)``,
       and ``I(Vinf; C)`` when ``Vinf`` is included.
   * - ``pairwise``
     - ``I(E; I)``, ``I(E; Vout)``, ``I(I; Vout)``, ``I(E; Vb)``,
       ``I(I; Vb)``, ``I(Vb; Vout)``, and the ``Vinf`` terms.
   * - ``conditional``
     - ``I(E; I | C)``, ``I(E; Vout | C)``, ``I(I; Vout | C)``,
       ``I(E; Vb | C)``, ``I(I; Vb | C)``, ``I(Vb; Vout | C)``, and the
       ``Vinf`` terms.
   * - ``ablation_aligned``
     - Conditional terms that mirror ablations: ``I(Vout; C | E)`` and
       ``I(Vout; C | I)``; with ``Vb`` also ``I(Vout; C | Vb)``,
       ``I(Vb; C | Vout)``, ``I(Vout; C | E, I)``, ``I(E; C | Vb)``,
       ``I(I; C | Vb)``, ``I(E, I; C | Vb)``, ``I(E; C | I, Vb)``,
       ``I(I; C | E, Vb)`` and the ``Vout`` terms given two signals; without
       ``Vb`` ``I(E; C | I)``, ``I(I; C | E)``, ``I(E; Vout | I)``,
       ``I(I; Vout | E)`` and ``I(Vout; C | E, I)``; with ``Vinf``
       ``I(Vinf; C | Vout)`` and ``I(Vout; C | Vinf)``. Computed only
       together with ``conditional``.
   * - ``upstream_unique_cmi``
     - ``I(Vb; C | E, I)`` when ``Vb`` exists; expensive for wide ``E`` and
       ``I``. Computed only together with ``conditional``.
   * - ``layer_proxies``
     - Layer-level summaries of the single-branch terms (``_sum``,
       ``_union``, ``_topK_sum`` with ``K = compute.layer_total_topk``,
       default 10). Computed only at ``computation_level: single_branch``.
   * - ``gaussian_fisher``
     - A Gaussian proxy from class-conditional means and variances
       (``I(E; C)_fisher`` and similar, and ``I(Vb; C | E, I)_fisher``),
       configured under ``compute.gaussian_fisher``: ``aggregation``
       (``sum``, ``mean``, ``max``), ``mi_transform`` (``half_log1p``,
       ``log1p``, ``none``) and ``eps``.
   * - ``pid``
     - Partial information decomposition of ``(E, I)``, ``(E, Vout)`` and
       ``(I, Vout)`` about ``C`` and of ``(E, I)`` about ``Vout``, with the
       minimum-mutual-information measure: redundancy
       ``min(I(S1; T), I(S2; T))``, unique terms ``I(Si; T)`` minus the
       redundancy, and synergy ``I(S1, S2; T) - I(S1; T) - I(S2; T) +
       redundancy``. With estimated informations the synergy can be slightly
       negative. ``pid.binarize`` thresholds ``E``, ``I`` and ``Vout`` first
       (``binarize_method``: ``median``, ``mean`` or ``quantile`` with
       ``binarize_threshold``); class labels are never thresholded.
   * - ``soma_coupling``
     - Information of branch currents and outputs about the output of the
       branch's soma.

Other selectors under ``params.selection``:

``scopes``
   Extra breakdowns: ``layer`` (per dendritic layer), ``einet`` (excitatory
   and inhibitory pathways separately) and ``neuron`` (per captured branch
   coordinate; expensive). ``neuron`` does not mean the soma population of a
   layer; use :ref:`representation_accessibility
   <analyzer-representation-accessibility>` for soma populations.

``signal_variants``
   ``network`` (the model's signals, always included), ``lda`` (signals
   projected on LDA weights, ``*_lin``), and ``random`` (a random-weight
   baseline for ``lda``, ``*_sh``; needs
   ``baselines.random_weights.n_shuffles`` above 0).

``params.compute`` sets how branches are combined:

``computation_level`` (default ``single_branch``)
   ``single_branch`` (each branch), ``layer_branch`` (the branches of a layer
   together) or ``all_branch`` (all branches together).

``branch_aggregation`` (default ``mean``)
   For ``layer_branch`` and ``all_branch``: ``mean`` (average the branches),
   ``multivariate`` (joint vectors) or ``sample`` (branches as separate
   observations).

``branch_sample_count_per_layer``, ``branch_sample_strategy``, ``branch_sample_seed``
   Analyze a label-independent sample of branches per layer at
   ``single_branch``; ``branch_sample_strategy`` is ``uniform`` or
   ``parent_soma_balanced`` (cycle through the parent somas).

``max_samples``, ``output_units``, ``normalize_mi``, ``network_population_information``
   A cap on the examples; ``bits`` (default) or ``nats``; division by
   ``H(C)``; and the raw information of the whole soma population (off by
   default).

``params.estimator.method`` chooses the estimator:

``kraskov`` (default; also ``auto``)
   The k-nearest-neighbour estimator, with ``estimator.kraskov.n_neighbors``
   (default 10).

``decoder``
   A cross-validated classifier bound for terms with ``C``, configured by
   ``estimator.decoder`` (``cv_folds``, ``C``, ``standardize``, ``seed``, and
   ``continuous`` for terms between continuous signals).

``binned``
   A plug-in estimator on binned signals: ``estimator.binned.n_bins`` (20),
   ``binning_strategy`` (``quantile`` or ``uniform``), ``bias_correction``
   (``none`` or ``miller_madow``).

``sklearn``
   The scikit-learn estimators. It has no options of its own: the
   ``estimator.sklearn`` keys are not read.

``copula``
   With ``estimator.copula.type: gaussian`` (default), the Gaussian-copula
   estimator of Ince et al. (2017): every column becomes the standard normal
   score of its rank, and the information is that of a Gaussian with the
   scores' covariance, with the analytic bias correction. It is exact for
   Gaussian dependence and a lower bound otherwise, needs more samples than
   dimensions (per class for class-conditional terms), and handles class
   labels through class-conditional entropies, so label order does not
   matter. ``type: kernel`` estimates the copula density of two continuous
   scalars with Gaussian kernels.

``params.baselines`` adds surrogate estimates with the same estimator:
``random_weights`` (``n_shuffles``, ``seed``) for the ``random`` variant, and
``label_shuffle_null`` (``n_shuffles``, ``seed``, and ``components``: the
families that get a ``*_null`` counterpart, from ``basic`` (default),
``pairwise``, ``conditional``, ``ablation_aligned``, ``upstream_unique_cmi``,
``soma_coupling`` or ``all``), which recomputes the terms with permuted labels
as a finite-sample bias estimate.

.. code-block:: yaml

   analysis:
     information:
       enabled: true
       params:
         selection:
           components: ["basic", "pairwise", "conditional", "gaussian_fisher"]
           scopes: ["layer"]
           signal_variants: ["network"]
         compute:
           max_samples: 1000
           computation_level: "single_branch"
           analysis_split: "validation"
         estimator:
           method: "kraskov"
           kraskov:
             n_neighbors: 10
         baselines:
           label_shuffle_null:
             n_shuffles: 20

Flat spellings such as ``params.method``, ``params.components`` or
``params.lda_n_shuffles`` load with a deprecation warning and are moved to the
grouped keys.

Output: ``information_analysis/<name>`` (JSON) and ``<name>_summary.txt``;
with the ``layer`` and ``einet`` scopes, per-layer summaries in
``per_layer_results/``; and figures with fixed names, replaced at every pass.
``dendritic_modeling.utils.information`` provides the estimators as functions.

.. _analyzer-correlation:

``correlation``
~~~~~~~~~~~~~~~

Requires a structured E/I core (an ``ExcitationInhibitionNetwork``; other
cores are skipped with a warning). Pearson correlations between excitatory,
inhibitory and output signals on the test split (at most
``compute.max_samples`` examples, default 5000). Selectors under
``params.selection``, all of them by default:

``components``
   ``total`` (all examples), ``noise`` (within class), ``signal`` (between
   class means; computed only for two-class tasks) and ``tuning`` (class
   selectivity).

``pairs``
   ``EE``, ``II``, ``EI``, ``E_output``, ``I_output``, ``output_output``.

``scopes``
   ``layer`` and ``einet`` (by default), ``neuron``.

``compute.computation_level`` is ``synaptic`` (individual synapses, sampled
with ``params.sampling``: ``sampling_strategy`` ``stratified`` (default) or
``random``, ``n_synapse_pairs``, ``max_pairs_per_category`` and
``stratified_categories``, default ``["within_branch", "within_layer"]``),
``single_branch``, ``layer_branch`` (default) or ``all_branch``. The category
``across_layer``, listed by earlier versions but never implemented, loads
with a warning and is dropped (``dendritic-upgrade-config`` removes it); any
other unknown category raises. Selector entries are compared in lower case with ``-`` and
spaces read as ``_``, and also accept ``all`` (``total``), ``tuning_curves``
(``tuning``), ``e_vout``/``exc_output``/``exc_vout`` (``E_output``), the
matching ``I_output`` spellings, ``vout_vout``/``out_out``
(``output_output``) and ``per_layer``/``per_einet``/``per_neuron``. An unknown
selector entry, computation level or sampling strategy raises when the
configuration loads, with the valid names. Flat spellings such as
``params.components`` or ``params.computation_level`` load with a
deprecation warning and are moved to the grouped keys.

.. code-block:: yaml

   analysis:
     correlation:
       enabled: true
       params:
         selection:
           components: ["total", "noise", "signal"]
           pairs: ["EE", "II", "EI"]
           scopes: ["layer"]
         compute:
           computation_level: "layer_branch"

Output: ``correlation_analysis/<name>`` (JSON), ``<name>_summary.csv`` and
figures with fixed names.

.. _analyzer-compartment-snr:

``compartment_snr``
~~~~~~~~~~~~~~~~~~~

Requires dendritic branch layers and a classifier; other models are skipped
with a warning. Discriminability-style signal-to-noise ratios on the test
split. This is not a simple signal/noise power ratio: branch-level scalar
signals report ``d'^2`` as between-class variance divided by total variance,
while multivariate input and layer signals report ``d_max``, ``d_total``,
``n_sig_dirs`` and ``ratio_sig_dirs`` from between-class versus total
covariance (the squared canonical correlations with the class labels,
computed in float64). They are in-sample, so even label-free activity gives
``d_total`` of about ``(C - 1) D / (n - 1)``; compare layers of different
widths through ``d_total_null_mean`` (the mean over label shuffles) and
``dim_over_n``. A leading direction counts in ``n_sig_dirs`` when its
eigenvalue exceeds the ``null_quantile`` (0.95) of the same-rank eigenvalue
over ``null_permutations`` (100) label shuffles, seeded by ``null_seed``.

``input_capture_mode`` chooses the input view: ``branch`` (default) measures
the tensors entering the dendritic branch layers; ``network`` measures the
model-level excitatory input and inhibitory output of an
``ExcitationInhibitionNetwork``. Further switches: ``input_analysis``,
``layer_analysis``, ``branch_analysis`` and the ``*_aggregation`` options,
all ``true``. Per-epoch runs skip the input analysis.

.. code-block:: yaml

   analysis:
     compartment_snr:
       enabled: true
       training: true
       params:
         input_capture_mode: branch
         null_permutations: 100

Output: ``compartment_snr_analysis/<name>.json`` (per epoch
``epochs/epoch<N>.json``), and after training figures in ``layer_analysis/``
and ``branch_analysis/``, with animations over epochs when per-epoch results
exist.

.. _analyzer-representation-accessibility:

``representation_accessibility``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Final-only. Requires a feedforward ``population_network`` whose populations are
excitatory or inhibitory; it analyzes the post-gate soma outputs of every
layer. It reports five separate estimands:

- class accessibility: Ridge or MLP probes fit on the validation split and
  evaluated on the test split, with label-shuffle nulls;
- joint ``I(X; C)`` of a label-independent population projection, on the
  validation split;
- scalar ``I(x_i; C)`` of individual somas, on the validation split;
- held-out decoding from label-independent nested subsets of population
  coordinates (individual somas of E-only or I-only representations);
- covariance and correlation effective dimension, on the validation split.

The MLP result is nonlinear class accessibility, not mutual information. The
mean of per-soma scalar information is redundancy-sensitive and is not the
joint information of the layer. Every probe is fit on the validation split,
and the test split is used only for the final probe evaluation.

Representations may be ``excitatory``, ``inhibitory``, ``joint``,
``readout`` or an exact name such as ``population:e0``. A network without an
inhibitory soma population records that representation as missing.
``balanced_countsketch`` projections give a label-independent matched width:
use the native width to compare mechanisms with identical architectures, and a
common projected width to compare E, I and joint populations of different
sizes. Projection draws, MLP initializations and random-subset draws are
technical repeats; average them within a trained checkpoint before inference
across training seeds. Within each draw, subset sizes are nested prefixes of
one random coordinate permutation, so the capacity curve is paired. Subset
decoding uses one deterministic ridge probe per coordinate set. Population
geometry reports participation-ratio and entropy effective ranks of the
covariance spectrum (native soma scales) and the correlation spectrum
(standardized somas); these describe dimensional concentration, not class
information.

The defaults are one identity projection ``native`` and one ridge probe
``linear`` with 20 label shuffles; ``information``, ``subset_decoding`` and
``geometry`` are off. ``information.projection_names`` (default
``[matched16]``) must name declared projections.

.. code-block:: yaml

   analysis:
     runtime:
       final:
         mode: "stream"
         batch_size: 256
     representation_accessibility:
       enabled: true
       params:
         representations: ["excitatory", "inhibitory", "joint"]
         projections:
           - name: "matched16"
             type: "balanced_countsketch"
             output_dim: 16
             draws: 5
         probes:
           - name: "linear"
             type: "ridge"
             ridge_alpha: 1.0
             n_label_shuffles: 20
           - name: "nonlinear"
             type: "mlp"
             max_projection_draws: 1
             hidden_dims: [32, 16]
             activation: "relu"
             alpha: 0.0001
             learning_rate_init: 0.001
             batch_size: 256
             max_iter: 300
             early_stopping: true
             validation_fraction: 0.1
             n_iter_no_change: 20
             draws: 3
             n_label_shuffles: 5
         information:
           enabled: true
           compute_joint: true
           compute_per_soma: true
           projection_names: ["matched16"]
           per_soma_representations: ["excitatory", "inhibitory"]
           n_neighbors: 5
           n_label_shuffles: 20
           standardize: true
         subset_decoding:
           enabled: true
           representations: ["excitatory"]
           subset_sizes: [1, 2, 4, 8, 16, 32, 64]
           draws: 10
           probe:
             name: "subset_linear"
             type: "ridge"
             ridge_alpha: 1.0
             n_label_shuffles: 5
         geometry:
           enabled: true
           representations: ["excitatory"]
           variance_epsilon: 1.0e-12
         seed: 42

Output: ``representation_accessibility/<name>.json``. Example profile:
``configs/examples/representation_accessibility_analysis.yaml``. Collect
checkpoint runs with ``dendritic-collect-representations``.

.. _analyzer-path-matched-information:

``path_matched_information``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Final-only. Joint-vector class information on the same nested anatomical
supports as :ref:`path_matched_intervention
<analyzer-path-matched-intervention>`: complete soma-to-distal paths are
selected without data, and their prefixes give the same number of compartment
coordinates at every depth. The signals are the pre-gate excitatory and
inhibitory currents, their joint vector, the child current, the pre-gate
voltage and the post-gate output (``signals``, all six by default). Every
coordinate is standardized on the validation examples without labels, and
the Ross continuous-discrete k-nearest-neighbour estimator gives the
information in bits, with a label-permutation distribution
(``n_label_shuffles``).

The analyzer reads only the validation split. Depth 0 (the soma) uses the
whole soma population once; every other depth uses ``support_draws`` path
draws, which are technical repeats: average them within a checkpoint before
treating independently trained checkpoints as units. ``depths`` defaults to
every depth, soma included. Set ``module_name_prefix`` to select one dendritic
tree when the model has more than one; ``expected_n_somas`` and
``expected_branch_factors`` make the analysis fail on a checkpoint with a
different architecture. An unset ``information_seed`` or ``support_seed`` is
derived from ``experiment.probe_seed`` (see :ref:`analysis-seeds`).

.. code-block:: yaml

   analysis:
     path_matched_information:
       enabled: true
       params:
         signals: [excitation_current, inhibition_current, joint_EI_currents,
                   upstream_child_current, pre_gate_voltage, post_gate_output]
         depths: [0, 1, 2, 3, 4]
         support_draws: 5
         support_seed: 271828
         information_seed: 314159
         n_neighbors: 10
         n_label_shuffles: 20
         expected_n_somas: 20
         expected_branch_factors: [3, 3, 3, 3]

Output: ``path_matched_information/<name>.json``. Example profile:
``configs/examples/path_matched_information_analysis.yaml``. Collect checkpoint
runs with ``dendritic-collect-path-information``.

Interventions
-------------

.. _analyzer-ablation:

``ablation``
~~~~~~~~~~~~

Requires E/I dendritic branch layers (a structured E/I core or a
``population_network``); other models are skipped without a message. Scores
the test split after disrupting one part of the network at a time. Methods
(``params.methods.ablation_methods``, default ``[lesion]``):

``lesion``
   Set the effective weights (after the weight transform) of the selected
   synapses to zero.

``shuffle``
   Permute the selected signal across the examples of a batch
   (``methods.shuffle_seed``). Keeps its distribution but breaks its alignment
   with the labels.

``mean_clamp``
   Replace the selected signal by its mean over the training split, computed
   once before the evaluation and recorded under ``mean_clamp_reference``.
   Keeps the operating point but removes trial-to-trial variability.
   ``methods.clamp_statistic`` accepts only ``mean``.

Another method name or clamp statistic raises when the configuration loads.
Selectors under ``params.selection`` must be set: with the default empty
lists nothing is ablated and only an empty baseline is written. Their entries
are compared in lower case with ``-`` and spaces read as ``_``; an entry that
names no level, target or score raises when the configuration loads, with the
valid names, instead of selecting nothing.

``levels``
   ``layer`` (whole dendritic layers) and ``compartment`` (each branch row in
   turn; lesion only, other methods are skipped).

``targets``
   ``all_synapses`` (or ``all``), ``excitation`` (``exc``), ``inhibition``
   (``inh``) and ``upstream`` (``vb``, ``branch_input``; the coupling from
   child branches).

``metrics``
   ``accuracy``, ``auc``, ``categorical_loglikelihood`` (``categorical_ll``,
   ``loglikelihood``, ``log_likelihood``), ``mse``, ``cosine_similarity``
   (``cosine``) and ``pred_label_mi_bits`` (``pred_label_mi``, ``pred_mi``;
   the information between predicted and true labels, in bits). Scores that do
   not fit the model are dropped.

.. code-block:: yaml

   analysis:
     ablation:
       enabled: true
       params:
         selection:
           levels: ["layer"]
           targets: ["excitation", "inhibition", "upstream", "all_synapses"]
           metrics: ["accuracy", "pred_label_mi_bits"]
         methods:
           ablation_methods: ["lesion", "shuffle", "mean_clamp"]

Output: ``ablation_analysis/<name>.json`` (per epoch ``epochs/epoch<N>.json``)
and figures per method and score, with an animation over epochs when per-epoch
results exist.

.. _analyzer-path-matched-intervention:

``path_matched_intervention``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Final-only. A fixed-count control for comparing causal sensitivity across
dendritic depth. One complete soma-to-distal path is selected per soma and
technical draw without inspecting data, labels, activations or the
checkpoint, and its prefixes give the same number of compartment coordinates
at every depth. Only those coordinates are changed; every other coordinate of
the targeted signal stays as the model produces it.

``targets`` (default: the first three) are ``excitation_current``,
``inhibition_current`` and ``joint_EI_currents``, which act on currents before
the compartment's transfer and gate, and ``post_gate_output``, which acts on
the branch-layer output before the next, more proximal layer consumes it.
``methods`` (default: all three) are ``zero``, ``fixed_train_mean`` (the mean
over the training split, computed once) and ``global_test_shuffle`` (one
test-donor permutation over the whole split, shared across depths and, for
joint targets, across E and I, so a joint shuffle keeps the donor's
sample-wise E/I covariance). ``depths`` are soma-relative and dendritic:
``1`` is proximal, larger values are more distal, ``0`` is rejected; the
default is every dendritic depth. ``paths_per_soma`` (for example
``[1, 2, 3]``) gives a nested dose-response in which each dose contains the
smaller ones; it may not exceed the proximal branch factor, and empty means
one path per soma. Each record reports the accuracy drop, the increase of the
mean cross entropy, and the accuracy drop normalized by the baseline accuracy
above ``chance_accuracy`` (default ``1 / n_classes``); the analysis raises
when the model is not above chance. ``max_samples`` caps the test examples.

Supports depend on ``support_namespace``, ``support_seed`` and
``support_draws``. Set ``module_name_prefix`` when the model has more than one
dendritic tree (any multi-layer stack, or an E/I network whose E and I cells
both have trees), and ``expected_n_somas`` and ``expected_branch_factors`` to
make the analysis fail on a checkpoint with a different architecture.

.. code-block:: yaml

   analysis:
     path_matched_intervention:
       enabled: true
       params:
         targets: ["excitation_current", "inhibition_current", "joint_EI_currents",
                   "post_gate_output"]
         methods: ["zero", "fixed_train_mean", "global_test_shuffle"]
         paths_per_soma: [1, 2, 3]
         support_draws: 5
         support_seed: 271828
         shuffle_seed: 314159
         expected_n_somas: 20
         expected_branch_factors: [3, 3, 3, 3]

The two path-matched analyzers select identical supports only when they use
the same ``support_namespace``, ``support_seed`` and ``support_draws``. An
unset ``support_seed`` is derived from ``experiment.probe_seed`` and the
analyzer's own configuration path, so the two analyzers then draw different
supports: write the same ``support_seed`` in both sections, and check the
support hash when collecting.

Output: ``path_matched_intervention/<name>.json`` with accuracy, cross
entropy, exact available and replaced coordinate counts and hashed supports.
Example profile: ``configs/examples/path_matched_intervention_analysis.yaml``.
Collect checkpoint runs with ``dendritic-collect-path-interventions``.

.. _analyzer-local-inhibitory-population-intervention:

``local_inhibitory_population_intervention``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Final-only. Replaces the complete post-gate output of the inhibitory
population of one network layer of a feedforward ``population_network`` and
re-evaluates the test split. Every layer must have exactly one inhibitory
population, other than the readout, with exactly one outgoing same-step
``ff_inhibitory`` route to a local excitatory population; other graphs raise.
The analysis therefore measures the complete local I-to-E signal of that
layer, not a compartment current or recurrent inhibition.

``methods`` (default: all three) are ``global_test_shuffle`` (donors from the
whole test split), ``within_class_test_shuffle`` (donors of the same class)
and ``zero``; the results report the accuracy drop and the increase of the
mean cross entropy. The global-minus-within-class contrast measures
class-aligned dependence. ``permutation_draws`` (default 3) donor permutations
are technical repeats within a checkpoint (``shuffle_seed``).
``expected_network_layers`` and ``expected_inhibitory_width`` make the
analysis fail on a different architecture; ``max_samples`` caps the test
examples.

.. code-block:: yaml

   analysis:
     local_inhibitory_population_intervention:
       enabled: true
       params:
         methods: [global_test_shuffle, within_class_test_shuffle, zero]
         permutation_draws: 5
         shuffle_seed: 1234
         expected_network_layers: 5
         expected_inhibitory_width: 16

Output: ``local_inhibitory_population_intervention/<name>.json``. Example
profile:
``configs/examples/local_inhibitory_population_intervention_analysis.yaml``.
Collect checkpoint runs with ``dendritic-collect-local-i-interventions``.

.. _analyzer-input-region-intervention:

``input_region_intervention``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Final-only. Paired interventions on image regions of a feedforward image
classifier (recurrent cores and models with typed synapses, spiking somas,
dendritic spikes or soma feedback raise). ``image_shape`` and at least one
region are required. Each region has a ``name`` and a ``kind``:
``central_square`` (the central ``center_fraction`` of the height and width,
default 0.5) or ``surround`` (the rest of the image), the same for every
example; or ``foreground`` (locations whose channel-mean intensity exceeds
``foreground_threshold``, default 0.5) or ``background`` (the other
locations), computed from each unmodified input. ``operations``
(``remove_only``, ``retain_only``) replace the region or everything except
it with zero or with the featurewise mean of the reference split
(``replacement_methods``: ``zero``, ``reference_mean``). The interventions
are evaluated on
``evaluation_split`` (default ``test``) with the reference mean from
``reference_split`` (default ``train``).

Outputs include accuracy and NLL changes, changes of the E and I soma
population responses, and, within matched target trees
(``target_population_polarities``, default ``[excitatory]``), changes of the
excitatory and inhibitory pre-gate drive, pre-gate voltage, post-gate branch
output and soma output. ``identity_soma_class_mapping`` with
``identity_soma_population_name`` and ``identity_soma_network_layer_index``
adds summaries for somas assigned to classes. Results are aggregate
statistics; no per-example values are saved.

Masking the raw image of directly copied E/I image banks perturbs both
routes, so the result is differential E/I response propagation, not a
pathway-specific intervention, and later-layer records are propagated
responses, not pixel receptive fields. Records from excitatory- and
inhibitory-target trees are separate estimands and must not be pooled. The
positive inhibitory drive is not a signed downstream effect: additive units
subtract it, whereas shunting units place it in the divisor.

.. code-block:: yaml

   analysis:
     input_region_intervention:
       enabled: true
       params:
         image_shape: [1, 28, 28]
         regions:
           - name: center
             kind: central_square
             center_fraction: 0.5
           - name: foreground
             kind: foreground
             foreground_threshold: 0.5
         operations: [remove_only, retain_only]
         replacement_methods: [zero, reference_mean]

Output: ``input_region_intervention/<name>.json``. Example profile:
``configs/examples/input_region_intervention_analysis.yaml``. Collect
checkpoint runs with ``dendritic-collect-input-region-interventions``.

.. _analyzer-branch-local-sensitivity:

``branch_local_sensitivity``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Final-only. Local diagonal derivatives of each branch's post-gate output with
respect to its excitatory current ``E``, inhibitory current ``I``, inherited
child current ``C`` and inherited conductance ``G``, at the operating points of
the test split. The analyzer differentiates the fitted gate at the observed
voltage, so learned gate gain is included. It supports shunting, raw
additive, tangent-matched additive and conductance-normalized additive
branches; across-feature additive normalization is rejected because it is
not a local diagonal operation.

The output reports mean signed and absolute derivatives, RMS derivatives, and
mean absolute first-order contributions by soma-relative depth. These are
descriptive operating-point sensitivities, not class information or evidence
that a signal is necessary; use :ref:`path_matched_intervention
<analyzer-path-matched-intervention>` for the latter. ``depths`` defaults to
every dendritic depth; ``include_soma: true`` adds depth 0. The tree
selection and architecture checks are those of ``path_matched_intervention``.

.. code-block:: yaml

   analysis:
     branch_local_sensitivity:
       enabled: true
       params:
         depths: [1, 2, 3, 4]
         module_name_prefix: core_network.layers.0.populations.e0.branch_layers
         expected_n_somas: 64
         expected_branch_factors: [3, 3, 3, 3]

Output: ``branch_local_sensitivity/<name>.json``. Example profile:
``configs/examples/branch_local_sensitivity_analysis.yaml``. Collect checkpoint
runs with ``dendritic-collect-branch-sensitivities``.

.. _analyzer-node-perturbation-credit:

``node_perturbation_credit``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Final-only. Estimates the error of every compartment of the selected
soma-relative ``depths`` from loss changes under antithetic perturbations of
that level (:ref:`node-perturbation`, applied with
:ref:`compartment perturbations <compartment-perturbations>`) and compares it
with the exact error, the gradient of the loss with respect to a zero
perturbation. ``coordinate`` is ``output``, ``voltage`` or, for shunting
levels, ``current``. ``dictionaries`` sets the perturbation directions:
``independent`` (every compartment), ``siblings`` (one direction per group of
children of a parent) or ``neuron`` (one per neuron). ``relative_scales``
multiply the RMS of the perturbed quantity over the level on the analysed
examples (for ``current``, the shunting numerator).
``match_perturbation_energy`` adds runs of the structured dictionaries scaled
by sqrt(D / m), so their expected perturbation energy equals that of
independent draws. ``loss`` is ``cross_entropy`` (class labels) or ``mse``
(any output shape, also one value per example). The first ``max_samples``
examples of the evaluation split are analysed (default 256; ``null`` for
all), and every forward pass takes at most ``analysis.runtime.batch_size``
rows, a row being one example under one draw. ``seed`` fixes the draws; left
unset, a run fills it from its probe seed (seed 0 without one), and every row
records the seed of its draws.

The tree selection and checks are those of ``path_matched_intervention``.
Each perturbed level must run ``DendriticBranchLayer.forward`` once per model
forward (feedforward population networks qualify). Recurrent cores, stateful
current paths, cores with ``autograd_credit_mode: soma_broadcast`` and models
with ``signal_rounding`` are refused: their gradients are not the derivatives
of the forward the perturbations probe (rounding makes it piecewise constant,
while autograd passes gradients through the casts).

The estimator assumes that examples are independent. Overrides
``sample_rolled_denominator``, ``sample_rolled_tangent`` and
``permuted_current_denominator`` are refused anywhere in the model, including
outside the selected tree: they can couple examples and change their pairings
when perturbation draws are batched.

A level that executes but does not influence the loss has zero exact credit,
including when all model parameters are frozen and the loss has no gradient
graph. Alignment to a zero reference is undefined. When both exact and
estimated credit are zero, the cosines and relative metrics are reported as
NaN.

The scale is relative to the RMS over the level, while the effect of a
perturbation depends on each compartment. For ``current`` the voltage
response is the injected current divided by the compartment's total
conductance, so weakly driven, high-resistance compartments receive much
larger relative perturbations, and a nonlinear tree can distort ``current``
estimates at scales where ``output`` and ``voltage`` estimates show no scale
dependence. List several ``relative_scales``: an estimate that changes with
the scale is distorted by it.

Each row reports a depth, dictionary, energy, relative scale and draw count:
cosine with the exact error over all compartments, the mean cosine within each
neuron, sign agreement, relative error, norm ratio, for structured
dictionaries the share of the exact error's energy in their span and the
cosine with that projection, the forward evaluations per example (two per
draw) and the wall time per draw (CUDA work included). The file also records
the seed, device, dtype, ``chunk`` and rows per forward. These are
measurements at a fixed state; they say nothing by themselves about learning
with the estimates.

.. code-block:: yaml

   analysis:
     node_perturbation_credit:
       enabled: true
       params:
         depths: [1, 2]
         module_name_prefix: core_network.layers.0.excitatory_cells.branch_layers
         coordinate: output
         dictionaries: [independent, siblings, neuron]
         relative_scales: [0.01, 0.1]
         draws: [1, 4, 16, 64, 256]
         match_perturbation_energy: true
         max_samples: 256

Output: ``node_perturbation_credit/<name>.json``. Example profile:
``configs/examples/node_perturbation_credit_analysis.yaml``.

Connectivity and weights
------------------------

.. _analyzer-weight:

``weight``
~~~~~~~~~~

Requires E/I dendritic branch layers; other models are skipped without a
message. Statistics of the synaptic weights (``synapse_weights``) and of the
branch coupling weights (``branch_weights``): means, variances, percentiles,
minimum and maximum (``compute_*`` switches), total dendritic strength, and
counts of active synapses above ``weight_threshold`` (1e-6), globally, per
layer and per branch. ``computation_level``, ``branch_aggregation``,
``per_einet_analysis`` and ``per_neuron_analysis`` are accepted but do not
change the result.

Source support
   ``compute_source_support: true`` adds exact source-contact frequency and
   mean effective-conductance profiles by network layer, target population,
   soma, dendritic depth and pathway (``source_support_target_polarities``,
   default ``[excitatory]``; ``source_support_pathways``, default
   ``[ff_excitatory, ff_inhibitory]``), saved as
   ``<name>_source_support_profiles.npz`` and
   ``<name>_source_support_records.csv``. It reads the populations of a
   ``population_network`` or a recurrent structured E/I core; a feedforward
   ``einet`` core yields no profiles. With
   ``source_support_image_shape`` the profiles of image inputs are also drawn
   (``source_support_plot_soma_indices``, ``source_support_fig_save_format``,
   default ``pdf``).

Source-region alignment
   ``compute_source_region_alignment: true`` aligns the image-space profile
   with class-conditional source occupancy estimated on
   ``source_region_reference_split`` (default ``validation``). It requires
   ``compute_source_support``, ``source_support_image_shape`` and an explicit
   class for every target soma (``source_region_soma_to_class``). For a source
   profile :math:`w_p` normalized to sum to one, it reports

   .. math::

      A_{\mathrm{pref}}=\sum_p w_p P(x_p>\tau\mid Y=c_s),\qquad
      T=A_{\mathrm{pref}}-A_{\mathrm{rest}}.

   Contact and conductance weighting are separate quantities, as are
   center/surround support mass and within-region preferred/rest alignment.
   These are structural and activity-alignment statistics, not information or
   causal attribution. Only the network layers in
   ``source_region_image_network_layers`` (default ``[0]``) count as image
   pixels; ``source_region_foreground_thresholds`` (default ``[0.0, 0.5]``),
   ``source_region_center_bounds`` and
   ``source_region_max_reference_samples`` complete the options.

.. code-block:: yaml

   analysis:
     weight:
       enabled: true
       params:
         compute_source_support: true
         source_support_image_shape: [28, 28]
         compute_source_region_alignment: true
         source_region_soma_to_class: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
         source_region_reference_split: validation

Output: ``weight_analysis/<name>`` (JSON), figures with fixed names after
training, and the source-support and alignment files above. Example profiles:
``configs/examples/synaptic_source_support_analysis.yaml`` and
``configs/examples/synaptic_source_region_alignment_analysis.yaml``. Collect
checkpoint runs with ``dendritic-collect-synaptic-support``.

.. _analyzer-source-tuning-support:

``source_tuning_support``
~~~~~~~~~~~~~~~~~~~~~~~~~

Final-only. Requires a feedforward ``population_network``. Asks whether the
exact source coordinates a synaptic pathway selects are tuned to the
preferred class of their target soma. The preferred class is chosen on
``reference_split`` (default ``train``) and the source moments are estimated
on ``evaluation_split`` (default ``validation``); with
``require_independent_splits`` (default ``true``) the two must differ. It
reports uniform-, contact- and conductance-weighted preferred/rest means,
threshold exceedance probabilities, standardized mean differences and the
excess over the uniform eligible-source baseline, and, for every target soma,
the centered class-profile correlation between that soma on the reference
split and every source coordinate on the evaluation split.

Only declared first-layer sources (``image_source_layers``,
``image_source_names``) with a matching ``image_shape`` are treated as pixels;
deeper sources use native neuron coordinates. The analysis keeps per-class
sufficient statistics and never stores example-by-source matrices. It works
with fixed indexed, indexed-release, standard Top-K and variance Top-K
contacts; release models use the mean transmission ``w * p(w)`` while keeping
the structural contact counts, and variance Top-K uses its saved training
statistics. See :doc:`conductance_options` for a contextual-input example.

.. code-block:: yaml

   analysis:
     source_tuning_support:
       enabled: true
       params:
         reference_split: train
         evaluation_split: validation
         pathways: [ff_excitatory, ff_inhibitory]
         target_polarities: [excitatory, inhibitory]
         image_shape: [28, 28]
         image_source_layers: [0]

Output: ``source_tuning_support_analysis/<name>.json``. Example profiles:
``configs/examples/source_tuning_support_analysis.yaml`` and
``configs/examples/conductance_source_tuning_analysis.yaml``. Collect
checkpoint runs with ``dendritic-collect-source-tuning``; its checkpoint
summaries average target somas with equal weight and never pool trained
seeds.

.. _analyzer-synapse-turnover:

``synapse_turnover``
~~~~~~~~~~~~~~~~~~~~

Requires ``training: true`` (the default). Counts the contacts of every
synapse bank (``exc``, ``inh``, ``rec_exc``, ``rec_inh``) of every dendritic
branch layer that stay active, form and disappear. The first snapshot is the
model before training, and later ones are taken every ``snapshot_interval``
epochs (default 1). ``incremental`` compares each snapshot with the previous
one and ``cumulative`` with the first, per module and summed per pathway. It
raises when the model has no masked synapse banks.

.. code-block:: yaml

   analysis:
     synapse_turnover:
       enabled: true
       params:
         snapshot_interval: 5

Output, after training only: ``synapse_turnover_analysis/<name>`` (JSON),
``<name>_turnover.png`` and ``<name>_turnover_by_layer.png``.

.. _analyzer-synapse-pruning:

``synapse_pruning``
~~~~~~~~~~~~~~~~~~~

Final-only. A census of the realized contacts of every synapse bank of every
dendritic branch layer: a contact is realized when the bank's mask selects it
and its effective weight is nonzero, so contacts that pruning set to zero do
not count even when a Top-K mask still selects them. When the run prunes
(``training.main.pruning``), the census before and after every pruning step is
recorded with the pruning statistics, the losses before and after it, and the
contacts removed per layer and pathway.

.. code-block:: yaml

   analysis:
     synapse_pruning:
       enabled: true
       params:
         save_figures: true

Output: ``synapse_pruning_analysis/<name>`` (JSON), ``<name>_summary.txt`` and,
with ``save_figures``, the realized contacts per branch and their mean and
standard deviation against depth.

.. _analyzer-receptive-field:

``receptive_field``
~~~~~~~~~~~~~~~~~~~

Requires dendritic branch layers. Gathers the input-level receptive-field
maps of all branch layers from the weights and plots sampled maps
(``n_rows`` by ``n_cols``). Set ``rf_shape`` when excitatory and inhibitory
maps share one image shape, or ``exc_rf_shape`` and ``inh_rf_shape``.
``save_raw_data`` (default ``true``) saves the raw maps, and
``load_raw_data: true`` reuses a saved file (``raw_filename``) instead of
reading the weights. Optional extensions, which read the test split:

- ``compute_activation_tuning``: class-preferred branch excitation,
  inhibition, E/I ratio and output summaries, and with
  ``activation_samples_per_class`` a class-balanced sample of scalar branch
  signals (``activation_sample_signals``: ``exc``, ``inh``, ``upstream``,
  ``depolarizing_drive``, ``vout``, ``ei_ratio``), restricted by depth and
  branch with the ``activation_sample_*`` options;
- ``compute_component_rfs``: E/I mean and conditioned component maps per soma;
- ``compute_rf_tuning``: aggregate tuning, confusion and entropy summaries,
  discriminability heatmaps, preferred and non-preferred compartment
  summaries for the E/I layer ``rf_tuning_layer_index``, drawn when
  ``plot_rf_tuning`` is true.

.. code-block:: yaml

   analysis:
     receptive_field:
       enabled: true
       params:
         rf_shape: [28, 28]
         compute_activation_tuning: true

Output: ``receptive_field_analysis/<name>_raw_receptive_fields.pt``,
``<name>_summary.json`` (in ``epochs/`` during training), optional tuning
files, and after training ``exc_receptive_fields`` and
``inh_receptive_fields`` figures in ``fig_save_format`` (default ``png``).
``configs/templates/analysis_full_template.yaml`` lists every receptive-field
option.

.. _analyzer-inhibitory-specialization:

``inhibitory_specialization``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Requires recurrent inhibition onto excitatory dendrites (a recurrent core with
``branch_rec_inhibition`` weights); other models are skipped with a warning.
For each local recurrent I-to-E route it computes the gate index
``G = f_soma - f_distal`` for every ``(I, E)`` pair, where ``f_level`` is the
fraction of the realized inhibitory weight mass that the I cell delivers to
the E cell's compartments at that level; pairs without a realized synapse are
excluded. Each I cell is classified from the mean and standard deviation of
its ``G`` values as gate-specialist (mean above ``gate_threshold``, default
0.1, and standard deviation below ``std_threshold``, default 0.15),
context-specialist (mean below ``-gate_threshold``), mixed-conditional
(standard deviation above ``std_threshold``) or diffuse. The level fractions
divided by the compartment count are reported as
``per_compartment_control``. For named ``population_network`` cores the
results are keyed by the excitatory target population (for example
``population_network/microcircuit.pyr``) and list the inhibitory source
populations.
``summarize_inhibitory_specialization_from_compartment_weights`` and
``ei_connectivity_motifs`` use the same computation.

.. code-block:: yaml

   analysis:
     inhibitory_specialization:
       enabled: true
       params:
         gate_threshold: 0.1
         std_threshold: 0.15

Output: ``inhibitory_specialization/<name>.json`` and, with ``save_figures``
(default ``true``), ``<name>.pdf``.

Activity
--------

.. _analyzer-compartment-statistics:

``compartment_statistics``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Requires dendritic branch layers; other models are skipped (recorded as
``skipped`` in ``analysis_status.json``, see :ref:`analysis-status`).
Statistics (mean, variance, percentiles, minimum and maximum, entropy) of the
weights, inputs and activations of the compartments, globally, per layer and
per branch, on the test split. It also runs on named recurrent population
graphs whose branch layers use ``TopKLinear`` synapses. Options: the content
switches ``synapse_weights``, ``branch_weights``, ``inputs``,
``activations``; the level switches ``global_analysis``, ``layer_analysis``,
``branch_analysis``; the statistic switches ``compute_*``;
``per_class_analysis`` (default ``false``); and ``n_samples`` (a cap on the
examples) with ``seed``, which only orders the capped examples.

.. code-block:: yaml

   analysis:
     compartment_statistics:
       enabled: true
       params:
         n_samples: 2000
         per_class_analysis: true

Output: ``compartment_statistics_analysis/<name>`` (JSON) and figures with
fixed names. The E/I ratio figure ``ratio_pairwise_comparison.png`` uses only
the branches that have both excitatory and inhibitory statistics: branches
without inhibitory synapses are left out, and an excitatory-only tree gets no
ratio figure. What was left out is logged and listed under
``missing_outputs`` in ``analysis_status.json``.

.. _analyzer-synaptic-activation:

``synaptic_activation``
~~~~~~~~~~~~~~~~~~~~~~~

Requires dendritic branch layers; other models are skipped without a message.
Records the activation of every synapse (input times effective weight) of the
excitatory, inhibitory and recurrent Top-K synapse banks on the test split,
and summarizes it for the whole network (``network_summary``), per layer
(``layer_summary``) and per branch (``branch_summary``), and as raw values
(``raw_summary``); all four are on. With ``downsample`` (default ``true``)
only ``samples`` examples (default 5) are used; ``logspace`` reports
logarithms. It also runs on named recurrent population graphs whose branch
layers use ``TopKLinear`` synapses.

.. code-block:: yaml

   analysis:
     synaptic_activation:
       enabled: true
       params:
         downsample: true
         samples: 32

Output: ``synaptic_activation_analysis/<name>`` (JSON).

.. _analyzer-branch-activation:

``branch_activation``
~~~~~~~~~~~~~~~~~~~~~

Requires dendritic branch layers; other models are skipped without a message.
Records, per branch layer, the excitatory current, the inhibitory current
(when the layer has inhibitory input) and the branch output on the test split,
with the summary, ``logspace``, ``downsample`` and ``samples`` options of
``synaptic_activation``. ``synapse_activations`` and ``branch_activations``
(both ``true``) choose the recorded signals.

.. code-block:: yaml

   analysis:
     branch_activation:
       enabled: true
       params:
         raw_summary: false
         samples: 32

Output: ``branch_activation_analysis/<name>`` (JSON).

Dynamics and learning
---------------------

.. _analyzer-reactivation-dynamics:

``reactivation_dynamics``
~~~~~~~~~~~~~~~~~~~~~~~~~

Runs after every epoch by default (``training: true``). Tracks every gate
(reactivation) module built from a ``param_*`` or ``linear_*`` activation
type, ``residual_silu`` or ``endpoint_tanh``; a model without such gates
writes nothing. For each it records the gate
parameters, the distribution of the pre-gate voltage (mean, standard
deviation, median, MAD, quantiles) and output statistics; for the unit-range
gates ``ParametricTanh`` and ``ParametricTanhOnlyM`` also the saturation
fractions ``sat_lo = P(output < 0.05)`` and ``sat_hi = P(output > 0.95)``. It
uses one batch of at most ``max_samples`` examples (default 1024) from the
training split during training and from the test split after training,
independently of ``analysis.runtime``. Useful for diagnosing gate-slope drift,
the quality of the gate calibration and the stationarity of the voltage
distribution.

.. code-block:: yaml

   analysis:
     reactivation_dynamics:
       enabled: true
       training: true
       params:
         max_samples: 1024

Output: ``reactivation_dynamics/epochs/<name>.json`` for every pass,
``final`` included. Collect checkpoint runs with
``dendritic-collect-reactivation``, which adds the network layer, population
and soma-relative depth of each gate as columns.

.. _analyzer-routing:

``routing``
~~~~~~~~~~~

Recurrent dendritic cores only (a recurrent structured E/I core or a recurrent
``population_network``); others are skipped. For each population it reports
the share of activity in each dendritic level, as raw L2 norms and as
width-normalized RMS values, and the fraction of each level's signal that
comes from feedforward rather than recurrent input, as means and standard
deviations over all time steps and examples of the test split (processed in
batches of 128). ``routing`` has no ``params``.

.. code-block:: yaml

   analysis:
     routing:
       enabled: true

Output: ``routing_analysis/<name>`` (JSON).

.. _analyzer-jacobian-spectrum:

``jacobian_spectrum``
~~~~~~~~~~~~~~~~~~~~~

Requires recurrent excitation or inhibition onto dendrites; other models are
skipped. The ``top_k`` (default 16) singular values and the spectral radii of
the effective recurrent excitatory and inhibitory weights at each dendritic
level, with the level's time constant when available. This summarizes weight
matrices; ``StateTransitionJacobianAnalyzer`` (see :doc:`analysis`)
differentiates the complete state update instead.

.. code-block:: yaml

   analysis:
     jacobian_spectrum:
       enabled: true
       params:
         top_k: 5
         save_figures: true

Output: ``jacobian_spectrum/<name>.json`` and, with ``save_figures``,
``<name>.pdf``.

.. _analyzer-dendritic-timetraces:

``dendritic_timetraces``
~~~~~~~~~~~~~~~~~~~~~~~~

Simulates how the learned dendritic cascade responds to a synthetic two-pulse
input, using the trained per-level time constants and ``ParametricTanh`` gate
parameters; populations without per-level time constants or with other gates
are skipped. The protocol has ``n_timesteps`` steps (default 90), stimulus
windows ``stim1_window`` and ``stim2_window`` with amplitudes ``stim1_amp`` and
``stim2_amp``, a ``response_window``, and noise ``noise_std`` (seeded by
``seed``). Structured recurrent E/I layers report ``layer_N`` keys; named
population networks report keys such as
``population_network/microcircuit.pyr``.

.. code-block:: yaml

   analysis:
     dendritic_timetraces:
       enabled: true
       params:
         n_timesteps: 80
         save_figures: true

Output: ``dendritic_timetraces/<name>.json`` and one PDF per layer.

.. _analyzer-spike-train:

``spike_train``
~~~~~~~~~~~~~~~

For models with spiking somas (``LIFSoma`` modules; see
:ref:`spiking-soma`); others are skipped. The analyzer hooks the soma
directly, so it works with ``spike_readout`` set to ``spikes``, ``rate`` or
``membrane``. On at most ``max_samples`` test examples (default 512) it
reports compact raster events, mean firing rate, silent-unit and silent-trial
fractions, spike-count Fano factors and ISI mean and coefficient of variation.
Options: ``dt`` (1.0), ``threshold`` (0.5), ``include_raster``,
``max_raster_samples`` (8), ``include_per_timestep_rate``.

.. code-block:: yaml

   analysis:
     spike_train:
       enabled: true
       params:
         max_samples: 512
         include_raster: true
         max_raster_samples: 8
         include_per_timestep_rate: true

Output: ``spike_train_analysis/<name>.json``.

.. _analyzer-local-rule-components:

``local_rule_components``
~~~~~~~~~~~~~~~~~~~~~~~~~

The factors of the local learning rule of ``local_ca`` (see
:ref:`local-credit-assignment`) by dendritic depth and synapse type: branch
voltage, input resistance, excitatory and inhibitory driving forces,
presynaptic drive, eligibilities and, with ``include_broadcast``, the
broadcast error (``broadcast_mode``: ``scalar`` or ``per_soma``;
``error_mode``: ``auto``, ``ce``, ``bce``, ``mse`` or ``none``).
``include_dendritic`` adds the dendritic factors. It reads at most
``max_samples`` examples (default 1024) of the training split during training
and of the test split after it. ``e_rev_exc`` and ``e_rev_inh`` must keep
their values 1.0 and 0.0. Models with indexed-release banks, learnable or
non-unit leak, ``endpoint_tanh`` gates, or additive layers other than raw
additive raise.

.. code-block:: yaml

   analysis:
     local_rule_components:
       enabled: true
       training: true
       params:
         max_samples: 1024
         broadcast_mode: per_soma
         error_mode: auto

Output: ``local_rule_components/epochs/epoch<N>.json`` and ``.csv`` per epoch,
rows appended to ``trajectory.csv`` (``append_training_csv``), and
``final/<name>.json`` and ``.csv`` after training (``save_json``,
``save_csv``).
