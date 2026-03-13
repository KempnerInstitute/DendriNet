Analysis
========

All analyses are configured under the ``analysis`` key in the YAML config.
Each module has ``enabled`` (bool) and ``training`` (bool, whether to run
during training vs only post-training) flags, plus module-specific ``params``.

.. contents:: On this page
   :local:
   :depth: 2

Performance Analysis
--------------------

Computes standard classification/regression metrics: accuracy, AUC,
categorical log-likelihood, MSE, cosine similarity.

.. code-block:: yaml

   analysis:
     performance_analysis:
       enabled: true

Information Analysis
--------------------

Information-theoretic analysis of network representations. The most
configurable analysis module.

Components
~~~~~~~~~~

Set via ``params.selection.components``:

``"basic"``
   ``I(X; C)`` -- mutual information between each signal and class labels.

``"pairwise"``
   ``I(X; Y)`` -- MI between pairs of signals (E, I, Vb, Vout).

``"conditional"``
   ``I(X; Y | C)`` -- conditional MI given class.

``"gaussian_fisher"``
   Fast Gaussian MI proxy from class-conditional means and variances.
   Config under ``params.compute.gaussian_fisher``: ``aggregation``
   (``"sum"``/``"mean"``/``"max"``), ``mi_transform``
   (``"half_log1p"``/``"log1p"``/``"none"``), ``eps``.

``"pid"``
   Synergy/redundancy decomposition (requires external PID library).

``"ablation_aligned"``
   Conditional-MI terms aligned to ablation interventions, e.g.,
   ``I(E; C | I, Vb)`` and ``I(Vout; C | Vb)``.

``"layer_proxies"``
   Aggregate per-branch MI into layer-level proxies: sum, union, topK_sum.

Signal Variants
~~~~~~~~~~~~~~~

Set via ``params.selection.signal_variants``:

``"network"``
   Network-learned weights (always included).

``"lda"``
   LDA-optimized weights for ``E_lin``, ``I_lin``, ``Vb_lin`` -- maximizes
   class separation.

``"random"``
   Random-weight baseline (requires ``lda_n_shuffles > 0``).

Computation Levels
~~~~~~~~~~~~~~~~~~

``"single_branch"``
   Per individual branch (most granular).

``"layer_branch"``
   Aggregate branches within each layer.

``"all_branch"``
   Aggregate all branches across the network.

Branch Aggregation
~~~~~~~~~~~~~~~~~~

``"mean"``
   Average branch activations to scalars.

``"multivariate"``
   Use raw multi-dimensional data for joint MI computation.

``"sample"``
   Treat each branch as a separate observation.

Scopes
~~~~~~

``"layer"`` -- per-layer breakdown. ``"einet"`` -- separate E vs I summaries.
``"neuron"`` -- per-neuron (expensive).

Estimator Methods
~~~~~~~~~~~~~~~~~

``"kraskov"`` (default) -- k-NN estimator (``n_neighbors`` configurable).
``"decoder"`` -- classifier-based MI bound. ``"binned"`` -- binned MI.
``"sklearn"`` -- sklearn functions. ``"copula"`` -- copula-based.

.. code-block:: yaml

   analysis:
     information_analysis:
       enabled: true
       training: false
       params:
         selection:
           components: ["basic", "pairwise", "conditional", "gaussian_fisher"]
           scopes: ["layer"]
           signal_variants: ["network"]
         compute:
           max_samples: 1000
           computation_level: "single_branch"
           branch_aggregation: "mean"
         estimator:
           method: "kraskov"
           kraskov:
             n_neighbors: 10

Ablation Analysis
-----------------

Evaluates the contribution of specific network components by disabling or
disrupting them.

Methods (``params.ablation_methods``):

``"lesion"``
   Structural ablation -- disable connection weights entirely.

``"shuffle"``
   Permute the signal across samples in a batch. Preserves the marginal
   distribution but destroys sample-label alignment.

``"mean_clamp"``
   Replace the signal with its batch mean. Preserves operating point but
   removes trial-to-trial variability.

``levels``: ``"layer"``, ``"compartment"``.
``targets``: ``"all_synapses"``, ``"excitation"``, ``"inhibition"``,
``"upstream"``.
``metrics``: ``"accuracy"``, ``"auc"``, ``"categorical_loglikelihood"``,
``"pred_label_mi_bits"``.

.. code-block:: yaml

   analysis:
     ablation_analysis:
       enabled: true
       params:
         ablation_methods: ["lesion", "shuffle", "mean_clamp"]
         levels: ["layer"]
         targets: ["excitation", "inhibition", "upstream", "all_synapses"]
         metrics: ["accuracy", "pred_label_mi_bits"]

Correlation Analysis
--------------------

Pearson correlations between excitatory, inhibitory, and output signals at
configurable granularity.

``components``: ``"total"`` (all samples), ``"noise"`` (within-class),
``"signal"`` (between-class), ``"tuning"`` (selectivity).

``pairs``: ``"EE"``, ``"II"``, ``"EI"``, ``"E_output"``, ``"I_output"``,
``"output_output"``.

``scopes``: ``"layer"``, ``"einet"``, ``"neuron"``.

``computation_level``: ``"synaptic"``, ``"single_branch"``,
``"layer_branch"``, ``"all_branch"``.

.. code-block:: yaml

   analysis:
     correlation_analysis:
       enabled: true
       params:
         computation_level: "layer_branch"
         components: ["total", "noise", "signal"]
         pairs: ["EE", "II", "EI"]
         scopes: ["layer"]

Other Analysis Modules
----------------------

Each module follows the same ``enabled``/``training``/``params`` pattern.

``weight_analysis``
   Weight distribution statistics and parameter counts.

``synapse_turnover_analysis``
   Tracks active synapse changes over training. Parameters:
   ``synaptic_analysis``, ``track_active_synapses``, ``turnover_frequency``.

``noise_perturbation_analysis``
   Evaluates robustness to input noise at configurable noise levels.

``compartment_statistics_analysis``
   Per-compartment (E, I, branch, soma) activation statistics.

``compartment_snr_analysis``
   Signal-to-noise ratio per compartment.

``multiplicative_gain_analysis``
   Measures gain modulation by inhibitory inputs.

``routing_analysis``
   Records which branches are active per input class.

``receptive_field_analysis``
   Computes effective receptive fields per neuron.

``adversarial_robustness_analysis``
   FGSM attack evaluation at configurable epsilon values.

``single_layer_contribution_analysis``
   Measures each layer's contribution to the network output.
