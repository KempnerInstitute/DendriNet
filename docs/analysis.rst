Analysis
========

All analyses are configured under the ``analysis`` key in the YAML config.
Each module has ``enabled`` (bool) and ``training`` (bool, whether to run
during training vs only post-training) flags, plus module-specific ``params``.

Dataset iteration is controlled centrally via ``analysis.runtime``:

.. code-block:: yaml

   analysis:
     runtime:
       training:
         mode: "auto"        # "auto" | "materialize" | "stream"
         batch_size: 256
         num_workers: 4
         pin_memory: true
         max_batches: null
         max_samples: null
         materialize_threshold_mb: 1024
       final:
         mode: "auto"
         batch_size: 256
         num_workers: 4
         pin_memory: true
         max_batches: null
         max_samples: null
         materialize_threshold_mb: 1024

These runtime profiles are shared by performance, information, ablation,
correlation, activation, and robustness analyses. For large datasets like
ImageNet, ``mode: "stream"`` avoids materializing the full dataset on the
device. For expensive analyses that still accumulate tensors on CPU
(``information``, ``correlation``, some compartment analyses), use
``max_samples`` to bound total analysis memory explicitly.
Analyzers may temporarily move the model to the requested analysis device,
but they restore the original device and train/eval mode before returning, so
post-training analysis does not leave the model in a different execution state.
When ``mode: "materialize"`` is selected and the dataset supports slicing,
hook-based analyses also use a single full batch again, matching the old
single-pass behavior on small single-GPU runs.
The default ``"auto"`` mode is preferred for most experiments: it preserves
the old materialized behavior on small datasets while switching to streaming
automatically when the estimated dataset footprint becomes large.
If a config omits ``analysis.runtime`` entirely, non-distributed runs default
to ``"materialize"`` for backward compatibility, while DDP/FSDP runs keep the
generic ``"auto"`` default.

.. contents:: On this page
   :local:
   :depth: 2

Performance Analysis
--------------------

Computes standard classification/regression metrics: accuracy, AUC,
categorical log-likelihood, MSE, cosine similarity.

.. code-block:: yaml

   analysis:
     runtime:
       training:
         mode: "auto"
         batch_size: 256
       final:
         mode: "auto"
         batch_size: 256
     performance:
       enabled: true
       training: false
       training_splits: ["valid"]
       final_splits: ["train", "valid", "test"]
       params:
         accuracy: true
         auc: true
         categorical_loglikelihood: true

Information Analysis
--------------------

Information-theoretic analysis of network representations. The most
configurable analysis module.

Components
~~~~~~~~~~

Set via ``params.selection.components``:

``"basic"``
   ``I(X; C)`` -- mutual information between each signal and class labels.
   Implemented examples include ``I(E; C)``, ``I(I; C)``, ``I(Vb; C)``,
   ``I(Vout; C)``, ``I(Vinf; C)`` when ``include_vinf`` is enabled, and the
   joint term ``I(E, I; C)``.

``"pairwise"``
   ``I(X; Y)`` -- MI between pairs of signals (E, I, Vb, Vout).
   Implemented examples include ``I(E; I)``, ``I(E; Vout)``,
   ``I(I; Vout)``, ``I(E; Vb)``, ``I(I; Vb)``, ``I(Vb; Vout)``, and the
   analogous ``Vinf`` pairwise terms when enabled.

``"conditional"``
   ``I(X; Y | C)`` -- conditional MI given class.
   Implemented examples include ``I(E; I | C)``, ``I(E; Vout | C)``,
   ``I(I; Vout | C)``, ``I(E; Vb | C)``, ``I(I; Vb | C)``,
   ``I(Vb; Vout | C)``, and ``Vinf`` variants when enabled.

``"gaussian_fisher"``
   Fast Gaussian MI proxy from class-conditional means and variances.
   Config under ``params.compute.gaussian_fisher``: ``aggregation``
   (``"sum"``/``"mean"``/``"max"``), ``mi_transform``
   (``"half_log1p"``/``"log1p"``/``"none"``), ``eps``.
   Implemented examples include ``I(E; C)_fisher``, ``I(I; C)_fisher``,
   ``I(Vb; C)_fisher``, ``I(Vout; C)_fisher``, ``I(Vinf; C)_fisher``, and
   the Fisher-style increment ``I(Vb; C | E, I)_fisher`` when ``Vb`` exists.

``"pid"``
   Synergy/redundancy decomposition (requires external PID library).

``"ablation_aligned"``
   Conditional-MI terms aligned to ablation interventions, e.g.,
   ``I(E; C | I, Vb)`` and ``I(Vout; C | Vb)``.
   Implemented terms depend on signal availability:

   - Always when enabled: ``I(Vout; C | E)`` and ``I(Vout; C | I)``.
   - With ``Vb`` present: ``I(Vout; C | Vb)``, ``I(Vb; C | Vout)``,
     ``I(Vout; C | E, I)``, ``I(E; Vout | I, Vb)``,
     ``I(I; Vout | E, Vb)``, ``I(Vb; Vout | E, I)``, ``I(E; C | Vb)``,
     ``I(I; C | Vb)``, ``I(E, I; C | Vb)``, ``I(E; C | I, Vb)``, and
     ``I(I; C | E, Vb)``.
   - Without ``Vb``: ``I(E; C | I)``, ``I(I; C | E)``,
     ``I(E; Vout | I)``, ``I(I; Vout | E)``, and ``I(Vout; C | E, I)``.
   - With ``Vinf`` enabled: ``I(Vinf; C | Vout)`` and ``I(Vout; C | Vinf)``.

``"upstream_unique_cmi"``
   Adds the extra conditional term ``I(Vb; C | E, I)`` when ``Vb`` exists.
   This can be expensive when ``E`` and ``I`` are high-dimensional.

``"layer_proxies"``
   Aggregate per-branch MI into layer-level proxies: sum, union, topK_sum.
   This derives layer-level metrics such as ``I(Vout; C)_sum``,
   ``I(Vout; C | E, I)_union``, and corresponding ``topK_sum`` summaries
   from single-branch results.

Implemented Metric Notes
~~~~~~~~~~~~~~~~~~~~~~~~

The information analyzer currently emits ``I(Vout; C)`` and
``I(Vout; C | E)``, ``I(Vout; C | I)``, and ``I(Vout; C | E, I)``.

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
     information:
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

Methods (``params.methods.ablation_methods``):

``"lesion"``
   Structural ablation -- disable connection weights entirely.

``"shuffle"``
   Permute the signal across samples in a batch. Preserves the marginal
   distribution but destroys sample-label alignment.

``"mean_clamp"``
   Replace the signal with its batch mean. Preserves operating point but
   removes trial-to-trial variability.

Selectors live under ``params.selection``:

``levels``: ``"layer"``, ``"compartment"``.
``targets``: ``"all_synapses"``, ``"excitation"``, ``"inhibition"``,
``"upstream"``.
``metrics``: ``"accuracy"``, ``"auc"``, ``"categorical_loglikelihood"``,
``"pred_label_mi_bits"``.

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
     correlation:
       enabled: true
       params:
         computation_level: "layer_branch"
         components: ["total", "noise", "signal"]
         pairs: ["EE", "II", "EI"]
         scopes: ["layer"]

Other Analysis Modules
----------------------

Each module follows the same ``enabled``/``training``/``params`` pattern.

``weight``
   Weight distribution statistics and parameter counts.

``synapse_turnover``
   Tracks active synapse changes over training. Parameters:
   ``synaptic_analysis``, ``track_active_synapses``, ``turnover_frequency``.

``noise_perturbation``
   Evaluates robustness to input noise at configurable noise levels.

``compartment_statistics``
   Per-compartment (E, I, branch, soma) activation statistics.

``compartment_snr``
   Signal-to-noise ratio per compartment.

``multiplicative_gain``
   Measures gain modulation by inhibitory inputs.

``routing``
   Records which branches are active per input class.

``receptive_field``
   Computes effective receptive fields per neuron.

``adversarial_robustness``
   FGSM attack evaluation at configurable epsilon values.

``single_layer_contribution``
   Measures each layer's contribution to the network output.

``reactivation_dynamics``
   Per-epoch trajectory tracker for parametric reactivation gates. For every
   ``ParametricTanh`` layer in the model, snapshots the pre-reactivation
   voltage distribution (mean, std, median, MAD, quantiles), the current
   learnable ``(m, b)`` parameters, and the post-gate saturation fractions
   (``sat_lo = P(output < 0.05)``, ``sat_hi = P(output > 0.95)``). Saves
   one JSON file per epoch under ``<run>/reactivation_dynamics/epochs/``.
   Useful for diagnosing gate-slope drift, init-calibration quality, and
   voltage-distribution stationarity during training.

   .. code-block:: yaml

      analysis:
        reactivation_dynamics_analysis:
          enabled: true
          training: true
          params:
            max_samples: 1024

Compatibility Notes
-------------------

Older configs that still use ``performance_analysis`` or the legacy
``training_runtime`` / ``final_runtime`` performance blocks are still accepted
and are migrated automatically at load time. New configs should prefer:

- ``analysis.runtime.training``
- ``analysis.runtime.final``
- ``analysis.performance.training_splits``
- ``analysis.performance.final_splits``

If ``analysis.runtime`` is omitted entirely:

- ``distributed.mode: none`` defaults to ``materialize`` for backward-compatible
  single-process analysis.
- ``distributed.mode: ddp`` and ``distributed.mode: fsdp`` keep the shared
  ``auto`` default.
- CPU analysis automatically disables ``pin_memory`` even if the runtime block
  leaves it enabled, so the YAML can keep one shared setting for CPU and GPU
  analysis.
