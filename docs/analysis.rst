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
         persistent_workers: true
         prefetch_factor: 4
         in_order: null
         max_batches: null
         max_samples: null
         materialize_threshold_mb: 1024
       final:
         mode: "auto"
         batch_size: 256
         num_workers: 4
         pin_memory: true
         persistent_workers: true
         prefetch_factor: 4
         in_order: null
         max_batches: null
         max_samples: null
         materialize_threshold_mb: 1024

These runtime profiles are shared by performance, information, ablation,
correlation, activation, and robustness analyses. For large datasets like
ImageNet, ``mode: "stream"`` avoids materializing the full dataset on the
device. For expensive analyses that still accumulate tensors on CPU
(``information``, ``correlation``, some compartment analyses), use
``max_samples`` to bound total analysis memory explicitly.
For ImageNet-scale streamed evaluation, ``in_order: false`` can reduce
DataLoader head-of-line blocking; leave it ``null`` to keep PyTorch's default
ordering behavior.
Analyzers may temporarily move the model to the requested analysis device,
but they restore the original device and train/eval mode before returning, so
post-training analysis does not leave the model in a different execution state.
When ``mode: "materialize"`` is selected and the dataset supports slicing,
hook-based analyses use a single full batch, matching single-pass behavior on
small single-GPU runs.
The default ``"auto"`` mode is preferred for most experiments: it preserves
materialized behavior on small datasets while switching to streaming
automatically when the estimated dataset footprint becomes large.
If a config omits ``analysis.runtime`` entirely, non-distributed runs use
``"materialize"`` by default, while DDP/FSDP runs keep the shared ``"auto"``
default.

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

``"layer"`` -- within-tree dendritic branch-layer breakdown.
``"einet"`` -- separate excitatory/inhibitory synaptic pathway summaries.
``"neuron"`` -- per captured branch coordinate (expensive). This legacy name
does not denote the joint soma population of a stacked network layer. Use
``representation_accessibility`` for soma-population analyses.

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

Representation Accessibility
----------------------------

``representation_accessibility`` analyzes the post-gate soma outputs of every
feedforward ``PopulationNetwork`` layer. It keeps three estimands separate:

- validation-fit/test-evaluated Ridge or MLP class accessibility;
- joint ``I(X; C)`` on a declared label-independent population projection;
- separately estimated scalar ``I(x_i; C)`` for individual somas.

The MLP result is nonlinear class accessibility, not mutual information. The
mean of per-soma scalar information is redundancy-sensitive and is not the
joint information in the layer. Every probe is fit on ``valid`` and the
untouched ``test`` split is used only for final evaluation. The analyzer is
therefore final-only; ``training: true`` is rejected.

Representations may be ``excitatory``, ``inhibitory``, ``joint``, ``readout``,
or an exact name such as ``population:e0``. Networks without an inhibitory soma
population record that representation as missing rather than confusing raw
inhibitory synaptic current with an I-population output.

``balanced_countsketch`` provides a label-independent matched-width control.
Use native width when comparing mechanisms with identical architectures, and a
common projected width when comparing E, I, and joint populations of different
sizes. Projection draws and MLP initializations are technical repeats; average
them within a trained checkpoint before inference across training seeds.

.. code-block:: yaml

   analysis:
     runtime:
       final:
         mode: "stream"
         batch_size: 256
         num_workers: 4
         max_samples: null
     representation_accessibility:
       enabled: true
       training: false
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
         seed: 42

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
   Discriminability-style compartment SNR. This is not a simple signal/noise
   power ratio: branch-level scalar signals report ``d'^2`` as between-class
   variance divided by total variance, while multivariate input/layer signals
   report ``d_max``, ``d_total``, ``n_sig_dirs``, and ``ratio_sig_dirs`` from
   between-class versus total covariance. The input view is controlled by
   ``params.input_capture_mode``:

   - ``"branch"`` (default): measure the actual tensors entering dendritic
     branch layers. This is the preferred view for E/I and population-network
     dendritic analyses.
   - ``"network"``: for explicit ``ExcitationInhibitionNetwork`` cores, measure
     model-level excitatory input and inhibitory-output diagnostics. Other
     dendritic cores still report branch/layer SNR, but do not add this
     input-output diagnostic.

``multiplicative_gain``
   Measures gain modulation by inhibitory inputs.

``routing``
   Records which branches are active per input class.

``receptive_field``
   Computes dendritic receptive-field weight maps for all branch layers. The
   default mode gathers E/I branch weight maps, plots sampled receptive fields,
   and writes a raw tensor checkpoint
   (``<filename>_raw_receptive_fields.pt`` by default) so plots can be
   regenerated without re-reading model weights. Use
   ``params.rf_shape`` when excitatory and inhibitory receptive fields share one
   image shape, or ``params.exc_rf_shape`` / ``params.inh_rf_shape`` when the two
   streams have different shapes.

   Two optional extensions are available:

   - ``params.compute_activation_tuning: true`` adds class-preferred branch
     excitation, inhibition, E/I ratio, and output activation summaries from the
     provided test split.
   - ``params.compute_rf_tuning: true`` enables aggregate RF tuning,
     confusion/entropy summaries, discriminability/d-prime heatmaps,
     preferred/unpreferred compartment summaries, and E-vs-I RF/tuning plots.
     ``params.plot_rf_tuning`` controls whether those plots are written during
     final analysis.

   A fully commented analysis block, including every preferred RF and SNR key,
   lives at ``configs/templates/analysis_full_template.yaml``.

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

``inhibitory_specialization``
   Structural targeting analysis for recurrent inhibitory populations. For each
   local recurrent I->E route, computes soma-vs-distal targeting indices per
   ``(I, E)`` pair and summarizes inhibitory neurons into targeting motifs such
   as ``mixed-conditional`` and ``context-specialist``. For named
   ``population_network`` cores, the analyzer reports the excitatory target
   population key, such as ``population_network/microcircuit.pyr``, and includes the
   contributing inhibitory source populations.

   .. code-block:: yaml

      analysis:
        inhibitory_specialization_analysis:
          enabled: true
          training: false
          params:
            gate_threshold: 0.1
            mixed_std_threshold: 0.15

``jacobian_spectrum``
   Static recurrent-weight spectrum snapshot for dendritic recurrent models.
   Reports the top singular values and spectral radii of recurrent excitation
   and recurrent inhibition at each dendritic level, together with per-level
   ``tau`` when available.

   .. code-block:: yaml

      analysis:
        jacobian_spectrum_analysis:
          enabled: true
          training: false
          params:
            top_k: 5
            save_figures: true

``dendritic_timetraces``
   Synthetic time-trace simulator that uses the trained per-level ``tau`` and
   reactivation parameters to illustrate how a learned dendritic cascade
   responds to a configurable input protocol. Structured recurrent E/I layers
   report ``layer_N`` keys; named population networks report keys such as
   ``population_network/microcircuit.pyr``.

   .. code-block:: yaml

      analysis:
        dendritic_timetraces_analysis:
          enabled: true
          training: false
          params:
            n_timesteps: 80
            save_figures: true

``spike_train``
   Spike-specific summary for models with opt-in ``LIFSoma`` modules. The
   analyzer hooks the soma directly, so it works with ``spike_readout`` set to
   ``spikes``, ``rate``, or ``membrane``. It reports compact raster events,
   mean firing rate, silent-unit and silent-trial fractions, spike-count Fano
   factors, and ISI mean/CV.

   .. code-block:: yaml

      analysis:
        spike_train:
          enabled: true
          training: false
          params:
            max_samples: 512
            include_raster: true
            max_raster_samples: 8
            include_per_timestep_rate: true

Compatibility Notes
-------------------

Recurrent analysis helpers share a duck-typed introspection layer in
``dendritic_modeling.analysis.utils.recurrent_introspection``. This keeps E/I
analysis outputs stable while letting analyzers and scripts discover named
populations, synapse-type support, and spiking soma state without knowing the
concrete core class.
The same compatibility pass lets hook-based synaptic activation and compartment
statistics run on named recurrent population graphs when their underlying
``TopKLinear`` branch modules are present.

Configs using ``performance_analysis`` or the
``training_runtime`` / ``final_runtime`` performance blocks are accepted and
migrated automatically at load time. Prefer:

- ``analysis.runtime.training``
- ``analysis.runtime.final``
- ``analysis.performance.training_splits``
- ``analysis.performance.final_splits``

If ``analysis.runtime`` is omitted entirely:

- ``distributed.mode: none`` defaults to ``materialize`` for single-process
  analysis.
- ``distributed.mode: ddp`` and ``distributed.mode: fsdp`` keep the shared
  ``auto`` default.
- CPU analysis automatically disables ``pin_memory`` even if the runtime block
  leaves it enabled, so the YAML can keep one shared setting for CPU and GPU
  analysis.
