Analysis
========

Analyzers measure a trained model: its performance, the information its
compartments carry, the effect of interventions, its connectivity and its
dynamics. Each analyzer has a section ``analysis.<name>`` in the experiment
configuration. Enabled analyzers run on the final model of a training run,
and optionally after every epoch, and write their results into the run
directory. :doc:`analyzers` describes every analyzer, and
:doc:`checkpoint_analysis` shows how to run analyzers on saved checkpoints and
how to collect and summarize the results of many checkpoints.

.. contents:: On this page
   :local:
   :depth: 2

Configuring analyses
--------------------

Every section has three keys:

``enabled`` (default ``false``)
   Run the analyzer. All analyzers are disabled by default.

``training`` (default ``false``, except where the catalogue says otherwise)
   Also run it after every training epoch (see :ref:`analysis-runtime`).

``params``
   The analyzer's options, documented in :doc:`analyzers`. ``routing`` has no
   ``params``.

``analysis.performance`` also has ``training_splits`` and ``final_splits``,
and ``analysis.runtime`` sets how analyzers iterate over data
(:ref:`analysis-runtime-profiles`).

.. code-block:: yaml

   analysis:
     performance:
       enabled: true
       training: true
       training_splits: ["validation"]
       final_splits: ["train", "validation", "test"]
     weight:
       enabled: true
     information:
       enabled: true
       params:
         selection:
           components: ["basic", "conditional"]

Section names are the analyzer names in the :ref:`analyzer-catalogue`, which
are also the field names in a run's saved ``config.json``. Unknown section
names and unknown keys raise when the configuration loads.
The older section names ending in ``_analysis`` (for example
``performance_analysis``) and ``spike_trains`` still load, with a warning; a
section may not be written under both names.

.. _analysis-runtime:

When analyzers run
------------------

After training
   Every enabled analyzer runs once on the model the run ends with (the
   restored best-validation weights when
   ``training.main.common.load_best_state_dict`` is true), whatever its
   ``training`` flag. Its results are named ``final``.

During training
   ``training: true`` also runs the analyzer after every training epoch, with
   results named ``epoch<N>`` (``N`` counts from 1). There is no interval
   option, so per-epoch analyses add their cost to every epoch.

Two analyzers default to ``training: true``: ``synapse_turnover``, which
compares snapshots taken during training and raises with ``training: false``,
and ``reactivation_dynamics``. Ten analyzers are final-only and raise with
``training: true``: ``representation_accessibility``,
``source_tuning_support``, ``input_region_intervention``,
``synapse_pruning``, ``path_matched_intervention``,
``branch_local_sensitivity``, ``node_perturbation_credit``,
``local_inhibitory_population_intervention``, ``path_matched_information``
and ``gain_load_perturbation``. Both errors are
raised before training starts.

Other rules:

- **Multi-stage training.** With ``training.main.strategy: multi_stage`` the
  analyzers run after each stage instead of after training (when
  ``training.main.strategies.multi_stage.analyze_between_stages`` is true,
  the default). Each stage writes into ``<run>/<stage name>/`` and names its
  results after the stage; per-epoch ``performance`` results of all stages
  are also gathered, renumbered, into ``<run>/performance/epochs/``.
- **Distributed training.** Under DDP the analyzers run in the main process
  only. Under FSDP per-epoch analyses are skipped, and the final analysis
  runs in the main process on an unsharded copy of the trained model.
- **Sealed test split.** ``experiment.sealed_test: true`` rejects every
  enabled analyzer (see :ref:`dataset-splits`).
- **Regression models.** For a regression model ``performance`` replaces
  ``accuracy``, ``auc`` and ``categorical_loglikelihood`` by ``mse`` and
  ``cosine_similarity``.
- **Device.** Analyses run on the GPU when one is visible and on the CPU
  otherwise. An analyzer may move the model, and restores its device and
  train/eval mode afterwards.
- **Random state.** Every analysis pass runs in an isolated random state
  seeded with ``experiment.evaluation_seed`` (``information`` also with
  ``experiment.probe_seed``), so enabling analyzers does not change the
  training random stream (see :ref:`analysis-seeds`).
- **Models an analyzer does not support.** Most analyzers that need a
  particular core (an E/I dendritic core, a recurrent core, spiking somas)
  skip other models, often with only a log message and without writing
  files; some raise. :doc:`analyzers` lists the requirement of each. Every
  pass records whether each analyzer completed
  (:ref:`analysis-status`).

.. _analysis-status:

Completion status
~~~~~~~~~~~~~~~~~

Every pass records the status of each analyzer it was asked to run (the
enabled analyzers; during training, those with ``training: true``) in
``analysis_status.json`` in the run directory (in a multi-stage run, in the
stage directory); a pass with no analyzer to run writes no record. The
status of an analyzer is one of:

``completed``
   The analyzer returned and wrote at least one file into its directory, or
   reported that it keeps its results for a later pass (``synapse_turnover``
   during training, recorded under ``deferred_output``).
``skipped``
   The analyzer did not run, with the ``reason``: it reported that it does
   not apply to the model or data, the pass stopped before reaching it, or
   the manager never runs it (a plugin analyzer registered without a phase).
``failed``
   The analyzer raised; ``error_type`` and ``error_message`` record the
   error, which still ends the pass and the run.
``empty``
   The analyzer returned without writing any file into its directory.

A completed analyzer can list ``missing_outputs``: optional outputs it did not
produce, each with the reason (for example a figure that needs inhibitory
statistics on a tree without inhibition). The file holds one record per pass,
under ``passes.<name>`` (``final``, a stage name, or the latest ``epoch<N>``:
per-epoch passes keep only the most recent record), with ``requested`` (the
analyzers asked for), ``finished`` (``false`` while the pass runs, or when an
error stopped it) and ``complete`` (every requested analyzer completed):

.. code-block:: json

   {
     "schema_version": "analysis_status_v1",
     "passes": {
       "final": {
         "training": false,
         "require_completion": false,
         "requested": ["performance", "correlation"],
         "finished": true,
         "complete": false,
         "analyses": {
           "performance": {"status": "completed", "output_dir": "performance"},
           "correlation": {
             "status": "skipped",
             "output_dir": "correlation_analysis",
             "reason": "the model has no ExcitationInhibitionNetwork core"
           }
         }
       }
     }
   }

``AnalysisManager.run_analysis`` returns the record of its pass. With
``analysis.require_completion: true`` (default ``false``) a pass whose
analyses did not all complete raises ``AnalysisIncompleteError`` after
writing the record, so a training run or a checkpoint analysis fails instead
of leaving an incomplete result directory. Result files are the same with
and without the record.

.. code-block:: yaml

   analysis:
     require_completion: true
     performance:
       enabled: true
     correlation:
       enabled: true

.. _analysis-runtime-profiles:

Data iteration: ``analysis.runtime``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``analysis.runtime.training`` applies to the per-epoch runs and
``analysis.runtime.final`` to the run after training. Both have the same keys:

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Key
     - Default
     - Meaning
   * - ``mode``
     - see below
     - ``materialize`` loads the whole (capped) split at once, ``stream``
       iterates in batches through a DataLoader, and ``auto`` materializes
       when the estimated size is at most ``materialize_threshold_mb`` and
       streams otherwise. Any other value raises when the configuration
       loads.
   * - ``batch_size``
     - ``256``
     - Batch size when streaming.
   * - ``num_workers``
     - ``4``
     - DataLoader worker processes when streaming.
   * - ``pin_memory``
     - ``true``
     - Pinned host memory; ignored when the analysis runs on the CPU.
   * - ``persistent_workers``
     - ``false``
     - Keep DataLoader workers between passes.
   * - ``prefetch_factor``
     - ``2``
     - Batches prefetched per worker.
   * - ``in_order``
     - ``null``
     - ``false`` lets workers deliver batches as they finish (useful for
       large streamed image datasets); ``null`` keeps PyTorch's default.
   * - ``max_batches``
     - ``null``
     - Cap on the number of batches per split.
   * - ``max_samples``
     - ``null``
     - Cap on the number of examples an analyzer sees (the first examples
       of the split).
   * - ``materialize_threshold_mb``
     - ``1024``
     - Size threshold of ``auto``.
   * - ``seed``
     - ``null``
     - Seed of the analysis data loaders and evaluations; unset takes
       ``experiment.evaluation_seed``.

``mode`` defaults to ``materialize`` in single-process runs
(``distributed.mode: none``) and to ``auto`` under DDP and FSDP. Keys left out
of a written phase keep these defaults, including the ``mode`` of the
distributed mode. In ``materialize`` mode, and when ``auto`` materializes,
hook-based analyzers process the split in one batch.
On a CUDA device, a split whose one-batch pass certainly cannot fit is
streamed instead (:ref:`analysis-activation-memory`).

.. code-block:: yaml

   analysis:
     runtime:
       training:
         mode: stream
         batch_size: 512
         max_samples: 2048
       final:
         mode: auto
         batch_size: 256

Every analyzer that reads data uses the phase's profile, with these
exceptions: ``routing`` iterates the test split with batches of 128,
``reactivation_dynamics`` takes one batch of ``params.max_samples`` examples,
and ``synapse_turnover``, ``synapse_pruning``, ``inhibitory_specialization``,
``jacobian_spectrum`` and ``dendritic_timetraces`` read no data. Analyzers
with their own sample option (for example ``params.max_samples``) use the
smaller of the two caps. ``performance`` does not apply ``max_samples``, and
applies ``max_batches`` only when it streams.

A ``training_runtime`` or ``final_runtime`` block under
``analysis.performance`` (an older layout) is moved to ``analysis.runtime``
when that section is absent and ignored, with a warning, otherwise; its
``splits`` list becomes ``training_splits`` or ``final_splits``.

.. _analysis-activation-memory:

Activation memory of one-batch passes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The size of a split does not bound the memory of a forward pass over it:
activations grow with the number of examples. Before a CUDA device runs a
materialized split in one forward pass, the library bounds the memory of that
pass from below: a copy of the model runs on the first 2, 4 and 8 examples,
with the random state restored, and the bytes of the tensors it holds at once
are extrapolated to the whole split. When the bound plus the split's own
tensors exceed what the device can still provide (its free memory, PyTorch's
unused cache, and at most a per-process memory fraction), the split is
streamed in ``batch_size`` batches instead of first failing with an
out-of-memory error (an INFO record says so). A pass that fits is unchanged:
it keeps its one batch and gives the same numbers.

The bound covers the scoring of the model through the shared metric
evaluation, used by ``performance``, ``noise_perturbation``,
``single_layer_contribution``, ``multiplicative_gain``,
``gain_load_perturbation`` and ``ablation`` (its scores without signal
interventions; the ``shuffle`` and ``mean_clamp`` interventions use forward
hooks, which disable the bound), and the passes of ``branch_activation``,
``synaptic_activation`` and ``branch_local_sensitivity``. Analyzers that
materialize a split and run their own pass over it are not covered and can
still run out of memory on a split too large for one batch: ``information``,
``correlation``, ``compartment_statistics``, ``compartment_snr``,
``receptive_field``, ``weight`` (source-region alignment),
``source_tuning_support``, ``representation_accessibility``,
``path_matched_information``, ``path_matched_intervention``,
``local_inhibitory_population_intervention``, ``input_region_intervention``,
``node_perturbation_credit``, ``local_rule_components``, ``spike_train`` and
the reference pass of
``ablation``'s ``mean_clamp``. Set ``mode: stream`` (or a smaller
``materialize_threshold_mb`` with ``mode: auto``, or ``max_samples``) in the
runtime profile of such runs.

No bound is used, and the pass runs as before (one full batch; a metric whose
pass fails is evaluated again by streaming, while an analyzer's own pass
raises) in these cases:

- on the CPU, where the memory available to a process (Slurm limits, page
  cache) cannot be measured reliably enough to prove that a pass fails;
- when forward hooks are registered, since the probes would run them;
- for models with ``workspace_mb`` chunking, and for modules that set
  ``batch_adaptive_memory = True`` (a module defined outside the library sets
  it when its allocations do not grow linearly with the batch size, for
  example when it picks its chunk size from the batch size);
- when the probes do not grow linearly with the batch size;
- when a probe fails for any reason, including an out-of-memory error while
  copying the model.

The probes cost one copy of the model on the device (the memory of its
parameters and buffers, freed afterwards) and four forward passes on at most
8 examples under a recording dispatch mode: about 0.07 s for a 14-million
parameter E/I network on an A100 once CUDA is warmed up, and about 0.4 s on
the first call of a process. They are reused for the same model, input shape
and device until a parameter or buffer of the model changes, so they run once
per evaluation pass of a fixed model, again after every training epoch (the
weights change), and again for every lesion of ``ablation``.

Data splits
~~~~~~~~~~~

Each analyzer reads a fixed split, listed in the :ref:`analyzer-catalogue`.
Most read the test split. ``performance`` reads the splits in
``training_splits`` (during training) and ``final_splits`` (after training),
both defaulting to all three. ``representation_accessibility`` fits its
probes, and ``path_matched_information`` estimates information, on the
validation split. Fixed reference statistics (for example the ``mean_clamp``
reference of ``ablation``) come from the training split.

.. _analysis-seeds:

Seeds
~~~~~

Seeds written in an analysis section are used unchanged. Every unset seed is
filled when the analyzers are built:

- ``analysis.runtime.<phase>.seed`` takes ``experiment.evaluation_seed``;
- every other unset field named ``seed`` or ending in ``_seed`` takes a seed
  derived from ``experiment.probe_seed`` and the field's configuration path,
  so each analyzer has its own stream, independent of which other analyzers
  are enabled.

Both experiment seeds fall back to ``experiment.seed`` (see
:ref:`random-seeds`). The filled values are saved in the run's
``config.json``. Because the derivation uses the configuration path, two
analyzers never share a derived seed, and runs whose ``experiment.probe_seed``
differs (by default, runs with different ``experiment.seed``) draw different
analysis seeds. Write a seed explicitly when an analysis must use the same
random numbers in every checkpoint (for example
``gain_load_perturbation.params.base_draw_seed``), and write the same
``support_seed`` (and ``support_namespace`` and ``support_draws``) in both
sections to make ``path_matched_intervention`` and
``path_matched_information`` select the same supports.

.. _analysis-outputs:

Where results are written
-------------------------

An analyzer writes into ``<run directory>/<directory>/``, where the run
directory is ``<outputs.results_dir>/<outputs.run_name>_<YYYYmmdd_HHMMSS>/``
(see :ref:`run-outputs`) and ``<directory>`` is the analyzer's directory in the
:ref:`analyzer-catalogue`. A result is named after its pass: ``final``,
``epoch<N>`` or a multi-stage stage name; this page writes ``<name>`` for it.
The directory is created when the analyzer first writes, so an analyzer that
skips the model leaves no directory.

The JSON result files follow four patterns:

- ``<name>.json``: ``performance``, ``representation_accessibility``,
  ``source_tuning_support``, ``input_region_intervention``, ``ablation``,
  ``path_matched_intervention``, ``branch_local_sensitivity``,
  ``node_perturbation_credit``,
  ``local_inhibitory_population_intervention``, ``path_matched_information``,
  ``compartment_snr``, ``multiplicative_gain``, ``gain_load_perturbation``,
  ``inhibitory_specialization``, ``jacobian_spectrum``,
  ``dendritic_timetraces``, ``spike_train``.
- ``<name>`` (JSON without a file extension): ``information``, ``weight``,
  ``correlation``, ``noise_perturbation``, ``single_layer_contribution``,
  ``compartment_statistics``, ``routing``, ``synaptic_activation``,
  ``branch_activation``, ``synapse_turnover``, ``synapse_pruning``.
- per-epoch results in an ``epochs/`` subdirectory: ``performance``,
  ``ablation``, ``compartment_snr`` and ``multiplicative_gain`` write
  ``epochs/epoch<N>.json``; ``reactivation_dynamics`` writes every pass,
  ``final`` included, to ``epochs/<name>.json``; ``local_rule_components``
  writes ``epochs/epoch<N>.json`` and ``.csv``, appends to
  ``trajectory.csv``, and writes ``final/final.json`` and ``.csv`` after
  training.
- other names: ``adversarial_robustness`` writes
  ``adversarial_robustness_<name>.json``, and ``receptive_field`` writes
  ``<name>_summary.json`` (in ``epochs/`` during training) and the raw maps
  ``<name>_raw_receptive_fields.pt``.

Many analyzers also save figures (PNG or PDF; some have a ``save_figures`` or
figure-format option). Some figures use fixed file names and are replaced at
every pass, and some are drawn only after training, including animations of
per-epoch results (MP4, or GIF when ``ffmpeg`` is missing). Series are never
told apart by colour alone: each line has its own marker or line style, each
bar series its own hatch, and overlaid histograms their own outline style
(``dendritic_modeling.plotting.styles``).

Part of the directory of a two-epoch run in which ``performance``,
``information``, ``ablation``, ``reactivation_dynamics`` and
``synapse_turnover`` ran with ``training: true`` (``synapse_turnover`` only
takes snapshots during training and writes after it):

.. code-block:: text

   <run directory>/
     ablation_analysis/
       epochs/epoch1.json
       epochs/epoch2.json
       final.json
       final_lesion_accuracy.png
     information_analysis/
       epoch1
       epoch1_summary.txt
       epoch2
       epoch2_summary.txt
       final
       final_summary.txt
       aggregated_basic_mi_comparison_depth.png
     performance/
       epochs/epoch1.json
       epochs/epoch2.json
       final.json
       performance_evolution_accuracy.png
     reactivation_dynamics/
       epochs/epoch1.json
       epochs/epoch2.json
       epochs/final.json
     synapse_turnover_analysis/
       final
       final_turnover.png
       final_turnover_by_layer.png

.. _analyzer-catalogue:

Analyzer catalogue
------------------

Every analyzer registered in ``dendritic_modeling.analysis.registry``. The
configuration block of each is ``analysis.<analyzer>``. "During training"
says whether ``training: true`` is possible: "yes", "no" (final-only), or the
default when it is ``true``. "Directory" is the output directory inside the
run directory.

.. list-table::
   :header-rows: 1
   :widths: 23 37 10 15 15

   * - Analyzer
     - Measures
     - During training
     - Data
     - Directory
   * - :ref:`performance <analyzer-performance>`
     - Accuracy, AUC, log-likelihood, MSE, cosine similarity
     - yes
     - ``training_splits`` / ``final_splits``
     - ``performance``
   * - :ref:`noise_perturbation <analyzer-noise-perturbation>`
     - Scores under uniform and Gaussian input noise
     - yes
     - test
     - ``noise_perturbation_analysis``
   * - :ref:`multiplicative_gain <analyzer-multiplicative-gain>`
     - Scores over a sweep of input gains
     - yes
     - test
     - ``multiplicative_gain_analysis``
   * - :ref:`gain_load_perturbation <analyzer-gain-load-perturbation>`
     - Scores over a factorial of shared gain and input load
     - no
     - test
     - ``gain_load_perturbation``
   * - :ref:`adversarial_robustness <analyzer-adversarial-robustness>`
     - Accuracy under FGSM attacks
     - yes
     - test
     - ``adversarial_robustness_analysis``
   * - :ref:`single_layer_contribution <analyzer-single-layer-contribution>`
     - Scores with only one dendritic layer intact
     - yes
     - test
     - ``single_layer_contribution``
   * - :ref:`information <analyzer-information>`
     - Mutual information between compartment signals and classes
     - yes
     - test (or validation)
     - ``information_analysis``
   * - :ref:`correlation <analyzer-correlation>`
     - Signal, noise and total correlations of E, I and output signals
     - yes
     - test
     - ``correlation_analysis``
   * - :ref:`compartment_snr <analyzer-compartment-snr>`
     - Class discriminability of inputs, layers and branches
     - yes
     - test
     - ``compartment_snr_analysis``
   * - :ref:`representation_accessibility <analyzer-representation-accessibility>`
     - Probe decodability and information of soma populations
     - no
     - validation, test
     - ``representation_accessibility``
   * - :ref:`path_matched_information <analyzer-path-matched-information>`
     - Class information on matched-width dendritic supports by depth
     - no
     - validation
     - ``path_matched_information``
   * - :ref:`ablation <analyzer-ablation>`
     - Score drops after lesioning, shuffling or clamping synapses
     - yes
     - test (reference: train)
     - ``ablation_analysis``
   * - :ref:`path_matched_intervention <analyzer-path-matched-intervention>`
     - Score changes after replacing matched-width supports by depth
     - no
     - test (reference: train)
     - ``path_matched_intervention``
   * - :ref:`local_inhibitory_population_intervention <analyzer-local-inhibitory-population-intervention>`
     - Score changes after replacing one layer's inhibitory population
     - no
     - test
     - ``local_inhibitory_population_intervention``
   * - :ref:`input_region_intervention <analyzer-input-region-intervention>`
     - Effects of removing or keeping image regions
     - no
     - test (reference: train)
     - ``input_region_intervention``
   * - :ref:`branch_local_sensitivity <analyzer-branch-local-sensitivity>`
     - Local derivatives of branch outputs by depth
     - no
     - test
     - ``branch_local_sensitivity``
   * - :ref:`node_perturbation_credit <analyzer-node-perturbation-credit>`
     - Compartment errors estimated from loss changes, against exact errors
     - no
     - test
     - ``node_perturbation_credit``
   * - :ref:`weight <analyzer-weight>`
     - Weight statistics; optional exact source-support profiles
     - yes
     - none (alignment: validation)
     - ``weight_analysis``
   * - :ref:`source_tuning_support <analyzer-source-tuning-support>`
     - Class tuning of the inputs each sparse pathway selects
     - no
     - train, validation
     - ``source_tuning_support_analysis``
   * - :ref:`synapse_turnover <analyzer-synapse-turnover>`
     - Contacts kept, formed and removed during training
     - required (default ``true``)
     - none
     - ``synapse_turnover_analysis``
   * - :ref:`synapse_pruning <analyzer-synapse-pruning>`
     - Census of realized contacts and of pruning steps
     - no
     - none
     - ``synapse_pruning_analysis``
   * - :ref:`receptive_field <analyzer-receptive-field>`
     - Receptive-field maps of branch layers
     - yes
     - none (tuning options: test)
     - ``receptive_field_analysis``
   * - :ref:`inhibitory_specialization <analyzer-inhibitory-specialization>`
     - Targeting of recurrent inhibition across dendritic levels
     - yes
     - none
     - ``inhibitory_specialization``
   * - :ref:`compartment_statistics <analyzer-compartment-statistics>`
     - Statistics of weights, inputs and activations
     - yes
     - test
     - ``compartment_statistics_analysis``
   * - :ref:`synaptic_activation <analyzer-synaptic-activation>`
     - Distributions of synaptic activations
     - yes
     - test
     - ``synaptic_activation_analysis``
   * - :ref:`branch_activation <analyzer-branch-activation>`
     - Distributions of branch currents and outputs
     - yes
     - test
     - ``branch_activation_analysis``
   * - :ref:`reactivation_dynamics <analyzer-reactivation-dynamics>`
     - Gate parameters and pre-gate voltage statistics
     - default ``true``
     - train during training, test after
     - ``reactivation_dynamics``
   * - :ref:`routing <analyzer-routing>`
     - Activity shares of dendritic levels in recurrent cores
     - yes
     - test
     - ``routing_analysis``
   * - :ref:`jacobian_spectrum <analyzer-jacobian-spectrum>`
     - Singular values of recurrent weights per level
     - yes
     - none
     - ``jacobian_spectrum``
   * - :ref:`dendritic_timetraces <analyzer-dendritic-timetraces>`
     - Simulated responses of the learned dendritic cascade
     - yes
     - synthetic input
     - ``dendritic_timetraces``
   * - :ref:`spike_train <analyzer-spike-train>`
     - Firing statistics of spiking somas
     - yes
     - test
     - ``spike_train_analysis``
   * - :ref:`local_rule_components <analyzer-local-rule-components>`
     - Factors of the local learning rule by depth
     - yes
     - train during training, test after
     - ``local_rule_components``

``configs/templates/analysis_full_template.yaml`` writes out the options of
every analyzer except ``representation_accessibility``,
``source_tuning_support`` and ``input_region_intervention``, which have their
own example files in ``configs/examples/`` (see :doc:`analyzers`).

Python interface
----------------

``dendritic_modeling.analysis.AnalysisManager`` runs the configured analyzers
on a model outside the training scripts:

.. code-block:: python

   from dendritic_modeling.analysis import AnalysisManager
   from dendritic_modeling.config import load_analysis_config

   analysis_config = load_analysis_config("analysis_profile.yaml")
   manager = AnalysisManager(
       model,
       data={"train": train_ds, "valid": valid_ds, "test": test_ds},
       analysis_config=analysis_config,
       save_root="analysis_output",
       device="cpu",
       evaluation_seed=42,
       probe_seed=42,
   )
   manager.run_analysis(filename="final", training=False)

``load_analysis_config(path, distributed_mode="none")`` reads a file whose
root is ``analysis:`` or that is the analysis mapping itself. ``save_root`` is
required. ``run_analysis`` writes files and returns ``None``; read the results
from ``save_root``. Without ``evaluation_seed`` and ``probe_seed`` unset seeds
stay unset and each analyzer uses its own default.

The package also exports analyzers that are used directly from Python rather
than configured in ``analysis``: ``ClassAccessibilityProbeAnalyzer`` and
``LinearClassAccessibilityAnalyzer`` (array-level probe decodability),
``MatchedClassInformationAnalyzer`` (matched-support class information),
``SameCheckpointCausalInterventionAnalyzer`` (phase, pathway or depth
interventions on one checkpoint), ``StateTransitionJacobianAnalyzer``
(leading singular value of a recurrent core's one-step state Jacobian),
``InhibitoryRoutingComparison`` (seed-level records and paired summaries at
one contact budget) and ``PoissonExposureTransferAnalyzer``, which
``dendritic-poisson-exposure-transfer`` runs (see :doc:`checkpoint_analysis`).

Adding an analyzer
------------------

An analyzer outside the library is a plugin component: its package registers
the analyzer with ``register_analyzer(..., phase=...)`` and its configuration
section ``analysis.<name>`` with ``register_config_section``, and the manager
runs it in that phase of the order (:doc:`extending`).

An analyzer of the library itself needs:

#. a parameter and section dataclass, and a field of ``AnalysisConfig``, in
   ``dendritic_modeling/config/analysis.py``;
#. an ``AnalyzerSpec`` in ``_BUILTIN_ANALYZERS``
   (``dendritic_modeling/analysis/registry.py``) with its output directory,
   constructor style and whether it supports or requires per-epoch runs;
#. a step in the execution plan
   (``dendritic_modeling/analysis/managers/analysis_plan.py``) that calls it
   with the right data;
#. tests, an entry in ``configs/templates/analysis_full_template.yaml`` and a
   section in :doc:`analyzers`.

See :ref:`extension-registries` for the other extension points.
