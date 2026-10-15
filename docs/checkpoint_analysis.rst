.. _checkpoint-analysis:

Analyzing saved checkpoints
===========================

Analyzers can run on saved checkpoints without training again. The workflow
has four steps, each with its own command:

#. ``dendritic-analyze-checkpoint`` runs an analysis profile on one
   checkpoint;
#. ``dendritic-submit-checkpoint-analysis`` runs it on many checkpoints as one
   Slurm job array, from a manifest;
#. a collector checks every finished job and gathers its results into
   checkpoint-level tables;
#. ``dendritic-summarize-checkpoint-metrics`` summarizes those tables across
   independently trained checkpoints, with bootstrap intervals and paired
   comparisons.

The collectors keep technical repeats (support draws, noise draws,
permutations) within each checkpoint and never pool trained checkpoints;
inference across checkpoints happens only in the summary step.
:ref:`cli-reference` lists every flag of these commands. This page also covers
``dendritic-select-validation-epochs``, the Poisson exposure-transfer
commands and the analyses of recurrent models by task condition
(:ref:`recurrent-condition-analyses`).

.. contents:: On this page
   :local:
   :depth: 2

Running analyzers on one checkpoint
-----------------------------------

.. code-block:: bash

   dendritic-analyze-checkpoint \
     --config run_dir/config.json \
     --checkpoint run_dir/final_model.pt \
     --analysis-config analysis_profile.yaml \
     --output-dir posthoc/run_0

``--config`` is the resolved training configuration of the run (the
``config.json`` of its run directory, or an equivalent YAML file).
``--analysis-config`` is the analysis profile: a file with an ``analysis:``
root, or the analysis mapping itself, in the format of
:doc:`analysis`. It replaces the training configuration's ``analysis`` section;
the two are not merged. An unset ``analysis.runtime`` mode follows the training
configuration's ``distributed.mode`` (see :ref:`analysis-runtime-profiles`).

.. code-block:: yaml

   analysis:
     runtime:
       final:
         mode: stream
         batch_size: 512
     performance:
       enabled: true
       final_splits: ["validation", "test"]
     noise_perturbation:
       enabled: true
       params:
         n_samples: 10
         mse: false
         cosine_similarity: false
     multiplicative_gain:
       enabled: true
       params:
         gain_max: 4.0
         n_steps: 5
         accuracy: true

The command rebuilds the datasets and the model from the training
configuration with its recorded seeds, loads the checkpoint, and runs the
profile's analyzers once with the result name ``--filename`` (default
``final``) on ``--device`` (default: the GPU when one is visible). It never
trains or changes the checkpoint. The checkpoint may be a state dictionary or
a mapping that holds one under ``model_state_dict``, ``state_dict`` or
``model``; a ``module.`` prefix on every key is removed. Loading is strict:
missing or unexpected keys raise unless ``--no-strict`` is given, which records
them instead (the collectors reject such runs).

``--output-dir`` becomes the run directory: the analyzer directories of
:ref:`analysis-outputs` are written there, together with

- ``checkpoint_analysis_provenance.json``: the paths and SHA-256 hashes of the
  configuration, checkpoint and analysis profile, the resolved seeds, the
  dataset sizes, the checkpoint-loading record (strict loading, missing and
  unexpected keys), the enabled analyzers, ``analysis_status`` (the record of
  the pass, with the status of every analyzer; see :ref:`analysis-status`)
  and the code identity;
- ``analysis_status.json``, the same status record;
- ``resolved_seeds.json``, ``runtime_environment.json``, ``train.log`` and
  ``dendritic_modeling.log``, as in a training run (see :ref:`run-outputs`).

Dataset fingerprints are not written. The training-time rule that a sealed
test split rejects every analyzer is not applied here; the size bound
``experiment.max_test_samples`` still is (see :ref:`dataset-splits`).

Job files
~~~~~~~~~

Arrays of checkpoint analyses, and every collector, use a job file in JSON
Lines format: one JSON object per line, one line per checkpoint, with the
four arguments of ``dendritic-analyze-checkpoint``:

.. code-block:: json

   {"config": "runs/seed_0/config.json", "checkpoint": "runs/seed_0/final_model.pt", "analysis_config": "profiles/noise.yaml", "output_dir": "posthoc/seed_0"}
   {"config": "runs/seed_1/config.json", "checkpoint": "runs/seed_1/final_model.pt", "analysis_config": "profiles/noise.yaml", "output_dir": "posthoc/seed_1"}

Optional keys are ``filename`` and ``device``, and ``config_sha256``,
``checkpoint_sha256`` and ``analysis_config_sha256``, which some collectors
use (see :ref:`checkpoint-collectors`). Collectors copy every key into their
tables as a ``job_<key>`` column, so further keys (for example ``seed`` or an
arm label) become the design columns of the summary step. Relative paths
resolve against the working directory of the command that reads the file, and
every path must match the one recorded in the job's provenance.

Running many checkpoints as a job array
---------------------------------------

``dendritic-submit-checkpoint-analysis --manifest manifest.yaml`` prepares the
job files of a set of checkpoints and submits them as one Slurm array, one
task per checkpoint. Each task runs
``python -m dendritic_modeling.scripts.analysis.run_checkpoint_analysis_job``,
which checks the source checkout and then runs ``dendritic-analyze-checkpoint``
with strict loading.

.. code-block:: json

   {
     "schema_version": "checkpoint_analysis_manifest_v1",
     "name": "noise-robustness",
     "output_root": "posthoc",
     "analysis_config": "profiles/noise.yaml",
     "completion_artifacts": ["noise_perturbation_analysis/final"],
     "jobs": [
       {"id": "seed_0", "config": "runs/seed_0/config.json", "checkpoint": "runs/seed_0/final_model.pt", "seed": 0},
       {"id": "seed_1", "config": "runs/seed_1/config.json", "checkpoint": "runs/seed_1/final_model.pt", "seed": 1}
     ],
     "pilot_job_ids": ["seed_0"],
     "scheduler": {"time": "02:00:00", "cpus_per_task": 4, "memory": "16G", "gpus": 1}
   }

Manifest keys (paths are relative to the manifest file, including the paths
inside a ``jobs_file``):

``schema_version``
   ``checkpoint_analysis_manifest_v1``.

``output_root`` (required)
   Directory of the job outputs and of the ``jobs/`` ledger.

``analysis_config`` (required), ``analysis_config_sha256``
   The analysis profile of every job, and an optional hash it must match.

``completion_artifacts`` (required)
   Relative paths, inside a job's output directory, that mark the job as
   complete together with ``checkpoint_analysis_provenance.json``.

``jobs`` or ``jobs_file`` (exactly one)
   The jobs, or a YAML file with ``schema_version: checkpoint_analysis_jobs_v1``
   and a ``jobs`` list. Every job needs a unique ``id``, ``config`` and
   ``checkpoint``; ``output_dir`` defaults to ``<output_root>/<id>``;
   ``config_sha256`` and ``checkpoint_sha256`` must match when given. Other
   keys (for example ``seed``, ``filename`` or ``device``) are copied into the
   job file.

``repository``, ``required_commit``, ``require_clean_worktree``
   Jobs run from a Git checkout of the library: ``repository`` (default: the
   checkout that contains the imported package) must hold the imported package
   under ``src/``. Its HEAD is recorded, must start with ``required_commit``
   when that is given, and must match when each task starts; with
   ``require_clean_worktree`` (default ``true``) tracked files must be
   unmodified.

``pilot_job_ids``
   The jobs that ``--pilot`` runs.

``name`` (default ``checkpoint-analysis``), ``scheduler``
   The Slurm job name, and ``account``, ``partition``, ``time``
   (``06:00:00``), ``cpus_per_task`` (8), ``memory`` (``32G``), ``gpus`` (0)
   and ``max_concurrent_jobs`` (a positive integer; default: all pending
   jobs).

The command writes into ``<output_root>/jobs/``: ``jobs_all.jsonl`` with every
declared job (a later preparation with different content raises),
``jobs_<label>.jsonl`` with the pending jobs and ``submission_<label>.json``
with the hashes of the manifest, job files and source checkout. The label is
``full`` or ``pilot``, with ``_retry_001``, ``_retry_002``, ... for later
submissions. Jobs whose provenance file and completion artifacts exist are
skipped unless ``--rerun`` is given. ``--generate-only`` writes the files and
stops before submitting. ``--account``, ``--partition`` and
``--max-concurrent`` override the manifest. Submission needs an account and a
partition; if a scheduler policy file lists approved pairs
(``$DENDRITIC_SCHEDULER_PROFILES``, or
``$XDG_CONFIG_HOME/dendritic_modeling/scheduler_profiles.yaml``; ``none``
disables it), the pair must be one of them. ``--max-concurrent`` must be a
positive integer. These settings are checked before the command writes any
file, so a submission that fails them leaves no ledger behind.

.. _checkpoint-collectors:

Collecting results
------------------

A collector reads the job file of an array, checks every job and writes
tables with one row per checkpoint (and per record within it). For every job
it checks the provenance written by ``dendritic-analyze-checkpoint``: its
``schema_version`` (``checkpoint_analysis_run_v1``), strict checkpoint loading
without missing or unexpected keys, the recorded paths, the input hashes, and
that the analyzers it reads were enabled.
Every collector except ``dendritic-collect-poisson-transfer`` (whose runs
record no analysis status) also rejects a run whose status record shows that
an analyzer it collects did not complete: the pass of the run's
``analysis_status.json`` named by the provenance ``filename`` (``final`` by
default), or else the ``analysis_status`` copy in the provenance (see
:ref:`analysis-status`). Runs written before status records existed have
neither and are collected as before. With ``allow_incomplete`` such a run is
listed with the skipped jobs instead. A collector writes
``collection_provenance.json`` with the hashes of its inputs and outputs, and
performs no aggregation across checkpoints.

Most collectors take either ``--config collection.yaml`` or the explicit
options (``--jobs``, which may be repeated, ``--output-dir``,
``--expected-count``, ...), not both. A collection configuration names its
schema and holds the same settings; relative paths in it resolve against the
configuration file:

.. code-block:: json

   {
     "schema_version": "path_matched_intervention_collection_config_v1",
     "jobs": ["jobs/jobs_all.jsonl"],
     "output_dir": "collected/interventions",
     "expected_count": 30,
     "expected_support_hash": "<selected_index_supports_sha256 of the analysis>"
   }

Common settings:

``expected_count``
   The number of declared jobs the collection must contain.

``allow_incomplete``
   (``dendritic-collect-representations``, ``-path-interventions``,
   ``-path-information``, ``-branch-sensitivities``) Collect the finished jobs
   and list the missing or incompletely recorded ones instead of failing.

``input_hash_validation``
   (``dendritic-collect-checkpoint-analysis``, ``-perturbation-response``,
   ``-gain-load-response``, ``-source-tuning``,
   ``-input-region-interventions``) ``rehash`` (default) hashes the input
   files again and compares them with the provenance;
   ``recorded_hash_chain`` compares the ``*_sha256`` values of the job file
   (as written by ``dendritic-submit-checkpoint-analysis``) with the
   provenance without reading the files again. The other collectors always
   rehash.

The collectors:

.. list-table::
   :header-rows: 1
   :widths: 30 22 48

   * - Command and configuration schema
     - Reads
     - Main outputs
   * - ``dendritic-collect-checkpoint-analysis``
       (``checkpoint_analysis_collection_config_v1``)
     - any analyzer
     - ``checkpoint_scalar_metrics.csv`` (one row per numeric value),
       ``checkpoint_analysis_files.csv``, ``checkpoint_analysis_records.jsonl``
   * - ``dendritic-collect-representations``
       (``representation_accessibility_collection_config_v1``)
     - ``representation_accessibility``
     - ``checkpoint_summary.csv``, probe, information, subset-decoding and
       geometry records with within-checkpoint technical summaries
   * - ``dendritic-collect-path-interventions``
       (``path_matched_intervention_collection_config_v1``)
     - ``path_matched_intervention``
     - ``support_draw_records.csv`` (one row per support draw),
       ``checkpoint_records.csv`` (draws averaged within each checkpoint),
       ``checkpoint_summary.csv``
   * - ``dendritic-collect-path-information``
       (``path_matched_information_collection_config_v1``)
     - ``path_matched_information``
     - the same three tables
   * - ``dendritic-collect-branch-sensitivities``
       (``branch_local_sensitivity_collection_config_v1``; ``--config``
       only)
     - ``branch_local_sensitivity``
     - ``checkpoint_records.csv`` with integration mode, component and depth
       columns, ``checkpoint_summary.csv``
   * - ``dendritic-collect-local-i-interventions`` (``--jobs``,
       ``--output-dir`` and ``--expected-count``, all required)
     - ``local_inhibitory_population_intervention``
     - ``intervention_records.csv``, ``paired_donor_contrasts.csv``, their
       technical summaries, ``checkpoint_summary.csv``; provenance in
       ``manifest.json``
   * - ``dendritic-collect-perturbation-response``
       (``perturbation_response_collection_config_v1``; ``--config`` only)
     - ``performance``, ``noise_perturbation`` and ``multiplicative_gain``
     - ``checkpoint_perturbation_response.csv``: one row per checkpoint and
       perturbation level, with clean and perturbed accuracy and
       chance-corrected retention
   * - ``dendritic-collect-gain-load-response``
       (``gain_load_response_collection_config_v1``; ``--config`` only)
     - ``gain_load_perturbation``
     - ``gain_load_draws.csv``, ``gain_load_checkpoint_response.csv`` (draws
       averaged within each checkpoint), input-statistics and
       branch-sensitivity tables
   * - ``dendritic-collect-reactivation``
       (``reactivation_dynamics_collection_config_v1``)
     - ``reactivation_dynamics``
     - ``checkpoint_modules.csv`` (one row per gate module, with network
       layer, population and soma-relative depth),
       ``checkpoint_metric_records.csv``
   * - ``dendritic-collect-synaptic-support``
       (``synaptic_source_support_collection_config_v1``)
     - ``weight`` with ``compute_source_support``
     - ``checkpoint_support_records.csv``, ``checkpoint_support_summary.csv``
       (equal-soma means), alignment records and summaries
   * - ``dendritic-collect-source-tuning``
       (``source_tuning_support_collection_config_v1``)
     - ``source_tuning_support``
     - preference and alignment records, and an equal-soma checkpoint summary
       per source block
   * - ``dendritic-collect-input-region-interventions``
       (``input_region_intervention_collection_config_v1``)
     - ``input_region_intervention``
     - classification, aggregate-response, per-class and assigned-soma
       response tables

Notes on individual collectors:

- ``dendritic-collect-checkpoint-analysis`` reads ``<directory>/final.json``,
  ``<directory>/final`` or ``<directory>/epochs/final.json`` of every
  analyzer named with ``--require-analyzer`` (``required_analyzers``), or of
  every enabled analyzer when none is named. ``--metric-pattern``
  (``metric_patterns``) keeps only the numeric values whose dotted path (for
  example ``accuracy.test`` or ``gaussian.0.1.accuracy``) matches a
  shell-style pattern such as ``*accuracy*``; a pattern list that selects
  nothing from one of the analyzers raises. ``[`` is a pattern character:
  write ``records[[]0]`` to match ``records[0]``. ``--omit-full-results``
  (``include_full_results: false``) leaves the complete result dictionaries
  out of the JSON Lines file, and ``--derive-information-null-excess`` adds
  the observed-minus-null value of every information term with a ``*_null``
  counterpart.
- ``dendritic-collect-path-interventions`` and
  ``dendritic-collect-path-information`` require identical support plans
  across checkpoints, and ``expected_support_hash`` pins the selected
  supports.
- ``dendritic-collect-perturbation-response`` and
  ``dendritic-collect-gain-load-response`` require ``chance_accuracy`` and a
  clean test accuracy above it, and report the chance-corrected retention
  :math:`R = (A_{\mathrm{perturbed}} - A_{\mathrm{chance}}) /
  (A_{\mathrm{clean}} - A_{\mathrm{chance}})`. The perturbation collector also reads
  ``analysis.noise_perturbation.params.n_samples`` from the analysis profile,
  so that profile must have an ``analysis:`` root and set ``n_samples``.
- ``dendritic-collect-reactivation`` with ``--require-structured-coordinates``
  requires the module names of ``population_network`` layers. With
  ``--require-clean-analysis-source`` (``require_clean_analysis_source: true``
  in the configuration) every run's ``checkpoint_analysis_provenance.json``
  must record a Git checkout without tracked changes or untracked files under
  ``src/``, all runs must share its commit, and ``collection_provenance.json``
  records that commit as ``analysis_source_commit``. The commit that trained a
  checkpoint is not checked: a reanalysis may use another validated commit.
  Its configuration file rejects unknown keys and takes ``true`` or ``false``
  for both requirements.
- ``dendritic-collect-source-tuning`` reads
  ``source_tuning_support_analysis/final.json`` unless
  ``--result-subdirectory`` or ``--result-filename`` say otherwise.
- The generic, perturbation, gain-load, reactivation, synaptic-support and
  input-region collectors read the results named ``final``; the other
  collectors read the job's ``filename``.

Summaries across checkpoints
----------------------------

``dendritic-summarize-checkpoint-metrics --config summary.yaml`` reads one
collected table and reports, for each group of rows, the mean over independent
experimental units (normally trained checkpoints) with a percentile-bootstrap
interval, and optional paired contrasts between arms. ``--output-dir``
overrides the configuration's ``output_dir``.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Key
     - Meaning
   * - ``schema_version``
     - ``checkpoint_metric_summary_v1``.
   * - ``input``
     - ``path`` of the CSV table (relative to the configuration file) and an
       optional ``sha256`` it must match.
   * - ``unit_columns`` (required)
     - Columns that identify one experimental unit; a unit may appear only
       once per group.
   * - ``group_by``
     - Columns that define the summary groups (default: one group).
   * - ``value_column``
     - The value to summarize (default ``value``).
   * - ``derived_columns``
     - Added columns, computed in order before filtering:
       ``{name: {operation, columns}}`` with ``sum`` (any number of columns),
       ``difference``, ``ratio`` (zero denominators raise) or ``equal`` (two
       columns each).
   * - ``filters``, ``regex_filters``, ``any_of_filters``
     - Row selection: column values (a value or a list), full-match regular
       expressions, and a list of alternative ``filters`` mappings. An empty
       selection raises.
   * - ``expected_units_per_group``
     - The number of units every group must have.
   * - ``bootstrap``
     - ``draws`` (10000), ``seed`` (0) and ``confidence`` (0.95) of the group
       intervals.
   * - ``paired_comparisons``
     - A list of paired contrasts (below).
   * - ``output_dir``
     - Output directory (relative to the configuration file).

Example (unknown keys are ignored):

.. code-block:: json

   {
     "schema_version": "checkpoint_metric_summary_v1",
     "input": {"path": "collected/perturbation/checkpoint_perturbation_response.csv"},
     "unit_columns": ["job_seed"],
     "group_by": ["job_arm", "perturbation", "perturbation_level"],
     "value_column": "retention_above_chance",
     "filters": {"perturbation": "gaussian"},
     "expected_units_per_group": 5,
     "bootstrap": {"draws": 10000, "seed": 0, "confidence": 0.95},
     "paired_comparisons": [
       {
         "name": "shunting_minus_additive",
         "arm_column": "job_arm",
         "reference": "additive",
         "comparison": "shunting",
         "pair_columns": ["job_seed"],
         "stratum_columns": ["perturbation_level"],
         "contrast_operation": "difference"
       }
     ],
     "output_dir": "summary/perturbation"
   }

Outputs: ``checkpoint_units.csv`` (the selected rows, derived columns
included), ``group_summary.csv`` (per group: mean, interval, standard
deviation and unit count), ``paired_units.csv``, ``paired_summary.csv`` and
``summary_provenance.json`` (configuration, input and output hashes, and the
source identity).

.. _paired-comparisons:

Paired comparisons
~~~~~~~~~~~~~~~~~~

An entry of ``paired_comparisons`` compares two values of ``arm_column``
(``reference`` and ``comparison``) within each experimental unit defined by
``pair_columns``, and reports a bootstrap interval of the mean within-unit
contrast, separately for each combination of ``stratum_columns``. Units
missing one arm, and duplicate rows, raise. Further keys: ``name`` (default
``contrast_<index>``), ``value_column`` (default: the top-level one),
``filters``, ``regex_filters`` and ``any_of_filters`` (applied to the selected
rows), ``expected_pairs_per_stratum``, ``draws`` and ``bootstrap_seed``
(default: the ``bootstrap`` values), and ``arm_sets`` with
``arm_set_reduction: mean``, which averages several arms within each unit
before pairing (``reference`` and ``comparison`` then name sets). The paired
intervals use a confidence of 0.95 whatever ``bootstrap.confidence`` says.

``contrast_operation`` is ``difference`` (default), ``comparison -
reference``, or ``symmetric_difference``,
``(comparison - reference) / (comparison + reference)``, which raises when a
denominator is zero.

``protocol_version`` selects how the bootstrap seed identity and the output
columns are formed; it never changes the contrast or the observations.
``legacy_difference`` and ``legacy_grouped`` reproduce summaries made with
the older seed identities:

.. list-table::
   :header-rows: 1
   :widths: 20 30 22 28

   * - ``protocol_version``
     - Bootstrap seed identity
     - ``symmetric_difference``
     - Output columns
   * - ``current`` (default)
     - includes the operation
     - accepted
     - ``delta__<comparison>_minus_<reference>``; the summary records
       ``contrast_operation``
   * - ``legacy_difference``
     - omits the operation
     - rejected
     - ``delta__...``; no ``contrast_operation``
   * - ``legacy_grouped``
     - omits the operation for ``difference``, includes it for
       ``symmetric_difference``
     - accepted
     - ``symmetric_delta__...`` for ``symmetric_difference``, ``delta__...``
       otherwise; records ``contrast_operation``

Choose a non-default protocol only to reproduce an existing summary, and only
from an identified source record with unchanged inputs, never to obtain a
different interval. The bootstrap seed of each group is derived from the
``repr`` of its key values, which differs between NumPy 1.x and 2.x (for
example ``1`` and ``np.int64(1)``); to reproduce a summary exactly, use the
NumPy and pandas versions it was made with, together with the library
version. The configuration's path and hash are recorded in
``summary_provenance.json``; keep the configuration file with the results.

The same contrasts are available from Python as
``dendritic_modeling.scripts.sweeps.paired_comparison.paired_arm_comparison``
(with the keywords ``contrast_operation`` and ``protocol_version``) and, for
the ``performance`` results of a sweep, from the command line:

.. code-block:: bash

   python -m dendritic_modeling.scripts.sweeps.paired_comparison sweep_root \
     --output paired_results \
     --arm-column use_shunting \
     --reference False --comparison True \
     --metric test_accuracy \
     --pair-column seed \
     --contrast-operation difference \
     --protocol-version current

It collects every configuration of the sweep, one row per configuration, with
the parameters the sweep collector extracts (for example ``use_shunting`` and
``seed``, from ``experiment.seed``) and the ``performance`` results as
``<split>_<metric>`` columns (for example ``test_accuracy``); arm values are
compared as text. It writes ``paired_units.csv`` and ``paired_summary.json``.

Selecting epochs on validation metrics
--------------------------------------

``dendritic-select-validation-epochs --config selection.yaml`` compares
checkpoint-selection rules without selecting on held-out data. It reads the
long ``training_history.csv`` and ``checkpoint_units.csv`` tables written by
``dendritic-collect-sweep-metrics`` (see :doc:`sweeps`). For every trained
checkpoint, each selector picks one epoch by a validation metric
(``metric`` must start with ``valid_``; ``objective`` ``min`` or ``max``;
``tie_break`` ``earliest`` or ``latest``), and the declared
``report_metrics``, test metrics included, are reported at that epoch.

The configuration (``schema_version: validation_epoch_selection_v1``) declares
the two input tables (``history`` and ``units``, each with ``path`` and an
optional ``sha256``), ``join_columns``, ``metadata_columns`` to carry forward,
``expected_unit_count``, ``expected_epochs_per_unit`` (every unit needs
epochs 1 to N), the column names (``epoch_column``, ``metric_column``,
``value_column``), ``selectors``, ``report_metrics`` and ``output_dir``.
Unknown keys raise. ``configs/examples/validation_epoch_selection.yaml`` is a
complete example.

Outputs: ``selected_epochs.csv`` (one row per checkpoint and selector),
``selected_epoch_metrics.csv`` (long form, which
``dendritic-summarize-checkpoint-metrics`` reads for paired comparisons
between selection rules) and ``selection_provenance.json`` (input and output
hashes).

Poisson exposure transfer
-------------------------

``dendritic-poisson-exposure-transfer`` evaluates one checkpoint trained on
``poisson_generator`` data (see :ref:`dataset-poisson`) at several stimulus
durations. The ``data.dataset_params.poisson_generator`` block of its training
configuration is read as the loader read it, with the loader defaults for the
keys it leaves out. The training data must have had Poisson sampling
(``poisson_sampling``, default ``true``), a finite positive
``stimulus_duration`` (default ``1.0``) and a multiplicative gain of exactly
1: ``multiplicative_gain: false``, ``fixed_gain_factor: 1``, or no
``max_gain_factor`` and ``max_gain_tau_ratio`` (or values that give an upper
bound of 1). Any other gain raises. For each
duration the command rebuilds the test split (flattened as in training),
draws ``draws_per_duration`` Poisson samples with seeds shared by all
checkpoints, and records the accuracy of each draw.

.. code-block:: bash

   dendritic-poisson-exposure-transfer \
     --config run_dir/config.json \
     --checkpoint run_dir/final_model.pt \
     --transfer-config transfer.yaml \
     --output-dir transfer/run_0

The transfer configuration has no ``schema_version`` and accepts only these
keys (defaults): ``test_durations`` (``[0.05, 0.1, 0.3162, 1.0, 3.1623]``),
``draws_per_duration`` (5), ``base_draw_seed`` (73000), ``batch_size`` (256),
``max_samples`` (``null``) and ``fingerprint_realized_datasets``
(``false``). The command writes ``poisson_exposure_transfer.json`` and
``poisson_exposure_transfer_provenance.json`` besides the run files of
``dendritic-analyze-checkpoint``.

``dendritic-collect-poisson-transfer --jobs jobs.jsonl --output-dir collected
--expected-count N`` collects such runs. Its job lines hold ``config``,
``checkpoint``, ``transfer_config`` and ``output_dir``; it writes
``draw_records.csv``, ``checkpoint_records.csv`` (mean accuracy per duration
and checkpoint), ``checkpoint_inventory.csv`` and
``collection_provenance.json``.

.. _recurrent-condition-analyses:

Recurrent models by task condition
----------------------------------

Four programs analyze a saved recurrent E/I model by task condition, for
example by the delay of ``variable_delay_dms`` trials. They are not installed
as commands; run them as modules with the library importable:

.. code-block:: bash

   python -m dendritic_modeling.scripts.analysis.analyze_routing_by_condition      --model_path run_dir/final_model.pt --config_path run_dir/config.json      --output_dir posthoc/routing --n_samples 2000 --n_bins 10
   python -m dendritic_modeling.scripts.analysis.lesion_analysis      --model_path run_dir/final_model.pt --config_path run_dir/config.json      --output_dir posthoc/lesion
   python -m dendritic_modeling.scripts.analysis.evaluate_ood_delay      --model_path run_dir/final_model.pt --config_path run_dir/config.json      --output_dir posthoc/ood --min_delay 100 --max_delay 500 --n_samples 2000
   python -m dendritic_modeling.scripts.analysis.recurrent_inhibitory_structure      --config_path run_dir/config.json --checkpoint_path run_dir/final_model.pt      --output_json posthoc/inhibition.json --plot_path posthoc/inhibition.png

The checkpoint is the state dictionary of the whole model, such as the run's
``final_model.pt``; the configuration is the run's ``config.json`` or its
training YAML file. The routing and lesion analyses need a core built as a
recurrent ``EINetwork`` (``unified_ei`` and its aliases, or a structured E/I
core with recurrent connectivity); the out-of-distribution evaluation takes
any classifier trained on ``variable_delay_dms``. Like
``dendritic-analyze-checkpoint``, these three rebuild the model and the
datasets from the configuration with the run's recorded seeds, load the
checkpoint strictly, evaluate on ``--device`` (default ``cpu``), and write the
rebuilt run's ``resolved_seeds.json``, ``runtime_environment.json``,
``train.log`` and ``dendritic_modeling.log`` into ``--output_dir`` besides
their results. The ``--dataset`` option of the routing and lesion analyses is
not read; the dataset is the one of the configuration.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Module
     - What it computes and writes
   * - ``analyze_routing_by_condition``
     - Runs the first ``--n_samples`` trials of the test split with routing
       recording on and takes, at every time step, each excitatory level's
       share of the summed voltage norms of the levels of ``--layer_key``
       (default ``layer_0``) and the feedforward fraction of each level's
       synaptic drive. Trials are binned into ``--n_bins`` equal-width bins of
       their delay (``--condition_key delay``, read from the test set's
       ``delays``) or, with another condition key or a dataset without
       delays, of their class label. Writes ``routing_by_condition.json``
       (per bin the mean and time-resolved level shares and feedforward
       fractions and the trial count; the Spearman correlation between the most
       distal level's share and the bin centre, with at least three bins),
       ``routing_heatmap.png`` (level shares over time and condition) and
       ``beta_vs_condition.png`` (mean share of each level, labelled with its
       soma-relative depth and time constant). The shares measure activity, not
       causal routing; the lesion analysis tests causal use.
   * - ``lesion_analysis``
     - Evaluates the test split intact and with each excitatory dendritic level
       of the first layer silenced in turn, through the population's level
       mask (the model is restored after each). Writes ``lesion_results.json``
       (overall accuracy, and accuracy and trial count per delay bin when the
       test set records delays, for ``intact`` and every
       ``Lesion depth <d> (tau=<t>)``), ``lesion_accuracy_vs_delay.png`` and,
       with at least four delay bins, ``lesion_bar_chart.png`` (accuracy drop
       on the shorter and the longer half of the delays).
   * - ``evaluate_ood_delay``
     - Needs a run trained on ``variable_delay_dms``. Draws ``--n_samples``
       new trials with delays in [``--min_delay``, ``--max_delay``] and the
       training trial shape (``stimulus_dim``, ``stimulus_duration``,
       ``probe_duration``), with a seed derived from the run's evaluation
       seed unless ``--seed`` is given, and measures accuracy overall and in
       ``--n_bins``
       delay bins. Writes ``ood_delay_metrics.json`` (delay range, trial
       count, seed, trial shape, overall accuracy and accuracy per bin) and
       ``ood_delay_accuracy.png``.
   * - ``recurrent_inhibitory_structure``
     - Reads the configuration and the checkpoint without building the model.
       For layer ``--layer_idx`` of a unified E/I core whose configuration
       lists ``model.core.unified_ei.layers``, it rebuilds the active
       recurrent inhibitory synapses onto each excitatory level from the
       stored ``branch_rec_inhibition.pre_w`` weights, keeping the
       ``rec_inhibitory_synapses`` largest weights of every compartment (by
       magnitude with ``weight_transform: identity``) as a standard TopK
       synapse does; a level without these weights in the checkpoint, for
       example with indexed synapses, is reported as absent with no synapses.
       It classifies each inhibitory neuron as ``soma_only``,
       ``dendrite_only``, ``mixed`` or ``none`` by the compartments it
       reaches, counts inhibitory-to-excitatory pairs by relation
       (``soma_only``, ``dendrite_only``, ``both``, ``none``) and marks
       neurons that reach some excitatory neurons only at the soma and others
       only on dendrites as compartment selective. It prints a summary;
       ``--output_json`` writes the full result with per-level and
       per-neuron records, and ``--plot_path`` a bar and pie chart of the
       pair relations.
