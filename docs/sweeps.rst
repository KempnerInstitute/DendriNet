.. _sweeps:

Parameter sweeps
================

.. highlight:: yaml

A sweep trains one model for each combination of configuration values. A
*sweep file* holds a complete training configuration (``base_config``) and the
configuration paths to vary. ``dendritic-sweep`` expands the file into one
training config per condition and replicate seed, loads every generated config
through the strict config loader, and writes Slurm job scripts.
``dendritic-analyze`` collects the finished runs into tables and plots.

Quick start
-----------

.. code-block:: bash

   # Generate the configs and job scripts; nothing is submitted
   dendritic-sweep --config configs/sweep_config.yaml --generate-only

   # Generate and submit (a Slurm account and partition are required)
   dendritic-sweep --config configs/sweep_config.yaml \
       --slurm-account ACCOUNT --slurm-partition PARTITION

   # Tabulate and plot the finished runs of one sweep directory
   dendritic-analyze <sweep_dir>

``configs/sweep_config.yaml`` is the annotated template: a population-network
MNIST model with axes on the E/I synapse counts and the branching of the
excitatory population, five replicate seeds each. Keep project-specific sweep
files in the project's own repository. The full option lists of the commands
are in :doc:`cli_reference`.

Sweep files
-----------

The following sweep file is complete. It trains the three E/I synapse pairs
with E + I = 50 at two sparse-synapse learning rates, with three replicate
seeds each: 18 runs::

   sweep_config:
     model.core.population_network.layers.0.population_defaults.ff_excitatory_synapses: [20, 30, 40]
     model.core.population_network.layers.0.population_defaults.ff_inhibitory_synapses: [10, 20, 30]
     training.main.common.param_groups.topk_lr: [0.001, 0.01]

   filter_config:
     mode: ei-sum            # keep the E/I pairs with E + I = 50
     value: 50
     tolerance: 0

   experiment_settings:
     seeds_per_condition: 3  # replicate seeds 0, 1 and 2 for every condition
     base_seed: 0

   slurm_config:
     time: "04:00:00"
     gpus_per_node: 1
     cpus_per_task: 8
     mem: 32G

   base_config:              # the complete training config that the axes modify
     data:
       dataset_name: mnist
       processing:
         flatten: true
         normalize: false
     model:
       encoder:
         type: identity
       core:
         type: population_network
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
       decoder:
         type: MLP
         params:
           hidden_dims: []
           output_dim: 10
     training:
       main:
         common:
           epochs: 20
     analysis:
       performance:
         enabled: true
     outputs:
       run_name: ei_sum_topk_lr
       results_dir: outputs/sweeps

``analysis.performance.enabled: true`` makes every run write
``performance/final.json``, which the analysis and the rerun tool read; the
performance analyzer is off by default. The top-level sections are:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Section
     - Content
   * - ``base_config``
     - Required. The complete training config every run starts from, in the
       schema ``dendritic-train`` reads.
   * - ``sweep_config``
     - Axes: dot paths into ``base_config`` and their values
       (:ref:`sweeps-axes`).
   * - ``sweep_variants``
     - Named conditions that change several paths together
       (:ref:`sweeps-variants`).
   * - ``filter_config``
     - E/I pair filter and branch sweeps (:ref:`sweeps-ei-filter`,
       :ref:`sweeps-branch`).
   * - ``experiment_settings``
     - ``seeds_per_condition``, ``base_seed`` and ``seed_paths``
       (:ref:`sweeps-seeds`).
   * - ``slurm_config``
     - Resources and environment of the jobs (:ref:`sweeps-slurm`).
   * - ``execution_results_dir``
     - Separate root for the run outputs (:ref:`sweeps-layout`).
   * - ``sweep_contract``
     - ``expected_config_count``: the number of configs generation must
       produce (:ref:`sweeps-count`). A top-level ``expected_config_count`` is
       read when ``sweep_contract`` does not set one.
   * - ``output_dir``, ``outputs``
     - Base directory (``output_dir`` or ``outputs.dir``) and name prefix
       (``outputs.run_name``) of the sweep directory (:ref:`sweeps-layout`).
   * - ``extends``
     - Parent sweep file (:ref:`sweeps-extends`).

The sweep must define at least one condition through ``sweep_config``,
``sweep_variants`` or a branch sweep. Before any job script is written, every
generated config is loaded with the strict config loader, so a misspelled axis
path, or a path into a section that the base core type does not read, fails at
generation. The loader does not catch an option that the selected component
ignores (for example an ``indexed_*`` option under ``topk_type: standard``),
and generation does not build the models. To check that a generated config
builds, run it once with ``--validate-only``:

.. code-block:: bash

   dendritic-train <sweep_dir>/configs/unified_config_0.yaml --validate-only --output_dir <scratch_dir>

Without ``--output_dir`` the check writes its files into the sweep's
``results/config_0/``.

The sweep file's own sections are checked when it loads. An unknown key in
``slurm_config``, ``experiment_settings``, ``filter_config`` or ``outputs``
raises with the closest valid key, and so does a top-level key that is a
near-miss of a section name (``sweep_confg``). Other top-level keys are
reported in a warning and ignored; put study annotations under ``metadata:``
(keys starting with ``x-`` hold YAML anchors and pass silently). The
generated training configs are checked by the strict config loader. Check the
generated config count after editing a sweep file.

.. _sweeps-axes:

Axes
----

Each key of ``sweep_config`` is a dot path into ``base_config``. Numeric path
components index lists: ``layers.0`` is the first layer of
``model.core.population_network.layers`` and ``populations.1`` the second
population of that layer. A path past the end of a list fails at generation;
missing mapping keys along a path are created, and the strict loader then
rejects any key the schema does not have. The sweep is the Cartesian product of
all axes. An axis value is one of:

- a list: the literal values, for example ``[0.001, 0.01]``; a list of lists
  sweeps a list-valued option, for example ``[[3, 3], [4, 4]]``;
- a scalar: one value, written into every generated config;
- ``{range: {start, stop, step}}``: start, start + step, ... up to and
  including stop;
- ``{linspace: {start, stop, num}}`` or ``{geomspace: {start, stop, num}}``:
  ``num`` evenly or geometrically spaced values, both ends included;
- ``{values: [...]}``: the literal values.

A plain list of three numbers that reads as ``[start, stop, step]``, such as
``[1, 10, 2]``, is rejected as ambiguous. Write
``{range: {start: 1, stop: 10, step: 2}}`` for the range or
``{values: [1, 10, 2]}`` for the three values.

The following excerpt of a sweep file whose ``base_config`` is the one of
``configs/sweep_config.yaml`` defines 4 × 3 × 3 × 3 = 108 conditions:

.. code-block:: yaml

   sweep_config:
     # 64, 128, 192, 256
     model.core.population_network.layers.0.populations.1.n_neurons: {range: {start: 64, stop: 256, step: 64}}
     # 0.0001, 0.001, 0.01
     training.main.common.param_groups.topk_lr: {geomspace: {start: 1.0e-4, stop: 1.0e-2, num: 3}}
     # 1.0, 3.0, 5.0
     training.main.common.grad_clip_value: {linspace: {start: 1.0, stop: 5.0, num: 3}}
     # the plain list [5, 15, 10] would be rejected as ambiguous
     model.core.population_network.layers.0.population_defaults.ff_inhibitory_synapses: {values: [5, 15, 10]}
     # one value for every condition
     training.main.common.epochs: 50

In a population network, options shared by the populations of a layer sit in
``model.core.population_network.layers[].population_defaults``, and a
population or connection is addressed by its position in the layer's
``populations`` or ``connections`` list (see :ref:`population-connections`).
With the template's base config, this excerpt sweeps the E synapse count, the
size of the excitatory population ``e`` and the probability of the third
connection (``i`` to ``e``):

.. code-block:: yaml

   sweep_config:
     model.core.population_network.layers.0.population_defaults.ff_excitatory_synapses: [20, 30]
     model.core.population_network.layers.0.populations.1.n_neurons: [64, 128]
     model.core.population_network.layers.0.connections.2.probability: [0.9, 1.0]

A connection probability below 1 leaves each target fewer candidate sources. A
target with fewer candidates than its synapse count can make the model fail
to build, in the job rather than at generation, so check such an axis with
``--validate-only``.

.. _sweeps-learning-rates:

Learning-rate axes
------------------

With the default ``training.main.common.param_groups.split_params: true``, the
optimizer trains the core's sparse synapses at ``topk_lr``, its BlockLinear
branch couplings at ``blocklinear_lr``, its reactivation gates at
``reactivation_lr`` and the decoder at ``decoder_lr``. ``lr`` applies only to
the remaining parameters: the encoder (unless ``encoder_lr`` is set), MLP
inhibitory cells and any parameter outside these groups. The population
network of ``configs/sweep_config.yaml`` has no such parameter, so the
conditions of a ``param_groups.lr`` axis train identical models. Sweep the
group rates instead (an excerpt; ``base_config`` as in the template):

.. code-block:: yaml

   sweep_config:
     training.main.common.param_groups.topk_lr: [0.0003, 0.001, 0.003]
     training.main.common.param_groups.decoder_lr: [0.0003, 0.001]

or set ``base_config.training.main.common.param_groups.split_params: false``,
which trains every parameter at ``lr``. The generator warns about a
``param_groups.lr`` axis without group-rate axes only when ``base_config``
writes ``split_params: true`` and at least one group rate. Every parameter
group carries its own rate, so an axis on ``training.main.optimizer.lr`` has
no effect either. The groups are described in :ref:`parameter-groups`.

.. _sweeps-variants:

Named variants
--------------

``sweep_variants`` defines conditions that change several paths together. Each
variant is one condition: its ``overrides`` (dot paths, as in ``sweep_config``)
are applied together, before the ``sweep_config`` axes, and every variant is
crossed with every axis. Variant names must be unique. With the template's
``base_config``, these sections compare shunting with normalized additive
inhibition at two learning rates, four conditions::

   sweep_variants:
     - name: shunting
       overrides:
         model.core.population_network.layers.0.population_defaults.use_shunting: true
         model.core.population_network.layers.0.population_defaults.use_additive_normalization: false
     - name: normalized_additive
       overrides:
         model.core.population_network.layers.0.population_defaults.use_shunting: false
         model.core.population_network.layers.0.population_defaults.use_additive_normalization: true
   sweep_config:
     training.main.common.param_groups.topk_lr: [0.001, 0.01]

Each generated config stores its variant name in the top-level
``_sweep_variant`` key, which the config loader keeps as metadata, and the
analysis tables have it in the ``sweep_variant`` column. A variant may change
only paths that the base core type reads: an override of
``model.core.sparsity.type`` on a ``population_network`` base fails at
generation, because population networks read only
``model.core.population_network``.

.. _sweeps-ei-filter:

Filtering E/I pairs
-------------------

When ``sweep_config`` has an excitatory and an inhibitory synapse-count axis,
``filter_config.mode`` can keep only the pairs that satisfy a constraint. The
axes are a pair of keys ending in ``.population_defaults.ff_excitatory_synapses``
and ``.population_defaults.ff_inhibitory_synapses`` (population networks), or
``model.core.connectivity.ee_synapses_per_branch_per_layer`` and
``model.core.connectivity.ie_synapses_per_branch_per_layer`` (structured E/I
cores). On a structured core the selected pair is written as one-element lists,
which the core repeats for every layer.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - ``mode``
     - Pairs kept
   * - ``all``
     - Every pair (the default; ``default`` is the same).
   * - ``ei-sum``
     - Pairs with \|E + I − ``value``\| ≤ ``tolerance``.
   * - ``ei-ratio``
     - Pairs with I > 0 and \|E / I − ``value``\| ≤ ``tolerance``.
   * - ``e-const``
     - E = ``value`` with every I; ``value`` must be on the E axis.
   * - ``i-const``
     - I = ``value`` with every E; ``value`` must be on the I axis.
   * - ``branch_sweep``
     - Every pair. Required for a branch sweep when the sweep has no other
       axis or variant (:ref:`sweeps-branch`).

The four E/I modes need both axes and ``value``; ``ei-sum`` and ``ei-ratio``
also need ``tolerance`` (``0`` for an exact match). A filter that keeps no
pair raises. The other axes are crossed with the pairs that remain. With the
template's ``base_config``, this excerpt keeps (20, 10), (30, 20) and (40, 20):

.. code-block:: yaml

   sweep_config:
     model.core.population_network.layers.0.population_defaults.ff_excitatory_synapses: [10, 20, 30, 40]
     model.core.population_network.layers.0.population_defaults.ff_inhibitory_synapses: [10, 20, 30, 40]
   filter_config:
     mode: ei-ratio
     value: 2.0
     tolerance: 0.5

.. _sweeps-branch:

Branch sweeps
-------------

``filter_config.branch_sweep_mode`` adds one dimension of dendritic trees to
the grid. Each tree replaces ``branch_factors`` of every population whose
``polarity`` is ``excitatory`` (or that has no ``polarity`` key) in every layer
of a population network, or ``model.core.architecture.excitatory_branch_factors``
of other cores. Do not also sweep those ``branch_factors`` in
``sweep_config``: the tree overwrites the axis, and conditions that differ only
in that axis become identical runs. A factor ``b`` at depth ``d`` gives ``d``
levels of ``b`` branches. Depths and factors are axis values, so the range
forms of :ref:`sweeps-axes` apply.

.. list-table::
   :header-rows: 1
   :widths: 18 36 46

   * - ``branch_sweep_mode``
     - Keys
     - Trees
   * - ``uniform_sweep`` (default)
     - ``dendrite_tree_depth``, ``dendrite_branches_per_layer``
     - Every depth with every factor.
   * - ``depth_sweep``
     - ``dendrite_tree_depth``, ``uniform_branches_per_layer`` (one factor)
     - One tree per depth.
   * - ``width_sweep``
     - ``fixed_dendrite_tree_depth`` (one depth), ``dendrite_branches_per_layer``
     - One tree per factor.
   * - ``pattern_sweep``
     - ``dendrite_tree_depth``, ``base_branches``, ``branch_patterns``
     - One tree per depth and pattern (below).
   * - ``custom_configs``
     - ``custom_branch_configs``: a list of ``{depth, branches}`` mappings
     - The listed trees; a list in ``branches`` is used as written, a number
       is repeated ``depth`` times.

The patterns of ``pattern_sweep`` give, for ``base_branches: 3``:

.. list-table::
   :header-rows: 1
   :widths: 18 42 20 20

   * - Pattern
     - Branch factors
     - Depth 3
     - Depth 4
   * - ``uniform``
     - ``base_branches`` at every level
     - ``[3, 3, 3]``
     - ``[3, 3, 3, 3]``
   * - ``decreasing``
     - ``base_branches``, ``base_branches`` − 1, ..., at least 1
     - ``[3, 2, 1]``
     - ``[3, 2, 1, 1]``
   * - ``increasing``
     - 1, 2, ..., depth (ignores ``base_branches``)
     - ``[1, 2, 3]``
     - ``[1, 2, 3, 4]``
   * - ``pyramid``
     - Level ``i`` (from 0) gets ``i + 1`` up to ``i = depth // 2``, then
       ``depth - i`` (ignores ``base_branches``)
     - ``[1, 2, 1]``
     - ``[1, 2, 3, 1]``

A branch sweep combines with the other axes and with an E/I filter (set the E/I
``mode`` and ``branch_sweep_mode`` together). With the template's
``base_config``, this excerpt trains trees ``[4]``, ``[4, 4]`` and
``[4, 4, 4]`` at two learning rates, six conditions:

.. code-block:: yaml

   sweep_config:
     training.main.common.param_groups.topk_lr: [0.001, 0.01]
   filter_config:
     branch_sweep_mode: depth_sweep
     dendrite_tree_depth: {range: {start: 1, stop: 3, step: 1}}
     uniform_branches_per_layer: 4

Without other axes or variants, set ``mode: branch_sweep`` as well; with an
empty ``sweep_config`` and ``mode: all``, generation fails.

.. _sweeps-seeds:

Replicate seeds
---------------

``experiment_settings`` sets the replicates of every condition:

- ``seeds_per_condition`` (default 1) and ``base_seed`` (default 42): replicate
  ``r`` of every condition gets ``experiment.seed = base_seed + r``, so all
  conditions use the same seeds. The ``experiment.seed`` of ``base_config`` is
  replaced.
- ``seed_paths``: further dot paths that receive the replicate seed.

Every random stream that ``base_config`` leaves unset follows
``experiment.seed`` (see :ref:`random-seeds`), so replicates differ in data
split, initialization, connectivity and data order. To hold one stream fixed
across replicates, set it in ``base_config.experiment``: for example, a fixed
``topology_seed`` gives every replicate the same connection masks and
indexed-synapse indices. ``seed_paths`` does the opposite for a component seed
that ``base_config`` pins, which would otherwise stay constant across
replicates (an excerpt; ``base_config`` as in the template):

.. code-block:: yaml

   experiment_settings:
     seeds_per_condition: 5
     base_seed: 0
     seed_paths:
       - model.core.population_network.layers.0.population_defaults.indexed_seed

.. _sweeps-count:

Config count
------------

The number of generated configs is the product of the axis lengths (after the
E/I filter), the number of trees and the number of variants, times
``seeds_per_condition``. Conditions are ordered with the variants outermost,
then the ``sweep_config`` axes in file order, then the trees, then the filtered
E/I pairs; the last varies fastest. Replicates are innermost:
``unified_config_<i>.yaml`` is replicate ``i % seeds_per_condition`` of
condition ``i // seeds_per_condition``.

To make generation fail when the grid has another size, declare the expected
count::

   sweep_contract:
     expected_config_count: 18

.. _sweeps-extends:

Composing sweep files
---------------------

``extends`` names a parent sweep file, relative to the file that names it. The
child's mappings are merged into the parent's key by key, and a child list
replaces the parent's list. A child therefore keeps every ``sweep_config``
axis of its parent; it can change their values or add axes. With the complete
example above stored as ``base_sweep.yaml``, this file in the same directory
sweeps three sparse-synapse learning rates over the parent's three E/I pairs
with five seeds: 45 runs::

   extends: base_sweep.yaml
   sweep_config:
     training.main.common.param_groups.topk_lr: [0.0003, 0.001, 0.003]
   experiment_settings:
     seeds_per_condition: 5
   slurm_config:
     account: my_account
     partition: my_partition
   base_config:
     outputs:
       run_name: topk_lr
   execution_results_dir: /scratch/my_project/topk_lr

Cyclic or empty ``extends`` declarations fail before any file is written. The
frozen manifest hashes every file of the chain and the merged
``original_config.yaml``.

.. _sweeps-layout:

Sweep directories
-----------------

Each generation creates a new directory ``<prefix>_<YYYYmmddHHMMSS>`` (a second
sweep in the same second gets a numeric suffix). The base directory is
``--output-dir``, else the sweep file's ``output_dir``, else ``outputs.dir``,
else ``base_config.outputs.results_dir``, else ``output``; a relative path
resolves against the current directory. The prefix is the sweep file's
``outputs.run_name``, else ``base_config.outputs.run_name``, else
``unified_sweep``.

.. code-block:: text

   <prefix>_<timestamp>/
   |-- original_config.yaml         composed sweep file (after extends and command-line values)
   |-- frozen_sweep_manifest.json   hashes and source identity of the generation
   |-- submit_all_jobs.sh           submits the job scripts and records their IDs
   |-- configs/
   |   |-- unified_config_<i>.yaml  one training config per condition and replicate
   |   `-- sweep_metadata.yaml      config count and file names
   |-- jobs/
   |   |-- run_array_sweep.sh       one array job (run_sweep.sh or run_sweep_<i>.sh for sequential jobs)
   |   |-- submitted_jobs.tsv       job ID, script and UTC time of each submission
   |   `-- output_<job>_<task>.out, error_<job>_<task>.err
   |-- results/
   |   `-- config_<i>/              run directory of unified_config_<i>.yaml
   `-- plots/                       tables and plots of dendritic-analyze

Each generated config sets ``outputs.results_dir`` to its
``results/config_<i>`` directory. A results directory whose path contains
``/config_`` is used as the run directory itself, without a timestamped
subdirectory, so every run's files (see :ref:`run-outputs`) land directly in
``results/config_<i>/``.

The top-level ``execution_results_dir`` keeps the light sweep files (configs,
scripts, logs) in one place and the run outputs under a separate root, for
example on a large-data filesystem. Each generated config then writes to
``<execution_results_dir>/config_<i>``, and ``results`` is a symbolic link to
that root. Generation does not create the root, so it may exist only on the
compute nodes.

.. _sweeps-slurm:

Slurm settings
--------------

``slurm_config`` sets the ``#SBATCH`` lines, the environment and the launch
command that every job script of the sweep (array, sequential and rerun jobs)
shares:

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Key
     - Default
     - Meaning
   * - ``account``, ``partition``
     - none
     - ``--account`` and ``--partition``; required to submit
       (:ref:`sweeps-commands`).
   * - ``qos``, ``constraint``
     - unset
     - ``--qos`` and ``--constraint``, written only when set.
   * - ``time``
     - ``"02:00:00"``
     - ``--time`` of each job or array task (format below).
   * - ``nodes``, ``ntasks_per_node``
     - 1, 1
     - ``--nodes`` and ``--ntasks-per-node``. Keep ``nodes: 1``.
   * - ``gpus_per_node``
     - 1
     - ``--gres=gpu:<n>``; also the number of processes of ``launcher:
       torchrun``.
   * - ``cpus_per_task``
     - 4
     - ``--cpus-per-task``.
   * - ``mem``
     - ``"32G"``
     - ``--mem``, passed to Slurm as written.
   * - ``exclude_nodes``
     - ``[]``
     - ``--exclude``. ``--exclude-nodes`` on the command line replaces it.
   * - ``max_concurrent_jobs``
     - 10
     - Maximum number of array tasks running at once (the ``%`` limit of
       ``--array``).
   * - ``modules_to_load``
     - ``[]``
     - Environment modules loaded after ``module purge``.
   * - ``conda_env_path``
     - unset
     - Environment whose ``bin/python`` runs the training (its ``lib`` is
       added to ``LD_LIBRARY_PATH``). Unset: the ``python`` first on ``PATH``
       after ``module purge``.
   * - ``launcher``
     - ``python``
     - ``python`` or ``torchrun`` (:ref:`sweeps-distributed`).
   * - ``training_script``
     - ``src/dendritic_modeling/scripts/training/train_experiments.py``
     - Training entry point, relative to the library checkout or absolute.

The job name is the sweep's prefix (a ``job_name`` key is not used). Every
job runs in the library checkout that generated the sweep
(``#SBATCH --chdir``), puts that checkout's ``src/`` first on ``PYTHONPATH``
and runs ``module purge``, so the cluster must provide environment modules.
Each array task trains one config.

``time`` takes a quoted Slurm duration with a colon or a day prefix
(``"HH:MM:SS"``, ``"MM:SS"``, ``"D-HH"``, ``"D-HH:MM:SS"``), or an integer
number of seconds. YAML reads an unquoted ``12:00:00`` as the integer 43200,
which becomes ``12:00:00``. A string of digits alone, such as ``"120"``, is
rejected as ambiguous (Slurm reads it as minutes).

A site may restrict submissions to approved account/partition pairs with a
scheduler policy file::

   profiles:
     - {id: lab_gpu, account: my_lab, partition: gpu}
     - {id: lab_gpu_priority, account: my_lab, partition: gpu_priority}

``DENDRITIC_SCHEDULER_PROFILES`` names the file (``none`` disables it);
otherwise ``$XDG_CONFIG_HOME/dendritic_modeling/scheduler_profiles.yaml``
(``XDG_CONFIG_HOME`` defaults to ``~/.config``) is used when it exists. Each
profile needs exactly ``id``, ``account`` and ``partition``. With a policy,
sweep generation (also with ``--generate-only`` and ``--dry-run``), reruns and
checkpoint-analysis submission reject any other explicit pair. Without one,
any pair is accepted and recorded in the frozen manifest as unvalidated; the
manifest records the policy file that approved a pair.

.. _sweeps-distributed:

Multi-GPU runs
--------------

With ``launcher: torchrun``, each array task runs
``python -m torch.distributed.run --standalone --nproc_per_node=<gpus_per_node>``:
one process group per config, on one node, one process per GPU. The ranks of a
run are not sweep replicates. For DDP keep the default ``training_script`` and
set ``base_config.distributed.mode: ddp``. For FSDP use the FSDP entry point
and ``mode: fsdp``, as in this excerpt of a sweep file::

   slurm_config:
     gpus_per_node: 4
     launcher: torchrun
     training_script: src/dendritic_modeling/scripts/training/train_experiments_fsdp.py
   base_config:
     distributed:
       mode: fsdp

The launcher always passes ``--standalone``, so one config cannot span several
nodes. The model and training options each mode supports are described in
:ref:`distributed-training`.

.. _sweeps-commands:

Generating and submitting
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - ``dendritic-sweep`` option
     - Effect
   * - ``--config FILE``
     - Generate from a sweep file; without a mode option, also submit.
   * - ``--run``
     - Generate and submit (the default).
   * - ``--generate-only``
     - Generate the configs and job scripts; submit nothing.
   * - ``--dry-run``
     - Generate, then print the ``sbatch`` commands without running them.
   * - ``--split-jobs N``
     - Write at most ``N`` sequential jobs (``jobs/run_sweep_<i>.sh``, nearly
       equal consecutive groups of configs) instead of one array job.
   * - ``--output-dir DIR``
     - Base directory of the sweep directory (with ``--analyze``: output
       directory of the analysis).
   * - ``--exclude-nodes NODE ...``
     - Replace ``slurm_config.exclude_nodes``.
   * - ``--slurm-account``, ``--slurm-partition``
     - Replace ``slurm_config.account`` and ``slurm_config.partition``.
   * - ``--analyze --results-dir SWEEP_DIR``
     - Analyze a finished sweep, with the analyzer type detected
       (:ref:`sweeps-analysis`).

Submitting needs an account and a partition, from ``slurm_config`` or the
command line. With ``--generate-only`` or ``--dry-run``, missing values are
written as ``PLACEHOLDER``; such a sweep cannot be submitted or rerun, so
regenerate it with real values. Explicit values are checked against the site
policy in every mode. Submission runs ``submit_all_jobs.sh``, which submits
each job script with ``sbatch``, stops at the first failure and appends job
ID, script and UTC time to ``jobs/submitted_jobs.tsv``. A sweep generated with
``--generate-only`` and real values can be submitted later with
``bash <sweep_dir>/submit_all_jobs.sh``. Generation loads every generated
config and checks the declared config count and ``execution_results_dir``
before it creates the sweep directory, so a sweep that fails these checks (for
example on an invalid axis) writes nothing. When writing the configs, job
scripts or manifest fails, the generation removes the sweep directory it
created, and any base directory it created that is still empty.

Running without Slurm
~~~~~~~~~~~~~~~~~~~~~

The generated configs are ordinary training configs. Generate them, then train
each one:

.. code-block:: bash

   dendritic-sweep --config my_sweep.yaml --generate-only
   for config in <sweep_dir>/configs/unified_config_*.yaml; do
       dendritic-train "$config"
   done
   dendritic-analyze <sweep_dir>

Each run writes into ``<sweep_dir>/results/config_<i>/`` (or under
``execution_results_dir``), as a sweep job would.

Source guard and reruns
-----------------------

``frozen_sweep_manifest.json`` records the generation: the mode
(``generate_only``, ``dry_run`` or ``submission``), the expected config count,
SHA-256 hashes of the sweep file, of ``original_config.yaml``, of every file of
the ``extends`` chain and of every generated config, the scheduler profile, and
the source identity of the library checkout (commit, digests of its tracked
changes and of its untracked files under ``src/``, and a digest of the
imported package).

Every job script starts with checks that exit before training:

- exit code 66: the checkout's commit differs from the one recorded at
  generation;
- exit code 67: its tracked changes differ;
- exit code 68: its untracked files under ``src/`` differ;
- exit code 75: a mount point holding an absolute path named in the configs
  does not answer within 60 seconds.

A checkout that is not a git repository gets no source guard; generation logs a
warning.

``rerun_missing_configs`` lists the configs of a sweep whose result directory
lacks ``performance/final.json`` or ``training_summary.json`` (under
``execution_results_dir`` when the sweep sets it). With ``--write-script`` it
writes ``jobs/rerun_<time>.sh``: the sweep's array script restricted to those
configs and bound to the source identity of the manifest. It refuses to write
the script from another source state or for a sweep generated with
placeholders; it does not submit it.

.. code-block:: bash

   python -m dendritic_modeling.scripts.sweeps.rerun_missing_configs <sweep_dir>
   python -m dendritic_modeling.scripts.sweeps.rerun_missing_configs <sweep_dir> --write-script
   sbatch <sweep_dir>/jobs/rerun_<time>.sh

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Option
     - Effect
   * - ``--timestamp T`` and ``--base-dir DIR``
     - Find the sweep directory ``<DIR>/*_<T>`` instead of naming it
       (``--base-dir`` defaults to the current directory).
   * - ``--config-ids 3,7``
     - Rerun these configs instead of every unfinished one.
   * - ``--force``
     - Allow ``--config-ids`` to name finished configs; their results are
       rewritten.
   * - ``--write-script``
     - Write the rerun script; without it, only list the configs.
   * - ``--max-concurrent N``
     - Array concurrency of the rerun job.

.. _sweeps-analysis:

Analyzing results
-----------------

.. code-block:: bash

   dendritic-analyze <sweep_dir>
   dendritic-analyze <sweep_dir> --type ei --output <analysis_dir>
   dendritic-sweep --analyze --results-dir <sweep_dir>

Pass the sweep directory, not its ``results/``: the analyzer reads the
generated configs from ``configs/`` and the runs from ``results/config_<i>/``.
For each run it collects configuration columns from the generated config and
these results:

- ``performance/final.json`` as ``<split>_<metric>`` columns, for example
  ``test_accuracy`` and ``valid_accuracy``;
- summaries of the information, weight, ablation, compartment SNR,
  multiplicative-gain and noise-perturbation analyses the runs wrote (see
  :ref:`analysis-outputs`);
- ``nparams``;
- from ``model_resources.json``: ``resource_<field>`` for
  ``total_parameters``, ``trainable_parameters``, ``core_total_parameters``,
  ``core_trainable_parameters``, ``synapse_candidate_parameters``,
  ``non_synaptic_parameters``, ``candidate_synapse_slots``,
  ``active_synapses`` and ``persistent_state_scalars_per_sample``, and per
  synaptic pathway ``resource_<pathway>_active_synapses``,
  ``_candidate_slots`` and ``_candidate_parameters``. The core fields count
  the core network alone; the synaptic fields count sparse connectivity;
- from ``training_summary.json``: ``training_best_epoch``,
  ``training_best_valid_loss``, ``training_final_train_loss``,
  ``training_final_valid_loss`` and ``training_epochs_completed``. Epoch
  histories are not copied; compare ``training_best_epoch`` with ``epochs`` to
  find runs that were still improving.

``--type`` selects the analyzer:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - ``--type``
     - Sweep
   * - ``ei``
     - E or I synapse counts vary; results by E/I ratio.
   * - ``stoch``
     - The stimulus duration or gain factor of a stochastic dataset varies.
   * - ``local_learning``
     - A local-learning rule setting varies.
   * - ``branch``
     - Layer sizes or branch factors vary.
   * - ``general``
     - Any other sweep.
   * - ``noise``
     - Runs with noise-perturbation results.

Without ``--type`` (and always with ``dendritic-sweep --analyze``), the type is
detected from the first five files of ``configs/`` in name order: ``ei``,
``stoch``, ``local_learning`` and ``branch`` are tried in turn, and ``general``
handles any sweep directory. ``noise`` is used only when requested.

The analyzer writes ``<type>_processed_data.csv`` (one row per run) and
``<type>_aggregated_data.csv`` to ``<sweep_dir>/plots/`` (or ``--output``).
The aggregated table has one row per condition, with ``<metric>_mean``,
``<metric>_std`` (sample standard deviation, empty for a single replicate) and
``n_replicates``. Plots go to the ``performance``, ``information``,
``weights``, ``ablation`` and ``noise`` subdirectories: performance,
information and weight plots against the sweep variables the analyzer
recognizes (the E/I ratio, the stimulus duration and gain factors of the
stochastic datasets, and the parameter count across dendritic network
categories), and ablation and noise plots when the runs wrote those analyses.
When no plot applies, the ``general`` analyzer draws bar charts of the mean
metrics in ``summary/``; analyze other sweep variables from the CSV files.
Every plotted series has a marker and a line style of its own besides its
colour from the Okabe-Ito palette (``dendritic_modeling.plotting.styles``),
so the plots read without colour: network categories, dendritic depths, data
splits, the metrics of a group, noise types, ablation types, local-rule
variants and the layer-by-category series of the weight plots.

Conditions are told apart only by the configuration columns the collector
extracts. These include the network type; for a population network, the first
layer's readout population (E and I synapse counts, branch factors, size) and
the synapse, gate and routing options; the training strategy, batch size,
epochs, ``param_groups.lr`` (as ``learning_rate``) and optimizer; local-rule
settings; the dataset and its parameters; and ``sweep_variant``. Runs that
differ only in another swept path, for example ``topk_lr``, are averaged as
replicates of one condition. Express such comparisons as ``sweep_variants``,
whose names separate conditions, or compare seed-level results with the tools
below.

Seed-level comparisons
~~~~~~~~~~~~~~~~~~~~~~

``paired_comparison`` pairs two arms of one sweep seed by seed and writes
``paired_units.csv`` and ``paired_summary.json``, with a bootstrap interval of
the mean difference. It fails when a seed lacks one of the arms.

.. code-block:: bash

   python -m dendritic_modeling.scripts.sweeps.paired_comparison <sweep_dir> \
       --output <comparison_dir> --arm-column sweep_variant \
       --reference shunting --comparison normalized_additive --metric test_accuracy

``dendritic-collect-sweep-metrics`` collects declared, complete sweeps into
one table with one row per trained run; it does not average over seeds. Its
manifest (relative paths resolve against the manifest's directory)::

   schema_version: sweep_metric_collection_v1
   output_dir: collected/topk_lr          # or --output-dir on the command line
   required_columns: [test_accuracy, valid_accuracy, seed]
   sweeps:
     - name: topk_lr                      # unique; becomes the sweep_name column
       root: ../sweeps/topk_lr_<timestamp>  # a sweep directory
       expected_config_count: 45          # collection fails on another count
       labels: {protocol: baseline}       # optional scalar columns
       # artifact_set_sha256: ...         # optional; must match when given
   training_history:                      # optional
     fields: {train_losses: train_loss, valid_losses: valid_loss}
     performance_epoch_fields: {accuracy.valid: valid_accuracy}

.. code-block:: bash

   dendritic-collect-sweep-metrics --manifest collection.yaml

It writes ``checkpoint_units.csv`` (the collected columns plus the labels,
``sweep_name``, ``sweep_root`` and ``sweep_artifact_set_sha256``) and
``collection_provenance.json``: the collection records hashes of every
generated config and ``performance/final.json`` it read. With
``training_history.fields``, it also writes ``training_history.csv``, one row
per run, metric and epoch, from the lists in ``training_summary.json``. Rows of
``performance_epoch_fields``, read from ``performance/epochs/epoch<N>.json``
(written when ``analysis.performance.training`` is true), are written to that
file only when ``fields`` is declared too. A run whose ``analysis_status.json``
records that ``performance`` did not complete, in its final pass or in a
per-epoch pass that ran it (see :ref:`analysis-status`), fails the
collection; runs written before status records existed are collected as
before.
