Configuration guide
===================

A DendriNet run is described by one YAML file. ``dendritic-train`` loads the
file into a typed ``Config`` object (``dendritic_modeling.config.Config``),
checks every key, builds the datasets, the model and the trainer from it, and
saves the resolved configuration in the run directory. This page explains how
a config is written, run and checked, how the random seeds and the numerics
flags work, and where each option is documented.

.. _config-files:

Writing and running a config
----------------------------

Top-level sections
~~~~~~~~~~~~~~~~~~

A config file is a mapping of the sections below. Every section is optional:
an omitted section, like every omitted option, takes its default.

.. list-table::
   :header-rows: 1
   :widths: 16 54 30

   * - Section
     - What it configures
     - Documented in
   * - ``experiment``
     - Random seeds, determinism and numerics flags, the train/validation
       split, test-split sealing, dataset fingerprints and periodic
       checkpoints.
     - :ref:`random-seeds`, :ref:`dataset-splits`, :ref:`run-outputs`
   * - ``data``
     - The dataset (``dataset_name``), the data root (``base_dir``), common
       processing (``processing``) and the options of the selected dataset
       (``dataset_params.<dataset_name>``).
     - :ref:`dataset-catalogue`
   * - ``model``
     - The task (``classification`` or ``regression``), the encoder, the core
       network, the decoder, the scaling of the classifier output and a warm
       start from saved weights (``initial_checkpoint``).
     - :ref:`model-overview`
   * - ``training``
     - The main training phase, ``training.main``: the strategy, the options
       every strategy shares (``common``), the optimizer, the per-strategy
       blocks (``strategies``), the local learning rules
       (``learning_strategy_config``), regularization and pruning.
     - :ref:`training-strategies`
   * - ``analysis``
     - How the analyzers iterate over the data (``runtime``) and one section
       per analyzer.
     - :ref:`analyzer-catalogue`
   * - ``outputs``
     - Where the run directory goes (``results_dir``, ``run_name``,
       ``exact_run_dir``).
     - :ref:`run-outputs`
   * - ``distributed``
     - The runtime (``mode``: ``none``, ``ddp`` or ``fsdp``), the collective
       timeout (``timeout_seconds``) and the FSDP options (``fsdp``).
     - :ref:`distributed-training`
   * - ``metadata``
     - Free-form annotations, saved in ``config.json`` and never read.
     - :ref:`strict-loader`

A file can also name a parent file with ``extends``; see :ref:`config-extends`.

A minimal config
~~~~~~~~~~~~~~~~

This config trains a one-layer population network on MNIST. An inhibitory
population and an excitatory population both read the input, the inhibitory
population also projects to the excitatory one, and a linear readout of the
excitatory somas classifies the ten digits. Each excitatory soma has two
dendritic branches, each with two branches of its own
(``branch_factors: [2, 2]``). The inhibitory population sets
``ff_inhibitory_synapses: 0`` because no inhibitory source projects to it.

.. code-block:: yaml

   experiment:
     seed: 1
   data:
     dataset_name: mnist
   model:
     core:
       type: population_network
       population_network:
         layers:
           - name: layer0
             readout_population: e
             population_defaults:
               ff_excitatory_synapses: 16
               ff_inhibitory_synapses: 4
             populations:
               - name: i
                 polarity: inhibitory
                 n_neurons: 16
                 branch_factors: [1]
                 population:
                   ff_inhibitory_synapses: 0
               - name: e
                 polarity: excitatory
                 n_neurons: 32
                 branch_factors: [2, 2]
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
         epochs: 5
         batch_size: 128
   analysis:
     performance:
       enabled: true
   outputs:
     results_dir: results
     run_name: mnist_population

The options left out take their defaults. For example, the MNIST loader
flattens each image into 784 inputs, and training sets the encoder's input
size from the data. :ref:`reference-config` shows complete configs with a
comment on every option.

Running a config
~~~~~~~~~~~~~~~~

Train with ``dendritic-train``:

.. code-block:: bash

   dendritic-train mnist_population.yaml

``--validate-only`` builds everything a run needs and exits without training
(see :ref:`strict-loader`):

.. code-block:: bash

   dendritic-train mnist_population.yaml --validate-only

Command-line options replace values of the file. They are applied to the YAML
before it is checked, so an override is checked exactly like a value written
in the file, and the saved ``config.json`` records the result. Each override
is also logged when the run starts.

.. code-block:: bash

   dendritic-train mnist_population.yaml --epochs 1 --seed 3 \
       --output_dir runs --run_name smoke --data_dir /path/to/data

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Option
     - Sets
   * - ``--output_dir DIR``
     - ``outputs.results_dir``
   * - ``--run_name NAME``
     - ``outputs.run_name``
   * - ``--data_dir DIR``
     - ``data.base_dir``
   * - ``--seed N``
     - ``experiment.seed``
   * - ``--epochs N``
     - ``training.main.common.epochs`` (at least 1)
   * - ``--learning_strategy NAME``
     - ``training.main.strategy``
   * - ``--error_broadcast_mode MODE``
     - ``training.main.learning_strategy_config.error_broadcast_mode``
   * - ``--lr``, ``--topk_lr``, ``--blocklinear_lr``, ``--reactivation_lr``,
       ``--decoder_lr``
     - ``training.main.common.param_groups.<name>`` (not negative)
   * - ``--train_samples_per_class N``, ``--val_samples_per_class N``
     - ``data.dataset_params.imagenet.<name>``; only with
       ``data.dataset_name: imagenet``

The FSDP entry point, launched with ``torchrun`` as
``-m dendritic_modeling.scripts.training.train_experiments_fsdp``, takes the
same options (see :ref:`distributed-training`). :ref:`cli-reference`
reproduces the full ``--help`` output of every command.

Loading a config from Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``load_config`` reads a file the way ``dendritic-train`` does and returns the
typed ``Config``. ``config_from_mapping`` checks an in-memory mapping with the
same rules. Both raise for unknown keys and invalid values (see
:ref:`strict-loader`).

.. code-block:: python

   from dendritic_modeling.config import config_from_mapping, load_config
   from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

   config = load_config(
       "mnist_population.yaml",
       overrides={"training.main.common.epochs": 1},
   )
   print(config.model.core.type)  # population_network
   model, encoder = initialize_model(config.model)

   defaults = config_from_mapping({"data": {"dataset_name": "mnist"}})

``overrides`` maps dotted paths to values and creates missing sections; the
values are checked like values written in the file. ``load_config()`` without
arguments returns the checked defaults. A path to a sweep file loads the
sweep's ``base_config`` (see :doc:`sweeps`).

``initialize_model`` builds the encoder, the core and the decoder as training
does, but without the rest of the training setup. Before it builds the model,
training fills unset component seeds from the seed streams
(:ref:`random-seeds`) and sets the encoder's input size from the data. A
direct call uses the components' own seed defaults and
``model.encoder.params.input_dim`` (784 when it is unset and the encoder is
``identity``).

``load_analysis_config(path)`` reads a standalone analysis profile: a file
with an ``analysis`` section, or the section's mapping alone.
``dendritic-analyze-checkpoint --analysis-config`` uses it (see
:ref:`checkpoint-analysis`).

.. _config-extends:

Composing configs with ``extends``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A file can name a parent file with a top-level ``extends: <path>`` key. A
relative path is resolved against the directory of the file that names it.
The child is merged over the parent: mappings merge key by key, and a list in
the child replaces the parent's list. A parent can extend another file; a
cycle raises. For example, a file that contains

.. code-block:: text

   extends: mnist_population.yaml
   training:
     main:
       common:
         epochs: 20

trains the model of ``mnist_population.yaml`` for 20 epochs. ``load_config``
reads the whole chain on every call, so an edit to any file of the chain takes
effect at the next load. ``extends`` works in files only;
``config_from_mapping`` rejects it as an unknown section.

.. _strict-loader:

How configs are checked
-----------------------

Unknown keys
~~~~~~~~~~~~

The loader checks every section that has a typed schema against the keys the
library reads. A key it does not read raises ``UnknownConfigKeyError`` (a
``ValueError``), which names the section and suggests the closest valid key:

.. code-block:: text

   UnknownConfigKeyError: Unknown training.main.common option(s): epoch (did you mean 'epochs'?). These keys would be silently ignored.

When no valid key is close, the message lists the valid keys instead. An
unknown top-level section raises the same way. Sections that installed
plugins register are checked by their plugins (:ref:`plugin-config-sections`).

Encoder and decoder ``params`` share one schema for all encoder and decoder
types, and each type reads only some of its fields. A field that the selected
type does not read raises unless it holds its default value. A ``params``
block that lists every field, as a saved ``config.json`` does, logs a warning
for such a field instead.

Annotations
~~~~~~~~~~~

Annotations for people and tools go in the top-level ``metadata`` mapping or
in top-level keys that start with an underscore, such as
``_sweep_config_id``. Both end up in ``Config.metadata`` and in the saved
``config.json``; the library never reads them. One key given in both places
with different values raises. Inside a section, a key that starts with an
underscore is an unknown key.

Retired, renamed and deprecated keys
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The loader accepts some keys that are not part of the schema:

- A retired key, listed in ``dendritic_modeling.config.keys.RETIRED_KEYS``,
  loads with a warning that says why it is ignored. A retired key whose value
  requests a behaviour the library cannot reproduce raises instead, for
  example ``model.core.reactivation.match_additive_init_to_shunting: true``.
  Sections of an older configuration layout, such as ``model.core_network`` or
  a top-level ``task`` or ``train`` section, raise in the same way.
- A renamed field loads under its current name with a warning. Giving both
  names with different values raises.

  .. list-table::
     :header-rows: 1
     :widths: 55 45

     * - Written as
       - Loads as
     * - ``training.main.common.weight_decay_rate``
       - ``synapse_decay_rate``
     * - ``training.main.strategies.multi_stage.use_best_from_each_stage``
       - ``final_state`` (``true`` becomes ``best_stage``)
     * - ``training.main.strategies.recurrent.grad_clip``
       - ``grad_clip_value``
     * - population ``topk_strategy``, ``reactivation_strategy``,
         ``blocklinear_strategy``
       - ``topk_gradient_scaling``, ``reactivation_gradient_scaling``,
         ``blocklinear_gradient_scaling``

- A deprecated spelling loads with a warning: ``model.core.ei_unified`` for
  ``model.core.unified_ei``, ``model.core.spiking`` for
  ``model.core.dynamics``, ``model.core.connectivity.structured_connectivity``
  for ``model.core.connectivity.structured``, and
  ``model.core.sparsity.annealed_topk`` for
  ``model.core.sparsity.dense_to_sparse``. Giving both spellings raises,
  because one of them would be ignored.
- An analysis section under its ``*_analysis`` name, for example
  ``performance_analysis``, loads as the analyzer's section
  (``performance``) with a warning.
- A top-level ``fsdp`` section is not read: FSDP options belong in
  ``distributed.fsdp``. It raises when the config has no ``distributed``
  section.
- ``model.vision_replacement``, ``model.pretrained_replacement``,
  ``training.vision_replacement``, ``training.transformer_replacement`` and
  the strategy name ``vision_distillation`` are reserved for optional
  extensions that this package does not include. The sections load when they
  are absent or set ``enabled: false``; enabling one raises.
  ``model.transformer_replacement`` is read only by
  ``dendritic_modeling.networks.architectures.transformer``, and
  ``dendritic-train`` raises when it is enabled.

.. _config-version:

Configuration format version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The top-level ``config_version`` names the configuration format a file is
written for. The current format is ``4``
(``dendritic_modeling.config.CONFIG_VERSION``): the meaning this library gives
every key, including what an omitted key means. A file without
``config_version`` loads with that meaning; an earlier version raises with a
pointer to ``dendritic-upgrade-config``, and a later one asks for a newer
library. Runs record the version in ``config.json``, and the sweep tools write
it into every generated config.

Write ``config_version: 4`` in new files, as the shipped configs do:
``dendritic-upgrade-config`` reads a file without it as written for an
earlier version, whose omitted keys could mean something else.

Version 4 enables the biological constraint check by default for standard
dendritic E/I cores (:ref:`biological-constraints`). The upgrade tool writes
``biological_neuron: false`` for older omitted declarations; it preserves the
original computation rather than redesigning a circuit.

Version 3 (2026-10-13) changed what two omitted structured-connectivity keys
mean (:ref:`structured-connectivity`): a ``distance`` rule of
``model.core.connectivity.structured`` without ``distance_kernel`` is
``gaussian``, as in population rules (version 2: ``none``), and training
fills the unset seed of a population-network or unified E/I rule from
``experiment.topology_seed`` (version 2: its masks drew from the global
random stream). Version 2 (2026-10-11) made the earlier defaults that
``dendritic-upgrade-config`` writes below explicit.

.. _config-upgrade:

Upgrading configs written for earlier versions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``dendritic-upgrade-config`` writes copies of experiment and sweep files in
the current form, describing the same run:

.. code-block:: bash

   dendritic-upgrade-config old/*.yaml --output-dir upgraded/ --report upgrade.json

It prints the number of changes per file; ``--report`` records every change,
and ``--in-place`` overwrites the files (YAML comments are lost). Copies keep
their paths relative to the common directory of the inputs, and the tool
stops before writing anything when two inputs would be written to one file.
Every upgraded file starts with ``config_version: 4``; a file that already has
it is left unchanged, so upgrading twice changes nothing. Each rewrite applies
to the files of the versions before the one that changed the meaning it
records, so a version-2 file gets the version-3 and version-4 rewrites. For files
written for version 2 or earlier the tool

- writes ``distance_kernel: none`` into a ``model.core.connectivity.structured``
  block whose ``distance`` rules have no kernel, removes the second spelling
  of a key (``kernel``, ``sigma``, ``p_connect``, ``in_degree``) that a root
  key shadowed in a pathway entry and the second of two spellings or pathway
  entries written for one key or pathway, and gives a population-network or
  unified E/I rule without a seed ``seed: global_rng`` (``seed: 0`` for a
  spatial rule). It compares the rewritten rules with their version-2 reading
  for every population and pathway and reports any difference;
- sets inhibitory counts of unified E/I populations that have no input to
  ``0``, and removes the ``input_dim`` of the recurrent baseline payloads (set
  from the encoder) and the ``baseline_rnn`` ``cell_type`` (set by the core
  type).

For files written before version 2 it also

- moves top-level annotations (keys that are not configuration sections),
  and annotation blocks inside ``analysis``, under ``metadata``, and gives
  renamed fields their current names, including dotted sweep override paths;
- removes retired keys that hold their old default, and settings the library
  never applied: ``learned_output_scale`` next to an ``output_scale_mode``,
  morphology values that the core-type alias replaced, encoder or decoder
  params their type does not read, population
  sparse-synapse options their synapse type does not read, additive-integration
  options of shunting layers, ``recurrent_ei`` settings of feedforward
  cores, a learned output-scale mode without an identity decoder (written as
  ``fixed``), ``quantile`` gate updates without recalibration (written as
  ``frozen``), ``data.processing``
  flags the dataset's loader does not apply, step options such as
  ``enable_nan_checking`` that the selected training loop does not apply,
  population-network and unified E/I ``input_dim`` (set from the encoder),
  and a layer ``population_defaults.indexed_seed`` that its indexed
  populations replaced with the topology stream;
- writes out wiring that the retired ``transfer.input_mode`` options or the
  stricter synapse checks require: declared inhibitory cells that a
  feedforward core with ``input_mode: 1`` (the former default) never built
  get size ``0`` (:ref:`structured-inhibitory-wiring`), a short
  ``inhibitory_layer_sizes`` list gets one entry per layer, counts of
  synapses without a source or target become ``0``, a delayed population
  connection without recurrent synapses gets an explicit
  ``rec_*_synapses: 0``, and a unified E/I layer without an ``inhibitory``
  entry gets the default 16-cell population it built;
- writes the stochastic Top-K settings that omitted keys meant before
  2026-10-08: ``model.core.sparsity`` of type ``stochastic`` gets
  ``temperature: 0.0`` (the noisy Top-K it ran), population options get
  ``topk_temperature: 0.0`` or ``topk_ultrafast: false``, and indexed
  dynamic synapses with stochastic selection at the former temperature 0,
  which selected exactly the standard candidates, get ``selection:
  standard``. The same selection on the stochastic layer type is reported,
  because the standard layer draws its initial weights differently;
- removes a top-level ``fsdp`` block identical to ``distributed.fsdp`` or
  with ``use_fsdp: false`` (a run without FSDP) and reports any other: whether
  a run read it depended on the library version.

Retired keys the current library cannot reproduce, and routing it cannot
express, are reported as unresolved and left in place. The copy also lists
every unresolved item under the top-level key ``unresolved_migration``, after
``config_version``: the loader refuses such a partly upgraded file and names
the items, and running the tool on it again rewrites nothing and reports the
same items, until they are resolved by hand and the key is deleted. In a sweep file a rewrite of
``base_config`` that depends on settings a sweep axis or variant also sets is
applied only when it rewrites every generated config the same way, and a
rewrite that depends on an ``extends`` parent is reported; generate the sweep
and upgrade the generated configs in those cases. Some changes need the
user: sequence loaders report the dataset parameters and processing flags
they do not apply only when the dataset is built, and datasets defined
outside the library must be registered with ``register_dataset`` before a
run. The tool does not change
a result's library pin: reproduce a published result at the revision that
produced it, and compare a migration on fixed inputs before relying on it.

What is checked when
~~~~~~~~~~~~~~~~~~~~

``load_config`` and ``config_from_mapping`` check:

- the keys of every typed section: ``experiment``, ``data`` (except the
  contents of ``dataset_params``), ``model`` with the encoder, the decoder
  and their ``params``, the typed core sections (``architecture``,
  ``connectivity``, ``transfer``, ``morphology``, ``sparsity``,
  ``reactivation``, ``blocklinear``, ``implementation``, ``spatial``,
  ``affine_mlp``, ``dynamics``, ``dendritic_spikes``, ``soma_feedback`` and
  ``recurrent_ei``), every ``training`` section including the strategy
  blocks and the stages of ``multi_stage``, the analysis section names and
  the keys of each analyzer section, ``outputs`` and ``distributed``;
- many option values, for example ``distributed.mode``, the FSDP choices
  (``sharding_strategy``, ``backward_prefetch``, the dtypes,
  ``auto_wrap_policy``), the selectors of the information analysis, the
  local-rule choices, ``grad_clip_mode`` and the gate ``init_policy`` of a
  structured E/I core;
- the structure of the model: per-layer synapse-list lengths of every core
  type built from the structured sections, duplicate layer or population
  names, unknown connection endpoints and readout populations, delayed
  population connections without ``recurrent: true``, and same-step
  population cycles. These raise one ``ConfigValidationError`` that lists
  every problem.

``dendritic-train`` then runs a training-stability check on the loaded
config. It logs warnings about risky option combinations and stops the run
for a critical one, such as normalized (possibly negative) inputs to a
shunting structured E/I core with positively transformed weights.

The other checks run when the run is built:

- When the model is built: the keys inside ``model.core.population_network``
  (at every level), ``unified_ei``, ``vision``, ``baseline_rnn``,
  ``heterogeneous_leak_ctrnn`` and ``legendre_memory`` and in the entries of
  ``synapse_types``, and the names of the core type, the encoder and decoder
  types, the synapse type (``sparsity.type``, ``topk_type``), the gate type
  (``reactivation.type``, ``reactivation_type``), ``weight_transform``,
  ``dbl_init_method`` and a population's ``reactivation_init_policy``. A
  misspelled key raises an ``UnknownConfigKeyError`` (a ``ValueError``) that
  names the section and suggests the closest key, for example:

  .. code-block:: text

     UnknownConfigKeyError: Unknown model.core.population_network.layers[0].populations[e] option(s): ff_excitatory_synapse (did you mean 'ff_excitatory_synapses'?). These keys would be silently ignored.

  Synapse counts without a source or target raise a ``ValueError`` when the
  network is built: the inhibitory synapses of a structured E/I core (see
  :ref:`inhibitory-wiring`) and the written inhibitory counts of a unified
  E/I population.

- When the datasets are built: ``data.dataset_name`` and the keys of
  ``data.dataset_params.<dataset_name>``. Only the block of the selected
  dataset is read, so the blocks of other datasets are not checked.
- When the trainer is built: ``training.main.strategy``,
  ``training.main.common.loss_function``, ``training.main.optimizer.name``,
  ``training.main.common.lr_schedule`` and the options that the selected
  strategy cannot apply.
- When the run starts: the determinism and numerics flags (see
  :ref:`random-seeds`).

``dendritic-train config.yaml --validate-only`` runs all of these steps. It
loads the config, builds the datasets and the model, calibrates data-driven
gates, builds the analysis manager, the optimizer and the trainer, writes the
run's configuration and construction records (see :ref:`run-outputs`) and
exits without training. The FSDP entry point with ``--validate-only`` also
wraps the model and builds its optimizer. Run it before a long or queued job.

``--validate-only`` does not reach every value. The ``learning_strategy`` of a
``multi_stage`` stage is checked only when that stage starts. Some analyzer
options are not checked when the config loads:
``ablation.params.methods.clamp_statistic`` other than ``mean`` raises when
the analyzer runs after training, an unknown correlation or ablation selector
selects nothing, and an unknown ``analysis.runtime`` mode behaves like
``auto``. Try an analysis setup on a short run before a long one.

.. _random-seeds:

Random seeds and reproducibility flags
--------------------------------------

Seed streams
~~~~~~~~~~~~

``experiment.seed`` (default ``42``) is the root of every random stream. The
seven streams below default to it (``null``); set one to hold it fixed while
``seed`` varies.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Option
     - What it seeds
   * - ``dataset_seed``
     - Generated datasets, data noise and label noise. The global random
       generators are seeded with it while the datasets are built.
   * - ``split_seed``
     - The train/validation split.
   * - ``model_seed``
     - Parameter initialization. The global generators are reseeded with it
       right before the model is built, so training-time noise (dropout,
       stochastic synapses) also draws from this stream.
   * - ``topology_seed``
     - Connectivity: indexed synapse contacts, structured connectivity masks,
       population connection seeds and the random input split of the transfer
       layer.
   * - ``loader_seed``
     - The order of the training batches.
   * - ``evaluation_seed``
     - Sampling in the analysis runtime.
   * - ``probe_seed``
     - Analysis probes, decoders, nulls and the other analyzer draws.

To vary initialization and data order over a fixed connectivity, set
``experiment.topology_seed`` and vary ``experiment.seed`` (for example with
``dendritic-train --seed``).

Component seeds
~~~~~~~~~~~~~~~

A seed written on a component, such as ``population_defaults.indexed_seed``, a
layer's ``connection_seed``, ``model.core.sparsity.indexed.seed`` or
``model.core.connectivity.structured.seed``, is kept and pins that component.
Training fills only the seeds left unset, from the matching stream. A layer's
``population_defaults.indexed_seed`` is the base seed of all its populations;
the builder derives a distinct stream for each layer and population from it.

Feedforward structured E/I cores, spatial cores and ``ss_mlp`` cores keep
``model.core.initialization_seed`` unset and initialize from the global
generator, which ``model_seed`` seeds. A feedforward structured E/I core with
an explicit ``initialization_seed`` derives a stream for each layer from it
instead; ``ss_mlp`` cores reject the option. For every other core type,
training fills an unset ``initialization_seed`` from ``model_seed``.
Population networks and unified E/I cores derive a separate initialization
stream for each population from it, so changing one population does not
change the initial weights of the others.

Analysis seeds
~~~~~~~~~~~~~~

An unset analysis-runtime seed (``analysis.runtime.training.seed``,
``analysis.runtime.final.seed``) takes ``evaluation_seed``. Every other unset
analysis seed (decoder, null-baseline, shuffle, support and draw seeds) takes a
seed derived from ``probe_seed`` and the seed's config path, so the analyzers
draw independent streams that do not change when other analyses are enabled.
A written analysis seed is used as it is. Analyses run with the global random
state saved and restored, so they neither depend on nor disturb training.
Checkpoint analyses fill unset seeds in the same way. An analyzer built
directly from Python, outside these runs, uses its own default for an unset
seed.

Determinism and numerics flags
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These ``experiment`` options control PyTorch's numerics:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Option (default)
     - Effect
   * - ``fast_mode`` (``false``)
     - The flags below that are unset follow it. ``false``: deterministic
       cuDNN kernels, no cuDNN autotuning, no TF32 and
       ``float32_matmul_precision: highest``. ``true``: nondeterministic
       cuDNN kernels, autotuning, TF32 and ``high``.
   * - ``deterministic`` (unset)
     - Sets ``torch.backends.cudnn.deterministic``; unset means
       ``not fast_mode``. It does not turn on PyTorch's deterministic
       algorithms; ``strict_deterministic`` does.
   * - ``cudnn_benchmark`` (unset)
     - The cuDNN autotuner, always off while ``deterministic`` is true. Unset
       means ``fast_mode`` and not ``deterministic``.
   * - ``allow_tf32`` and ``float32_matmul_precision`` (unset)
     - One setting: ``highest`` disables TF32, ``high`` and ``medium`` allow
       it. Writing one sets the other; writing both requires that they agree.
   * - ``strict_deterministic`` (``false``)
     - Exact math: deterministic cuDNN kernels, no autotuning, no TF32,
       ``highest`` precision and ``torch.use_deterministic_algorithms(True)``,
       so an operation without a deterministic implementation raises when it
       runs. It sets ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` when the variable is
       unset (and warns about a value other than ``:4096:8`` or ``:16:8``),
       and data loaders with workers deliver batches in order
       (``in_order: false`` raises). A written flag that contradicts it
       raises.

A written flag takes precedence over ``fast_mode``. The flags are checked when
the run starts, a step that ``--validate-only`` includes, and training writes
the resolved values into ``config.json``.

What a run records
~~~~~~~~~~~~~~~~~~

``config.json`` records the filled seeds, including the component and
analysis seeds, and the resolved flags. ``resolved_seeds.json`` records the
eight resolved stream seeds, ``strict_deterministic`` and the numerics state
that PyTorch reports after setup: the float32 matmul precision, TF32 for
matmuls and cuDNN, cuDNN determinism and autotuning, and deterministic
algorithms. ``runtime_environment.json`` records the imported library, its
checkout state and the numerical stack. :ref:`run-outputs` describes these
files and the other records of a run.

Where options are documented
----------------------------

- Model structure, core types, encoders and decoders:
  :doc:`network_architectures`.
- Population networks: :doc:`population_networks`.
- Structured E/I cores: :doc:`structured_ei_cores`.
- Dendritic trees, integration modes and gates: :doc:`dendritic_neurons`.
- Synapses and connectivity: :doc:`synapses`.
- Receptor synapses, spiking somas, dendritic spikes and soma feedback:
  :doc:`temporal_dynamics`.
- Training strategies, loss functions and multi-stage training:
  :ref:`training-strategies`, :ref:`loss-functions`,
  :ref:`multi-stage-training`.
- Datasets and their options: :ref:`dataset-catalogue`.
- Analyzers and their options: :ref:`analyzer-catalogue`.
- Distributed training: :ref:`distributed-training`.
- Parameter sweeps over configs: :ref:`sweeps`.
- Run directories and records: :ref:`run-outputs`.

:ref:`reference-config` shows two complete configs with a comment on every
option.
