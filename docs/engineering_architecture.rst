Engineering Architecture
========================

This page is for engineers working inside the repository. It focuses on how
the codebase is organized, where the main control points live, and how a run
moves from YAML to artifacts.

System Overview
---------------

The framework has a small number of stable layers:

.. image:: _static/architecture/system-overview.svg
   :alt: System overview of configs, datasets, model factory, training, analysis, and artifacts.
   :width: 100%

For neuron-level and operator-level diagrams of the dendritic core itself, see
:doc:`network_architectures`. That page shows the actual ``DendriNet`` tree
layout, ``TopKLinear`` synapse banks, ``BlockLinear`` branch aggregation, and
the E/I layer stack used by the factory.

The practical rule of thumb is:

- ``config/`` and ``configs/`` define intent.
- ``src/dendritic_modeling/networks`` defines model behavior.
- ``src/dendritic_modeling/scripts`` turns config into a runnable experiment.
- ``src/dendritic_modeling/training`` defines update rules and training loops.
- ``src/dendritic_modeling/analysis`` turns checkpoints and trajectories into diagnostics.

Repository Layout
-----------------

The main directories are:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Path
     - Responsibility
   * - ``src/dendritic_modeling/config``
     - Structured config dataclasses and config-facing defaults.
   * - ``src/dendritic_modeling/datasets``
     - Standard dataset loaders, synthetic tasks, and dataset utilities.
   * - ``src/dendritic_modeling/networks/architectures``
     - Feedforward E/I cores, recurrent E/I cores, dendritic operators, and baselines.
   * - ``src/dendritic_modeling/training``
     - Training strategies such as backprop and local learning variants.
   * - ``src/dendritic_modeling/analysis``
     - Reusable analyzers and the analysis manager used by training and evaluation scripts.
   * - ``src/dendritic_modeling/scripts``
     - Top-level experiment entry points, environment setup, and config canonicalization.
   * - ``configs/``
     - Example and experiment YAMLs.
   * - ``docs/``
     - User-facing and engineering-facing documentation.

Build Path: From YAML to Model
------------------------------

Most experiment entry points follow the same build sequence:

.. image:: _static/architecture/training-analysis-flow.svg
   :alt: Experiment build and training flow from YAML through canonicalization, model init, training, analysis, and saved artifacts.
   :width: 100%

The important engineering points are:

- ``config.model_aliases.canonicalize_model_core_flags`` and
  ``canonicalize_rnn_core_flags`` normalize alias-driven settings before model
  construction. ``scripts.script_utils.config_utils`` provides script-facing
  wrappers around the same behavior.
- ``setup_utils.initialize_model`` is the standard entry point for creating a
  model from the resolved config.
- ``networks.architectures.factory.get_architecture`` is the central dispatch
  point from config type strings to actual modules. Project-local
  architectures can register extension builders with
  ``networks.architectures.register_architecture`` without editing the factory.
- post-build gate diagnostics write ``init_gate_stats.json`` when learnable
  reactivation gates are present.

Feedforward and Recurrent E/I Consistency
-----------------------------------------

The feedforward and recurrent E/I stacks share the same branch-level building
blocks. The recurrent stack adds temporal state and recurrent routing, but it
does not redefine the dendritic branch semantics.

.. image:: _static/architecture/recurrent-ei-flow.svg
   :alt: Recurrent E/I architecture showing shared branch logic, stateful dendritic traces, recurrent masking, and outputs.
   :width: 100%

The consistency contract is:

- both feedforward and recurrent E/I models use the same branch-layer logic
  for dendritic integration and reactivation-gate initialization
- recurrent aliases only add type-implied flags and recurrent connectivity;
  they do not change the underlying dendritic branch implementation
- ``allow_self_recurrence`` is a recurrent-only extension that masks
  same-population self-feedback while preserving other recurrent connectivity
- positive shunting safety is enforced at the transfer-layer boundary, so raw
  inputs may be normalized as long as the transfer output is clamped back to a
  non-negative range before the first shunting layer

Recurrent Dynamics Boundaries
-----------------------------

The recurrent dendritic implementation is split by runtime responsibility:

- network orchestration: ``population_network.py`` builds ordered
  population-network layers, manages sequence state, and applies readout
  reduction
- layer routing: ``population_layer.py`` maps named populations and directed
  connections into ``StatefulDendriNet`` modules
- config/state helpers: ``population_configs.py``, ``population_sources.py``,
  ``population_state.py``, and ``population_constants.py`` define the
  dataclasses and source normalization used by both feedforward and recurrent
  population-network execution
- population configuration: ``ei_config.py`` defines synapse counts, dendritic
  morphology, temporal constants, synapse-type options, soma dynamics, branch
  spike options, and soma-feedback options
- temporal population execution: ``stateful_dendrinet.py`` owns branch-layer
  construction and the recurrent step
- typed synapse dynamics: ``stateful_typed_dynamics.py`` integrates configured
  receptor traces, short-term plasticity, voltage gates, and typed shunting or
  additive currents

LIF soma dynamics live in ``spiking_soma.py``. Local dendritic branch
spike/plateau updates live in ``dendritic_spikes.py``. ``StatefulDendriNet``
remains the orchestration point for per-timestep state updates; geometry
helpers live in ``dendritic_geometry.py``. New recurrent features should follow
the same boundary rule: standalone dynamics belong in a focused helper or
mixin, while branch construction and the ordered step remain in
``StatefulDendriNet``.

Key Runtime Objects
-------------------

When you are tracing a bug or extending the code, these are the most useful
landmarks to keep in mind:

- ``src/dendritic_modeling/scripts/training/train_experiments.py``:
  experiment entry point for standard training
- ``src/dendritic_modeling/scripts/script_utils/setup_utils.py``:
  environment setup and model initialization
- ``src/dendritic_modeling/config/model_aliases.py``:
  config alias canonicalization and type-implied model defaults
- ``src/dendritic_modeling/scripts/script_utils/config_utils.py``:
  script compatibility wrappers and config-to-runtime flattening helpers
- ``src/dendritic_modeling/networks/architectures/factory.py``:
  model factory and architecture dispatch
- ``src/dendritic_modeling/networks/architectures/registry.py``:
  extension registry for project-local architecture builders
- ``src/dendritic_modeling/networks/architectures/excitation_inhibition``:
  feedforward E/I implementation
- ``src/dendritic_modeling/networks/architectures/excitation_inhibition/synapse``:
  sparse synapse implementations and sparse-layer factory. ``factory.py`` owns
  sparse layer registration and lookup. ``indexed.py`` is the public facade for
  fixed indexed, adaptive rewiring, and candidate-pool indexed TopK layers;
  implementation files are split by behavior.
- ``src/dendritic_modeling/networks/architectures/recurrent``:
  recurrent unified-EI and population-network implementation
- ``src/dendritic_modeling/networks/architectures/replacement``:
  shared replacement-cell components: input adapters, dendritic core/output
  adapter contracts, and layerwise replacement-stack helpers
- ``src/dendritic_modeling/networks/architectures/transformer``:
  transformer FFN replacement modules, config translation, parameter estimates,
  and Hugging Face patching utilities. Transformer code owns placement inside
  decoder blocks; the reusable replacement-cell contract lives in
  ``architectures/replacement``. Population-network translation helpers live in
  ``config_translation_population.py`` while ``config_translation.py`` remains
  the stable public builder surface.
- ``src/dendritic_modeling/networks/architectures/classical/pretrained.py``:
  vision-backbone splitter for ``model.vision_replacement``; resolves contiguous
  ``target_modules`` into encoder/core/decoder tensor contracts, including
  spatial CNN spans when the core is convolutional
- ``src/dendritic_modeling/training/vision_replacement.py``:
  block-output distillation loop for vision replacement cores or patched full
  vision models using cached tensors
- ``src/dendritic_modeling/training/_transformer_replacement``:
  transformer replacement training internals split by common result types, data
  sources, checkpoint loading, evaluation, joint LM losses, and training loops.
  ``training/transformer_replacement.py`` remains the public import facade.
- ``src/dendritic_modeling/training/strategies/local_learning.py``:
  public local-learning strategy. Signal construction, recorders, broadcast
  helpers, STDP rules, and modulators live in focused ``local_learning_*``
  mixins.
- ``src/dendritic_modeling/training/optimizers/factory.py``:
  registry-backed optimizer construction
- ``src/dendritic_modeling/training/loss/functions.py``:
  registry-backed loss-function construction
- ``src/dendritic_modeling/datasets/registry.py``:
  extension registry for project-local dataset loaders
- ``src/dendritic_modeling/datasets/sequence``:
  sequence dataset package split by task family. ``sequence_datasets.py``
  remains a compatibility facade.
- ``src/dendritic_modeling/analysis/registry.py``:
  analyzer metadata registry for discovery and future manager refactors
- ``src/dendritic_modeling/analysis/managers/analysis_manager.py``:
  lifecycle coordinator for analyzers
- ``src/dendritic_modeling/analysis/core/information_parts/information_helpers.py`` and
  ``information_parts/information_results.py``: pure information-analysis
  utilities for estimator math, unit conversion, metric filtering, result
  aggregation, and layer-level summaries. ``information_metrics_mixin.py``,
  ``information_hooks_mixin.py``, and ``information_signal_mixin.py`` separate
  metric computation, hook collection, and LDA/E-I signal summaries.
  ``information.py`` keeps the public ``InformationAnalyzer`` entry point.
- ``src/dendritic_modeling/networks/architectures/excitation_inhibition/dendritic/initialize.py``:
  compatibility facade for dendritic branch initialization. Expectation math,
  reactivation calibration, scaling, and named initialization methods live in
  the adjacent ``initialize_*`` modules.

Artifacts and Outputs
---------------------

A normal run can produce:

- resolved ``config.json``
- checkpoints and optimizer state
- ``performance`` summaries
- ``init_gate_stats.json`` for gate-aware models
- analyzer-specific JSON, tensors, and figures

When debugging reproducibility or regressions, the first artifacts to compare
are usually:

1. ``config.json``
2. ``init_gate_stats.json``
3. the first few epoch metrics
4. analyzer outputs tied to the affected subsystem

How To Extend Safely
--------------------

When adding model, data, optimizer, loss, or analysis behavior, use this
extension pattern:

1. add or extend a config field in ``src/dendritic_modeling/config``
2. normalize any alias behavior in ``config/model_aliases.py`` or the relevant
   config helper
3. register the component when a registry exists:
   ``register_architecture``, ``register_optimizer``, ``register_loss_function``,
   or ``register_dataset``
4. add at least one focused test near the changed subsystem
5. document the control point in the relevant doc page

That sequence keeps feedforward and recurrent behavior aligned and reduces the
chance of config-only drift.

Current Safe Extension APIs
---------------------------

These extension points let project-local components add named behavior without
shadowing built-in config strings.

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Component family
     - API
     - Builder contract
   * - Architecture
     - ``register_architecture(name, builder)``
     - ``builder(parameters, input_dim=None, suffix_input_dim=None) -> nn.Module``
   * - Optimizer
     - ``register_optimizer(name, builder)``
     - ``builder(model_parameters, optimizer_config) -> torch.optim.Optimizer``
   * - Loss function
     - ``register_loss_function(name, builder)``
     - ``builder() -> LossFunction``
   * - Dataset
     - ``register_dataset(name, builder)``
     - ``builder(task_cfg) -> (train_ds, valid_ds, test_ds)``
   * - Analyzer metadata
     - ``register_analyzer(name, ...)``
     - Records config field, class path, save directory, and manager attribute

The analyzer registry records analyzer metadata. ``AnalysisManager`` owns
execution order and analyzer-specific call signatures, so registry entries do
not change analysis output by themselves.
