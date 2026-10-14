.. _extending:

Extending DendriNet
===================

DendriNet finds its components by name in registries: core architectures,
sparse synapse layers, activations, additive integration rules, datasets,
trainer strategies, losses, optimizers, analyzers, model builders and
configuration sections. Code outside the library registers new components, and
configurations then select them by name. :ref:`extension-registries` lists the
registries of components; this page describes how a separate package provides
components as a plugin, together with the configuration sections that set them
up.

.. _plugins:

Plugins
-------

A plugin is an installed distribution that declares an entry point in the
``dendritic_modeling.plugins`` group of its ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."dendritic_modeling.plugins"]
   my_plugin = "my_package.plugin:register"

The entry point names a function without arguments that makes the plugin's
registrations:

.. code-block:: python

   # my_package/plugin.py
   from dendritic_modeling.config.extensions import register_config_section
   from dendritic_modeling.training import register_trainer_strategy

   from my_package.settings import BoundaryDistillationSettings
   from my_package.trainer import BoundaryDistillationTrainer


   def register() -> None:
       register_config_section(
           "training.strategies.boundary_distillation",
           BoundaryDistillationSettings.from_mapping,
       )
       register_trainer_strategy("boundary_distillation", BoundaryDistillationTrainer)

DendriNet loads every installed plugin once, in the order of the entry-point
names: before the first configuration loads, whenever a registry is asked for
a name it does not know, and whenever a registry lists its names (such as
``get_available_architectures()`` or ``get_registered_dataset_names()``). A
plugin that cannot be imported, or whose registration raises, stops the
program with :class:`~dendritic_modeling.plugins.PluginLoadError`; a run never
continues without a plugin it may depend on. The failure is kept for the rest
of the process, because the plugin may have registered part of its
components: every later configuration load or registry lookup raises it again,
so a caller that catches the first error cannot go on with a half-registered
plugin. Setting ``DENDRITIC_MODELING_PLUGINS=0`` turns loading off, so that
only built-in components exist. The library's test
suite runs this way unless the variable is set.

Training runs record each loaded plugin in ``runtime_environment.json``
(:ref:`run-outputs`): its entry point, distribution and version, the path of
its module and, when it is imported from a git checkout, that checkout's
commit and state.

In a notebook or a test, :func:`~dendritic_modeling.plugins.activate_plugin`
runs a registration function that is not installed and records it as a loaded
plugin.

.. _plugin-config-sections:

Configuration sections
----------------------

A plugin's components are set up in configuration sections that the plugin
registers with
:func:`~dendritic_modeling.config.extensions.register_config_section`. When a
configuration loads, the registered builder receives the section's mapping
(empty when the configuration omits the section) and returns the typed
section, which the configuration stores. Like the library's own sections, a
builder raises for unknown keys and invalid values (:ref:`strict-loader`).
Plugin sections are placed at these paths:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Written at
     - Stored at
   * - ``model.vision_replacement`` (also ``model.pretrained_replacement``)
     - ``config.model.vision_replacement``
   * - ``model.transformer_replacement``
     - ``config.model.transformer_replacement``. Without a plugin that
       registers it, this is the library's transformer replacement section.
   * - ``training.transformer_replacement``,
       ``training.vision_replacement``
     - ``config.training.transformer_replacement``,
       ``config.training.vision_replacement``
   * - ``training.main.strategies.<name>`` (or
       ``training.encoder.strategies.<name>``), registered as
       ``training.strategies.<name>``
     - ``strategies.extensions["<name>"]`` of that training stage
   * - ``analysis.<name>``
     - ``config.analysis.extensions["<name>"]``, also returned by
       ``config.analysis.section("<name>")``

Configurations constructed in code (``Config()``) hold the defaults of the
registered sections, ``builder({})``.

Without the plugin, a fixed slot such as ``model.vision_replacement`` loads
when it is empty or has ``enabled: false`` (its other keys are ignored, so the
saved configuration of a run made with the plugin still loads) and raises when
it is enabled. A named section such as ``analysis.<name>`` is an unknown key.

A run saves its configuration with ``dataclasses.asdict``, and loading the
saved file passes each plugin section back to its builder. Return a dataclass,
ideally a subclass of ``dendritic_modeling.config.base.BaseConfig``, whose
``asdict`` output the builder accepts.

.. _plugin-config-migrations:

Configuration migrations
------------------------

When a plugin renames or moves its options, configurations written for its
earlier versions still have to load, and to describe the same run. The plugin
registers a migration that rewrites the old keys into their current form:

.. code-block:: python

   from dendritic_modeling.config.extensions import register_config_migration


   def move_run_name(data):
       notes = []
       experiment = data.get("experiment", {})
       if "run_name" in experiment:
           data.setdefault("outputs", {}).setdefault(
               "run_name", experiment.pop("run_name")
           )
           notes.append("experiment.run_name -> outputs.run_name")
       return notes


   register_config_migration("my_plugin", move_run_name)

The loader runs the registered migrations, in the order of their names, on a
copy of every configuration before checking it, and logs each returned note as
a warning; standalone analysis profiles are migrated as the ``analysis``
section of a configuration. ``dendritic-upgrade-config`` applies the same
migrations, so that the rewritten file loads without warnings. A migration
changes only keys it recognises and leaves current configurations unchanged.

.. _model-builders:

Model builders
--------------

``initialize_model`` builds the encoder, core and decoder that ``model``
names and wraps them for the task. A plugin can build the three components
itself for the configurations it recognises, for example a pretrained network
whose layers are replaced by dendritic cores. It registers a model builder:

.. code-block:: python

   from dendritic_modeling.models.builders import register_model_builder

   register_model_builder(
       "pretrained_replacement",
       applies=lambda model: model.vision_replacement.enabled,
       build=build_components,
   )

``applies(model_config)`` decides whether the builder handles a
configuration, and ``build(model_config)`` returns
``(encoder, core, decoder, encoder_network)``, where ``encoder_network`` is the
module whose parameters the encoder parameter group trains (often the encoder
itself). ``initialize_model`` then wraps the components as it wraps those of
built-in models. At most one registered builder may apply to a configuration.

Trainer strategies
------------------

``register_trainer_strategy(name, trainer_class)`` adds a strategy that
``training.main.strategy: <name>`` selects (:ref:`training-strategies`).
``trainer_class`` may be a ``"module:Class"`` reference, imported when the
strategy is first used, which keeps loading the plugin light. Its
settings are the plugin section ``training.strategies.<name>``, written under
``training.main.strategies.<name>``. The trainer receives the settings that
differ from the section's defaults as the keyword argument ``<name>_config``,
a plain mapping, together with the common trainer options.

Analyzers
---------

An analyzer registered with a ``phase`` runs in the analysis order
(:doc:`analysis`) when its section ``analysis.<config_field>`` is enabled:

.. code-block:: python

   from dendritic_modeling.analysis.registry import register_analyzer
   from dendritic_modeling.config.extensions import register_config_section

   register_config_section("analysis.boundary_probe", BoundaryProbeSection.from_mapping)
   register_analyzer(
       "boundary_probe",
       config_field="boundary_probe",
       class_path="my_package.analysis.BoundaryProbeAnalyzer",
       attribute_name="boundary_probe_analyzer",
       save_subdir="boundary_probe",
       phase="model_only",
   )

The section has an ``enabled`` field, a ``training`` field when the analyzer
also runs during training (``supports_training``), and a ``params`` field that
the analyzer is built from. ``constructor_style`` selects the call:
``analyzer_class(params)`` for ``params`` (through
``params.to_analyzer_params()`` when it exists),
``analyzer_class(model=..., params=...)`` for ``model_params``, or
``analyzer_class()`` for ``none``. Each run calls the analyzer's ``analyze``
method:

- ``pre_weight`` and ``post_reactivation`` run with the test-dataset analyses,
  before the weight analyses or after the reactivation analyses. ``analyze``
  receives ``model``, ``test_dataset``, ``device``, ``save_path``, ``filename``
  and ``runtime``, and also ``training`` when the analyzer is registered with
  ``pass_training=True``.
- ``model_only`` runs with the model-only analyses. ``analyze`` receives
  ``model``, ``device``, ``save_path`` and ``filename``.

An analyzer registered without a phase is built when its section is enabled
but never run.

.. _plugin-additive-rules:

Additive integration rules
--------------------------

The built-in rules of additive branches are listed in
:ref:`integration-modes`. A plugin adds a rule with
:func:`~dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_dynamics.register_additive_rule`:

.. code-block:: python

   import torch
   from torch import nn

   from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_dynamics import (
       AdditiveRule,
       register_additive_rule,
   )


   def setup(layer, options):
       layer.drive_log_gain = nn.Parameter(
           torch.full((layer.n_branches,), float(options["start_log_gain"]))
       )


   def voltage(layer, *, excitation, inhibition, child_current,
               child_conductance, raw_voltage, inputs):
       gain = layer.drive_log_gain.exp()
       return raw_voltage / (1 + gain * torch.as_tensor(excitation).abs())


   register_additive_rule(
       "drive_normalized",
       AdditiveRule(voltage=voltage, options={"start_log_gain": -6.0}, setup=setup),
   )

A configuration selects the rule with ``additive_mode`` and sets its options
with ``additive_options``, in ``model.core.morphology`` or in the options of a
population:

.. code-block:: yaml

   model:
     core:
       morphology:
         use_shunting: false
         additive_mode: drive_normalized
         additive_options:
           start_log_gain: -4.0

- ``voltage`` returns the compartment voltage before the gate. It receives
  the currents of the built-in rules: ``excitation``, ``inhibition``,
  ``child_current`` and ``child_conductance``, and
  ``raw_voltage = excitation - inhibition + child_current``. A current that a
  layer does not have is the scalar 0.
- ``inputs`` maps ``x``, ``inhibitory_input`` and ``branch_input`` to the
  layer's inputs in the forward pass. Where voltages are computed from stored
  currents (recurrent and stateful paths, and analyses that substitute
  currents), ``inputs`` is ``None``, and a rule that needs the inputs raises.
- ``options`` declares the rule's option names and defaults. A layer stores
  its merged options as ``layer.additive_options``; unknown names raise, and
  the built-in modes take no ``additive_options``.
- ``setup(layer, options)`` runs once while a layer is constructed, right
  after its leak parameter and before its synapses. It creates the rule's
  parameters and buffers, which checkpoints save with the layer, and checks
  the layer.

Registered rules apply to additive integration only. They are rejected
together with shunting, with ``use_additive_normalization``, with typed
reversal-potential synapses and with the analytic local learning rules; their
gradients come from autograd. Analyses that use closed-form derivatives of the
integration rule, such as branch local sensitivities, support the built-in
rules only.

A rule that pools over a layer's contacts can use two methods of
:class:`~dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse.IndexedSparseLinear`:
``effective_sparse_weight()`` returns the ``[out_features, K]`` weights that
the forward projection applies, and ``project_with_weights(x, weight)``
projects ``x`` through the layer's contacts with other per-contact weights.
For example,
``layer.branch_excitation.project_with_weights(x.abs(), w.abs())``, with
``w = layer.branch_excitation.effective_sparse_weight()``, sums the magnitudes
of the excitatory drives of each branch.
