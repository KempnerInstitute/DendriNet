Component catalogue
===================

A model is an encoder, a core network and a decoder (see
:ref:`model-overview`), each selected by a name in the configuration. This
page lists the core types, the encoder and decoder types, and the registries
through which code adds new components. The other lists of names (synapse
types, activations, training strategies, datasets, analyzers and more) live on
the pages that explain them; `Other component lists`_ links to them.

Component names are resolved when the component is built, not when the
configuration loads, so a misspelled name passes the loader and fails when
the model, dataset or trainer is built. ``dendritic-train --validate-only``
builds every component without training and reports such errors.

.. _core-types:

Core types
----------

``model.core.type`` selects the core. Names are case-insensitive.
``dendritic_modeling.networks.architectures.factory.get_available_architectures()``
returns every built-in name and every registered extension.

Population networks
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 46 28

   * - ``model.core.type``
     - Core
     - Configuration
   * - ``population_network``
     - Layers of named excitatory and inhibitory populations with explicit,
       directed connections, feedforward or recurrent; each population has
       its own morphology and options. See :ref:`population-networks`.
     - ``model.core.population_network``

Structured E/I cores
~~~~~~~~~~~~~~~~~~~~

Stacks of excitation/inhibition (E/I) layers: each layer has an excitatory
population of dendritic trees and an optional inhibitory population, wired by
synapse counts per branch. They read the sections
``model.core.architecture``, ``connectivity``, ``transfer``, ``morphology``,
``sparsity``, ``reactivation``, ``blocklinear`` and ``implementation``; a
recurrent structured core also reads ``dynamics``, ``synapse_types``,
``dendritic_spikes``, ``soma_feedback`` and ``recurrent_ei``. See
:ref:`structured-ei-cores`, and :ref:`integration-modes` for the integration
rules.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - ``model.core.type``
     - Core
   * - ``einet``
     - Structured E/I core whose integration rule comes from
       ``model.core.morphology`` (``use_shunting``, default ``true``).
   * - ``dendritic_shunting``
     - Shunting (divisive) inhibition.
   * - ``dendritic_additive``
     - Raw additive (subtractive) inhibition.
   * - ``dendritic_normalized_additive``
     - Additive inhibition with voltage normalization.
   * - ``dendritic_signed``
     - Signed weights (``weight_transform: identity``) in separate excitatory
       and inhibitory synapse banks, combined additively.
   * - ``dendritic_mlp``
     - Signed weights in one synapse bank per branch that reads the
       excitatory and inhibitory inputs together, combined additively.
   * - ``flat_shunting``, ``flat_additive``, ``flat_normalized_additive``,
       ``flat_signed``, ``flat_mlp``
     - The ``dendritic_*`` type of the same suffix with a flattened tree: the
       same number of branch compartments, all attached directly to the soma.

Recurrent unified E/I cores
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Layered E/I networks with an excitatory and an optional inhibitory population
per layer, feedforward or recurrent, configured by ``model.core.unified_ei``.
See :ref:`unified-ei`.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - ``model.core.type``
     - Core
   * - ``unified_ei`` (alias ``ei_unified``)
     - Each population's options set its integration rule and morphology.
       The names ``ei_net`` and ``unified_einet`` build the same core with a
       deprecation warning.
   * - ``rnn_dendritic_shunting``, ``rnn_dendritic_additive``,
       ``rnn_dendritic_normalized_additive``
     - ``unified_ei`` with shunting, raw additive or normalized additive
       integration set on every population.
   * - ``rnn_flat_shunting``, ``rnn_flat_additive``,
       ``rnn_flat_normalized_additive``
     - As the ``rnn_dendritic_*`` types, with ``branch_factors: [1]`` on every
       population.

Spatial cores
~~~~~~~~~~~~~

Cores for image-shaped inputs ``(batch, channels, height, width)``. See
:ref:`spatial-cores`.

.. list-table::
   :header-rows: 1
   :widths: 30 44 26

   * - ``model.core.type``
     - Core
     - Configuration
   * - ``hierarchical_dendritic_tensor_map`` (aliases
       ``dendritic_tensor_map``, ``tensor_map``)
     - Flattens the input and applies a structured E/I core to it, without
       spatial locality.
     - Structured E/I sections
   * - ``hierarchical_dendritic_conv`` (alias ``spatial_dendritic_conv``)
     - Applies one shared structured E/I core to every local patch and
       returns a ``(batch, channels, height, width)`` map.
     - Structured E/I sections and ``model.core.spatial``
   * - ``dendritic_alexnet``
     - An AlexNet-shaped network of dendritic convolution stages and a
       dendritic classifier.
     - ``model.core.vision`` and the structured E/I sections of its cells

Baselines and controls
~~~~~~~~~~~~~~~~~~~~~~

Conventional networks for comparisons, described in :ref:`baseline-models`.

.. list-table::
   :header-rows: 1
   :widths: 34 40 26

   * - ``model.core.type``
     - Core
     - Configuration
   * - ``affine_mlp``
     - Dense, fixed-sparse or learned Top-K multilayer perceptron.
     - ``model.core.affine_mlp``
   * - ``point_mlp``, ``total_param_mlp``, ``active_param_mlp``
     - Point-neuron MLPs matched to the parameter count of a dendritic
       reference.
     - Structured E/I sections or ``model.core.population_network``
   * - ``ss_mlp``, ``ss_mlp_flat``
     - Sparse structured MLPs with the branch structure of a structured E/I
       reference.
     - Structured E/I sections
   * - ``sparse_active_matched_point``, ``direct_active_matched_point``
     - One-layer point controls matched to the effective parameter count of a
       structured E/I reference.
     - Structured E/I sections
   * - ``sparse_active_matched_point_conv`` (alias
       ``spatial_sparse_active_matched_point``)
     - The sparse matched control applied to local image patches.
     - Structured E/I sections and ``model.core.spatial``
   * - ``gru``, ``lstm``, ``vanilla_rnn`` (alias ``rnn``)
     - PyTorch recurrent networks.
     - ``model.core.baseline_rnn``
   * - ``heterogeneous_leak_ctrnn`` (alias ``heterogeneous_ctrnn``)
     - Sparse point-neuron E/I network with leaky channels of several time
       constants.
     - ``model.core.heterogeneous_leak_ctrnn``
   * - ``legendre_memory``
     - Fixed Legendre delay-network memory with sparse learned input and
       readout.
     - ``model.core.legendre_memory``

Pass-through
~~~~~~~~~~~~

``identity`` (alias ``none``) passes the encoder output to the decoder
unchanged; it reads no section.

Names that are not core types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The architecture factory also builds encoder, decoder and Python building
blocks: ``mlp``, ``effectivematchedparammlp``, ``alexnet``, ``cnn``,
``cnn_down``, ``cnn_downsample``, ``cnndownsample``, ``cnn_up``,
``cnn_upsample``, ``cnnupsample``, ``mlpautoencoder``,
``mlpvariationalautoencoder``, ``cnnautoencoder``,
``cnnvariationalautoencoder``, ``pathway_router`` and ``router``. They take
flat parameters that ``model.core`` cannot express, so ``model.core.type`` set
to one of them raises an error that names the alternatives (``affine_mlp`` for
a plain MLP core, ``point_mlp`` for a matched control).

.. _encoder-decoder-types:

Encoder and decoder types
-------------------------

``model.encoder.type`` and ``model.decoder.type`` select the encoder and the
decoder; their options go under ``model.encoder.params`` and
``model.decoder.params``. Names are case-insensitive. The loader rejects an
option that the selected type does not read, unless it holds its default
value.

Encoders:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``model.encoder.type``
     - Encoder and the options it reads
   * - ``identity`` (alias ``none``)
     - Passes the input through. ``input_dim`` and ``input_shape`` are set
       from the dataset by ``dendritic-train``.
   * - ``pathway_router`` (alias ``router``)
     - Routes the input features into separate pathways and concatenates
       their outputs. ``router_mode`` (``fixed`` or ``learned``),
       ``pathway_groups`` (input feature indices of each pathway; by default
       the features are split evenly into ``n_pathways`` groups),
       ``shared_indices`` (features every pathway receives), ``pathway_dim``
       or ``pathway_dims`` (output width of each pathway; by default its
       number of features), ``router_activation`` (an activation type,
       default ``none``), and for ``learned`` routing
       ``learned_router_temperature`` and ``learned_router_init_scale``.
   * - ``mlpautoencoder``, ``mlpvariationalautoencoder``
     - The encoder half of a multilayer-perceptron (variational) autoencoder:
       ``encoder_hidden_dims``, ``latent_dim``, ``decoder_hidden_dims``,
       ``activation``.
   * - ``cnnautoencoder``, ``cnnvariationalautoencoder``
     - The encoder half of a convolutional (variational) autoencoder:
       ``encoder_hidden_dims``, ``latent_shape``, ``decoder_hidden_dims``,
       ``activation``.

Decoders:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``model.decoder.type``
     - Decoder and the options it reads
   * - ``MLP``
     - A multilayer perceptron from the core output to ``output_dim``
       (default 10) outputs: ``hidden_dims`` (default ``[16]``; ``[]`` gives
       a single linear readout), ``activation`` (an activation type, default
       ``relu``), ``input_normalization`` (``none`` or ``layer_norm``),
       ``input_norm_eps`` (default ``1e-5``) and ``output_init`` (``default``,
       ``zero``, ``xavier``, ``xavier_zero_bias`` or ``xavier_uniform``).
   * - ``identity`` (alias ``none``)
     - Passes the core output through. A written ``output_dim`` must equal the
       core output size.

The decoder's ``input_dim`` is always set from the core output size. Other
names, such as ``linear`` or the ``CNN`` building blocks, load but fail when
the model is built.

.. _encoder-pretraining:

Pretraining autoencoder encoders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An autoencoder encoder is trained before the main phase. With
``training.encoder_pretraining.epochs`` positive, ``dendritic-train`` first
trains the whole autoencoder to reconstruct the training inputs and keeps the
state with the lowest validation loss; its encoder then feeds the core. A
trained autoencoder can instead come from ``model.initial_checkpoint`` (with
``epochs: 0``). Without either, the run raises.

.. code-block:: yaml

   model:
     encoder:
       type: mlpautoencoder
       params: {encoder_hidden_dims: [256], latent_dim: 32, decoder_hidden_dims: [256]}
   training:
     encoder_pretraining:
       epochs: 50                # 0 (default): no pretraining
       batch_size: 256
       loss_function: mse        # mse | log_mse | bce | vae_elbo (variational only)
       optimizer: adam
       learning_rate: 0.001
       weight_decay: 0.0
       early_stopping: true
       patience: 10
       grad_clip_value: null
       freeze_encoder: true      # false: the encoder trains with the model

With ``freeze_encoder`` (the default) the encoder receives no gradients in
the main phase and stays in evaluation mode, so a variational encoder returns
its latent mean; the run directory holds the pretraining losses in
``encoder_pretraining/summary.json``, and the model checkpoints contain the
encoder. Pretraining runs in single-process training: start a DDP run from the
checkpoint of a single-process run, and use another encoder with FSDP. A
positive ``epochs`` with an encoder that is not an autoencoder raises.
``dendritic-train`` and the FSDP entry point check these requests for the
built-in encoder types before they create the run directory, so a rejected
request leaves no files behind; an encoder that a plugin builds is checked once
the model is built.

.. _extension-registries:

Extension registries
--------------------

Code outside the library adds named components by registering them. A
registered name is then accepted wherever the configuration selects that kind
of component.

.. list-table::
   :header-rows: 1
   :widths: 30 26 22 22

   * - Function
     - Registers
     - Selected by
     - Name rules
   * - ``register_architecture(name, builder, *, aliases=(),
       description="", allow_override=False)`` in
       ``dendritic_modeling.networks.architectures``
     - A core builder, called as ``builder(parameters, input_dim=...,
       suffix_input_dim=...)``. It returns an ``nn.Module`` with an
       ``output_dim`` attribute; ``parameters`` is the typed ``model.core``
       section.
     - ``model.core.type``
     - Built-in names are reserved; an existing extension name needs
       ``allow_override=True``.
   * - ``register_sparse_layer(name, layer_class, *,
       allow_override=False)`` in
       ``dendritic_modeling.networks.architectures.excitation_inhibition.synapse``
     - A sparse synapse class, built as ``layer_class(in_features=...,
       out_features=..., K=..., **options)``.
     - ``topk_type`` (population options) and ``model.core.sparsity.type``
     - Existing names, built-in or registered, need ``allow_override=True``.
   * - ``register_activation_builder(act_type, builder, *,
       allow_override=False)`` in ``dendritic_modeling.networks.activations``
     - An activation or reactivation-gate builder.
     - ``reactivation_type``, ``model.core.reactivation.type``, decoder
       ``activation``
     - Existing names need ``allow_override=True``.
   * - ``register_trainer_strategy(name, trainer_class, *,
       fixed_kwargs=None, allow_override=False)`` in
       ``dendritic_modeling.training``
     - A ``Trainer`` subclass, constructed with the trainer options plus
       ``fixed_kwargs``.
     - ``training.main.strategy``
     - Existing names, built-in or registered, need ``allow_override=True``.
       Settings go in a plugin section ``training.strategies.<name>``
       (:doc:`extending`).
   * - ``register_optimizer(name, builder, *, aliases=(),
       allow_override=False)`` in ``dendritic_modeling.training.optimizers``
     - An optimizer builder, called as ``builder(model_parameters,
       optimizer_config)``.
     - ``training.main.optimizer.name``
     - Existing names need ``allow_override=True``.
   * - ``register_loss_function(name, builder, *, aliases=(),
       allow_override=False)`` in ``dendritic_modeling.training.loss``
     - A loss builder, a callable without arguments that returns a
       ``LossFunction``.
     - ``training.main.common.loss_function``
     - Existing names need ``allow_override=True``.
   * - ``register_dataset(name, builder, *, aliases=(), description="",
       sequence=False, required_strategy=None, processing=None,
       allow_override=False)`` in ``dendritic_modeling.datasets``
     - A dataset builder, called with the task configuration; it returns
       ``(train, valid, test)`` datasets and defines its own splits.
     - ``data.dataset_name``
     - Registered names are looked up before the built-in loaders, so
       registering a built-in name such as ``mnist`` replaces that loader.
       Existing registered names need ``allow_override=True``.
   * - ``register_sequence_dataset_builder(name, builder, *,
       allow_override=False)`` in ``dendritic_modeling.datasets.sequence``
     - A sequence-task builder, called as ``builder(dataset_name, data_path,
       parameters)``.
     - ``data.dataset_name``
     - Existing names need ``allow_override=True``.
   * - ``register_analyzer(name, *, config_field, class_path,
       attribute_name, save_subdir, ...)`` in ``dendritic_modeling.analysis``
     - An analyzer: configuration field, class, output directory and the
       phase of the analysis order it runs in.
     - ``analysis.<config_field>.enabled``, a plugin section
       (:doc:`extending`)
     - Existing names need ``allow_override=True``. Without a ``phase`` the
       analyzer is built but never run.
   * - ``InformationEstimatorFactory.register_estimator(method,
       estimator_class)`` in ``dendritic_modeling.utils.information.estimators``
     - The estimator class of one information estimation method.
     - ``analysis.information.params.estimator.method``
     - Only the members of ``EstimationMethod``; registering replaces the
       built-in estimator of that method.

Registrations run in the process that uses them. A package makes them
available to every DendriNet command and script by declaring them as a plugin,
which the library loads before the first configuration and whenever a registry
meets a name it does not know; plugins can also register configuration
sections, model builders and additive integration rules (:doc:`extending`).

Other component lists
---------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Components
     - Where they are listed
   * - Sparse synapse types (``topk_type``, ``model.core.sparsity.type``)
     - :ref:`synapse-variants`
   * - Weight transforms (``weight_transform``)
     - :ref:`weight-transforms`
   * - Synapse initialization (``topk_init_method``, ``dbl_init_method``)
     - :ref:`synapse-initialization`
   * - Activations and reactivation gates (``reactivation_type``)
     - :ref:`activation-types`
   * - Indexed projection backends (``projection_backend``)
     - :ref:`indexed-backends`
   * - Population options, connections and time constants
     - :ref:`population-options`, :ref:`population-connections`,
       :ref:`temporal-constants`
   * - Receptor synapse types, spiking soma, dendritic spikes and soma
       feedback
     - :doc:`temporal_dynamics`
   * - Training strategies (``training.main.strategy``)
     - :ref:`training-strategies`
   * - Loss functions (``training.main.common.loss_function``)
     - :ref:`loss-functions`
   * - Optimizers (``training.main.optimizer.name``)
     - :doc:`training_strategies`
   * - Datasets (``data.dataset_name``)
     - :ref:`dataset-catalogue`
   * - Analyzers (``analysis.<name>``)
     - :ref:`analyzer-catalogue`
   * - Baselines and matched controls
     - :ref:`baseline-models`
