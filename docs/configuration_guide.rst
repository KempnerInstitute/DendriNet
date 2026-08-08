Configuration Guide
===================

The primary configuration reference is ``configs/config_exp.yaml``.  That file
is the runnable single-GPU template for the ``population_network``
schema. This page summarises the structure and highlights the most important
settings; focused feature examples live under ``configs/examples``.

Quick Start
-----------

.. code-block:: bash

   # Standard training
   python src/dendritic_modeling/scripts/training/train_experiments.py configs/config_exp.yaml

   # With overrides
   python src/dendritic_modeling/scripts/training/train_experiments.py \
     configs/config_exp.yaml --output_dir results/my_run --run_name test1

Templates:

- ``configs/config_exp.yaml`` -- single-GPU population-network reference
- ``configs/config_exp_fsdp.yaml`` -- FSDP multi-GPU training
- ``configs/sweep_config.yaml`` -- generic parameter-sweep template

Top-Level Structure
-------------------

.. code-block:: yaml

   experiment:   # seed, profiling, checkpointing
   data:         # dataset, processing, dataset-specific params
   model:        # encoder, core network, decoder, vision/transformer replacement
   training:     # encoder + main training, strategies, regularization, pruning, AMP
   analysis:     # 20 analysis modules
   wandb:        # Weights & Biases logging
   outputs:      # run_name, results_dir
   distributed:  # mode (none/ddp/fsdp), FSDP tuning knobs

Core Network Types
------------------

Set ``model.core.type`` to one of:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Type string
     - Description
   * - ``dendritic_shunting``
     - Shunting (divisive) inhibition
   * - ``dendritic_additive``
     - Raw additive (subtractive) inhibition, ``V = E - I``
   * - ``dendritic_normalized_additive``
     - Additive inhibition with explicit voltage normalization
   * - ``dendritic_mlp``
     - Dendritic morphology, no E/I sign constraint
   * - ``point_mlp``
     - Point-neuron MLP baseline (matched params)
   * - ``einet``
     - Legacy generic E-I type; uses ``morphology.use_shunting``
   * - ``flat_shunting``
     - Single-branch shunting variant
   * - ``flat_additive``
     - Single-branch raw additive variant
   * - ``flat_normalized_additive``
     - Single-branch normalized additive variant
   * - ``ei_unified``
     - Unified E-I, supports feedforward and recurrent
   * - ``rnn_dendritic_shunting``, ``rnn_dendritic_additive``, ``rnn_dendritic_normalized_additive``
     - Unified recurrent E-I aliases with shunting/raw-additive/normalized-additive defaults applied
   * - ``population_network``
     - Population-network notation for feedforward stacks and recurrent microcircuits
       with named E/I populations, per-population morphology, explicit
       directed connections, synapse types, spiking soma, and feedback
   * - ``gru``, ``lstm``
     - Baseline RNNs
   * - ``vanilla_rnn``
     - Plain point-RNN baseline
   * - ``mlp``
     - MLP baseline

See :doc:`network_architectures` for architecture details. For recurrent
``ei_unified`` models, the biology-facing configuration supports
``allow_self_recurrence: false`` inside each population block to disable
same-population E->E and I->I self-feedback while keeping cross-population
recurrence intact. Structured feedforward EINet configs and recurrent
PopulationConfig blocks share the same positive-weight default
(``weight_transform: softplus``). Use ``weight_transform: exp`` explicitly
when you want the sharper positive-conductance parameterization.

Shunting/additive precedence is path-specific:
``dendritic_shunting`` / ``dendritic_additive`` /
``dendritic_normalized_additive`` / ``flat_*`` types define the effective
feedforward mode, and contradictory
``model.core.morphology.use_shunting`` or ``use_additive_normalization`` values
are warned about and overwritten. The generic ``einet`` type is the explicit
escape hatch when you want morphology flags to choose the mode. For recurrent
``rnn_*`` aliases, the alias is applied to every excitatory and inhibitory
``PopulationConfig`` block; ``rnn_flat_*`` also forces ``branch_factors: [1]``.
Plain ``ei_unified`` and ``population_network`` keep population-level fields as
their source of truth. Transformer replacement builders follow the same
``model.core.type`` alias when one is present, otherwise they read
``model.core.morphology.use_shunting``.

Fair Additive Comparison Controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``DendriticBranchLayer`` and population configs expose an explicit
``additive_mode`` for mechanism-controlled comparisons. The default is
``"raw"``, so existing configurations and checkpoints retain their historical
behavior. ``use_shunting: true`` takes precedence; ``additive_mode`` is only
evaluated when ``use_shunting: false``.

For the implemented branch decomposition, let ``E`` be the summed local
feedforward and recurrent excitatory pathway output, ``I`` the corresponding
inhibitory output, ``C`` the child-branch current, and ``G`` the child coupling
conductance. Then ``N = E + C`` and ``T = E + I + G``. The available modes are:

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Configuration
     - Pre-reactivation voltage
     - Intended comparison
   * - ``use_shunting: true``
     - ``N / (1 + T + epsilon)``
     - Implemented conductance-based shunting circuit.
   * - ``additive_mode: raw``
     - ``E - I + C``
     - Historical subtractive additive baseline.
   * - ``additive_mode: conductance_normalized``
     - ``(E - I + C) / (1 + T + epsilon)``
     - Additive-sign numerator with the same total-conductance normalization.
   * - ``additive_mode: tangent_matched``
     - ``V0 + ((N-N0) - V0*(T-T0)) / D0``, where
       ``D0=1+T0+epsilon`` and ``V0=N0/D0``
     - Affine first-order control matching the shunting value and Jacobian at a
       frozen operating point.

The tangent anchor is never estimated or updated during ``forward``. Supply a
fixed scalar anchor with ``additive_tangent_n0`` and
``additive_tangent_t0``, or calibrate explicitly from a shunting reference:

.. code-block:: python

   reference_layer.set_branch_diagnostics(True)
   reference_layer(calibration_input, inhibitory_input, child_input)
   diagnostics = reference_layer.get_branch_diagnostics()

   tangent_layer.calibrate_additive_operating_point(diagnostics)
   # The resulting (N0, T0) is now frozen and is saved in state_dict().

For multi-level models, calibrate and copy the anchor for each corresponding
``DendriticBranchLayer``. Use held-out calibration data or a predetermined
analytical anchor, record the resulting values with the run metadata, and do
not recalibrate one condition during training unless that adaptation is itself
the experimental manipulation.

``use_additive_normalization`` has a different legacy meaning: it z-scores the
raw additive voltage across output features within each sample. It is not a
conductance normalization. To prevent mislabeled controls, it can only be used
with ``additive_mode: raw``; the two new modes reject it. Typed
reversal-potential synapses likewise currently support only ``raw`` when
``use_shunting: false``, because their signed current equation does not share
the legacy ``N/T`` decomposition.

Calling ``set_branch_diagnostics(True)`` exposes detached ``E``, ``I``, ``C``,
``G``, ``N``, ``T``, and pre-reactivation ``V`` tensors through
``get_branch_diagnostics()``. Its ``realized_K`` mapping reports the exact
active synapse count per output for ``E``, ``I``, ``E_rec``, and ``I_rec`` on
the most recent forward. This is preferable to reporting only requested K when
connection masks or dynamic sparsity can reduce the realized count.

For additive modes without a validation sweep, prefer
``reactivation.init_policy: "occupancy_quantile"`` as a safe operating-point
fallback.
When an additive YAML config omits ``reactivation.init_policy``, typed config
loading chooses ``occupancy_quantile`` by default; explicit policies such as
``analytical`` or ``fixed`` are preserved for reproduction. The deprecated
spelling ``quantile`` is accepted as an alias for ``occupancy_quantile``.
When you use a data-driven gate init policy
(``occupancy_quantile``, ``analytical_slope_occupancy_center``,
``occupancy_slope_analytical_center``, or ``empirical``), each run
writes both the post-build ``init_gate_stats.json`` and the executed
``post_calibration_gate_stats.json`` into its run directory, together with
``reactivation_calibration.json``. Use these artifacts to verify convergence
and the actual gate state before trusting the run. For an attainable-performance
comparison, the initializer and learning rate may be selected separately for
each architecture, but the validation information and search budget must be
equal. A same-named initializer is a robustness control, not necessarily an
effective-operating-point match when the mechanisms have different voltage
scales.

The two center/slope component policies require no fitted mixing coefficient.
``analytical_slope_occupancy_center`` keeps the mechanism-aware analytical
slope and recenters the gate on the observed voltage quantiles.
``occupancy_slope_analytical_center`` uses the quantile-derived slope while
retaining the analytical expected-voltage center. Together with the analytical
and occupancy endpoints, these policies form a 2-by-2 diagnostic that separates
center mismatch from scale mismatch.

Note that ``occupancy_quantile`` already combines the two information sources:
the code analytically initializes synaptic weights from morphology and contact
statistics before it measures branch voltages on unlabeled training inputs.
The empirical step calibrates only the gate operating point; it does not refit
the weights or use validation/test labels.

Config loading validates structural model contracts before model creation.
For example, invalid ``input_mode`` values, impossible per-layer synapse-list
lengths, duplicate population names, unknown connection endpoints, delayed
population edges without ``recurrent: true``, and same-step population cycles
raise a clear ``ConfigValidationError`` from ``load_config``. Training-stability
and scientific plausibility checks remain in the training validators so existing
experiments are not rejected for warning-level choices.

Population-Network Config Choices
---------------------------------

Use these defaults for population-network configs. They keep the code path
modular and make feedforward, recurrent, additive, shunting, local-learning,
and sweep runs easier to compare.

.. list-table::
   :header-rows: 1
   :widths: 28 34 38

   * - Decision
     - Preferred setting
     - Rationale
   * - Architecture notation
     - ``model.core.type: population_network``
     - One schema covers original E/I layers, heterogeneous populations,
       multilayer feedforward routing, delayed recurrent edges, spiking, typed
       synapses, and feedback.
   * - Shunting vs additive
     - Choose the type or per-population ``use_shunting`` directly; use
       ``dendritic_normalized_additive`` only when normalization is the
       intended model
     - Raw additive and normalized additive are scientifically different
       models, so normalization should be explicit.
   * - Gate initialization
     - ``reactivation.init_policy: occupancy_quantile``
     - Data-driven gate placement is the fairest default across shunting,
       raw additive, normalized additive, shallow, and deep networks.
   * - Positive conductances
     - ``weight_transform: softplus``
     - Keeps conductances positive with smoother gradients than ``exp``.
   * - Developmental sparsity
     - ``topk_type: dense_to_sparse``
     - Lets all weights receive gradient early, then prunes to the target
       sparse budget on a controlled schedule.
   * - Multilayer routing
     - Qualified sources such as ``layer0.e0`` and ``layer0.i0``
     - Avoids hidden broadcasting assumptions and supports multiple readout
       streams from earlier layers.
   * - Recurrence
     - ``timing: delayed`` on a recurrent layer
     - Delayed edges read previous-timestep state and avoid same-step
       algebraic loops.
   * - Sweeps
     - Dot paths under ``model.core.population_network``
     - The same field names work for training, sweeps, and analysis metadata.

Training Strategies
-------------------

Set ``training.main.strategy`` to one of:

``standard``, ``fa``, ``dfa``, ``shunting_fa``, ``shunting_dfa``,
``local_ca``, ``freeze_layers``, ``freeze_branches``, ``freeze_branch_kl``,
``voltage_stabilization``, ``homeostatic_control``,
``multi_stage``, ``recurrent``, ``vision_distillation``

Strategy-specific configuration usually lives under
``training.main.strategies.<name>``. Local credit assignment uses the direct
path ``training.main.learning_strategy_config`` for ``3f``/``4f``/``5f`` local
rules and the separate ``stdp`` trace rule. See :doc:`training_strategies` for
details.

DataLoader options live with the shared training fields and are honored by
both the standard trainer and FSDP script:

.. code-block:: yaml

   training:
     main:
       common:
         batch_size: 512
         num_workers: null          # null = trainer default / auto
         pin_memory: null           # null = auto by device
         persistent_workers: null   # requires num_workers > 0
         prefetch_factor: null      # requires num_workers > 0
         in_order: null             # false can reduce image-loader stalls
         profile_dataloader: false  # log epoch data-wait timing

Loss Functions
--------------

Set ``training.main.common.loss_function`` (or ``training.encoder.common.loss_function``) to:

``mse``, ``log_mse``, ``ce``, ``bce``, ``cat_nll``,
``vae_elbo``, ``neg_cos_sim``, ``log_adj_neg_cos_sim``

Datasets
--------

Set ``data.dataset_name`` to one of:

- **Standard**: ``mnist``, ``fashion_mnist``, ``cifar10``, ``imagenet``,
  ``mnist_modulo10``, ``cifar10_modulo10``,
  ``double_cifar10_contextual_mod10``, ``double_mnist_contextual_mod10``
- **Synthetic**: ``correlated_gaussian``, ``nonlinear_interaction``,
  ``linear_separable``, ``nonlinear_separable``, ``block_correlated``,
  ``balanced_information``, ``contextual_stream_gain_shift``,
  ``poisson_generator``, ``multixor``, ``orientation_bars``, ``orthonet``
- **Sequence**: ``sequential_mnist``, ``copy_task``, ``adding_problem``,
  ``variable_delay_dms``, ``multi_frequency``, ``hierarchical_temporal``,
  ``context_dependent_decision``, ``context_dependent_decision_hard``,
  ``context_dependent_decision_positive``,
  ``context_dependent_decision_hard_positive``,
  ``switching_context_decision``, ``switching_context_decision_positive``,
  ``timescale_generalization``, ``timescale_generalization_positive``,
  ``odr_generalization``, ``odr_generalization_positive``,
  ``romo_delay_comparison``, ``romo_delay_comparison_positive``,
  ``gain_modulated_contextual_comparison``,
  ``gain_modulated_contextual_comparison_positive``, ``lorenz``,
  ``lorenz_sequence``, ``switching_lds``, ``multi_sine_forecast``,
  ``stringer_v1``

Dataset-specific parameters go under ``data.dataset_params.<name>``.
See :doc:`datasets` for details.

For ImageNet with a pretrained torchvision backbone, use the pretrained
transform preset and point the run at a real ImageNet root; missing roots fail
loudly instead of falling back to synthetic data:

.. code-block:: yaml

   data:
     dataset_name: "imagenet"
     base_dir: "/path/to/imagenet"
     processing:
       flatten: false
       normalize: true
     dataset_params:
       imagenet:
         transform_preset: "pretrained"
         loader_backend: "auto"
         allow_flatten_materialize: false

Set ``loader_backend: "imagefolder"`` when the configured root already has
``train/<class>`` and ``val/<class>`` directories and you want to bypass the
torchvision ImageNet metadata loader.  Keep ``"auto"`` for compatibility with
official torchvision ImageNet roots.

For AlexNet or other pretrained-backbone ImageNet runs, prefer node-local
storage when the cluster provides it.  Set ``IMAGENET_LOCAL_DIR`` to the staged
copy and use ``loader_backend: "imagefolder"`` when that copy is organized as
class directories.  Keep ``processing.flatten: false`` for these runs; the
flattened ImageNet path is memory-heavy and requires the explicit
``allow_flatten_materialize: true`` opt-in.

ImageNet/ImageFolder datasets use the shared dataloader auto-tuning path when
``num_workers``, ``pin_memory``, ``persistent_workers``, and
``prefetch_factor`` are left as ``null``.  The auto path keeps multi-worker,
pinned, persistent prefetching for file-backed image datasets, uses
non-blocking CUDA copies when pinned batches are available, and avoids worker
overhead for tensor-backed datasets. Set ``in_order: false`` for ImageNet jobs
where worker stragglers cause batch-delivery stalls. Set
``profile_dataloader: true`` to log per-epoch data-wait time and confirm whether
training is input-pipeline bound. For fixed cluster jobs, set explicit
per-process values under the active trainer config, for example
``num_workers: 8``, ``pin_memory: true``, ``persistent_workers: true``, and
``prefetch_factor: 4``.

For shunting E-I models with positive weight transforms, keep the tensors that
enter the first shunting layer non-negative. If preprocessing uses
``normalize: true``, either disable normalization for that run or set
``model.core.transfer.output_activation`` to a nonnegative activation such as
``"relu"`` so the transfer-layer outputs remain non-negative.

Analysis Modules
----------------

Each module lives under ``analysis.<module_name>`` with ``enabled`` and
``training`` flags:

.. code-block:: yaml

   analysis:
     runtime:
       training:
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
     performance:
       enabled: true
       training: true    # run every epoch
       training_splits: ["valid"]
       final_splits: ["train", "valid", "test"]
     information:
       enabled: true
       training: false   # run after training only

``analysis.runtime`` is the shared dataset-iteration policy for all analysis
modules. Use ``mode: "stream"`` for large datasets that should not be fully
materialized on the device, and use ``max_samples`` for expensive analyses
that still accumulate tensors on CPU. If you want exact small-dataset
single-GPU behavior for hook-based analyses, set ``mode: "materialize"``:
the runtime will use one full batch again when the dataset supports slicing.
Analyzers may temporarily move the model to the requested analysis device, but
the original device and train/eval mode are restored automatically after the
analysis finishes.
For most runs, keep the default ``mode: "auto"``: it materializes small
datasets and switches to streaming when the dataset estimate is large.
For ImageNet-scale streamed analysis, set ``in_order: false`` to let DataLoader
yield ready worker batches without waiting for slower earlier batches. Leave it
``null`` for PyTorch's default ordering behavior.
If you omit ``analysis.runtime`` entirely, non-distributed runs default to
``materialize``, while DDP/FSDP runs keep the shared ``auto`` default.
The ``pin_memory`` flag is also interpreted per device: CPU analysis disables
it automatically even if the shared runtime block leaves it set to ``true``.

Preferred short names are ``performance``, ``information``, ``ablation``,
``path_matched_intervention``, ``local_inhibitory_population_intervention``,
``path_matched_information``, and
``correlation``. The longer ``*_analysis`` names are accepted as aliases.

Available modules: ``performance``, ``information``, ``correlation``,
``ablation``, ``path_matched_intervention``,
``local_inhibitory_population_intervention``, ``synapse_turnover``,
``path_matched_information``,
``noise_perturbation``, ``weight``,
``single_layer_contribution``, ``synaptic_activation``, ``branch_activation``,
``compartment_statistics``, ``compartment_snr``, ``multiplicative_gain``,
``receptive_field``, ``adversarial_robustness``, ``routing``,
``reactivation_dynamics``, ``inhibitory_specialization``,
``jacobian_spectrum``, ``dendritic_timetraces``, ``local_rule_components``,
``spike_train``.

Use ``path_matched_intervention`` when comparing causal sensitivity across
dendritic depth. It prespecifies one complete soma-to-distal path per soma and
uses nested path prefixes, so every depth replaces the same number of
compartment coordinates. It can replace excitation, inhibition, or joint E/I
currents with zero, a training-split mean, or a dataset-wide shuffled test
donor. Joint E/I shuffles share the donor permutation and therefore preserve
the donor pair's sample-wise E/I covariance. Technical coordinate draws are
reported separately from independently trained model seeds. For multi-layer
models, set ``module_name_prefix`` to select exactly one dendritic tree.
The optional ``post_gate_output`` target instead replaces the compartment
output after its configured transfer/reactivation and before propagation to
the next more-proximal layer. Depths are soma-relative and non-somatic: 1 is
proximal, larger values are more distal, and 0 is excluded.

Use ``local_inhibitory_population_intervention`` for complete local-I
population dependence across the positions of a feedforward population stack.
It requires one inhibitory population per layer and exactly one same-step
I-to-E route, then compares global-donor, within-class-donor, and zero
replacement controls. This is a network-position estimand and must not be
described as dendritic depth or lateral/recurrent inhibition.

Use ``path_matched_information`` for matched-width information comparisons
along dendritic depth. It shares the intervention analyzer's prespecified
nested paths, but receives only validation data and estimates joint
continuous--discrete class information for E, I, joint E+I, child current, and
post-reactivation output. Its path draws are retained as technical support
sensitivity records and must be averaged within checkpoint before inference
across independently trained seeds.

See :doc:`analysis` for per-module parameter details. For a copy/paste YAML
reference with every preferred analysis key, default value, and option hint,
use ``configs/templates/analysis_full_template.yaml``.

Compatibility:

- Configs using ``performance_analysis`` instead of ``performance`` still load.
- Performance-only ``training_runtime`` / ``final_runtime`` blocks still load
  and are migrated into the shared ``analysis.runtime`` structure.

Vision Backbone Replacement
---------------------------

Replace one or more layers of a pretrained torchvision backbone (e.g. AlexNet,
VGG) with an EINet core.  The backbone is loaded once and split into three
segments:

- **prefix** (encoder): all layers before the replacement range
- **core**: the dendritic EI network that replaces the removed layers
- **suffix** (decoder): all layers after the replacement range

When ``model.vision_replacement.enabled`` is ``true``, the standard
``model.encoder`` and ``model.decoder`` sections are ignored.  Older configs
using ``model.pretrained_replacement`` still load as a compatibility alias.

.. code-block:: yaml

   model:
     vision_replacement:
       enabled: true
       backbone: "alexnet"                    # torchvision model name
       weights: "IMAGENET1K_V1"               # pretrained weights (null = random init)
       core_source: "configured"               # or "pretrained_span" control
       input_shape: [3, 224, 224]

       # Backbone modules to replace. They must form one contiguous span.
       # Use list_backbone_layers("alexnet") to see all available names.
       target_modules: ["classifier.4", "classifier.5"]  # fc7 Linear + ReLU

       # Drop layers entirely (e.g. Dropout that is not useful with EI nets).
       omit_layers: ["classifier.0", "classifier.3"]

       # Freeze/train policy for encoder and decoder segments.
       trainability:
         encoder:
           mode: "frozen"                     # freeze conv backbone + fc6
         decoder:
           mode: "trainable"                  # train final classifier

     core:
       type: "einet"
       architecture:
         excitatory_layer_sizes: [4096]       # must match suffix input dim
         # ... rest of EINet config ...

Set ``core_source: pretrained_span`` to retain the exact modules and loaded
weights named by ``target_modules`` as the core. The prefix and suffix follow
the same split and trainability policy, while the retained span remains
trainable. This is useful as a training-matched dense reference; it is not a
parameter-matched sparse control. ``configured`` remains the default and uses
``model.core`` as the replacement.

Alternative pretrained-replacement cores:

- ``hierarchical_dendritic_tensor_map``
  Flattens the prefix tensor, runs a newly initialized EINet-style hierarchical
  dendritic core, then projects to the suffix input dimension when needed. Use
  this as the baseline/control for late classifier-adjacent replacements where
  the suffix expects flat input. The pretrained part is the surrounding
  backbone split, not the tensor-map core itself.
- ``hierarchical_dendritic_conv``
  Uses local spatial patches (``core.spatial.kernel_size/stride/padding``) and
  a shared hierarchical dendritic core to produce a spatial feature map for the
  suffix. Use this for convolutional suffixes that still expect spatial input.

The convolutional output contract is channel aligned. At each output position,
one excitatory soma produces one output channel; the same soma/channel mapping
and the same E/I parameters are shared over all positions. Only the excitatory
soma map is propagated in the standard ``(B,C,H,W)`` tensor. The inhibitory
population is local to each output position and modulates the excitatory tree
there, but is not exposed as a second feature stream. Consequently, two
``hierarchical_dendritic_conv`` layers can be composed with ordinary pooling or
normalization layers without losing channel or spatial identity. The method
``spatial_output_contract()`` exposes this mapping for audits and experiment
manifests.

Current spatial-replacement scope:

- ``hierarchical_dendritic_tensor_map`` is intended for flat suffixes or small
  spatial maps that can be flattened safely.
- ``hierarchical_dendritic_conv`` preserves the prefix/suffix spatial contract,
  but its local geometry is set by ``core.spatial.*``; it does not infer the
  effective receptive field of an arbitrary multi-layer CNN block automatically.
- Sequential backbones such as AlexNet and VGG are the primary supported cases.

Supervised full-model training uses the standard trainer.  Set the replacement
trainability fields to decide whether the prefix/suffix are frozen or trainable;
the replacement core is trainable by default.  Learning rates for the prefix
and suffix are controlled in the training config::

   training.main.common.param_groups.encoder_lr    # prefix LR (0.0 = frozen)
   training.main.common.param_groups.decoder_lr    # suffix LR

To discover valid layer names for a backbone, use the ``list_backbone_layers``
utility:

.. code-block:: python

   from dendritic_modeling.networks.architectures.classical.pretrained import (
       list_backbone_layers,
   )
   print(list_backbone_layers("alexnet"))
   # ['features.0', 'features.1', ..., 'avgpool', 'classifier.0', ..., 'classifier.6']

**Scope**: The current splitter linearizes ``nn.Sequential`` containers.  This
works for AlexNet, VGG, and coarse splits of ResNet (e.g. at ``avgpool`` or
``fc``).  Replacing layers *inside* residual blocks requires an FX-graph
backend (planned).

Block-output distillation is configured separately with
``training.vision_replacement``.  For ``teacher_source: tensor_cache``, cache
the inputs to the replaced span and the original span outputs, then train the
replacement core directly.  For ``teacher_source: image_tensor_cache``, cache
images and teacher logits/features and choose ``train_target: replacement_only``,
``train_target: replacement_and_decoder``, or ``train_target: full_model``.
When checkpoint saving is enabled, the trainer always writes the replacement
core and also writes the decoder or full student state for train targets that
update those parameters.

Complete AlexNet-shaped students can instead use online matched-boundary
distillation through ``training.main.strategy: vision_distillation``. The
trainer keeps a frozen pretrained AlexNet on each training rank and combines a
weighted mean of relative feature MSE at ``conv1``--``conv5``, ``fc6``, and
``fc7`` with temperature-scaled logit KL and ordinary label cross-entropy.
Boundary weights are normalized by their sum, so adding a boundary does not
silently increase the feature-loss scale. For example:

.. code-block:: yaml

   training:
     main:
       strategy: vision_distillation
       strategies:
         vision_distillation:
           teacher_backbone: alexnet
           teacher_weights: IMAGENET1K_V1
           boundary_weights:
             conv1: 1.0
             conv2: 1.0
             conv3: 1.0
             conv4: 1.0
             conv5: 1.0
             fc6: 1.0
             fc7: 1.0
           feature_weight: 1.0
           logit_kl_weight: 1.0
           supervised_weight: 1.0
           temperature: 2.0

This strategy requires the complete ``dendritic_alexnet`` tensor contract;
single-span replacement distillation continues to use
``training.vision_replacement``.

The runnable examples page includes ``configs/vision/alexnet_fc7_dendritic_500.yaml``,
which replaces AlexNet ``fc7`` with a 500E-only dendritic tensor-map core. See
:doc:`examples` for all four primary example configs.

Additional vision replacement configs:

- ``configs/vision/alexnet_k1_einet_single.yaml`` — replace fc7 only (K=1), single-GPU / non-DDP
- ``configs/vision/alexnet_k1_einet_ddp.yaml`` — same K=1 replacement configured for DDP launch
- ``configs/vision/pretrained_alexnet_fc.yaml`` — tensor-map vision
  replacement around AlexNet FC layers
- ``configs/vision/pretrained_alexnet_conv.yaml`` — spatial vision
  replacement example for the exact AlexNet conv5 plus ReLU span
- ``configs/vision/alexnet_conv5_replacement_benchmark.yaml`` — matched conv5
  comparison of shunting/additive dendritic operators, a fixed-index sparse
  point convolution, and the exact retained pretrained span
- ``configs/vision/alexnet_fc6_replacement_benchmark.yaml`` — AlexNet fc6
  replacement benchmark with compact global/random-local/tree-aligned
  dendritic variants, fixed-index sparse point and factorized dense controls,
  and the exact retained pretrained span
- ``configs/vision/alexnet_fc6_learned_topk.yaml`` — full learned-TopK
  companion, separated for its much larger candidate storage and runtime
  request
- ``scripts/test_pretrained_training.sh`` — generates a temporary 1-epoch AlexNet smoke config

Transformer MLP / FFN Replacement
---------------------------------

Transformer replacement follows the same idea as AlexNet replacement: define
the source network and replacement sites, then define the dendritic replacement
with ``model.core``.  The source-network block is
``model.transformer_replacement`` instead of ``model.vision_replacement``
because transformer MLPs live inside decoder blocks rather than in a flat
``nn.Sequential`` layer list.

Both vision and transformer replacements use the same internal replacement-cell
contract: an input adapter maps source activations into a dendritic-compatible
domain, the dendritic core performs the replacement computation, and an output
adapter maps activity back to the source model's expected tensor shape.  The
placement logic is domain-specific: vision splits a backbone into
prefix/core/suffix segments, while transformer replacement patches the target
module in-place inside each decoder layer.

DeepSeek-V3/R1-style models are a natural first target because the first three
decoder blocks use dense FFNs before the later MoE blocks.  Public DeepSeek-V3
configs use ``hidden_size=7168``, ``intermediate_size=18432``, and
``first_k_dense_replace=3``.  The dense gated FFN parameter count per layer is
approximately ``3 * hidden_size * intermediate_size``.

.. code-block:: yaml

   model:
     transformer_replacement:
       enabled: true
       model_name: deepseek-ai/DeepSeek-V3
       layers: [0, 1, 2]
       target_module: mlp          # or a dotted path such as self_attn.o_proj
       layers_attr: model.layers
       input_transform: signed_split
       model_kwargs:
         device_map: auto
         torch_dtype: auto

     core:
       type: dendritic_ffn
       population_network:
         input_transform: signed_split
         layers:
           - name: transformer_mlp
             readout_population: ffn
             population_defaults:
               ff_excitatory_synapses: 40
               ff_inhibitory_synapses: 0
               somatic_synapses: true
               use_shunting: true
               reactivate: true
               reactivation_type: param_tanh
               weight_transform: softplus
               topk_type: indexed_rewire
               indexed_seed: 42
               indexed_rewire_frequency: 100
               indexed_rewire_quantile: 0.05
               indexed_rewire_until_step: null
               indexed_output_chunk_size: 2048
               efficient_blocklinear: true
             populations:
               - name: ffn
                 polarity: excitatory
                 n_neurons: 512
                 branch_factors: [3, 3]
             connections:
               - source: input
                 target: ffn

Training the replacement is configured separately under
``training.transformer_replacement``.  The most controlled training mode is
layerwise distillation: freeze the teacher module, feed hidden states through
the original target module, and train the dendritic module to match that
output.

.. math::

   \mathcal{L}_{\mathrm{distill}}
      = \frac{1}{|\mathcal{B}|}
        \sum_{h \in \mathcal{B}}
        \left\|D_{\theta}(h) - \mathrm{MLP}_{\mathrm{teacher}}(h)\right\|_2^2.

.. code-block:: yaml

   training:
     main:
       optimizer:
         name: adamw
         lr: 3.0e-4
         weight_decay: 0.01

     transformer_replacement:
       enabled: true
       mode: layerwise_distillation
       teacher_source: hf_text          # synthetic_mlp | hidden_cache | hf_text
       train_target: replacement_only
       layers: [0, 1, 2]
       device: auto
       dtype: bfloat16
       distributed_mode: ddp            # none | ddp (launch with torchrun)
       sequence_length: 128
       batch_size: 1
       max_steps: 1000
       eval_every: 100
       loss: relative_mse                # stable across layer-output scales
       cosine_weight: 0.1
       reactivation_calibration_batches: 3
       checkpoint_every: 100
       replacement_checkpoint_encoding: bitmask
       save_dir: results/deepseek_v3_dendritic_mlp_distillation
       text:
         dataset_name: json
         data_files: /path/to/shards/*.json.gz
         text_field: text
         streaming: true
         validation_documents: 64
         tokenizer_name: deepseek-ai/DeepSeek-V3

``checkpoint_every`` writes a common replacement/optimizer/scaler/history
checkpoint plus rank-local RNG state and packed-stream progress. Resume
layerwise training with ``resume_checkpoint`` pointing to
``training_checkpoint.pt``. Joint/FSDP checkpointing is not yet provided by
this dedicated runner.

``replacement_checkpoint_encoding: bitmask`` applies only to final layer
exports. It stores fixed row-wise topology compactly and is decoded
transparently when benchmarking or assembling replacements. Restart and
validation-best checkpoints retain slot-wise compact integers so optimizer
state stays aligned.

After layerwise distillation, ``joint_lm_distillation`` trains the patched
causal LM on token windows from ``hf_text``.  Use
``train_target: replacement_only`` to keep the original backbone frozen, or
``train_target: full_model`` to fine-tune the entire patched student.
``student_initialization: from_config`` constructs a randomly initialized
student from the Hugging Face architecture config; the default
``student_initialization: pretrained`` preserves the fine-tuning/distillation
path. ``save_student_model: true`` additionally writes the full patched state
dict and is intentionally opt-in for multi-billion-parameter models.

For model-weight-free stored-parameter estimates and the all-layer OLMo
checkpoint workflow, see :doc:`transformer_llm_scaling`.

Run the distillation trainer with:

.. code-block:: bash

   PYTHONPATH=src:$PYTHONPATH python -m \
     dendritic_modeling.scripts.training.train_transformer_replacement \
     configs/transformer_replacement_benchmarks/templates/deepseek_v3_mlp_dendritic_replacement.yaml

For a quick local learning check that does not load a Hugging Face model, use
``configs/examples/transformer_replacement_synthetic_distillation.yaml``.

Apply the YAML-style config to a loaded Hugging Face model:

.. code-block:: python

   from dendritic_modeling.networks.architectures.transformer import (
       apply_transformer_replacement_config,
       load_transformer_with_dendritic_replacements,
   )

   model, records = load_transformer_with_dendritic_replacements(
       config.model.transformer_replacement,
       config.model.core,
   )

   # Or, if the model has already been loaded:
   records = apply_transformer_replacement_config(
       model,
       config.model.transformer_replacement,
       config.model.core,
   )

Important replacement parameters:

``dendritic_units``
   Number of dendritic soma units in the replacement bottleneck.

``branch_factors``
   Dendritic morphology, distal-to-soma.  ``[3, 3]`` means each soma receives
   three proximal branches and each proximal branch receives three distal
   branches.

``synapses_per_branch``
   Number of active presynaptic features per dendritic branch.

``topk_type``
   ``standard``, ``stochastic``, ``variance``, ``dense_to_sparse``, and
   ``deepst`` preserve the dense-score/dynamic behavior of the previous
   DendriNet TopK family. ``dense_to_sparse`` starts dense or high-K and
   anneals to the configured synapse count; ``annealed_topk`` is an alias.
   ``indexed`` stores only exactly active fixed synapses. ``indexed_dynamic``
   stores a larger candidate pool and dynamically selects the active ``K`` from
   that pool. ``indexed_rewire`` stores only ``K`` active synapses and
   periodically replaces the weakest adaptive-quantile slots with fresh random
   allowed inputs.

``sparsity.dense_to_sparse``
   Developmental pruning schedule for ``topk_type: dense_to_sparse``. Common
   fields are ``initial_density`` or ``initial_k``, ``start_step``,
   ``end_step``, ``update_interval``, ``schedule`` (``linear``, ``cubic``,
   ``cosine``, or ``step``), ``freeze_on_end``, and ``prune_metric``.

``sparsity.indexed.candidate_size``
   Candidate-pool size ``C`` for ``indexed_dynamic``.  Stored synapse
   parameters scale as ``out_features * C`` instead of
   ``out_features * input_features``. If omitted, the implementation uses an
   automatic pool size; when hard connection masks are present, that automatic
   pool is capped to the largest value every masked output row can support.

``sparsity.indexed.rewire_frequency``
   Number of optimizer steps between ``indexed_rewire`` updates.  Use larger
   values for real training; every-step rewiring is useful for fast tests but
   can be too exploratory on fixed supervised tasks.

``sparsity.indexed.rewire_quantile``
   Fraction of active same-row synapses replaced at each update.  The
   threshold is adaptive: the layer replaces the weakest quantile of current
   active transformed weights, so no absolute weight cutoff has to be tuned
   across layers.

``sparsity.indexed.rewire_until_step``
   Optional cutoff step after which indexed rewiring stops and the sampled
   sparse morphology is frozen.

``sparsity.indexed.projection_backend``
   Execution backend for fixed-index projection: ``eager`` (compatibility
   default), ``recompute`` (portable memory-saving backward), ``triton``,
   ``cuda``, or ``auto``. Explicit native choices fail when unavailable;
   ``auto`` selects CUDA, then Triton, then portable recompute. The selector is
   shared by feedforward, recurrent, transformer, and configured vision
   replacement cores. See :doc:`performance_kernels` for scope and dtype rules,
   ``configs/templates/indexed_runtime_template.yaml`` for the full option
   block, and ``configs/examples/indexed_backends/`` for complete large-model
   recipes covering every selector value.

``reactivation.memory_efficient``
   Opt-in custom autograd for ``param_tanh`` and ``param_tanh_only_m``. In
   shunting layers it also fuses division with reactivation unless diagnostics
   need the intermediate voltage. The default is ``false``.

Minimal ``indexed_rewire`` block:

.. code-block:: yaml

   model:
     core:
       sparsity:
         type: indexed_rewire
         indexed:
           seed: 42
           projection_backend: eager
           rewire_frequency: 100
           rewire_quantile: 0.05
           rewire_until_step: null

``input_transform``
   Use ``signed_split`` for transformer hidden states.  It maps ``x`` to
   ``[ReLU(x), ReLU(-x)]``, allowing positive dendritic conductances to encode
   signed residual-stream features.

Recommended experimental order:

1. Patch one layer only and confirm a forward pass.
2. Freeze the original model and distill the dendritic replacement against the
   original MLP output for that layer.
3. Evaluate perplexity with layer 0 replaced.
4. Repeat for layers ``[0, 1]`` and then ``[0, 1, 2]``.

Template config:
``configs/transformer_replacement_benchmarks/templates/deepseek_v3_mlp_dendritic_replacement.yaml``.

Recurrent Networks
------------------

Any E-I type (``dendritic_shunting``, ``dendritic_additive``, ``einet``)
becomes recurrent when ``connectivity.rec_*`` synapse counts are nonzero.
Recurrence is controlled solely by these counts; no separate ``enabled``
flag is needed:

.. code-block:: yaml

   model:
     core:
       type: dendritic_shunting       # same type as feedforward
       architecture:
         excitatory_layer_sizes: [64]
         excitatory_branch_factors: [3, 3]
       connectivity:
         ee_synapses_per_branch_per_layer: [12]
         ie_synapses_per_branch_per_layer: [6]
         # Recurrent synapse counts (nonzero = recurrent)
         rec_ee_synapses_per_branch: [8]   # E->E recurrent
         rec_ie_synapses_per_branch: [4]   # I->E recurrent
         rec_ei_synapses_per_branch: [8]   # E->I recurrent
         rec_ii_synapses_per_branch: [4]   # I->I recurrent
       transfer:
         input_mode: 1
         inhibitory_mode: "first"     # none, first, all
       # morphology, sparsity, reactivation as usual...

       recurrent_ei:                  # temporal params only
         output_mode: last            # last, all, mean
         use_transfer: true           # default true (set false to bypass TransferLayer)
         dt: 1.0
         tau_base: 50.0               # scalar or per-layer list
         tau_ratio: 3.0               # scalar or per-layer list

   training:
     main:
       strategy: recurrent

For full per-layer, per-population control use ``core.type: ei_unified``
with the ``unified_ei`` config instead. For baseline RNNs use
``model.core.type: gru`` (or ``lstm``, ``vanilla_rnn``).
See :doc:`network_architectures` for full details.

Named Cell Types
----------------

The recurrent extensions use a few related terms that are easy to mix up:

.. list-table::
   :header-rows: 1
   :widths: 20 34 46

   * - Term
     - Where it is defined
     - What it means
   * - Population network
     - ``model.core.type: population_network`` and
       ``model.core.population_network``
     - The canonical feedforward/recurrent named-population backend. It owns
       ordered layers, populations per layer, directed source-target
       connections, readout population, timestep, and output mode.
   * - Named population
     - ``population_network.layers[].populations``
     - One named dendritic population, such as ``pyr``, ``pv``, or ``sst``.
       It defines how many neurons that population has, whether it is
       excitatory or inhibitory, and what morphology/config overrides it uses.
   * - Population connection
     - ``population_network.layers[].connections``
     - A source-target rule between named populations or input streams. It says
       which population can drive another, whether the edge is same-step or
       delayed, and can optionally sample a fixed pair-specific mask.
   * - Structured connectivity
     - ``model.core.connectivity.structured``
     - A lower-level TopK/DeepST mask on E/I pathways such as ``ee``,
       ``ie``, and ``rec_ee``. It controls which presynaptic units are
       eligible for sparse synapses, independent of biological population
       names.
   * - Synapse type
     - ``synapse_types`` inside ``population_defaults``, a population's
       ``population`` block, or ``model.core.synapse_types``
     - A conductance/receptor subtype within an excitatory or inhibitory
       stream, such as ``AMPA``, ``NMDA``, ``GABA_A``, or a custom continuous
       class. It changes kinetics, fractions, voltage gates, and reversal
       potentials; it does not create a separate neuron population.
   * - Soma dynamics
     - ``dynamics_mode`` / ``spike_*`` or ``model.core.dynamics``
     - The output rule for each dendritic population. ``rate`` is the
       continuous-output rule; ``spike`` routes the soma through a LIF module.

In short: populations define **who the neurons are**, population connections
define **who can talk to whom**, structured connectivity defines **which
individual synapse slots are allowed**, synapse types define **how a signal is
filtered and driven once it arrives**, and soma dynamics define **how the
population emits its output**.

For population-network experiments, use ``core.type: population_network``.
It is the notation for feedforward stacks and recurrent microcircuits:
each layer contains named populations, and each edge is an explicit directed
connection. ``timing: same_step`` is feedforward/default; ``timing: delayed``
uses previous-timestep outputs and enables recurrence. Each population is still
a dendritic population internally, so the usual ``PopulationConfig`` fields can
be placed in ``population_defaults`` or in an individual population's
``population`` block.

.. code-block:: yaml

   model:
     core:
       type: population_network
       population_network:
         output_mode: last
         layers:
           - name: microcircuit
             recurrent: true
             readout_population: pyr
             connection_seed: 123
             population_defaults:
               ff_excitatory_synapses: 12
               rec_excitatory_synapses: 8
               rec_inhibitory_synapses: 4
               somatic_synapses: true
               reactivate: true
             populations:
               - name: pyr
                 polarity: excitatory
                 n_neurons: 64
                 branch_factors: [3, 3]
               - name: pv
                 polarity: inhibitory
                 n_neurons: 16
                 branch_factors: [1]
               - name: sst
                 polarity: inhibitory
                 n_neurons: 16
                 branch_factors: [1]
             connections:
               - source: input
                 target: pyr
               - source: input
                 target: pv
               - source: input
                 target: sst
               - source: pv
                 target: pyr
                 timing: delayed
                 probability: 0.45
               - source: sst
                 target: pyr
                 timing: delayed
                 probability: 0.25
               - source: pyr
                 target: pyr
                 timing: delayed
                 enabled: false

   training:
     main:
       strategy: recurrent

Connection polarity is inferred from the source population or input stream.
``probability`` optionally samples a fixed pair-specific mask. Missing
probability means dense/all allowed; the target population's E/I synapse-count
fields determine how many active synapses per branch are selected inside the
allowed mask.

Each named-population entry has four direct fields plus an optional nested
population override:

``name``
   A stable string key used by ``readout_population``, ``connections``, saved
   routing metadata, and analysis outputs. Use short names such as ``pyr``,
   ``pv``, ``sst``, or ``vip`` when you want biologically motivated labels.

``polarity``
   Either ``excitatory`` or ``inhibitory``. This is a circuit-level sign/source
   classification, not a receptor type. Outgoing edges from excitatory
   populations enter the target's E stream; outgoing edges from inhibitory
   populations enter the target's I stream.

``n_neurons``
   Number of dendritic neurons in this named population.

``branch_factors``
   The morphology for this population. It overrides any default morphology in
   ``population_defaults``.

``population``
   Optional dictionary of any ``PopulationConfig`` field. Use it when one
   population should have different synapse counts, time constants,
   reactivation, structured masks, synapse types, or spiking dynamics from the
   defaults.

The ``population_defaults`` block is copied into every named population
before the cell-specific ``population`` override is applied. This gives a
compact way to share most morphology and integration parameters while changing
only the biological identity fields in ``populations``.

``population_network`` uses ``population_defaults`` and per-population
``population`` blocks as its population source of truth. Top-level feedforward fields such as
``model.core.morphology.use_shunting`` are not automatically inherited; set the
corresponding ``PopulationConfig`` fields, for example ``use_shunting`` or
``weight_transform``, directly in ``population_defaults`` when they should apply
to all named populations.

Synapse Types
-------------

Recurrent dendritic populations can optionally split excitatory and inhibitory
conductance traces into named synapse types. This is useful for experiments
with receptor-like kinetics such as fast AMPA, slower NMDA, GABA-A, and GABA-B,
or for continuous conductance classes defined by their reversal potential.
The default is disabled, which preserves the original single E trace and
single I trace exactly.

Place ``synapse_types`` either in a ``PopulationConfig`` payload, such as
``population_network.layers[].population_defaults`` or a
``populations[].population`` override, or at ``model.core.synapse_types`` for
recurrent EINet-family configs translated through the factory.

Synapse types are intentionally orthogonal to named populations:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Question
     - Named-population answers
     - Synapse type answers
   * - What object is being described?
     - A named population of dendritic neurons.
     - A named conductance/receptor component inside E or I input.
   * - Examples
     - ``pyr``, ``pv``, ``sst``, ``vip``.
     - ``AMPA``, ``NMDA``, ``GABA_A``, ``GABA_B``, ``E_rev_0p4``.
   * - Main fields
     - ``polarity``, ``n_neurons``, ``branch_factors``, ``population``.
     - ``fraction``, ``tau``/``tau_rise``/``tau_decay``,
       ``reversal_potential``, ``magnesium_block``, ``stp``.
   * - Scientific role
     - Models heterogeneity in neuron classes and circuit motifs.
     - Models heterogeneity in synaptic kinetics and driving forces.
   * - Where it changes computation
     - Adds another ``StatefulDendriNet`` population and recurrent state.
     - Splits raw E/I currents into typed traces before dendritic integration.

You can use biologically named receptors, continuous abstract conductance
classes, or both. For example, ``AMPA``/``NMDA`` is a receptor-motivated
excitatory split; a custom type named ``E_rev_0p4`` with
``reversal_potential: 0.4`` is a continuous conductance class. The code treats
both as named synapse types with kinetics and reversal parameters.

.. code-block:: yaml

   population_network:
     layers:
       - name: recurrent_layer
         recurrent: true
         readout_population: pyr
         population_defaults:
           ff_excitatory_synapses: 12
           ff_inhibitory_synapses: 4
           rec_excitatory_synapses: 8
           rec_inhibitory_synapses: 4
           synapse_types:
             enabled: true
             excitatory:
               - name: AMPA
                 fraction: 0.8
                 tau_rise: 0.5
                 tau_decay: 2.0
                 reversal_potential: 1.0
               - name: NMDA
                 fraction: 0.2
                 tau_rise: 3.0
                 tau_decay: 70.0
                 reversal_potential: 1.0
                 magnesium_block: true
                 voltage_scale_mv: 100.0
             inhibitory:
               - name: GABA_A
                 fraction: 0.9
                 tau_rise: 1.0
                 tau_decay: 7.0
                 reversal_potential: 0.0
               - name: GABA_B
                 fraction: 0.1
                 tau_rise: 20.0
                 tau_decay: 150.0
                 reversal_potential: -0.25
                 stp:
                   enabled: true
                   u0: 0.25
                   depression: 0.2

When ``enabled: true``, define both an ``excitatory`` list and an
``inhibitory`` list explicitly. This keeps a partially specified receptor block
from silently falling back to the single-trace path. Fractions are
normalized within each polarity. ``tau: null`` inherits the population's
existing per-level dendritic time constant, so a single ``AMPA``/``GABA_A``
pair with ``tau: null`` matches the standard recurrent dynamics. In shunting mode,
``reversal_potential`` is used directly in the conductance numerator. In
additive mode, typed synapses intentionally support only the normalized additive
reversal convention: excitatory types use ``reversal_potential: 1.0`` and
inhibitory types use ``reversal_potential: 0.0``. Use shunting mode for
continuous or biophysical reversal potentials such as GABA-B-like negative
values. For stability and interpretability, typed reversal potentials use the
normalized conductance convention and must lie in ``[-1.0, 2.0]``.

For receptor-like dynamics, use ``tau_rise`` and ``tau_decay`` to create a
double-exponential trace. ``tau`` remains as a shorthand for a single decay
trace. ``magnesium_block: true`` applies the standard NMDA-style voltage gate
``1 / (1 + Mg * exp(-V_mV / slope) / scale)`` using the previous level voltage.
Because branch voltages are normalized and unitless, the gate first computes
``V_mV = voltage_scale_mv * V + mg_v_offset``. The default
``voltage_scale_mv: 100.0`` makes NMDA-like gates vary across the model's
typical ``0`` to ``1`` voltage range while preserving the canonical
``mg_scale`` and ``mg_slope`` constants. You can also override
``mg_concentration``, ``mg_scale``, ``mg_slope``, and ``mg_v_offset``.
``nonlinearity`` can be ``linear``, ``relu``, ``sigmoid``, ``tanh``, or
``softplus``; ``conductance_power`` applies a signed power transform to the raw
current. The optional ``stp`` block adds aggregate short-term
utilization/resource dynamics for rate or spike-trace inputs. By default, STP
state updates use a detached activity proxy so the current multiplicative
factor is differentiable but its history is treated as a fixed state update.
Set ``stp.differentiable_state: true`` to allow BPTT through the STP state
history.

Common definition patterns:

* **Legacy-equivalent**: keep ``enabled: false``, or enable one excitatory type
  and one inhibitory type with ``tau: null`` and fractions of ``1.0``.
* **Biophysical receptors**: use names such as ``AMPA``, ``NMDA``,
  ``GABA_A``, and ``GABA_B`` with different rise/decay constants, Mg block,
  and reversal potentials.
* **Continuous conductance classes**: use custom names and vary
  ``reversal_potential``, ``tau``, ``nonlinearity``, or
  ``conductance_power`` without implying a specific receptor identity.

Spiking Soma Dynamics
---------------------

Recurrent dendritic populations can optionally replace the rate-valued soma
output with leaky integrate-and-fire dynamics. This is configured at the
``PopulationConfig`` level for ``population_network`` or via
``model.core.dynamics`` for recurrent EINet-family configs. The default
``dynamics_mode: rate`` preserves the original behavior.

.. code-block:: yaml

   population_network:
     layers:
       - name: recurrent_layer
         recurrent: true
         readout_population: pyr
         population_defaults:
           dynamics_mode: spike
           spike_threshold: 0.6
           spike_reset: 0.0
           spike_tau: 20.0
           spike_refractory_steps: 2
           spike_surrogate_beta: 10.0
           spike_readout: rate       # spikes, rate, membrane
           spike_readout_tau: 20.0

For recurrent EINet-style configs translated through the factory:

.. code-block:: yaml

   model:
     core:
       type: dendritic_shunting
       dynamics:
         mode: spike
         threshold: 0.6
         readout: rate
         tau: 20.0
         refractory_steps: 2

``spike_readout: spikes`` returns binary surrogate-gradient spikes.
``spike_readout: rate`` returns a leaky spike trace, which keeps the downstream
decoder interface dense and is usually the easiest first training target.
``spike_readout: membrane`` returns the LIF membrane before thresholding for
ablation and debugging.

Dendritic Spike / Plateau Dynamics
----------------------------------

Soma spiking and dendritic spiking are separate switches.  Soma spiking changes
only the final soma readout.  Dendritic spike / plateau dynamics act earlier:
after a dendritic level computes its branch voltage and before that voltage is
aggregated into the parent level.  This makes local branch events propagate
toward the soma while preserving the existing branch-factor morphology.

The implementation supports two local event modes:

``plateau``
   A branch emits a surrogate-gradient threshold event and stores it in a leaky
   plateau trace.  The plateau trace then modifies the branch output.

``threshold``
   The branch emits a binary surrogate-gradient local event without a persistent
   plateau trace.

.. code-block:: yaml

   population_network:
     layers:
       - name: recurrent_layer
         recurrent: true
         readout_population: pyr
         population_defaults:
           dendritic_spikes_enabled: true
           dendritic_spike_mode: plateau       # plateau or threshold
           dendritic_spike_levels: distal      # non_soma, all, distal, proximal, none, or [0, 1]
           dendritic_spike_threshold: 0.7
           dendritic_spike_plateau_amplitude: 1.0
           dendritic_spike_plateau_tau: 20.0
           dendritic_spike_refractory_steps: 5
           dendritic_spike_surrogate_beta: 10.0
           dendritic_spike_propagation: additive  # additive, replace, gated

For recurrent EINet-style configs translated through the factory, the same
fields can be nested under ``model.core.dynamics.dendritic_spikes``:

.. code-block:: yaml

   model:
     core:
       type: dendritic_shunting
       dynamics:
         dendritic_spikes:
           enabled: true
           mode: plateau
           levels: non_soma
           threshold: 0.7
           plateau_amplitude: 1.0
           plateau_tau: 20.0
           refractory_steps: 5
           surrogate_beta: 10.0
           propagation: additive

Level indices are distal-to-soma.  ``non_soma`` and ``all`` mean all dendritic
levels but exclude the soma; soma spiking is controlled by ``dynamics_mode:
spike`` and the ``spike_*`` fields above.  ``distal`` means level ``0`` and
``proximal`` means the last dendritic level before the soma.  The propagation
rule controls how the local plateau/event changes the branch output:

``additive``
   ``V_modified = V_branch + amplitude * plateau``.

``replace``
   ``V_modified = V_branch * (1 - plateau) + amplitude * plateau``.

``gated``
   ``V_modified = V_branch * (1 + amplitude * plateau)``.

The recurrent state stores ``dendritic_spike_plateau``,
``dendritic_spike_refractory``, and ``dendritic_spike_events`` only when this
component is enabled.

Soma-to-Dendrite Feedback
-------------------------

Soma feedback is an optional recurrent component that sends the previous
timestep's soma signal back into selected dendritic levels. The default is off,
so existing feedforward and recurrent runs keep the original dendrite-to-soma
flow. Because the source is delayed by one recurrent step, the implementation
does not introduce a same-timestep circular dependency.

For ``population_network`` configs, place the fields in ``population_defaults``
or in an individual population's ``population`` block:

.. code-block:: yaml

   population_network:
     layers:
       - name: recurrent_layer
         recurrent: true
         readout_population: pyr
         population_defaults:
           soma_feedback_enabled: true
           soma_feedback_mode: additive      # additive, shunting, gated
           soma_feedback_source: output      # output, voltage, spike_readout
           soma_feedback_levels: non_soma    # non_soma, all, distal, proximal, soma, none
           soma_feedback_strength: 0.15
           soma_feedback_learnable_strength: true
           soma_feedback_per_level: true
           soma_feedback_init_std: 0.01

For recurrent EINet-family configs translated from the structured core config,
use ``model.core.soma_feedback``. The
``model.core.recurrent_ei.soma_feedback`` location is also accepted:

.. code-block:: yaml

   model:
     core:
       type: dendritic_shunting
       soma_feedback:
         enabled: true
         mode: gated
         source: voltage
         levels: distal
         strength: 0.1

``additive`` adds feedback as a branch-current term. ``shunting`` creates a
rectified feedback conductance and adds ``conductance * reversal`` to the
current numerator. ``gated`` multiplies the branch output by
``1 + tanh(feedback)``. ``soma_feedback_init_std`` defaults to ``0.0``, making
the projection initially no-op; set a small nonzero value when an experiment
should start with active feedback.

Spike-specific analysis is available through ``analysis.spike_train``:

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

This analyzer hooks every ``LIFSoma`` module and reports firing rate, silent
unit/trial fractions, spike-count Fano factors, ISI mean/CV, per-timestep rate,
and compact raster events.

Structured Connectivity
-----------------------

Optional hard topology constraints can be added under
``model.core.connectivity.structured``.  The existing synapse-count fields still
control the TopK budget per branch; the structured mask controls which
presynaptic cells are allowed to compete for those TopK slots.

.. code-block:: yaml

   model:
     core:
       type: dendritic_shunting
       architecture:
         excitatory_layer_sizes: [64]
         inhibitory_layer_sizes: [16]
         excitatory_branch_factors: [3, 3]
       connectivity:
         ee_synapses_per_branch_per_layer: [12]
         ie_synapses_per_branch_per_layer: [6]
         ei_synapses_per_branch_per_layer: [8]
         ii_synapses_per_branch_per_layer: [4]

         structured:
           enabled: true
           seed: 123
           pathways:
             ee:                 # E input onto E target cells
               method: fixed_indegree
               indegree: 24
             ie:                 # I input onto E target cells
               method: bernoulli
               probability: 0.35

Pathway names are ``ee``, ``ie``, ``ei``, ``ii`` for feedforward/current-layer
connections, and ``rec_ee``, ``rec_ie``, ``rec_ei``, ``rec_ii`` for recurrent
connections. Supported mask methods are ``fixed_indegree``, ``bernoulli``, and
``distance``. With DeepST, rewiring is constrained to stay inside the structured
mask.

``method: spatial_morphology`` is the compact image-specific alternative. It
does not create a dense mask. It directly assigns exactly K fixed pixel indices
to every branch according to the tree's nested spatial regions:

.. code-block:: yaml

   model:
     core:
       type: tensor_map
       architecture:
         excitatory_layer_sizes: [9916]
         inhibitory_layer_sizes: []
         excitatory_branch_factors: [3, 3, 3, 3]
       connectivity:
         ee_synapses_per_branch_per_layer: [32]
         ie_synapses_per_branch_per_layer: [8]
         structured:
           enabled: true
           method: spatial_morphology
           seed: 42
           spatial:
             input_shape: [3, 224, 224]
             split_axes: [height, width, height, width]
       sparsity:
         type: indexed
         indexed:
           index_dtype: int32
           projection_backend: recompute
           persistent_indices: false

``split_axes`` is listed from soma to distal leaves and must have the same
length as the branch-factor list. The raw tensor is interpreted in CHW order.
Every soma has independent weights and independently sampled indices; only the
geometric partition is shared. Thus the mapping preserves image geometry
without convolution or parameter sharing. The current implementation requires
fixed ``indexed`` sparsity and rejects global ``indexed_rewire``. The
``local_ca`` 3F/4F/5F rules update only the stored indexed contacts and remain
compact under DDP; local gradients are averaged explicitly across ranks.

``spatial.region_mode`` selects the geometry used by the fixed indices:

``partition``
   Recursively partitions the image into disjoint morphology-aligned regions.
``overlapping_grid``
   Uses nested morphology-aligned windows, permitting overlap and enforcing
   ``minimum_window_shape`` where a disjoint terminal region would be too small.
``random_windows``
   Starts from the exact ``overlapping_grid`` window shape assigned to each
   branch, then places each window independently and uniformly over the image.
   This preserves the number of contacts and per-level window-size distribution
   while removing parent--child spatial alignment. ``spatial.region_seed``
   optionally fixes window placement separately from contact sampling. E and I
   banks use the same branch windows but independently sample their contacts.

All three modes generate the index map once at model construction; the indices
remain fixed throughout training.

The same method is available to feedforward populations in a
``population_network``. In that schema, put the rule and fixed-index controls
inside the target population defaults (or population override):

.. code-block:: yaml

   population_defaults:
     ff_excitatory_synapses: 40
     ff_inhibitory_synapses: 10
     topk_type: indexed
     indexed_seed: 42
     indexed_index_dtype: int32
     indexed_persistent_indices: false
     structured_connectivity:
       enabled: true
       method: spatial_morphology
       seed: 42
       spatial:
         input_shape: [3, 32, 32]
         region_mode: random_windows
         minimum_window_shape: [4, 4]
         region_seed: 42

Both direct image banks (for example ``input_e`` and ``input_i``) then use the
same branch regions while sampling their configured E and I contact counts
independently. Image-grid routing is not defined for recurrent population-state
pathways; configure pathway-specific rules when a population also has active
recurrent connections.

Multi-Stage Training
--------------------

.. code-block:: yaml

   training:
     main:
       strategy: multi_stage
       strategies:
         multi_stage:
           analyze_between_stages: true
           stages:
             - learning_strategy: standard
               epochs: 50
               trainer_config:
                 loss_function: ce
             - learning_strategy: local_ca
               epochs: 50

See :doc:`training_strategies` for per-stage options.

Distributed Training
--------------------

Set ``distributed.mode`` to choose the runtime:

.. code-block:: yaml

   distributed:
     mode: ddp            # none | ddp | fsdp
     fsdp:                # FSDP-specific knobs (only used when mode=fsdp)
       sharding_strategy: "FULL_SHARD"
       mixed_precision: true
       cpu_offload: false
       auto_wrap_policy: "none"  # or "indexed_synapses"
       cpu_init: false           # shard wrapped units before moving to GPU

- ``none`` — single-GPU, plain ``python`` (default)
- ``ddp`` — DistributedDataParallel via ``torchrun``.  Best when the model
  fits on one GPU and you want data-parallel speedup.  Supports multi-node.
- ``fsdp`` — FullyShardedDataParallel via ``torchrun``.  Use when the model
  is too large for a single GPU.  Supports multi-node.

Node and GPU counts are controlled by the SLURM allocation (``sbatch --nodes``,
``--gpus-per-node``) or by the launcher CLI argument, **not** by the YAML
config.  The training config only selects the *mode*; the scheduler decides
the resources.

FSDP-specific knobs (sharding strategy, mixed precision, CPU offload, etc.)
live under ``distributed.fsdp``.  For contact-heavy models, combine
``auto_wrap_policy: indexed_synapses`` with ``cpu_init: true`` so FSDP wraps
and moves each indexed sparse-synapse unit without first placing the complete
unsharded model on every GPU.  ``cpu_init`` is opt-in; the default preserves
the historical eager device transfer.

Launch Commands
~~~~~~~~~~~~~~~

Use the package entry points directly; wrap them in a site-local SLURM script
when running on a cluster:

.. code-block:: bash

   # DDP
   torchrun --nproc_per_node=4 \
     -m dendritic_modeling.scripts.training.train_experiments \
     configs/my_config.yaml

   # FSDP
   python -m dendritic_modeling.scripts.training.train_experiments_fsdp \
     configs/my_config.yaml

See :doc:`fsdp_troubleshooting` for FSDP-specific troubleshooting.

Reference Config
----------------

The annotated population-network reference config is in
``configs/config_exp.yaml`` and reproduced in :doc:`config_reference`.
