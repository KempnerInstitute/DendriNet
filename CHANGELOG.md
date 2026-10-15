# Changelog

## Unreleased

- Make native YAML documentation examples construct under the biological
  default, with explicit input-rate and external-source contracts. Test model
  construction as well as schema loading; the signed soma-feedback example
  explicitly opts out.

- Unified E/I networks also support explicit nonnegative/split input adapters
  and positive input projections. Omitted options keep the historical signed
  projection and identity input; biological adoption requires an explicit
  encoding and retraining when those choices change the model.

- Add explicit biological configuration preparation for expanded experiments.
  It upgrades historical keys before applying caller-selected input/source
  contracts, rejects ambiguous upgrades and leaves circuit choices to callers.
  A source-pinned recipe runner validates study inventories and compares
  diagnostic states, outputs and gradients before writing isolated new recipes.

- Configuration format 4 enables ``biological_neuron`` by default for standard
  dendritic E/I cores. Explicit ``false`` remains available for exploratory
  models; signed/MLP controls and unrelated families retain their contracts.
  The upgrader writes the historical ``false`` default in older files and
  leaves explicit declarations intact. Shipped historical recipes preserve
  their previous computation rather than silently adopting new circuitry.
- Add checked-identity ``nonnegative`` inputs, preserving values and gradients
  while rejecting negative or nonfinite rates. Transfer layers can explicitly
  declare distinct, identically tuned external E/I ``paired_afferents``;
  shared input channels still fail the Dale constraint when used with both signs.
- Population networks support an opt-in positive input projection via
  ``input_projection_weight_transform: softplus``. This is a new model with
  positive weights and a rectified output; it requires retraining when replacing
  a signed projection. The omitted option preserves the signed Linear path.
  Analytic local updates reject trainable positive projections instead of
  leaving their parameters without updates; frozen projections are supported.
- Correct biological operator-card checks to cover the neuronal core, excluding
  task encoders and decoders. Recognize a rectified internal projection when
  checking the dendritic input domain, while still rejecting its signed weights.
  Operator-card schema v3 records this scope and the external-source declaration.
- Conventional parameter-budget benchmarks explicitly retain their historical
  non-biological contract in generated configurations and reports. Their model
  states, outputs and numerical report fields are unchanged.

- The README coverage badge uses a relative image path so private-repository
  readers can load it. CI proposes measured badge updates through a pull
  request instead of pushing directly to ``develop`` and ignoring failures.
  Refreshed the badge to 80.3% line coverage from the Python 3.11 report in
  [CI run 195](https://github.com/KempnerInstitute/DendriNet/actions/runs/38682938459)
  for ``fdf49d0a81aa039a873bbf538fabf1545fb9ee0e``.
  The CI Python setup action now uses its supported Node 24 runtime.
- Add opt-in ``shared_projection_shunting``: normalized sparse projections
  reused across native flat DendriNet banks, signed block pooling and an affine
  output. Its typed configuration declares supports, initial transition nodes,
  precision and frozen parameter groups; epsilon is included in the node-to-
  conductance mapping. It exposes native features and complete resource counts.
  This composition supports standard backpropagation; it is not a new branch
  integration rule or a qualification of nonlinear-soma dendritic trees.
  Existing model defaults and initialization paths are unchanged.
- ``dendritic-collect-reactivation --require-clean-analysis-source`` (or
  ``require_clean_analysis_source: true``) rejects runs whose analysis
  provenance records no Git checkout, tracked changes, untracked source files
  or a commit other runs do not share, and records the shared commit. It checks
  the source of the analysis only, not the commit that trained the checkpoint.
  Off by default; collections without it are unchanged.
- The reactivation collection configuration rejects unknown keys and requires
  ``true`` or ``false`` for ``require_structured_coordinates`` and
  ``require_clean_analysis_source`` and a positive integer ``expected_count``:
  a quoted ``"false"`` turned a check on, a misspelt key was ignored, and
  ``expected_count: 2.5`` became 2.
- A regression test pins the variational autoencoders' reparameterized
  sampling: in training mode the reconstruction loss reaches the layers that
  produce the latent mean and scale. An August 2025 fix had replaced
  ``sample()`` with ``rsample()``; no test guarded it.

## 0.2.0 (2026-10-14)

Changes since the ``pre-refactor`` tag (version 0.1.0), newest first.
Configuration files carry ``config_version: 3``; ``dendritic-upgrade-config``
rewrites files written for earlier versions so that they describe the same
runs.

### 2026-10

Repository housekeeping:

- The default-path goldens run their cases in a child process with fixed MKL,
  oneDNN and PyTorch CPU code paths, so they hold on CPUs without AVX-512 such
  as the pull-request runners; re-recorded, they change the initial-state
  digest of one case by one rounding step in one 2x2 tensor. Tests that start
  their own worker processes run one at a time in a separate CI step.
- Tests remove the logging handlers they add. A program run in-process
  attaches its console handler to the test's captured stdout, and once that
  stream closed, log records of later tests in the same worker failed with a
  logging error on stderr, depending on which tests shared the worker.
- The configuration-options reference renders parameterized field types (for
  example ``list[int]`` and ``dict[str, Any]``) identically on every Python
  version, so its check passes on the Python 3.11 of the pull-request checks
  as on 3.10.
- Installation instructions select ``develop`` explicitly. The example guide
  distinguishes complete experiment configurations from section templates,
  and documents the contact-policy and gate-initialization differences between
  the single-process and FSDP recipes. Removed inactive indexed-backend options
  from the variance Top-K target-level example; its initialized state, forward
  output and gradients are unchanged. The quickstart and documentation-check
  instructions now describe the configured early stopping and test coverage.
- Removed the empty submodule manifest and consolidated the ignore rules for
  local environments, credentials, editor state, generated runs and scratch
  files. Generated Slurm files are ignored in their output directories;
  maintained scripts, example configurations and test CSV fixtures remain
  addable without force. No library code or configuration defaults changed.

Corrections change results only where the earlier behaviour was wrong;
results produced before them reproduce at the library revision that produced
them.

Release review of 2026-10-13:

- Configuration format version 3. In ``model.core.connectivity.structured``
  a ``distance`` rule without ``distance_kernel`` is gaussian, as in
  population-network rules (version 2: ``none``, which ignored positions).
  Training fills the unset seed of population and unified E/I structured rules
  from ``experiment.topology_seed`` (offset per unified layer), so a fixed
  topology seed fixes their masks (version 2: the global random stream; the
  new value ``seed: global_rng`` keeps that). Files of version 2 load only
  after ``dendritic-upgrade-config``, which writes ``distance_kernel: none``,
  removes aliases a canonical key shadowed, writes ``seed: global_rng`` (or
  ``seed: 0`` for spatial rules) where version 2 drew from the global stream,
  removes retired keys, and checks the result against the version-2 reading,
  reporting what it cannot keep. Version-1 rewrites apply only to files
  before version 2. The shipped configurations carry version 3.
- Structured connectivity reads ``p_connect``, ``in_degree``, ``kernel`` and
  ``sigma`` as ``probability``, ``indegree``, ``distance_kernel`` and
  ``distance_sigma`` in the root and in each pathway entry before merging
  them, so a pathway's alias overrides the root key. In
  ``model.core.connectivity.structured`` such an alias was ignored
  (``kernel``, ``sigma``) or failed (``p_connect``, ``in_degree``). A rule that
  writes both spellings of a key, or names one pathway twice, raises.
- A ``distance`` rule with neither ``base_probability`` nor a ``probability``
  to stand in for it raises a configuration error naming the rule (at load
  for ``model.core.connectivity.structured``) instead of a bare TypeError at
  build.
- ``model.core.spatial`` options that differ from their defaults raise at load
  for built-in core types that never read them;
  ``sparse_active_matched_point_conv`` also rejects the adapter options it
  ignores. A saved ``config.json`` warns instead.
- Per-layer synapse lists longer than the number of layers raise at load for
  every core type built from the structured sections, as they already did for
  ``einet``; they were truncated silently.
- Building a model raises when ``sparsity.deepst.synapses_per_branch`` (the
  population ``synapses_per_branch``) is set but unused: dendritic layers keep
  each pathway's synapse count as the ``constant_branch`` budget.
- Unified E/I populations raise before construction when a written inhibitory
  synapse count has no input source (it built no synapses);
  ``dendritic-upgrade-config`` sets such counts to 0.
- Reloading a saved ``config.json`` no longer warns that the default
  ``init_m``/``init_b`` have no effect.
- Recurrent-baseline payload typos raise an error that names the
  ``model.core`` section and suggests the key; a written ``input_dim`` or
  ``cell_type`` that the build would replace raises (the upgrade tool removes
  them).
- The ImageNet protocol loaders resolve an empty ``data.base_dir`` as
  training does.
- The gate-statistics files record the initialization policy each dendritic
  layer applied to its gate.
- ``sparsity.init_method`` does not set the weights of contacts that a
  configured ``indexed_rewire`` regrows (they keep the pruned weight); the
  documentation and comments said it did.
- The library no longer reads ``training.transformer_distillation``, an
  earlier spelling of ``training.transformer_replacement``; the optional
  extension that defines that section rewrites it when it loads a
  configuration. Without the extension the key is unknown and raises.
- ``ZeroPadOutputAdapter``, ``LinearOutputAdapter`` and
  ``FlatAffineOutputAdapter`` left ``networks.architectures.replacement``:
  no DendriNet model or configuration uses them, and the optional extension
  that does now carries them. The adapters DendriNet's models use (input
  transforms, ``make_nonmixing_output_adapter`` and its modes,
  ``PopulationToFeatureReduction``) are unchanged.
- ``dendritic-train`` and the FSDP entry point check encoder requests (an
  autoencoder encoder without pretraining or a trained checkpoint,
  pretraining with an encoder that is not an autoencoder, and the other
  encoder rules) before they create the run directory, so a rejected request
  leaves no partial run directory. Valid runs are unchanged.
- Training logs a warning when a request cannot act: periodic reactivation
  recalibration that would recalibrate no gate, ``use_amp`` without CUDA, and
  a ``local_ca`` auxiliary term (``hsic``, ``inhibitory_homeostasis``,
  ``voltage_homeostasis``, ``gate_homeostasis``) enabled with weights that
  make it inactive. Training itself is unchanged. A scheduled recalibration
  that recalibrates no gate still draws training batches, so removing the
  setting can change the data order of later epochs; the warning says so, and
  an earlier run is reproduced by keeping it.
- The error raised when data-driven gate calibration finds no complete
  training batch names the training split size and
  ``reactivation_initialization_batch_size``.
- The FSDP rejection of scalar parameters no longer advises against
  ``output_scale_mode: global`` (its scale is a one-element vector that FSDP
  shards); the comments on ``param_groups.encoder_lr``, the FSDP dtype fields
  and the sweep template's key checks describe current behaviour.
- Figures no longer tell series apart by colour alone: the sweep plots (noise,
  weight, layer-wise information, ablation, local rules), the library's
  plotting modules (ablation, weight, performance, loss, information,
  synaptic-weight and correlation plots) and the analyzer figures (ablation,
  layer contribution, compartment SNR and statistics, weights, time traces,
  spectra, receptive fields) give lines their own marker and line style, bar
  series their own hatch (drawn in black or white, whichever stands out
  against the fill), overlaid histograms their own outline style and
  depth-coloured scatters their own markers. Series that used matplotlib's
  default colour cycle use the Okabe-Ito colours, and noise types are drawn
  in sorted order. New helpers in ``dendritic_modeling.plotting.styles``:
  ``HATCHES``, ``indexed_hatch``, ``indexed_linestyle``, ``hatch_color``,
  ``bar_kwargs``; ``plotting_utils.add_error_bars`` takes ``marker`` and
  ``linestyle``. Numerical outputs are unchanged.
- ``dendritic-sweep`` checks every generated configuration, the declared
  configuration count and ``execution_results_dir`` before it creates the
  sweep directory, and removes what it created when writing the
  configurations, job scripts or manifest fails.
- The pull-request documentation build treats Sphinx warnings as errors, like
  the deployment build, and the pull-request checks fail when
  ``docs/cli_reference.rst`` or ``docs/options_reference.rst`` is out of date.
- Source contracts (``dendrinet.project.json``) of the ``core`` and ``llm``
  lines must name ``library.maintained_ref``; the implicit default branches
  were removed. An optional ``library.release_bridge`` (``source_commit``,
  ``release_commit``) admits a release built by squashing: the pinned commit
  must be in ``source_commit``'s history, ``release_commit`` in the maintained
  ref's history, and both commits must have the same tree; the receipt records
  the bridge and the tree. The provenance utilities are documented on the new
  Project provenance page and in the API reference.
- ``analyze_routing_by_condition`` no longer fails after collecting routing
  (it read a field that a plain dictionary does not have, so it wrote no
  results); its levels are labelled with the built model's soma-relative
  depths and time constants, each drawn with its own marker and line style.
  The four recurrent condition analyses are documented in Analyzing saved
  checkpoints.
- Every option of the installed ``dendritic-*`` commands has help text; the
  command-line reference was regenerated.
- ``blocklinear_gradient_scaling: conductance_dynamic`` works on additive
  branch layers, as before 2026-10-08 when it was made to raise: a modified
  backward rule (not exact backpropagation) that divides the gradient passed
  to each child by its coupling weight (``|w| + epsilon`` for identity
  weights), with forward outputs and coupling-weight gradients unchanged and
  gradients identical to the earlier ones. Additive layers with ``relu``
  couplings raise, since a zero coupling has no inverse.
- Populations stepped through time with ``blocklinear_gradient_scaling:
  conductance_dynamic`` and no Top-K scaling (recurrent populations and
  layers, and populations with synapse types, spiking somas, dendritic spikes
  or soma feedback) raised at their first training step since 2026-10-09;
  each step's coupling gradients are again scaled with that step's scales,
  with gradients identical to those before.
- ``population_network.compile_step`` raises for shunting populations with
  ``conductance_dynamic`` gradient scaling, whose compiled steps failed at
  the first training step; additive populations keep it.
- ``implementation.compile_forward`` raises for shunting structured E/I cores
  with BlockLinear ``conductance_dynamic`` gradient scaling, whose compiled
  graph failed at the first training step; Top-K conductance scaling of
  shunting cores and the BlockLinear rule of additive cores still compile.
- Models can be deep-copied after a training forward: the caches modules keep
  for analyses (population outputs, routing information, branch currents and
  diagnostics, DeepST's per-forward weights) are copied without their autograd
  graph. ``copy.deepcopy`` raised for every population network after a
  training step. The model, its forward pass and numerics are unchanged.
- ``ExcitationInhibitionNetwork.branch_layers``, ``n_branch_layers`` and
  ``get_effective_params`` work with MLP inhibitory cells
  (``inhibitory_network_type: mlp``): the cells have no branch layers, and
  every parameter of the MLP counts as effective. They raised AttributeError.
- The compact-checkpoint documentation states that compact checkpoints do not
  record module versions, and that the contact-age warning shown when
  ``indexed_release`` layers load from them can be ignored.
- Compact checkpoints: the ``bitmask`` topology encoding keeps the contact
  order of layers whose updates depend on it (``indexed_release``, and
  ``indexed_rewire`` and ``indexed_dynamic`` layers that are not converted to
  fixed layers), so their ``impact``, ``contact_age`` and next training update
  survive the round trip; fixed layers are encoded as before. Files written
  earlier reordered only the weights and cannot be repaired; when one is
  loaded, the layers whose order matters are listed under
  ``contact_order_warnings``. ``compact_sparse_bitmask_state_dict`` raises
  instead of misaligning per-contact state unless ``keep_slot_order`` names
  the layers. ``dendritic-compact-checkpoint --report`` refuses an existing
  file before anything is written.
- ``dendritic-poisson-exposure-transfer`` reads the training run's
  ``poisson_generator`` options with the loader's defaults and accepts only
  training data whose multiplicative gain is exactly 1. A run trained with
  gains (for example ``max_gain_factor: 4`` with ``multiplicative_gain`` left
  out, which the loader enables) raises; an omitted ``poisson_sampling``,
  ``stimulus_duration``, ``base_dataset`` or block is read with its default,
  and the rebuilt test split is flattened as in training.
- Unknown entries of the ablation ``levels``, ``targets`` and ``metrics``,
  unknown ablation methods and clamp statistics, unknown correlation
  ``components``, ``pairs`` and ``scopes``, and unknown correlation
  ``computation_level`` or ``sampling_strategy`` values (including the
  never-implemented ``importance``) raise when the configuration loads,
  listing the valid names; they silently disabled the requested computations.
- ``analysis.runtime.<phase>.mode`` must be ``auto``, ``materialize`` or
  ``stream``; other values raise instead of running as ``auto``.
- Analyzer loggers are named ``dendritic_modeling.analysis.<name>`` (the
  information estimators ``dendritic_modeling.utils.information.estimators.<class>``),
  so their records reach the console and ``dendritic_modeling.log``; their
  INFO records were lost before.
- ``analysis.correlation.params.compute.batch_process`` (and its flat
  spelling) is retired: it was stored but never read. Configurations that set
  it load with a warning, and ``dendritic-upgrade-config`` removes it.
- The correlation ``stratified_categories`` entry ``across_layer``, a default
  since it was introduced but never implemented, is retired: the default is
  ``["within_branch", "within_layer"]`` (results are unchanged), a written
  ``across_layer`` loads with a warning and is dropped,
  ``dendritic-upgrade-config`` removes it, and any other unknown category
  raises.
- ``compartment_statistics`` leaves branches without inhibitory statistics out
  of its E/I ratio figures, and draws none for excitatory-only trees, instead
  of logging a KeyError; outputs of trees with both are unchanged.
- ``compartment_statistics`` with ``per_class_analysis`` writes
  ``per_class_analysis.png``: each class's mean input and activation per
  branch depth. The figure function existed but was never called, and its key
  parsing could not match the per-class statistics.
- Every analysis pass records the status of each requested analyzer
  (``completed``, ``skipped`` with its reason, ``failed`` with the error,
  ``empty``) in ``analysis_status.json``; ``dendritic-analyze-checkpoint``
  also stores it in its provenance, and ``AnalysisManager.run_analysis``
  returns the record. The opt-in ``analysis.require_completion: true`` raises
  ``AnalysisIncompleteError`` for an incomplete pass. Every collector of
  analyzer outputs (the ``dendritic-collect-*`` commands except
  ``-poisson-transfer``) rejects a run whose record (the pass in the run's
  ``analysis_status.json``, else the copy in its provenance) shows that an
  analyzer it collects did not complete, and with ``allow_incomplete`` lists it
  among the skipped jobs; ``dendritic-collect-sweep-metrics`` checks
  ``performance`` this way. Runs without a status record are collected as
  before. Result files are unchanged.
- Before a CUDA device evaluates a materialized split in one forward pass, the
  library bounds that pass's activation memory from below by probing a copy
  of the model on 2, 4 and 8 examples; a pass that certainly cannot fit is
  streamed in ``batch_size`` batches instead of first failing with an
  out-of-memory error. Passes that fit are unchanged. The analysis documentation lists the passes
  the bound covers, the analyzers that can still run out of memory on
  oversized splits, and what the probes cost.
- ``dendritic-submit-checkpoint-analysis`` checks account, partition and
  concurrency before it writes its ledger, and rejects a concurrency limit
  below 1 instead of ignoring 0; the path-matched intervention, path-matched
  information and branch-sensitivity collectors check the provenance
  ``schema_version``.

Corrections that change what a configuration builds or how it runs (review of
2026-10-11). Every shipped configuration and every flat core type builds a
byte-identical model (state, forward output and input gradient) before and
after them:

- Structured connectivity and connection masks: a population-network stream
  with a connection mask (a connection ``probability`` below 1,
  ``target_levels``, or ``allow_self_recurrence: false``) used that mask and
  ignored the ``structured_connectivity`` rule of its pathway. Both masks now
  restrict the stream. A combination that leaves a compartment without an
  allowed input raises, and so does a connection mask on a pathway whose rule
  places exact contacts (``spatial_morphology``, index manifests). Affects
  population networks that set both.
- Structured connectivity of later population-network layers: list-valued
  ``probability`` and ``indegree`` were indexed by the population's position
  in its layer, and seeded masks and index-manifest keys used that position
  alone, so populations at the same position in different layers read the same
  list entry and drew identical masks. Lists are now indexed by the network
  layer (new population field ``structured_network_layer_idx``), and layers
  after the first draw their own masks and read manifest entries
  ``<pathway>/network_layer<i>/layer<j>/level<k>``. The first layer is
  unchanged. Affects population networks with more than one layer and
  structured connectivity.
- Image cores in classifiers: ``hierarchical_dendritic_conv`` and
  ``sparse_active_matched_point_conv`` (and their aliases) received the
  number of elements of one sample as their channel count, so a classifier
  built from a configuration failed on its first image. They now take the
  channels of ``model.encoder.params.input_shape``, and their feature maps are
  flattened for the decoder (``FlattenedSpatialCore``). ``dendritic_alexnet``
  without ``vision.input_channels`` likewise takes the image channels. These
  cores need image samples (``data.processing.flatten: false``) and the
  identity encoder; anything else raises when the model is built.
- Strict determinism: under ``torch.use_deterministic_algorithms(True)``
  (``experiment.strict_deterministic``), ``indexed_projection_backend: auto``
  no longer selects the atomic ``triton_fused`` (layers with at least 100,000
  inputs run ``triton_transposed``), and it runs ``recompute`` where the
  deterministic input gradient of ``triton_transposed`` cannot run the shape.
  Explicitly selected backends raise when they would compute a gradient with
  atomic additions. The gradients of such runs varied from run to run before.
  Affects strict runs on CUDA only.
- ``dendritic_signed``, ``flat_signed``, ``dendritic_mlp``, ``flat_mlp`` and
  ``mlp``: the structured E/I builder always built additive branches with
  signed (identity) weights and ignored ``morphology.use_shunting`` and
  ``weight_transform``. Conflicting values now raise, and the typed
  configuration and ``config.json`` record the values built. The transformer
  feed-forward translations of these types
  (``build_dendritic_ffn_kwargs_from_core_config`` and
  ``build_ei_stack_kwargs_from_core_config``) built shunting cells with
  softplus weights for the signed types and additive softplus cells for the
  mlp types; they now build the same additive identity-weight cells. No
  shipped or paper configuration uses these types for feed-forward
  translation.

Public names removed compared with ``develop`` of 2026-07-21 (4102bf536), and
what replaces them:

- Transformer and vision replacement training is not part of this library:
  the modules ``training.transformer_replacement``,
  ``training.vision_replacement``, ``training.replacement_common`` and
  ``training._transformer_replacement`` with everything they exported, among
  them ``run_transformer_replacement_training``,
  ``run_vision_replacement_training``,
  ``benchmark_saved_transformer_replacement`` and their result classes; the
  commands ``dendritic-train-transformer-replacement``,
  ``dendritic-benchmark-transformer-replacement``,
  ``dendritic-benchmark-transformer-operator``,
  ``dendritic-assemble-transformer-replacements``,
  ``dendritic-compact-transformer-checkpoint`` and
  ``dendritic-profile-transformer-scaling`` (``dendritic-compact-checkpoint``
  exports DendriNet models); ``TokenReplacementCell``,
  ``LayerwiseTokenReplacementStack``,
  ``make_linear_output_projection``, ``GatedDendriticFFNReplacement`` and
  ``estimate_gated_dendritic_ffn_params``; the language-model sizing helpers
  of ``networks.architectures.transformer`` (``TransformerParameterSpec``,
  ``profile_dendritic_lm``, ``profile_dendritic_lm_config``,
  ``profile_transformer_component_options``,
  ``estimate_dendritic_replacement_from_config``,
  ``estimate_factorized_lm_head_params``,
  ``estimate_factorized_token_embedding_params``,
  ``estimate_indexed_attention_params``); and the configuration classes
  ``TransformerReplacementTrainingConfig``,
  ``VisionReplacementTrainingConfig``, ``PretrainedReplacementConfig``,
  ``ReplacementRegionConfig``, ``TrainabilityConfig`` and
  ``SegmentTrainabilityConfig``. The feed-forward construction utilities of
  ``networks.architectures.transformer`` remain.
- The validators of ``dendritic_modeling.utils`` (``validate_full_config``,
  ``validate_architecture_only``, ``validate_training_only``,
  ``get_validation_summary``, ``ModelArchitectureValidator``,
  ``validate_model_architecture``): every configuration is validated when it
  loads, E/I wiring when a core is built (``config.inhibitory_wiring``), and
  training settings by ``validate_training_stability``.
  ``apply_automatic_safeguards`` and ``suggest_optimal_config_for_model_size``
  changed nothing a run used.
- ``config.BranchConfig`` and ``RecurrentConfig``, of the recurrent cell stack
  removed on 2026-02-27 (``model.core.recurrent`` raises with the
  replacement); ``EncoderTrainingConfig`` and
  ``scripts/training/train_encoder_network.py`` (the ``training.encoder``
  phase, whose autoencoder pretraining failed on ``develop``; regular
  encoders train in the main phase, and ``training.encoder_pretraining``
  pretrains autoencoder encoders); ``DatasetParamsConfig``
  (``data.dataset_params`` holds plain per-dataset mappings); ``WandbConfig``.
- ``training.factory.SHUNTING_FEEDBACK_ALIGNMENT_MODES``: the strategies
  ``shunting_fa`` and ``shunting_dfa`` load as ``fa`` and ``dfa`` with a
  warning (``DEPRECATED_STRATEGY_ALIASES``).
- Code that no part of the library ran: ``config.training.ReportingConfig``
  (no configuration built it; ``PruningConfig`` holds the same pruning-report
  settings); ``analysis.tools.integrate_pruning_with_turnover_analysis``, the
  former pruning integration that never ran (the ``synapse_pruning``
  analyzer replaces it); ``training.fsdp_utils.create_fsdp_wrap_policy``
  (the FSDP setting ``auto_wrap_policy``, ``size`` or ``indexed_synapses``,
  selects the wrapping); and the ``safe_hook`` decorator of the dendritic
  gradient scaler.
- The analysis scripts ``rec_inhibition_impact.py``,
  ``stability_analysis.py`` and ``trace_decomposition_analysis.py``, which
  needed models of the removed recurrent stack and could not run on
  ``develop``; the analyzers ``jacobian_spectrum`` and
  ``state_transition_jacobian`` replace the stability analysis.
- ``setup.py`` (``pyproject.toml`` holds the packaging), ``.ruffignore``
  (``ruff.toml`` configures ruff, which never read it), ``docs/make.py`` and
  two stylesheets nothing linked (``docs/Makefile`` builds the docs), the
  ``projects/`` scaffolding, and the pages ``docs/fsdp_troubleshooting.rst``
  and ``docs/initialization_comparison_policy.rst``, merged into
  ``docs/distributed_training.rst`` and ``docs/dendritic_neurons.rst``.

Deprecated names kept for compatibility, to be removed in a later release:

- ``ShuntingFeedbackAlignmentTrainer`` (``training.strategies``) runs
  ``FeedbackAlignmentTrainer`` and warns, as the strategy names
  ``shunting_fa`` and ``shunting_dfa`` do. FA and DFA compute each block's
  exact local gradient, which in shunting branch layers includes each
  pathway's conductance and driving force, the terms the class approximated
  by scaling the feedback with ``R_tot * (e_rev_exc - V)``. Its ``e_rev_exc``
  argument has no counterpart, because the gradient uses the model's own
  reversal potentials, and raises.
- ``check_pid_available`` and ``pid_synergy_redundancy``
  (``utils.information``) raise ``ImportError``, as they always did, and now
  name the replacements: they imported ``PID`` from ``external``, which never
  provided it (the former ``external/`` directory held only ``BROJA_2PID``).
  ``partial_information_decomposition`` (two sources, a target and a
  mutual-information estimator) and ``mmi_pid`` (from three mutual
  informations) compute the decomposition.

Configuration and run records:

- ``config_version`` names the configuration format a file is written for
  (``CONFIG_VERSION = 2``, exported from ``dendritic_modeling.config``). A
  file without it loads with the current meaning, an earlier version raises
  with a pointer to ``dendritic-upgrade-config``, and a later one asks for a
  newer library. Shipped configurations and templates carry
  ``config_version: 2``, generated sweep configurations get it, and
  ``config.json`` records it.
- ``dendritic-upgrade-config`` is idempotent: it writes ``config_version: 2``
  into every file it upgrades and leaves files at the current version
  unchanged, so a second pass no longer reads keys the first pass removed as
  omitted (it zeroed the inhibitory populations of feedforward EINets whose
  ``input_mode: 0`` the first pass had removed). It writes the stochastic
  Top-K settings that omitted keys meant before 2026-10-08 (temperature 0 for
  ``sparsity.type: stochastic``; ``topk_temperature: 0.0`` or
  ``topk_ultrafast: false`` for population options; ``selection: standard``
  for indexed dynamic synapses whose stochastic selection at temperature 0
  was exactly standard) and reports the stochastic layer type in that case.
  It removes a top-level ``fsdp`` block identical to ``distributed.fsdp`` or
  with ``use_fsdp: false`` and reports any other. Copies keep their paths relative to the inputs' common
  directory, and two inputs that would be written to one file stop the tool
  before it writes (same-named files overwrote each other).
- What ``dendritic-upgrade-config`` cannot rewrite is listed in the upgraded
  file under ``unresolved_migration``. The loader refuses such a partly
  upgraded file, and a later upgrade of it rewrites nothing and reports the
  same items, until they are resolved by hand and the key is deleted (a
  second pass used to report a partly upgraded file as clean).
- A plugin that fails to import or register keeps failing: every later
  configuration load or registry lookup in the process raises its
  ``PluginLoadError`` again, because the plugin may have registered part of
  its components (a second load used to succeed without it).
- Every function that lists a registry's names loads the installed plugins
  first, as the trainer-strategy, activation and sparse-layer listings did:
  ``get_available_architectures`` and ``get_registered_architecture_names``,
  the analyzer, dataset, sequence-dataset, model-builder, additive-rule, loss
  and optimizer listings, and the configuration-section and migration
  listings, as well as ``apply_config_migrations``. A plugin's components
  were missing from them until a lookup had loaded it. ``get_additive_rule``
  and ``config_section_builder`` load plugins for a name they do not know,
  like the other lookups.
- A top-level ``fsdp`` block raises when the configuration loads. Next to a
  ``distributed`` section it was ignored with a warning.
- Unknown keys in ``synapse_types`` raise: a misspelt ``enabled`` left the
  block disabled.
- ``utils.reproducibility.alert_not_deterministic`` raises (or warns in
  PyTorch's ``warn_only`` mode) when an operation without a deterministic
  implementation runs under deterministic algorithms.
  ``nondeterministic_gradient_reason`` and
  ``NONDETERMINISTIC_INDEXED_BACKENDS`` in the indexed kernels package report
  which backends compute gradients with atomic additions at a shape. The
  atomic input-gradient fallback of the transposed kernels raises under
  deterministic algorithms.
- ``replace_transformer_mlp_layers`` and
  ``replace_transformer_mlp_layers_with_ei_stack`` check every target and
  build every replacement before they change the model, so an invalid layer
  index, path or replacement setting leaves it untouched (a failure past the
  first layer left earlier layers replaced), and they reject a repeated layer
  index (it rebuilt a replacement from a replacement). ``mlp_attr`` accepts a
  dotted path, and ``replacement_class`` any module with ``from_mlp``, as on
  ``develop``. Valid requests build the same replacements as before.
- The unit tests of ``IndexedRewireLinear`` and of the rewiring that
  ``CustomWeightDecayOptimizer`` advances are back in the suite.
- Autoencoder encoders train again: ``training.encoder_pretraining``
  (``EncoderPretrainingConfig``, off by default) trains the whole autoencoder
  to reconstruct the training inputs before the main phase, keeping the state
  with the lowest validation loss, and its encoder then feeds the core,
  frozen in evaluation mode unless ``freeze_encoder: false``. A trained
  autoencoder can come from ``model.initial_checkpoint`` instead. The
  retired ``training.encoder`` phase pretrained them through a separate script
  that failed on ``develop`` and replaced the model's encoder by an identity,
  so checkpoints could not read raw inputs; the encoder now stays in the
  model. Pretraining runs in single-process training; the library function
  is ``dendritic_modeling.training.encoder_pretraining.pretrain_autoencoder``.

- Plugins: installed packages extend DendriNet through the
  ``dendritic_modeling.plugins`` entry-point group (``dendritic_modeling.plugins``).
  A plugin registers configuration sections (``config.extensions``: the
  ``model``/``training`` replacement slots, ``training.strategies.<name>`` and
  ``analysis.<name>``), model builders that ``initialize_model`` uses
  (``models.builders``), trainer strategies, analyzers with a run phase, and
  components of the other registries. Plugins load before the first
  configuration and when a registry is asked for an unknown name;
  ``DENDRITIC_MODELING_PLUGINS=0`` turns automatic loading off, and a plugin that
  fails to load raises. ``runtime_environment.json`` records each loaded plugin
  with its version and checkout. Saved configurations keep plugin strategy and
  analysis settings under ``strategies.extensions`` and
  ``analysis.extensions``, which load again.
- Removed the hooks that served the private replacement line inside this
  package: the ``vision_distillation`` placeholder strategy and its special
  cases in the loader, factory and trainer configuration; the conditional
  transformer re-exports (the library's transformer replacement API is now
  always exported from ``networks`` and ``networks.architectures``);
  ``utils.platform_line`` (``is_missing_module`` moved to
  ``utils.optional_imports``); and ``config.private_features`` (now
  ``config.extensions``). Without the plugin that provides them, enabled
  replacement slots raise, and ``training.main.strategies.vision_distillation``
  and ``analysis.vision_boundary_diagnostics`` are unknown keys (the strategy
  settings used to load as inactive metadata).
- Additive integration rules from plugins:
  ``register_additive_rule(name, AdditiveRule(voltage, options, setup))``
  adds a rule that ``additive_mode: <name>`` selects, configured by the new
  ``additive_options`` mapping of ``model.core.morphology`` and of population
  options (default ``{}``; the built-in rules reject it). A rule receives the
  currents of the built-in rules and, in the forward pass, the layer's inputs;
  its ``setup`` creates its parameters right after the leak parameter. The
  built-in rules, their outputs, random streams and checkpoint keys are
  unchanged. ``IndexedSparseLinear.effective_sparse_weight()`` and
  ``project_with_weights(x, weight)`` project inputs through a layer's
  contacts with other per-contact weights.
- Every registry (architectures, sparse layers, activations, datasets,
  sequence datasets, losses, optimizers, analyzers and additive rules) loads
  plugins when asked for a name it does not know. ``register_sparse_layer``
  stores names in the normalized form that lookups use (a name with capitals
  was never found), and ``unregister_sparse_layer`` removes one. The test
  suite runs with built-in components only unless
  ``DENDRITIC_MODELING_PLUGINS`` is set. The guide "Extending DendriNet"
  describes plugins, their configuration sections, model builders,
  strategies, analyzers and additive rules.
- The backbone-splitting helpers of ``networks.architectures.classical.pretrained``
  (``load_backbone``, ``flatten_model_layers``, ``names_and_modules``,
  ``filter_omitted_layers``, ``resolve_split_input_size``,
  ``resolve_target_module_span``) and the spatial core-type sets of
  ``networks.architectures.factory_spatial`` (``SPATIAL_DENDRITIC_TYPES``,
  ``SPATIAL_DENDRITIC_CONV_TYPES``, ``SPATIAL_PATCH_POINT_TYPES``, now
  frozensets) are public; they had leading underscores.
- Configurations constructed in code (``Config()``, ``ModelConfig()``,
  ``TrainingConfig()``) hold the defaults of the plugin sections registered for
  their slots, like loaded configurations.
- ``register_trainer_strategy`` accepts a ``"module:Class"`` reference that
  is imported when the strategy is first used.
- Plugins can register configuration migrations
  (``config.extensions.register_config_migration``) that rewrite keys
  historical configurations wrote for their components. The loader applies
  them to every configuration and analysis profile before checking it, and
  logs each change; ``dendritic-upgrade-config`` writes their result.

- Compact export fails closed: topology freezing raises for sparse layers it
  cannot freeze losslessly (stochastic and variance Top-K, DeepST), which it
  left unconverted with their dense candidates, unless
  ``sparse_topology.allow_unconverted`` is true. ``--verify-full-model`` takes
  its input shape from the config (the recorded ``config.json`` holds the one
  training saw) instead of the encoder's default ``[3, 32, 32]``.
- A run writes into a new ``<run_name>_<timestamp>`` directory under
  ``outputs.results_dir`` unless ``outputs.exact_run_dir`` is true (generated
  sweep configs now set it) or ``results_dir`` ends in ``config_<i>`` (configs
  generated by earlier sweep versions). Any ``results_dir`` containing
  ``/config_``, such as ``results/config_tests``, used to write every run into
  the same directory. A run that finds its timestamped directory taken by a
  run that started in the same second writes to ``<name>_1`` (``_2``, ...)
  instead of sharing it.
- Recurrent structured additive cores no longer warn that their alias
  overrides reactivation settings it does not set.
- ``register_dataset`` refuses built-in dataset names, and
  ``register_sparse_layer`` and ``register_trainer_strategy`` refuse names
  already registered, unless ``allow_override=True``; they replaced the
  existing entry silently.
- ``AnalysisManager`` raises a clear error without ``save_root`` (it failed
  with a ``TypeError``), and ``run_checkpoint_analysis`` returns the run
  directory and the files the analyzers wrote instead of an always-empty
  ``results`` entry.
- ``dendritic-upgrade-config`` removes ``model.core`` sections the core type
  never reads (for example ``morphology`` on a population network), which the
  loader rejects.
- ``dendritic-upgrade-config`` writes copies of experiment and sweep YAMLs in
  the current form: top-level annotations move under ``metadata``, renamed
  fields take their new names (including dotted sweep override paths),
  retired keys holding their old default are removed, and retired keys that
  this version cannot reproduce are reported and left in place. The retired
  ``transfer.input_mode`` routing becomes explicit counts: a feedforward EINet
  with ``input_mode: 1``, the former default, built none of the inhibitory
  cells it declared (first-layer excitatory cells read the transferred
  stream), so their sizes become 0, and a recurrent EINet with
  ``input_mode: 0`` or ``inhibitory_mode: none`` gets the first-layer ``ie``
  count it ran with. The tool also removes settings the library ignored
  (``learned_output_scale`` next to an ``output_scale_mode``, encoder/decoder
  params their type does not read, population sparse-synapse options their
  synapse type does not read, additive-integration options of shunting
  layers, ``recurrent_ei`` settings of feedforward EINets, a learned
  output-scale mode without an identity decoder (written as ``fixed``),
  ``quantile`` gate updates without recalibration (written as ``frozen``),
  morphology values a core-type alias replaced, ``flatten``/``normalize:
  true`` for a
  dataset whose loader does not apply them, step options a strategy's
  training loop does not apply, population-network and unified E/I
  ``input_dim``, and a layer ``indexed_seed`` that its indexed populations
  replaced with the topology stream) and writes the synapse counts the
  stricter wiring checks require with the values that were used (and moves
  annotation blocks written inside ``analysis`` under ``metadata``, and gives
  sweep ``ei-sum``/``ei-ratio`` filters the tolerance generation used):
  ``rec_*_synapses: 0`` for delayed population connections without them, the
  default inhibitory population of a unified E/I layer without an
  ``inhibitory`` entry, one
  ``inhibitory_layer_sizes`` entry per layer (the recurrent EINet repeated the
  last entry, the feedforward EINet built no inhibitory cells past the end),
  and zero inhibitory counts in layers without inhibitory cells. In sweeps,
  such a rewrite of ``base_config`` is applied only when it rewrites every
  generated config the same way, and reported otherwise or when an
  ``extends`` parent may set what it depends on.
- ``architecture.inhibitory_layer_sizes`` accepts 0 for a layer without
  inhibitory cells, as the wiring check asks; loading rejected it.
- ``learning_strategy_config.allow_unmodeled_forward_features: true`` lets
  ``local_ca`` train cores with typed synapses, spiking, dendritic spikes,
  soma feedback or the ``parallel_readout``/``all_active_star`` cross-level
  modes, as a rule variant whose updates ignore those features; without it
  such cores raise, as since 2026-10-08. Likewise
  ``learning_strategy_config.allow_scalar_fallback: true`` lets the path-based
  error modes use the scalar error where a model has no path to one soma,
  as they did silently before 2026-10-08. ``dendritic-upgrade-config`` writes
  both for configs that ran this way.
- ``ss_mlp`` and ``ss_mlp_flat`` build when the final layer's excitatory
  cells have no ``ie`` synapses; they failed with an ``AttributeError``.
- Options a sparse-synapse type does not read raise when they would change
  the model: non-default ``indexed_rewire_*`` (outside ``indexed_rewire``),
  ``release_*`` (outside ``indexed_release``), ``indexed_candidate_size`` and
  ``indexed_selection`` (outside ``indexed_dynamic``), structured indexed
  support, and ``dense_to_sparse_*`` (outside ``dense_to_sparse``) settings
  were silently ignored. Indexed execution settings (chunk size, index dtype,
  workspace, projection backend, ...) under other types are logged once.
- A misspelled key in a population-network or unified E/I payload raises a
  ``ValueError`` naming its section (for example
  ``model.core.population_network.layers[0].populations[e]``) and the closest
  valid key, instead of a bare ``TypeError`` from the constructor.
- Indexed synapses sample their fixed topology from the seed, layer shape and
  connection constraints only. Masked sampling used the layer's
  ``output_chunk_size``, an execution setting, as its rejection-chunk size; it
  now always uses 2048 rows (that setting's default), so layers with a
  connection mask and a non-default ``output_chunk_size`` sample a different
  graph and all others are unchanged. Layers that do not save their indices
  (``persistent_indices: false``) save a ``topology_fingerprint`` buffer:
  loading a checkpoint onto a different regenerated graph raises, and a
  checkpoint written before fingerprints existed loads with a warning.
- Sweep analysis no longer averages different conditions as if they were
  seeds: the collector adds every config path that varies across the sweep
  (other than replicate seeds) to the condition columns, so runs that differ
  only in a path the curated extractor does not name (for example
  ``param_groups.topk_lr``) stay separate.
- Sweep files are checked when they load: unknown keys in ``slurm_config``,
  ``experiment_settings``, ``filter_config`` and ``outputs`` raise with a
  did-you-mean hint (``memory`` instead of ``mem`` was ignored), a top-level
  key that nearly matches a section name raises, and other top-level keys
  are reported as ignored (put annotations under ``metadata:``; ``x-`` keys
  hold YAML anchors). ``experiment_settings.use_array_jobs``, which was never
  read, selects the job layout unless the command line sets one.
- ``distributed.mode`` must be ``none``, ``ddp`` or ``fsdp`` (any case) when
  the config loads; other values, such as ``fdsp``, raise with a hint instead
  of training in one process. The FSDP ``sharding_strategy``,
  ``backward_prefetch`` and dtype fields are checked and normalized at load as
  well, so ``full_shard`` works instead of failing later with a bare
  ``KeyError``.
- Information-analysis selectors (``selection.components``, ``scopes``,
  ``views``, ``signal_variants`` and ``baselines.label_shuffle_null.components``)
  raise for entries the analyzer does not act on, instead of silently
  computing less: a misspelled component disabled every component, and null
  components without a null counterpart (``layer_proxies``,
  ``gaussian_fisher``, ``pid``) were ignored.
- Messages, docstrings and tests no longer name internal development branches
  or programs: a section or strategy of an optional extension that is not
  installed raises "requires an optional extension of dendritic_modeling that
  is not installed". Parameter-budget reports with an explicit readout record
  the factory that built it (``networks.utils.readouts.make_signed_readout``)
  instead of a module that does not exist in this library.
- Moved to the papers that own them: the connectome co-sourcing
  (``analysis.connectome``), recorded-population gain
  (``analysis.public_gain``) and structured-inhibition simulation
  (``analysis.structured_inhibition``) packages, which are independent
  statistical analyses and simulations of one study rather than DendriNet
  capabilities. The ``public-data`` extra they needed is gone. The same holds
  for the paired local-gain memory tasks (``datasets.local_gain``), the
  conductance-based branching teacher (``datasets.biophysical``), and the
  choice-strategy, cascade-response and linear-memory-curve helpers of
  ``analysis.utils``, and the summary script of one initializer factorial
  (``scripts.sweeps.summarize_initializer_components``). Runs that used them
  reproduce at the library revisions that ran them. The empty
  ``scripts.experiments`` and ``scripts.vision`` packages are removed.
- The contextual-remapping study configurations (``configs/contextual_remapping``)
  and their study-specific parameter-budget and seed-contract tests moved to
  the project that specified them. Three of them remain as feature examples
  with unchanged settings: ``configs/examples/population_network_contextual_remapping.yaml``,
  ``population_network_target_levels.yaml`` and
  ``contextual_remapping_affine_mlp.yaml``.
- ``scripts.benchmarks.benchmark_spatial_imagenet`` is
  ``scripts.benchmarks.benchmark_training_steps``: it profiles bounded
  training steps of any configured model on its dataset (it was named for one
  ImageNet model but never depended on it).
- FSDP runs with checkpointing keep the trainer's resumable final checkpoint
  (model, optimizer and epoch) as ``final_checkpoint.pt``. It was written to
  ``final_model.pt`` and then overwritten by the plain final weights.
- ``model.core.type: identity`` (or ``none``) builds a pass-through core.
  Encoder/decoder building blocks (CNN, autoencoder, router and
  ``effectivematchedparammlp`` types) raise a ``ValueError`` naming the
  supported alternatives when ``model.core`` selects them, instead of a
  ``TypeError`` from the constructor.
- Internal project records (migration and boundary notes, paper release and
  source-contract procedures, the ``projects/`` release scaffolding) are no
  longer part of the repository; they are kept with the project records. The
  parameter-budget families, signed readouts, readout initialization,
  ``StreamingCovariance`` and ``passive_star_soma_voltage`` they described are
  documented in ``docs/comparison_models.rst``.
- The ImageNet study configurations (``configs/vision``: dendritic AlexNet
  pilots and the raw-ImageNet spatial DendriNet runs) and their contract test
  moved to the project that runs them; the earlier files remain in the
  history at the revisions that ran them. The ``dendritic_alexnet`` and
  spatial core types are unchanged.
- ``final_model.pt`` is a hard link to the run's best-validation checkpoint
  when it holds the same weights (``load_best_state_dict: true`` restored
  them), instead of a second copy of the model; weights that differ, such as
  after post-training pruning, are saved as before, and so is a run on a
  filesystem without hard links.
- An empty ``data.base_dir`` stores datasets under ``$DENDRITIC_DATA_DIR`` or
  ``~/.cache/dendritic_modeling`` instead of each run's results directory
  (sequential MNIST and the MNIST pixel-sampling dataset used ``./data``), so
  runs share one locked download; runs record ``DENDRITIC_DATA_DIR`` and
  ``DENDRITIC_SCHEDULER_PROFILES`` in ``runtime_environment.json``.
- Sweep, rerun and checkpoint-analysis submission validate the Slurm
  account/partition pair against a site policy file
  (``DENDRITIC_SCHEDULER_PROFILES`` or
  ``~/.config/dendritic_modeling/scheduler_profiles.yaml``) instead of one
  site's pairs built into the library. Without a policy the pair is accepted
  and recorded as unvalidated; manifests record the approving policy file.
- Generated sweep jobs probe the mount points holding the absolute paths
  their configs name (found when the scripts are generated; the root
  filesystem is not probed) instead of the ``/n/<mount>`` roots of one cluster
  layout. ``scripts/train_fsdp.sh`` sets no partition, account or NCCL
  interface: pass them with ``sbatch`` options or the environment.
- The indexed kernels no longer read value-changing environment variables.
  The atomic input-gradient scatter of the transposed Triton kernel
  (``DENDRITIC_ATOMIC_GRADX=1``) is the backend ``triton_transposed_atomic``,
  so it appears in configurations and compiled plans. ``DENDRITIC_FROZEN_TRITON``
  is retired: frozen layers run the fused Triton kernel or GEMM layout they
  are configured with (they used the eager tiled gather whatever the backend)
  and keep the bounded tiles for ``eager``, ``recompute`` and ``auto``; their
  outputs change at fp32 reassociation level. The setup-cache limits remain
  memory-only environment settings, and invalid values raise ``ValueError``.
- Importing the package no longer configures logging or writes
  ``src/logs/dendritic_modeling.log`` inside the installed package (it failed
  on read-only installs and grew to hundreds of MB in checkouts): the package
  loggers carry a ``NullHandler``, and programs call ``setup_logging()``. Run
  directories still get ``dendritic_modeling.log``. The 14 installed
  collector and analysis commands that returned their result dicts exited with
  status 1 on success; they now exit 0 (``main`` still returns the results).
- Unknown configuration keys raise with a did-you-mean hint at every level.
  Retired keys still load with one warning, or raise where the retired value
  changed the model. Free-form records go under a top-level ``metadata:``.
- Renamed: the ``context_dependent_decision`` option ``variable_delay`` is
  ``variable_integration`` (it samples the integration length, not the delay);
  ignored ``integration_duration`` or ``min/max_integration`` values warn, and
  the examples that wrote an ignored ``integration_duration`` now state the
  sampled 20-80-step integration they always used.
- Renamed: the recurrent override ``strategies.recurrent.grad_clip`` is
  ``grad_clip_value``, the name of the common setting it overrides, and
  ``RecurrentTrainer`` takes ``grad_clip_value`` (its constructor default
  follows ``Trainer``, 2.0). The old name still loads with one warning.
- Renamed: ``weight_decay_rate`` is ``synapse_decay_rate``; the population
  ``topk/reactivation/blocklinear_strategy`` settings are
  ``*_gradient_scaling``; ``learned_output_scale`` is folded into
  ``output_scale_mode``; analysis sections use their analyzer names; the
  performance split is ``validation``. Old names load with one warning.
- One seed policy: ``experiment.seed`` roots the component streams, a seed
  written on a component wins, and runs record the resolved seeds and numerics.
- ``transfer.input_mode`` and its routing flags are retired: declared
  inhibitory cells decide the wiring for feedforward and recurrent EINets, and
  synapses without a source raise. ``biological_neuron`` is checked on the
  built model.
- ``train_valid_split: 1.0`` means no validation set for every dataset; model
  selection on an empty or test-set validation raises.
- ``experiment.strict_deterministic: true`` delivers loader batches in order
  (an unset ``in_order`` resolved to ``false`` for image datasets with workers)
  and rejects ``in_order: false``; training results record the resolved loader
  settings under ``dataloader_tuning``.
- An unknown ``data.dataset_name`` raises at once with the known names, instead
  of importing a ``data_generation`` package from wherever it is on
  ``sys.path``; datasets defined outside the library register with
  ``datasets.registry.register_dataset``. Every library synthetic dataset is
  selectable by name (``heterogeneous_information``,
  ``nonlinear_transformation``, ``asymmetric_noise`` and ``population_code``
  were not). ``multixor``, ``orientation_bars``, ``orthonet`` and
  ``random_flip_mnist``, which no core loader provides, left the docs and the
  reference configs, and tests that exercised a stand-in class were replaced.
- A written ``data.processing.normalize: true`` for a dataset that does not
  scale its inputs, or ``flatten: true`` for a sequence dataset that keeps its
  time axis, raises instead of being ignored. Registered datasets can declare
  the processing their builder applies.
- Structured-connectivity blocks (``connectivity.structured`` and population
  ``structured_connectivity``) reject unknown keys, pathway names and
  ``spatial`` keys, and an enabled block without any rule; a typo had left the
  pathways dense. ``mlp_inhibitory_network_params`` rejects unknown keys and
  fills unset ones from its defaults (``hidden_dims: [200]``, where a
  partial programmatic dict had built no hidden layer).
- A ``synapse_types`` block that lists receptor types without ``enabled``
  raises: it had made the layer step through time while its synapses stayed
  single-trace. Entries of a ``types`` list state their polarity (an entry
  without one had joined both polarities).
- A population-network connection that reaches no synapses because its
  count was left at the default (``rec_*_synapses`` is 0 unless written, so
  delayed edges were silently inert) raises and names the field; a count
  written as 0 stays a supported lesion arm. A connection whose only targets
  are soma contacts with ``somatic_synapses: false`` also raises.
- A cell with no synapse at any level (a point cell, ``branch_factors: []``,
  with ``somatic_synapses: false``) raises instead of building an input-free
  population. ``autograd_credit_mode: soma_broadcast`` raises outside the
  feedforward hierarchical rate forward, where its gradient replacement does
  not exist (those combinations had trained with exact gradients).
- Short-term plasticity of typed synapses is driven by the type's rectified
  drive (``stp.activity``: ``saturating`` r/(scale + r) by default, or
  ``clamped``), zero without input. The former proxy ``sigmoid(drive)`` was at
  least 0.5 at rest, so resources depleted to a few percent with no input and
  STP acted as a near-constant gain cut (the shipped GABA_B example kept about
  5% of its conductance at rest).
- A ``unified_ei`` layer with ``recurrent: false`` raises when its populations
  set recurrent synapse counts or stateful dynamics; the counts were silently
  not built and the dynamics ran as one step from a zero state.
- Population networks warn once when the first batch feeds signed external
  input into shunting populations with positive weights (EINet raises for the
  same case). Recurrent task inputs such as the signed evidence of
  ``context_dependent_decision`` use this deliberately. The seven population
  examples that z-scored image inputs into shunting cells now load pixels in
  [0, 1] (``data.processing.normalize: false``), as positive conductances
  require.

Initialization:

- ``analytical_expectation`` couples a shunting compartment whose excitation
  reaches its inhibition plus leak (for example one without inhibition) to its
  children at the midpoint voltage instead of a coupling of 1e-6, which
  disconnected it.
- Adaptive scaling multiplies each bank's expected conductance by ``exp(-f)``
  for every weight transform (unchanged for ``exp``); ``relu`` banks keep
  their active contacts. A flat cell (``branch_factors: []``) counts as one
  branch, not 100.
- Gate ``init_m``/``init_b`` written under a policy other than ``fixed`` warn.
- The integration/gate pairing warning (``warn_integration_gate_pairing``,
  formerly ``warn_untrainable_init_integration_pairing``) states its mechanism
  instead of citing one experiment, and every branch layer checks its own
  settings (only feedforward EINets were checked, so population and recurrent
  cells never were). It warns for shunting with signed weights, and for
  positive-weight additive drive (``raw`` or ``conductance_normalized``) whose
  gate slope or midpoint is not calibrated on measured voltages
  (``analytical``, ``fixed`` and the two mixed policies). ``empirical``
  calibrates both and no longer warns; z-scored ``raw`` voltage is not
  flagged.
- Keyword arguments that contradict a typed ``config``, ``synapse_config`` or
  ``branch_config`` raise in ``DendriNet``, ``DendriticBranchLayer`` and
  ``DendriNetConfig.from_kwargs`` instead of being silently discarded.
- Branch layers reject contradictory integration options instead of ignoring
  them: a shunting layer with a non-raw ``additive_mode`` or the additive
  z-score, and tangent anchors outside ``tangent_matched``.
  Conv2d converters write kernels through the softplus inverse.

Training rules:

- Population networks take an opt-in ``compile_step`` that compiles each
  layer's per-timestep step with ``torch.compile``; a recurrent RNN-paper rate
  model trained 3.3x faster per step (about 700 to 210 ms on an A100). The
  default is unchanged.
- ``output_scale_mode: global`` stores its learned scale as a one-element
  vector instead of a 0-d tensor, which FSDP cannot shard (the FSDP entry
  point rejected these models). Outputs are unchanged, and checkpoints with
  the 0-d scale load as before.
- The FSDP trainer trains the registry loss (``training.main.common.
  loss_function``) like single-process training, instead of mapping the names
  ``ce``/``cat_nll`` to cross-entropy and ``mse`` to MSE itself and raising
  for any other; ``cat_nll`` runs now use the categorical NLL, as on one GPU.
  It receives ``model.task``, reports accuracy only for ``[batch, classes]``
  logits (NaN for regression, where argmax accuracy could exceed 1), refuses
  the recurrent trainer's sequence losses and class losses on sequence
  outputs, and takes every option from the typed configuration (its fallback
  defaults had drifted, for example ``grad_clip_value`` 5.0).
- ``cat_nll`` no longer validates its arguments on every call, which read the
  labels back to the host each step; its values are bitwise unchanged.
- ``train_experiments`` no longer sets TensorFlow environment variables when
  it is imported.
- FSDP checkpoints (``save_fsdp_checkpoint``) store every rank's random
  generators, and ``load_fsdp_checkpoint`` restores them: training resumed
  from a checkpoint drew new dropout masks and stochastic, variance or
  release synapse selections, so it diverged from the uninterrupted run. A
  checkpoint from a different number of ranks, or one without generator
  state, raises unless ``restore_random_state=False``.
- Under FSDP and DDP, variance Top-K input moments and the release synapse's
  forward impact and source activity were updated from each rank's own batch,
  so the ranks selected synapses from different statistics and could prune
  and regrow different contacts (DDP replaced the buffers with rank 0's before
  each forward, discarding the other ranks' batches). These synapses now
  reduce their batch statistics over the ranks: every rank keeps the
  statistics of the global batch, as one process on the combined batch would.
  Single-process training is unchanged.
- FA and DFA use the configured loss gradient and exact block-local
  derivatives (hidden nonlinearity, weight transform, Top-K masks, shunting
  driving forces); ``shunting_fa``/``shunting_dfa`` run as ``fa``/``dfa``.
- LocalCA: voltage homeostasis moves voltages toward its target; the output
  error follows the loss class (exact BCE for probability outputs); gate
  parameters receive exact gradients; per-soma and path errors reach only the
  readout tree; HSIC follows the weight transform and compact indexed banks;
  ``additive_gain_mode: learned_gain`` (a constant 0.5) is retired; ``g_tot``
  is the forward denominator (leak and recurrent conductances included,
  conductance terms no longer rectified). Models without branch
  layers, recurrent unrolls, unsupported population features, row weight
  normalization, STDP on non-TopK banks, reversal potentials other than the
  model's (1 and 0) and a forced ``dynamics_mode`` that disagrees with a
  layer raise; ``three_factor.allow_mismatched_dynamics_mode`` keeps the
  forced mode as an explicit rule variant.
- DeepST regrows and initializes relu and identity contacts at positive
  weights: relu contacts regrown at ``log(weight_threshold)`` had weight 0 and
  no gradient, so live contacts could only decrease. exp and softplus are
  unchanged. Its documentation now states that ``sigma`` perturbs only the
  prune test.
- ``credit_deepst`` keeps candidates until they mature, so swapping continues
  at the default cadence. Stochastic Top-K has one default (temperature 0.5,
  ultrafast) at every entry point. Shunting soma feedback needs a nonzero
  initial projection.
- The E/I weight-ratio regularizer validates its scope and metric, no longer
  swallows per-layer errors, and raises when enabled on a model without a
  layer that has both feed-forward excitation and inhibition (it returned a
  constant zero). Its statistics are unchanged; ``global`` is documented as
  the per-layer ratio averaged over layers.
- MuonH leaves matrices that start at zero norm outside its hyperball, with a
  warning; the projection had held them at zero (for example coupling
  log-weights initialized to 0).
- ``frozen_sparse_state()`` keeps auxiliary forwards from advancing
  dense-to-sparse schedules, variance statistics and release impact; the
  KL trainer's no-grad target pass uses it (it advanced them a second time
  per step). FSDP gradient checkpointing rejects models with such synapses,
  whose recomputation would compute gradients for a different mask.
- The post-training pruning step builds its trainer from the trainer config as
  it was before the main trainer consumed it, so multi-stage and recurrent
  runs keep their strategy settings and task when pruning.
- FSDP runs load ``model.initial_checkpoint`` and calibrate data-driven gates
  whether or not analyses run.
- ``distributed.mode`` is checked against the launch: ``fsdp`` under
  ``train_experiments`` (which had trained with DDP or on one GPU) and ``ddp``
  without ``torchrun`` raise, and ``config.json`` records the runtime used.
  ``scripts/train_fsdp.sh`` runs the checkout it is submitted from (it always
  ran the shared main checkout), takes Python from ``PYTHON_BIN`` or ``PATH``
  and requires a config.
- Sweep job scripts render their Slurm header and environment from one
  ``SlurmProfile``: array jobs now request ``slurm_config.constraint``, an
  integer ``time`` means seconds for every job type (sequential scripts wrote it
  raw, which Slurm reads as minutes), digit-only time strings raise, and the
  sequential default training script path exists. ``rerun_missing_configs``
  decides completion from ``performance/final.json`` and
  ``training_summary.json`` (it waited for a log line no run writes, so every
  rerun repeated the whole sweep) and writes the sweep's own array script for
  the unfinished configs, bound to the sweep's recorded source state.
- Sweep generation creates a new, absolute sweep directory each time (the
  timestamp was fixed when the manager was built, so a second sweep reused and
  overwrote the first one's configs); ``--split-jobs N`` writes at most ``N``
  sequential jobs covering every config (it was ignored for multi-config sweeps,
  and the API dropped the remainder configs); submission stops at a failed
  ``sbatch`` and records job IDs in ``jobs/submitted_jobs.tsv`` (it printed
  success regardless); ``--output-dir`` applies to submitted sweeps and
  ``--analyze`` runs the result analyzer. Run by path, ``sweep_manager.py``
  imports its own checkout; importing the module no longer edits ``sys.path``.
- Source identities fail when git fails or times out in a checkout instead of
  writing unguarded job scripts, include a digest of untracked files under
  ``src/`` (checked by every job script), detect checkouts only at the library
  root (an installed package inside another repository is not a checkout),
  raise for missing source files, and the sweep manifest records a digest of
  the imported package's files in place of a hand-picked file list.
- ``submit_checkpoint_analysis`` runs the checkout it validated (a relative
  ``repository`` was resolved against the working directory at submission),
  and each job checks that it imports ``dendritic_modeling`` from it.
  ``paired_comparison`` hashes the modules it imported (it failed for an
  installed package) and offers ``--contrast-operation``.
- Training runs write ``runtime_environment.json``: package path, checkout
  commit and source digests, Python and PyTorch builds, numerical dependency
  versions, GPUs and thread settings (``utils.runtime_environment``; the source
  identity helpers moved to ``utils.source_identity`` and run git without
  taking optional locks).
- ``setup_logging`` also routes the running program's ``__main__`` logger to
  its handlers, so the training scripts' own INFO lines (configuration loading,
  calibration, completion) are no longer dropped when they run as scripts or
  with ``python -m``/``torchrun``, and they reach the run's
  ``dendritic_modeling.log``. A run's ``train.log`` handler replaces the
  previous run's in the same process instead of accumulating.
- ``dendritic-analyze --verbose`` prints debug messages. It lowered the level
  of the root logger only, while the console handler that ``setup_logging``
  installs stayed at INFO.
- The information analysis ``"pid"`` component computes a partial information
  decomposition: the minimum-mutual-information measure on the configured
  estimator's ``I(S1;T)``, ``I(S2;T)`` and ``I(S1,S2;T)``, reporting
  redundancy, unique terms, synergy and the joint information
  (``utils.information.pid``). It was always switched off, and its helper
  imported a nonexistent ``external`` module. ``pid.binarize`` now defaults to
  false and never thresholds the class labels (a median split merged classes).
- The ``copula`` information estimator is the Gaussian-copula estimator of
  Ince et al. (2017) for scalars and vectors: normal scores of ranks,
  Gaussian log-determinants with the analytic bias correction, and class
  labels through class-conditional entropies. It used Spearman's rho in the
  Gaussian formula (0.79 instead of 0.83 nats at r = 0.9), returned 0 for
  |rho| >= 0.999, ordered class labels numerically after adding unseeded
  noise, gave H(C) = 0 for 32 or more classes (zeroing every normalized
  metric) and inserted ``<repo>/external`` into ``sys.path``. Entropy is the
  plug-in value for discrete variables and raises for continuous ones;
  ``copula.type`` is ``gaussian`` or ``kernel`` (``dvc`` and other values
  raise), and the kernel copula takes two continuous scalars. No shipped or
  paper config selects this estimator.
- Checkpoint version rules (``networks/state_versions.py``): the library's
  loaders keep the state dict's per-module version metadata (they rebuilt plain
  dicts, so every module loaded as legacy and a current file missing its
  calibration priors or ``fixed_b_buffer`` loaded silently with constructed
  values). Entries a file predates are kept from construction, warned and
  returned in the load records' ``legacy_state``; a current file lacking them
  fails strict loading. DeepST, indexed sparse/rewire and indexed-release
  modules are now version 2 (counter, mask and seed were accepted as missing
  from any file); a kept mask reports the loaded contacts it disallows, and
  version-1 release files note that their contact ages may count training
  forwards. ``compare_state_dict_keys`` separates keys a newer module version
  added from missing keys, and ``compare_develop_numerics`` lists such keys
  instead of failing parity on them.
- Trainer checkpoints (best, per-epoch, final, FSDP, multi-stage) and index
  manifests are written atomically and flushed to disk (a plain ``torch.save``
  truncated ``best_model.pt`` before rewriting it, so an interrupted save lost
  the previous best); atomic writes no longer create owner-only files on shared
  storage. Multi-stage checkpoints store their stage configuration and results
  as plain containers and are read with the weights-only loader (their pickled
  dataclass could not be loaded under torch's weights-only default).
- Cross-commit goldens (``tests/golden_cases.py``,
  ``tests/fixtures/default_path_goldens.json``,
  ``tests/test_default_path_goldens.py``) record the initial state digest,
  outputs and input gradients of twelve default-path models: the structured
  EINet families, indexed eager and recompute backends, shunting and additive
  population networks, a unified recurrent rollout and the affine MLP control.
  The earlier "golden" EINet tests compared two builds in one process. Against
  the pre-audit core 18b2a53e6 every case is bitwise identical except the two
  shunting cases with excitation-only compartments, which change only at the
  midpoint-coupling repair (d3c2c233c). ``golden_cases.py --write --reason``
  is the only way to regenerate the fixture and records why. Sixteen more
  cases cover branched trees ([4], [3,3], reduced density), a two-layer E-I
  EINet, every sparse-synapse family, a reduced-density population network,
  an RNN-paper-style recurrent rate-mode population (level taus,
  mechanism-neutral initialization), the spatial conv and tensor-map cores
  and the typed ``DendriNetConfig`` path; the fixture history records, from
  bisection, which branch commit changed each case relative to 18b2a53e6.
- Every YAML under ``configs/`` is tested with the loader of its kind
  (experiment, analysis profile, sweep with ``extends`` and strict loading of
  each generated config, compression policy, epoch selection, operator
  preset), and every experiment config builds and runs forward unless the test
  that builds it on its dataset's input is named; the smoke test no longer
  builds the default model for analysis-only files. The YAML examples in the
  documentation load through the strict loaders: the recurrent
  population-network examples now include their populations and connections,
  and the multi-stage example sits under ``training.main.strategies``.
  ``select_validation_epochs`` configs reject unknown keys.
- Test hygiene: pytest no longer hides every UserWarning and
  DeprecationWarning; trainers use the ``torch.amp`` autocast and GradScaler
  API (the ``torch.cuda.amp`` forms are deprecated); single-sample variances in
  the weight and compartment statistics are an explicit NaN. Process-spawning
  tests carry the ``slow`` and ``multiprocess`` markers, the unused ``unit``,
  ``integration`` and ``requires_data`` markers are gone, and the large
  memory-estimator cases use meta tensors. Removed vacuous tests (an empty
  timing loop, self-tests of mock classes, a doctest of modules without
  examples, a permanently skipped test, a commented-out block) and the
  unittest runner that ran 13% of the suite while reporting success; the
  stochastic Top-K tests assert their selection statistics and the spatial
  split asserts its specs instead of skipping.
- FSDP runtime: mixed precision defaults to bf16, and fp16 parameters or
  reductions without loss scaling (``use_amp``) raise (fp16 gradients
  underflowed silently); ``limit_all_gathers`` and ``forward_prefetch`` apply
  as written (``reduce_communication_overhead`` gated them and is retired);
  ``auto_wrap_policy: size`` uses PyTorch's size-based policy with
  ``min_num_params``, which nothing read before (writing it for another policy
  raises); ``cpu_init`` is retired because the model always reaches FSDP on the
  CPU (the code that read it never ran). The training loop no longer empties
  the CUDA cache before every batch, wrapping no longer overrides the
  configured sharding strategy of E/I networks above 10^9 parameters, and the
  FSDP entry point no longer sets NCCL and TensorFlow environment variables at
  import (``scripts/train_fsdp.sh`` sets the cluster's NCCL network settings).
- Distributed runs keep one reactivation state: under DDP, data-driven
  calibration (which every rank ran after wrapping) is followed by a broadcast
  of rank 0's gate parameters and buffers; under FSDP, the calibrated
  ``fixed_b`` floats the forward reads follow the buffers that
  ``sync_module_states`` broadcast (other ranks kept their constructed value).
- ``IndexedReleaseLinear`` refuses to run while a ``rank0_broadcast`` topology
  awaits its broadcast, like the other indexed layers (it computed outputs from
  all-zero indices).
- Trainers check ``loss_function`` against ``model.task``: classification
  losses (``ce``, ``cat_nll``, ``bce``) on a regression model raise
  (cross-entropy treated regression targets as soft class probabilities, or
  was identically zero for one output); the recurrent
  trainer keeps its documented task-loss substitution, and the FSDP entry
  point checks the same rule. Trainers of the standard family raise when a
  recurrent model's batches carry sequence lengths they would drop (the model
  ran at the padded length); use ``strategy: recurrent``.
- Packaging: torch 2.8 is the floor (compact checkpoints need the unsigned
  dtypes of 2.3, ``weights_only`` loading is safe from 2.6; the unexplained
  2.6.0 exclusion is gone); plotly, psutil, python-dateutil and tensorboard,
  which nothing imports, are no longer dependencies; h5py moved with fsspec
  and aiohttp to the ``public-data`` extra (the NWB reader raises an
  ImportError naming it), and the ``gpu``, ``plotting`` and ``transformers``
  extras declare Triton, networkx and transformers. ``networks/utils``,
  ``scripts/training`` and ``scripts/tools`` are regular packages, discovery
  ships only ``dendritic_modeling``, the license is an SPDX expression and the
  build needs setuptools 77; ``setup.py`` is gone.
- ``setup/constraints/core-py310-torch2.9.1.txt`` pins the stack the core suite
  passes on, written by ``setup/freeze_constraints.py``; ``setup/environment.yml``
  provides Python 3.10 and pip, and the install guides install through the
  constraints file, as does the Python 3.10 CI job (the 3.11 and 3.12 jobs
  keep testing the newest releases). Black and isort format, ruff lints: the unused ruff
  sections in ``pyproject.toml``, ``.ruffignore`` and inert ruff settings are
  gone, and the contributing guide no longer prescribes ``ruff format``.
- With ``load_best_state_dict: true``, a run in which no epoch had a finite
  validation loss raises instead of restoring the untrained weights and
  reporting best epoch 1. LocalCA restores its best state into the unwrapped
  model, so DDP runs no longer fail when loading it, and returns the base and
  regularization loss curves like the other trainers.
- ``train_experiments_fsdp`` sets runs up with the functions of
  ``train_experiments``: command-line overrides (so sweep launchers' ``--output_dir``
  works), ``validate_training_stability``, ``setup_environment`` (run
  directory, seeds, datasets with the same data root, input width from a sample
  and written to ``config.json``, model, warm start) and the run records. Its
  own model assembly had built recurrent classification cores as
  ``Classifier``, which adds a learned per-class output scale with an identity
  decoder; they are ``RecurrentClassifier`` as in single-process training.
  Autoencoder encoders raise instead of training from random weights. Every run
  saves a plain-state_dict ``final_model.pt`` (default runs saved no weights)
  and ``training_summary.json`` (was ``training_results.json``); the final
  analysis no longer gathers optimizer state, runs after the FSDP model is
  released, and raises on failure instead of logging it.
- FSDP builds its optimizer groups with ``BaseModel.get_param_groups``, so a
  configuration trains every parameter at the same rate as in single-process
  and DDP training. Its own rule had sent the gates without a trainable ``b``
  (``param_tanh_only_m``, ``param_linear_sigmoid``, ``param_linear_tanh``,
  ``param_linear_tanh_transition``, ``residual_silu``) and the learned output
  scales to ``lr`` instead of ``reactivation_lr``/``decoder_lr``, synapses
  and gates in the encoder or decoder to ``topk_lr``/``reactivation_lr``,
  and ignored ``encoder_lr``; ``configs/config_exp_fsdp.yaml`` groups exactly
  as before. The trainer takes ``optimizer_config={"optimizer": ...,
  "param_groups": ...}``. FSDP rejects MuonH unless parameters stay unsharded
  2-D matrices (``NO_SHARD`` with ``use_orig_params``; sharded units gave
  MuonH only 1-D views, so it ran AdamW), split groups over flat parameters
  (``use_orig_params: false``), and scalar parameters such as the
  ``output_scale_mode: global`` scale; the group-size count no longer fails
  under ``cpu_offload``.
- Loss names fall back to a case-insensitive match (``MSE`` is ``mse``), and
  unknown names suggest the closest registered one. ``presentation_stream``
  resolves ``loss_function`` through the registry, so cross-entropy aliases
  such as ``cross_entropy`` are accepted. ``LossFunction`` declares the
  ``temporal_event_loss`` and ``sequence_prediction_loss`` flags the
  recurrent trainer reads.
- ``distributed.timeout_seconds`` sets the collective timeout of DDP and
  FSDP runs (null keeps 600 and 1800 seconds). The entry points now load the
  configuration before creating the process group; the undocumented
  ``DENDRITIC_FSDP_PROCESS_GROUP_TIMEOUT_SECONDS`` variable is gone. A
  training-time analysis on rank 0 longer than the timeout had aborted DDP
  runs through the NCCL watchdog with no setting to raise it.
- ``training.state.capture_epoch_state`` records the trainer's LR schedule
  and ``restore_epoch_state`` continues it; a checkpoint and a trainer that
  differ in having one raise (a resumed cosine run had restarted or dropped
  its schedule). Checkpoints without a schedule keep their layout.
- Fewer host synchronizations and duplicate kernels, with bitwise-identical
  outputs and gradients (checked on CPU and GPU against the previous code):
  variance Top-K updates its statistics without reading its update counter
  on the host every training forward; ``TransferLayer`` activates an input
  shared by both streams once (identity and ReLU, whose gradients are then
  bitwise the same); the shunting input check reads one minimum for both
  streams; LIF, plateau and typed-synapse decays are filled on the device
  instead of copied from the host every timestep; and Top-K layers locate
  their forbidden (row, input) pairs once, so masking them no longer needs
  boolean indexing and host reads on every mask computation.
  ``DenseToSparseLinear`` keeps host mirrors of its schedule step and frozen
  flag (read back only after an outside change such as a state load) and
  builds its mask with ``scatter_``, so a forward reads no GPU scalar.
  The standard training loop keeps its per-epoch loss sums on the device
  (float64, in step order) instead of reading every step's losses on the
  host; loss histories and weights match bitwise on CPU and GPU, with and
  without AMP and regularization. ``local_ca`` and ``soma_dfa`` do the same
  for their logging loss; ``local_ca`` also picks the near-zero error
  fallback on the device and reads each rho and phi statistic once (one read
  for the batch moments or noisy samples instead of one per value), with the
  same Python-float arithmetic; under DDP its gradient presence check reads
  once. Loss histories and weights match bitwise on CPU and GPU for the
  pearson, dot, ema, ema+augment and augment rho estimators, the variance and
  conditional phi modes, and ``soma_dfa``. Dense Top-K banks with a
  deterministic selection decay the synapses their training forward selected
  instead of running a second top-k in the same step (stochastic, variance
  and dense-to-sparse selectors and selection noise still select again, as
  does a bank whose ``pre_w`` changed after the forward); weights and losses
  match bitwise on CPU and GPU for the exp, softplus, identity and relu
  transforms, with and without boosting, forbidden inputs and connection
  masks. The ``recurrent``, ``fa`` and ``dfa`` epochs also keep their loss
  sums on the device (bitwise-identical histories and weights on CPU and GPU,
  with and without regularization and recurrent AMP). DeepST layers keep host
  mirrors of their rewiring step and of whether the rewired-slot record
  changed since it was consumed, so optimizer steps between rewiring events
  (``rewire_frequency > 1``) read nothing from the GPU; training matches
  bitwise for the global, local and constant-branch modes.
- ``runtime_environment.json`` records the indexed-synapse kernel switches
  ``DENDRITIC_FROZEN_TRITON`` and ``DENDRITIC_ATOMIC_GRADX`` (they change
  values) and ``DENDRITIC_GX_SETUP_CACHE_MAX``/``_BYTES`` (memory), so a run
  that set them can be identified. Typed options for them are pending.
- Removed from ``DendriNet.forward``: assigning each level's inputs to its
  ``excitatory_input``/``inhibitory_input`` attributes, which branch layers
  never define, so the assignment never ran.
- ``BlockLinear`` and ``EfficientBlockLinear`` keep their last input and
  weight tensors only in forward-hook gradient-scaling mode, which a branch
  layer's ``set_forward_dynamic_grad_scaling`` now also sets on its
  aggregator (as it does on its synapses). Every forward had pinned the
  child activation and the weight graph until the next one, outside
  activation checkpointing. Registering per-forward scales on a forward that
  was not recorded raises. Gradients are bitwise unchanged.
- Data-driven gate calibration reuses each layer's finite float64 voltage
  sample for its quantiles and occupancy fit instead of filtering and
  concatenating it again for each (two full copies per statistic); the
  calibrated gates and diagnostics are bitwise unchanged.
- ``fa`` and ``dfa`` training losses are now averaged across DDP ranks, as
  in the other trainers (each rank had logged its own mean). The FSDP
  trainer keeps its loss and accuracy sums on the device and reduces them
  in float64: the float32 reduction had rounded the sample-weighted loss sum
  (the validation loss that selects the best epoch included) and, beyond
  2**24 samples, the counts.
- The ``analytical_expectation`` and ``ei_equivalence`` initializers raise a
  ``ValueError`` naming the pathway when a branch layer expects no contacts
  per branch (DeepST rounds ``target_density x in_features`` down, so the
  default density 0.1 gives none for fewer than ten inputs); they had failed
  with a ``ZeroDivisionError``.
- In the ``parallel_readout`` and ``all_active_star`` controls,
  ``conductance_dynamic`` synapse gradient scaling at the soma uses the
  soma's pooled denominator (its local terms plus the projection or arm
  conductances), in the feedforward and recurrent paths; it had used the
  local denominator, so the controls scaled soma gradients differently from
  the hierarchy they match. Other levels and configurations without
  ``conductance_dynamic`` are unchanged; no configuration uses the
  combination.
- Analysis rule overrides (``_analysis_rule_override``) raise on additive
  layers and in ``voltage_from_currents`` (recurrent and stateful paths and
  analyses that substitute currents), which had silently ignored them; they
  still apply to the feedforward shunting forward pass.
- ``voltage_from_currents`` (and ``shunt_from_currents``) count non-zero scalar
  ``trace_E_rec``, ``trace_I`` and ``trace_I_rec`` in the shunting
  denominator (they had entered the numerator only), and a shunting layer
  given a child-branch current raises when ``branch_conductance`` is omitted
  instead of dropping the coupling conductance (voltages of up to 18 had
  resulted).
  The library's callers pass tensors and the conductance, so their results
  are unchanged.
- Branch diagnostics record the compartment ``leak``, and
  ``calibrate_additive_operating_point`` rejects diagnostics of a reference
  whose leak is not 1: the ``tangent_matched`` control expands the unit-leak
  rule ``N / (1 + T + epsilon)``, so an anchor from a leak-0.03 reference had
  been tangent to a different function (value off by about 2x). The
  configuration guide gives the shunting rule as
  ``N / (leak + T + epsilon)``. No shipped configuration calibrates from a
  leaky reference.
- The ``analytical_expectation``, ``ei_equivalence``, ``naive`` and ``default``
  initializers and the centre-preserving conductance cap use the layer's
  leak where they had a literal 1: shunting gate centres are
  ``E / (E + I + leak)`` and child couplings balance ``E + I + leak``. With
  ``leak_conductance: 0.03`` the analytical centre had been placed for unit
  leak (0.25 instead of 0.485 at ``E = I = 0.5``). Initialization at the
  default leak of 1 is byte-identical (the cross-commit goldens are
  unchanged); no shipped configuration combines another leak with these
  initializers.
- Reactivation dynamics records every call of each gate in the forward
  pass (each timestep of a recurrent unroll; it had kept only the first) and
  tracks every parametric gate, not only ``param_tanh``: snapshots name the
  gate and its number of calls, summarize ``log_m``, the midpoint (``b`` or
  ``fixed_b``) and any other gate parameter, and report rail occupancy for
  the positive-tanh gates, whose outputs lie in ``(0, 1)`` (other gates
  report output moments and range; the collector checks occupancy where it
  is reported). Sampling and forward failures propagate instead of
  producing an empty snapshot. Snapshots of feedforward ``param_tanh`` models
  keep their values and gain the ``gate`` and ``n_calls`` fields.
- The dendritic time-trace cascade decays each level by
  ``exp(-dt / tau)`` with the population's ``dt`` (it had assumed ``dt = 1``;
  results at ``dt = 1`` are unchanged and the step is recorded), and it skips
  populations whose gates are not ``param_tanh`` with a warning: its
  ``0.5 * (1 + tanh(m (V - b)))`` gate had been applied to any gate with
  ``log_m`` and ``b`` parameters (ReLU, endpoint and linear gates included).
- ``DendriticBranchLayer.output_from_currents`` (and
  ``branch_dynamics.compute_branch_output_from_currents``) gives a layer's
  output for given currents under its own voltage rule, signal rounding and
  gate. The information projection and the component receptive fields use
  it instead of their own copies of the equation. Component RFs had assumed
  unit leak and raw or z-scored additive integration (with the epsilon added
  to the standard deviation instead of bounding it); conditioning a model
  without inhibitory input on its excitation now reproduces the soma output
  for every rule and leak. RFs of unit-leak shunting and raw additive models
  are unchanged.
- The information analysis's projected signals (``compute_lda_weights`` and
  the ``lda`` and ``random`` signal variants) apply each layer's own voltage
  rule (``compute_branch_voltage_from_currents``, in float64): its leak, its
  additive mode and, for a shunting layer with child branches, the child
  coupling conductance in the denominator. The copy it used had put the child
  current in the shunting denominator, assumed unit leak and raw additive
  integration, and ignored signal rounding, so projected outputs of shunting
  layers with child branches, non-unit leaks and normalized or tangent
  additive layers were wrong; recorded currents now reproduce every layer's
  output. ``compute_branch_voltage_from_currents(..., record_diagnostics=False)``
  leaves a layer's stored diagnostics untouched. Results that used these
  projections (``compute_lda_weights: true``) change.
- The ``alexnet`` core requests torchvision's ImageNet weights through the
  ``weights`` API instead of the deprecated ``pretrained`` flag (the loaded
  weights are identical), and its module docstring describes the class it
  defines.
- ``InputNetTransform`` builds its MLP input network again: a shared trunk
  (``hidden_dims``) with excitatory and inhibitory heads
  (``excitatory_hidden_dims``/``excitatory_dim`` and their inhibitory
  counterparts), or two independent MLPs with ``independent_heads``. The
  MLP path had failed at construction (``set_readout`` no longer exists), so
  ``network_type: MLP``, the default, could not be built. Removed, as they
  had no reachable function: the ``FeatureExtractorMLP`` type (its module
  does not exist), the ``AlexNetTransform`` and ``CNNInputTransform`` types
  (they only raised ``NotImplementedError``) with their ``input_shape``,
  kernel-size and stride arguments, the unused ``output_dim`` argument, and
  the ``pytorch`` weight source, which loaded into a missing attribute.
  Options that the index split would ignore now raise. Removed with them:
  ``training.transfer_mlp_weights_to_input_net`` and
  ``training.extract_weights_from_simple_mlp``, which required layer classes
  (``MLPExcInhLayer``, ``SimpleMLP``) that no longer exist and were tested
  only with mocks.
- Tests classify every one of the 63 registered architecture types into a
  family (a new type fails until it is classified), check that each alias
  builds the same network as its canonical type (CNN, tensor-map, spatial
  conv, router, identity, baseline RNN, CTRNN and unified E-I aliases), and
  run every autoencoder, which had no test. The integration-rule by gate
  contract (every branch gets the configured rule, with the exact voltage
  equation and gradients) now also covers the spatial conv and tensor-map
  builders, and StatefulDendriNet's rejection of leak, endpoint gates,
  release synapses and signal rounding outside feedforward hierarchical rate
  populations is tested in every context.
- GPU tests carry the ``requires_gpu`` marker (tests parametrized with a CUDA
  ``device`` get it automatically) and are skipped on hosts without CUDA, so
  ``pytest -m requires_gpu`` selects all 65 of them on a GPU node; the ad-hoc
  CUDA ``skipif`` decorators and in-test skips are gone, and the FSDP trainer
  test no longer writes a ``checkpoints`` directory into the working tree.
- The pre-training stability check warns when an FSDP ``FULL_SHARD`` run
  turns off ``limit_all_gathers`` or ``forward_prefetch``; it had read the
  retired ``reduce_communication_overhead`` key and could never warn. Its
  NaN-risk advice names the existing clipping and NaN-check settings instead
  of a non-existent adaptive clipping mode.
- ``pin_memory: null`` now pins host batches (and copies them without
  blocking) whenever the trainer trains on a GPU, as documented. Loaders were
  built before the trainer resolved its device, so the setting had resolved
  against the CPU in the standard, ``local_ca``, two-step and other trainers
  that build loaders first. Values and batch order are unchanged. The
  recurrent, FA/DFA and FSDP loops now also copy pinned batches without
  blocking (they had synchronized the stream on every batch copy).
- Standard and recurrent validation keep their batch and sample-weighted
  loss sums on the device (float64, read once per validation pass);
  validation losses match bitwise on CPU and GPU, locally and under DDP.
  Recurrent DDP validation without samples returns NaN, as the standard
  trainer does (it had returned 0).
- ``training.<phase>.common.best_state_device: cpu`` keeps the best-epoch
  weights in host memory instead of a second copy on the GPU, and
  ``save_best_checkpoint: end | never`` writes ``<prefix>best_model.pt`` once
  at the end of the phase or not at all (default: on every improvement, as
  before). The pruning step no longer copies the model to track a best state,
  and ``presentation_stream`` restores its best state through the shared rule
  (it raises when no epoch was selected).
- ``presentation_stream`` checks loss and gradient finiteness with one host
  read per step (one per parameter tensor before) and keeps its loss, penalty
  and accuracy sums on the device until the end of the epoch; metrics rows and
  weights are bitwise unchanged, and a non-finite loss or gradient still stops
  the run before the optimizer step with the same message.
- Importing ``dendritic_modeling.analysis.metrics`` no longer sets
  ``TORCHDYNAMO_DISABLE`` for the process (and its children), which had made
  every later ``torch.compile`` call a silent no-op.
  ``implementation.compile_forward`` compiles through ``Module.compile``:
  it no longer changes dynamo's process-wide recompile settings or swallows
  compilation errors, and copies or pickles of the network drop the compiled
  call instead of running the original module's forward.
- GPU memory estimates count fp32 parameters, gradients and optimizer states
  (training keeps fp32 master weights, also under AMP) for the configured
  optimizer, and fp16 activations only under ``use_amp``; every estimate had
  assumed fp16 storage and Adam, about half the real memory. The smallest
  sufficient GPU is recommended first (an A100 before an H100), and the
  estimator builds a typed core as training does, so every EINet alias type
  (``dendritic_shunting``, ``flat_additive`` and the others) can be
  estimated; they had raised.
- ``scripts.tools.generate_slurm_job`` and ``submit_fsdp_job`` read the
  current configuration schema (they read the removed ``task`` and ``train``
  sections and failed on every shipped config). They launch the entry point
  ``distributed.mode`` selects (``torchrun`` with one process per GPU for
  ``ddp`` and ``fsdp``, one GPU otherwise), take the FSDP GPU count from the
  memory estimate, require an explicit account and partition, render through
  the sweeps' ``SlurmProfile``, and no longer export debugging settings
  (``CUDA_LAUNCH_BLOCKING``, blocking NCCL waits) or fixed cluster limits.
  ``gpu_estimation.estimate_config`` returns the estimator with the model,
  batch size and input shape it resolved.
- FSDP checkpoints gather the optimizer state on rank 0 only (every rank had
  kept a full CPU copy at each save; the written file is unchanged). A failed
  optimizer-state gather or remapping raises instead of saving or loading a
  rank-local shard, which had given every rank rank 0's optimizer moments.
- FSDP: ``distributed.fsdp.buffer_dtype: keep`` leaves buffers in their own
  dtype (every other setting, including null, casts floating buffers for the
  whole run with no full-precision copy). ``training_summary.json`` records
  the resolved mixed-precision dtypes and the recurrent decay buffers whose
  factors the compute dtype rounds to exactly 1, and the entry point warns
  about those (in bf16, levels with tau above about 511 steps stop
  responding to input).
- Under FSDP, ``strategy: recurrent`` raises for ``strategies.recurrent``
  overrides it would drop: an LR schedule or warmup, or ``use_amp`` or
  ``grad_clip`` values that differ from the ``training.main.common`` values
  FSDP applies (they had been ignored); unset (null) overrides are accepted.
- ``presentation_stream`` follows the common warmup rule: a positive
  ``lr_warmup_epochs`` needs ``lr_schedule: cosine`` (it applied a
  warmup-only ramp), and negative values raise. ``lr_schedule: cosine`` with
  the default ``lr_terminal_multiplier: 1.0`` raises, since the rate would
  never change; the shipped stream configs set 0.01. The trainer declares
  that it applies its own schedule instead of being exempted by name.
- Multi-stage training uses one configuration class for loading and running
  (``MultiStageTrainingConfig`` is now an alias of
  ``MultiStageStrategyConfig``); stages are typed ``TrainingStageConfig``
  entries checked when the configuration loads. ``final_state:
  last_stage | best_stage`` replaces ``use_best_from_each_stage`` (the old key
  loads, ``true`` as ``best_stage``) and defaults to ``last_stage``: the run
  had reloaded the stage with the lowest validation loss after all stages,
  silently discarding later stages and comparing losses of different
  objectives. ``best_stage`` now requires every stage to use the same loss
  function and no ``presentation_stream`` stage. ``stage_name`` names the
  stage's directory and log entries (it was ignored); every stage writes its
  trainer artifacts and ``stage_checkpoint.pt`` into its own created
  directory (stage trainers had written into a directory nothing created,
  or into one shared directory that later stages overwrote). ``epochs``,
  ``batch_size``, ``param_groups`` and ``local_rule_config`` inside a stage's
  ``trainer_config`` raise (the stage-level value had silently won). The
  results name the ``final_stage`` and its ``best_loss`` and ``best_epoch``,
  and ``final_state_dict`` is a CPU copy of the final weights (it had shared
  storage with the live parameters). Removed: the default two-stage
  ``stages`` list, which could never run, and ``global_batch_size``,
  ``global_shuffle`` and ``global_grad_clip_value``, which no configuration
  could set and which the common values always overrode.
- Multi-phase trainers (``freeze_layers``, ``freeze_branches``,
  ``freeze_branch_kl``, the warm-up strategies) number epochs across their
  phases. Each phase had restarted the count, so later phases overwrote the
  per-epoch analysis files and checkpoints of earlier ones, ``best_epoch``
  could name an epoch of another phase, and logs showed totals such as
  ``epoch=15/10``. Epoch numbers now index the concatenated loss histories,
  and the results list the phases (``freeze_layers``: pretraining, layers,
  final tuning).
- Two-step training (``freeze_branches``, ``freeze_branch_kl``) pretrains
  each layer on its input in the model's own forward: stage 1 ran the core
  layers one by one on the raw input, skipping the encoder, the transfer
  layer (normalization, pathway split), the inhibitory stream and population
  routing. The new ``forward_until(x, layer_index)`` of
  ``ExcitationInhibitionNetwork`` and feed-forward ``PopulationNetwork``
  returns one layer's output as ``forward`` computes it; other cores raise.
  Stage 1 trains in training mode and validates like a standard epoch (it
  validated in whatever mode the model was in, averaging batch means), the
  ``freeze_branch_kl`` teacher runs in eval mode, local heads are scored with
  the configured loss (the KL variant hard-coded cross-entropy) and sized
  from the layers' real outputs (an unknown width fell back to 784 inputs),
  and the result is selected among stage-2 epochs (stage 2 inherited the
  stage-1 best loss and could end on a stage-1 state). The two trainers share
  one stage-1 loop. ``phases`` names both stages, and the stage-1 histories
  fill all four loss curves.
- ``strategy: ei_equilibrium`` selects the E/I equilibrium warm-up
  (``ExcitationInhibitionEquilibriumTrainer``), which existed without a
  registry entry or configuration section; ``training.main.strategies.
  ei_equilibrium`` takes ``pretrain_epochs``, ``freeze_reactivation`` and
  ``loss_metric``.
- Warm-up strategies: ``homeostatic_control`` with ``stabilize_mode:
  gate_occupancy`` optimizes the gate-occupancy objective (it pulled gate
  outputs toward ``target_voltage`` by MSE and ignored ``target_saturation``);
  other modes are unchanged. ``pretrain_epochs: 0`` raises (it ran one warm-up
  epoch), as do unknown ``stabilize_mode`` values and a ``target_saturation``
  outside [0, 1]. A warm-up loss that finds no layer to read raises
  (``homeostatic_control`` needs inhibitory synapses; its E/I term was zero
  without them), and the warm-up hooks record activations only during the
  loss's own forward (calibration forwards before the first loss had raised
  AttributeError, analysis forwards accumulated activations). The results
  keep all four loss histories aligned and name the phases, and the main
  phase selects its best state among its own epochs instead of falling back
  to a warm-up state.
- Runs record their data-parallel layout: ``training_summary.json`` has
  ``data_parallel`` (world size, per-rank and global batch size, training
  steps per epoch) and ``runtime_environment.json`` the process group, so a
  run directory tells a one-GPU run from a four-GPU one; ``batch_size`` is
  documented as per process.
- ``reactivation_update_mode: quantile`` without
  ``recalibrate_reactivation_every > 0`` raises (it warned and kept the gates
  fixed, which is ``frozen``). The update and recalibration modes are checked
  when the configuration loads and when a trainer is built, by one
  normalizer that the FSDP entry point also uses; their former alias
  spellings load with a deprecation warning. A recalibration start epoch
  below 1, fewer than one recalibration batch or an EMA factor outside
  ``(0, 1]`` raise instead of being clamped.
- ``training.main.learning_strategy_config`` reaches only the strategies
  that read it: ``local_ca``, and ``multi_stage``, which hands it to its
  ``local_ca`` stages that set none of their own. With another strategy the
  loader warns that it has no effect (trainers had begun to reject the
  forwarded block, so ``configs/config_exp.yaml`` and multi-stage runs with a
  top-level block failed to start), and sweep summaries no longer record its
  rule parameters for such runs (backpropagation runs from the template were
  tabulated with ``rule_variant: 3f``).
- Staged trainers keep fixed parameters fixed. ``freeze_layers``,
  ``freeze_branches`` and ``freeze_branch_kl`` made every parameter of the
  layer being trained, and afterwards of the whole model, trainable, so the
  slopes of fixed ``linear_sigmoid``/``linear_tanh`` gates, reactivation
  parameters under ``reactivation_update_mode: quantile``/``frozen`` and
  frozen encoders were trained. A parameter that does not require gradients
  when a phase starts now stays fixed (the ``freeze_branches`` local
  optimizers leave it out), and every flag returns to its value before the
  phase, also when the phase fails. ``freeze_layers`` raises instead of
  starting final tuning from the untrained weights when no pretraining or
  layer epoch had a finite validation loss. ``TrainOnlyMReactivation`` froze
  parameters named ``log_b``, which no gate has, and made every other
  parameter trainable; it now holds the midpoints ``b`` of the trainable gates
  fixed, restores the flags afterwards and raises for models without such
  gates.
- Options that change which parameters train after ``DistributedDataParallel``
  has wrapped the model raise before the first step; DDP's reducer had stopped
  such runs at the second step. ``freeze_layers`` and ``multi_stage`` reject
  distributed training, as two-step and ``presentation_stream`` training
  already did. ``reactivation_update_mode: quantile``/``frozen`` raises for the
  trainers whose gradients DDP reduces (``standard``, ``recurrent``,
  ``soma_dfa``; ``fa``, ``dfa`` and ``local_ca`` average their gradients
  themselves), and so do the ``voltage_stabilization`` and
  ``homeostatic_control`` warm-ups, whose loss gives the decoder no gradient.

Analysis:

- ``stable_seed`` prints NumPy scalars as NumPy 1 does (``0``, not
  ``np.int64(0)``). Under NumPy 2 an identity holding NumPy scalars, such as
  the pandas group keys of integer, float or boolean columns, had given
  another seed than the same values under NumPy 1, so
  ``summarize_checkpoint_metrics``, ``paired_comparison`` and the
  inhibitory-routing comparison drew other bootstrap resamples from identical
  inputs (accuracy intervals moved by up to 1e-4). Seeds under NumPy 1 are
  unchanged, and under NumPy 2 they now equal them: summaries computed under
  NumPy 1 are reproduced byte for byte from their inputs under NumPy 2.2.6.
- Removed ``scripts/analysis/stability_analysis.py``. It was written for the
  recurrent cell API that the March 2026 recurrent implementation replaced
  (``recurrent.state.RecurrentState``, ``cell.branches``) and could not be
  imported. The ``jacobian_spectrum`` and ``state_transition_jacobian``
  analyzers compute the Jacobian spectra of the current recurrent models.
- The receptive-field ``ei_ratio`` class summary is the ratio of the class
  means of excitation and |inhibition|, not the class mean of per-sample
  ratios, which samples with near-zero inhibition dominated; per-sample
  records divide each sample's mean excitation by its mean |inhibition|.
  Results that report the ``ei_ratio`` summary change.
- An undefined ROC-AUC (one class present) is NaN instead of 0.5, which read
  as chance level: training evaluation of a single-class set reports NaN, and
  ``roc_auc_score`` raises when the labels hold one class and returns NaN for
  single-class rows of a batch. Binary labels other than {0, 1} (for example
  {1, 2} or {-1, 1}) take the greater label as positive; they scored 0.5.
- Branch rule overrides are a typed analysis API: ``BranchRuleOverride``
  applied with ``override_branch_rules({layer: override})`` or
  ``DendriticBranchLayer.override_branch_rule``, replacing private
  ``_analysis_*`` attributes. Each rule takes exactly the tensors it reads,
  checked when the override is built; layers are checked on entry and restored
  on exit. A rule string written on a layer raises instead of being read.
  Renamed: ``learned_tangent`` is ``anchor_tangent`` (nothing is learned) and
  ``sample_shuffled_*`` are ``sample_rolled_*`` (a one-sample roll, not a
  shuffle). ``frozen_denominator`` and ``sample_rolled_denominator`` no longer
  need unused anchors, and ``external_denominator`` takes the conductance of
  each forward pass instead of consuming a stored sequence. Outputs of the
  rules that existed are bitwise unchanged.
- all_branch sample aggregation pairs rows with their own labels; lesions use
  each bank's transform; branch sensitivities use the layer's leak; synaptic
  correlation uses the realized synapses; AUC ranks class probabilities; the
  LDA comparison decodes through the model; information-analysis failures
  raise and unmeasured metrics are NaN; compartment SNR captures inputs per
  collection; reactivation collection reports soma-relative depth; resource
  accounting handles stochastic selectors; sweep aggregation never averages
  different conditions as seeds; sweep information columns take the
  analyzer's soma-relative depths (never key order), name the dendritic tree
  when a model has several, raise on column collisions and count layers
  without a depth; weight columns also appear as ``*_depth<N>``.
- Local-rule components label inhibitory populations from the model, use the
  trainer's gate slopes and total conductance, record indexed synapse banks,
  map errors through the decoder of frozen models, add ``error_mode: bce``,
  and reject the layers, modes and reversal potentials the trainer rejects.
- Synapse turnover measures, per synapse bank of every branch layer
  (PopulationNetwork included), the contacts that stay, form and disappear
  between snapshots, instead of per-layer averages estimated from the
  configuration; snapshots use the deterministic eval-mode masks and keep only
  two reference masks. It needs ``training: true`` (now its default); its
  never-read ``synaptic_analysis``/``save_synaptic_report`` are retired.
- Synapse pruning analysis (``analysis.synapse_pruning``) is a registered,
  final-only analyzer: a census of realized contacts (selected by the mask
  and with nonzero effective weight) for every synapse bank of every branch
  layer, and, when the run prunes, the census before and after the pruning
  step with the contacts removed per layer and pathway. Its per-branch and
  per-depth plots cover any branch-layer model. The previous integration
  never ran: it looked for an analyzer attribute that did not exist.
- Inhibitory specialization and E/I connectivity motifs share one targeting
  computation: level fractions of realized weight mass (the estimand of the
  public summary helper, unchanged), with disconnected pairs excluded. The
  analyzer JSON previously divided each level by its compartment count, and
  motifs counted disconnected pairs as balanced; the per-compartment view is
  reported as ``per_compartment_control``.
- State-transition linearizations (singular values, named-state responses)
  evaluate indexed projections on the portable eager kernel: the Triton
  kernels' backward is not differentiable, so Jacobian-vector products through
  them were wrong (they failed the adjoint test). Every linearization now
  checks ``<u, Jv> = <J^T u, v>`` and raises on a mismatch.
- Plot series no longer differ by colour alone: ``plotting.styles`` gives
  network categories, depths, data splits and metrics their own Okabe-Ito
  colour, marker and line style, used by the information, ablation, SNR,
  performance and lesion plots (depths beyond 9 had shared one colour).
- The recurrent condition scripts (routing by condition, level lesions, OOD
  delay) rebuild a run through ``setup_environment``, with its resolved seeds
  and dataset parameters, instead of regenerating data from builder defaults.
  Level lesions silence a level through the population's native mask (zeroing
  its raw parameters had saturated it under exp and softplus transforms), are
  labelled by soma-relative depth and the model's time constants, and their
  figures are written (sorting crashed before). OOD-delay trials are seeded
  and take the run's trial shape.
- Compartment SNR computes the multivariate discriminability in float64 on
  the whitened non-null space, and counts a direction as significant only
  when it beats a seeded label-shuffle null; label-free activity previously
  reported every one of the C - 1 directions as significant. It reports the
  null mean of ``d_total`` and ``D / n``, since in-sample ``d_total`` grows
  with layer width.
- Gain/load perturbation records the fraction of values its clamp moved in
  every cell, warns when the clean inputs already fall outside the clamp, and
  states in ``load_definition`` that clipping rectifies the load (results
  unchanged).
- Multiplicative gain validates its options, requires a score when enabled,
  raises instead of returning nothing when no requested score fits the model,
  rejects options a Poisson-generator dataset would ignore, and draws every
  gain level's random gains and noise from ``noise_seed``.

Earlier this month:

- Restored general indexed Triton execution kernels to core, including recurrent
  workloads. The direct-constructor default remains eager; explicit `auto`
  configurations can select restored GPU kernels and require renewed validation
  before replacing a paper's historical source pin.
- Teacher-student training and the transformer- and vision-replacement
  training workflows are not part of this library. The compact checkpoint
  export of DendriNet models (``dendritic-compact-checkpoint``) is.
- Removed the transformer-replacement training entry points, examples and
  benchmark configs; the transformer feed-forward construction utilities
  (``networks.architectures.transformer``) remain.
- Removed the Weights & Biases dependency. A ``wandb:`` block that does not
  enable logging loads with a warning; one that enables logging raises.
- Added a cached ImageFolder index and a checkpoint evaluation CLI.
- Added an opt-in NVTX range helper and a per-family component
  performance-matrix profiler.

### 2026-08

- Added one indexed-projection backend selector across dendritic, recurrent,
  and degree-grouped construction paths. Portable eager and
  recompute modes remain available without accelerator dependencies; optional
  Triton and architecture-aware CUDA kernels are loaded only when selected.
- Improved the portable indexed backward with compact index reuse and fp32
  accumulation for reduced-precision input gradients.
- Added opt-in memory-efficient positive-tanh autograd and fused shunt
  reactivation, with eager fallback whenever diagnostics require voltage.

### 2026-07

- Scaled indexed recurrent DendriNets to one million neurons with FSDP and
  added reproducible single-node and strong-scaling benchmark controls.
- Expanded checkpoint-bound empirical analysis for representation access,
  reactivation operating states, synaptic support, perturbations, and
  provenance-checked result collection.
- Added deterministic `spatial_morphology` connectivity for image DendriNets,
  including feedforward populations in the `population_network` schema.
  Dendritic paths recursively partition a `(C, H, W)` input and store compact
  fixed sparse indices without a dense branch-by-pixel mask or convolutional
  weight sharing.
- Sweep collection now records the effective structured-connectivity policy and
  a canonical `synaptic_routing` label, so global-indexed and
  morphology-aligned runs can be summarized without array-index inference.
- Added a raw-ImageNet `[3, 3, 3, 3]` DendriNet experiment with no learned
  vision backbone. Its 61,103,392 trainable parameters closely match AlexNet,
  and its fixed connectivity can be regenerated instead of stored in
  checkpoints.
- Added a bounded H200 profiler for the raw-ImageNet model, including exact
  optimizer groups, CUDA memory, input-pipeline throughput, early-learning
  windows, and held-out evaluation. The training entrypoint also accepts a
  main-epoch override for bounded canary runs.
- Made post-training resource accounting operate directly on compact indexed
  connectivity, avoiding dense compatibility matrices for large sparse models.
- Added compact fixed-index LocalCA gradients and explicit DDP averaging for
  manually assigned local-rule gradients, enabling large spatial DendriNets to
  use 3F/4F/5F credit assignment without dense branch-by-input updates.
- Reused recorded LocalCA predictions for pointwise training losses, removing a
  redundant full-model forward pass while preserving model-dependent fallbacks.
- Aggregated LocalCA training-loss totals across DDP ranks so distributed
  learning curves report the full epoch rather than one rank's data shard.
- Added LocalCA input-wait and step-time profiling and nonblocking pinned-memory
  transfers through the shared training data path.
- Made LocalCA decoder recording, soma-error mapping, and decoder updates unwrap
  DDP models, so distributed local training updates the configured readout.
- Added bounded ImageNet sample, optimizer-group learning-rate, and LocalCA
  error-routing CLI overrides for reproducible large-model canaries and tuning
  runs.
- Corrected the raw-ImageNet LocalCA reference to use ancestry-shared soma
  feedback at every dendritic stage. The legacy ``per_soma`` mode is
  width-matched with scalar fallback; ``per_soma_shared`` explicitly repeats
  each soma coordinate only along its own subtree.
- Added a seed-controlled shuffled-ancestry LocalCA feedback control that
  preserves feedback bandwidth and values while assigning them to the wrong
  soma subtrees. Learning-rate experiment overrides now accept zero so a
  decoder-only control can freeze every core optimizer group exactly.
- Added an explicit soma-level DFA baseline for dendritic classifiers. It uses
  a fixed random output-to-soma projection, an exact supervised decoder
  update, and exact differentiation only within each dendritic core. This
  avoids applying the legacy dense-layer DFA implementation to compact indexed
  synapses.
- Corrected exact path-transport feedback for raw additive dendrites to use
  unit parent voltage gain; conductance-based dendrites continue to use the
  exact input-resistance factor $1/G_{\mathrm{tot}}$.
- Expanded frozen sweep provenance to hash all model, synapse, and local-rule
  implementation files used by prospective experiments, and added matched
  peak CUDA-memory fields to post-training resource summaries.

### 2026-06

- Added population-network configs for feedforward, recurrent, spiking, synapse-type, and heterogeneous-cell examples.
- Standardized additive aliases and occupancy-quantile reactivation initialization.
- Added recurrent TopK weight caching and dense-to-sparse sparsity examples.
- Switched the population examples to `indexed_rewire` for
  memory-efficient sparse storage with adaptive same-dendrite synapse turnover;
  older sparse modes remain available by setting `topk_type` explicitly.
- Removed tracked cluster-specific `jobs/` artifacts; generated jobs/manifests
  are local output.
- Consolidated population-network config validation and warning-only legacy alias handling.
- Refactored local-learning and information-analysis helper modules.
- Verified feedforward and population-format numerical behavior against develop and pre-cleanup baselines.

### 2025-08

- Added regularization and pruning support.
- Expanded information analysis and training optimization options.
- Added computation-level controls and multivariate aggregation options.
- Added encoder-network caching and split encoder/main trainer flow.

### 2025-07

- Refactored codebase structure.
- Added SLURM generation for FSDP and distributed training.
- Added the newer sweep workflow.

## 0.1.0 (2025-03, tag ``pre-refactor``)

### 2025-03

- Pre-refactor version tagged as `pre-refactor` (package version 0.1.0).

### 2025-02

- Added analysis, branch ablation, and information tools.

### 2024-11

- Initial EI-layer and DendriNet codebase structure.
