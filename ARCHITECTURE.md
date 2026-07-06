# DendriNet — Architecture Walkthrough

> A bottom-up map of the `dendritic_modeling` package: start at the leaf utilities
> everything imports, climb to the network cores, then to training, analysis, and
> the scripts that orchestrate a full run. Built interactively; map-level detail
> (purpose + key symbols + dependencies per file), not line-by-line.

## What this repo is

**DendriNet** is a PyTorch research framework for **biologically-inspired dendritic
excitation–inhibition (E/I) neural networks**: models whose "neurons" represent real
dendritic trees with named E/I cell populations (pyramidal, PV, SST), shunting/divisive
inhibition, sparse synaptic connectivity, spiking somas, and soma→dendrite feedback.
These cores are used both as standalone classifiers/regressors and as drop-in
replacements for the FC / transformer-MLP layers of pretrained networks.

Scale: ~102k lines across ~190 Python files in `src/dendritic_modeling/`.

## The dependency ladder (reading order)

| Layer | Folder(s) | Role |
|------|-----------|------|
| 0 · Foundation | `utils/`, `config/` | Pure helpers + config schema/validation |
| 1 · Core | `networks/` | activations → synapses → layers → architectures |
| 2 · Models | `models/` | Wrap networks into classifiers/regressors |
| 3 · Data | `datasets/` | Standard/synthetic/sequence datasets + augmentation |
| 4 · Training | `training/` | Trainers, strategies, losses, optimizers, FSDP/DDP |
| 5 · Introspection | `analysis/`, `plotting/` | Analyzers, metrics, visualizations |
| 6 · Orchestration | `scripts/`, `configs/`, `jobs/` | CLI entry points, sweeps, Slurm launchers |

The defining trait of Layer 0: it imports almost nothing from inside the package.
Every layer above imports *down* into it. Understand this layer and you know the
vocabulary the rest of the codebase speaks.

---

## Layer 0 — Foundation

Two folders: `utils/` (computational helpers) and `config/` (the typed config system).

### `utils/` — pure helpers (a toolbox, not a pipeline)

Grouped by job rather than alphabetically. Four independent clusters.

**Group A — General glue (touched everywhere)**
- `general.py` — most-imported file in the repo. `set_seed()` (reproducibility entry
  point), `Shaper` (reshape tensors between flat and structured neuron→branch→compartment
  layouts), `convert_to_serializable()` / `save_dict()` (tensors/numpy/OmegaConf → JSON-safe).
- `stable_hash.py` — `stable_seed_offset()`: BLAKE2s-hashes structured strings (layer name +
  index) into deterministic 64-bit numbers. This is how the repo gets reproducible-yet-
  independent RNG streams across model parts.
- `logging_config.py` — singleton `LoggerManager`; `get_logger(__name__)`. Console=INFO, file=DEBUG.

**Group B — Validators (gatekeepers that run before training; catch bad configs at load time)**
- `model_architecture_validator.py` — `validate_model_architecture()`: structural sanity —
  layer dims chain, TopK ≤ synapse count, branch factors, transfer alignment, biological plausibility.
- `training_stability_validator.py` — `validate_training_stability()` /
  `apply_automatic_safeguards()`: flags risky optimization combos (AMP + grad hooks,
  signed-input shunting, aggressive FSDP) and can auto-fix them.
- `config_validation.py` — thin unifier: `validate_full_config()` calls both validators above.

**Group C — Numerical / ML helpers**
- `sparse_ops.py` — `get_sparse_layer()`: factory + registry mapping a string ("topk",
  "stochastic", …) to a sparse-synapse class in `networks/.../synapse/`. The indirection that
  lets a YAML string pick a connectivity impl. Plus `create_topk_mask()`, sparsity math.
- `lda.py` — non-negative LDA (scipy L-BFGS-B): `fit_nonnegative_lda()`,
  `compute_lda_importance_scores()`, `select_topk_features_by_lda()`. Feature importance +
  data-driven weight init; non-negativity respects synaptic sign constraints.
- `gpu_estimation.py` — `GPUMemoryEstimator` + `MemoryBreakdown`/`GPURecommendation`:
  predict params/grads/optimizer/activation memory incl. FSDP, to size GPUs before submitting.
- `math/auc.py` — torch-native `roc_auc_score()` (binary Mann-Whitney U + OvR multiclass), GPU/CPU, batched.

**Group D — Information theory (`utils/information/`) — math engine behind the analysis layer**
- `nearest_neighbors.py` — KSG / Frenzel-Pompe k-NN MI estimators: `compute_mi_cc`,
  `compute_mi_cd`, conditional MI, synergy/redundancy matrices, `compute_synaptic_information`.
- `estimators.py` — wraps those + binned/decoder/sklearn/copula backends behind
  `InformationEstimatorFactory`; Miller-Madow correction, permutation nulls.
- `pid.py` — optional Partial Information Decomposition wrapper (no-ops if external lib absent).

Two you'll keep bumping into later: `general.py` (seeds/reshaping/serialization) and
`sparse_ops.py` (string→sparse-layer factory).

### `config/` — the typed config system (YAML → validated dataclass tree)

The whole repo is config-driven: one YAML describes an entire experiment.

**Spine**
- `config.py` — the heart. Defines top-level **`Config`** dataclass and **`load_config(path)`**
  (the entry point the whole repo calls). `Config` sections: experiment, data, model, training,
  analysis, wandb, outputs, distributed.
- `base.py` — `BaseConfig` mixin: `load()`/`save()`/`asdict()` via OmegaConf.
- `__init__.py` — re-export hub.

**Schema sections (the actual fields)**
- `model.py` — `ModelConfig`, `CoreConfig`, `MorphologyConfig`, `ConnectivityConfig`,
  `SparsityConfig`, `TransferFunctionConfig` — the dendritic structure description.
- `training.py` — `TrainingConfig`, `OptimizerConfig`, strategy configs.
- `regularization.py` — `RegularizationConfig`, `PruningConfig`, `SelectiveConfig`.
- `analysis.py` — `AnalysisConfig` aggregating param blocks for every analyzer.

**Adapters (a backward-compat shield — legacy YAML quirks quarantined here, not in modern code)**
- `legacy.py` — old transfer-function / unified-EI spellings → modern keys.
- `model_aliases.py` — friendly core name (`"dendritic_additive"`) → explicit morphology flags.
- `local_learning.py` — flat legacy keys → typed 3F/4F/5F local-rule config.
- `reactivation.py`, `recurrence.py`, `conversion.py` — gate-init policies, recurrent-core
  detection, dict coercion (`to_plain_dict`, `deep_merge_dicts`, `layer_value`).
- `recurrent.py`, `multi_stage_training.py` — deprecated, kept so old configs still load.

**Validation**
- `validation.py` — `validate_loaded_config()` runs automatically at the end of `load_config()`:
  invalid core types, illegal population-network cycles.

**Flow**
```
load_config("exp.yaml")
   → OmegaConf loads YAML to a dict
   → builders construct typed dataclasses,
       applying legacy.py / model_aliases.py / local_learning.py normalizers
   → validate_loaded_config() checks structural sanity
   → returns a deep-copied, validated Config object
```

Takeaway: one entry point (`load_config`), one root object (`Config`), a typed tree
underneath, and a deliberate wall of adapter files isolating legacy YAML from clean schema.
When later code does `config.model.core...`, this is where those attributes are defined.

---

## Layer 1 — `networks/` (the core)

The heart of DendriNet and the largest layer. Everything else feeds these modules a
config + batch, or inspects what they computed. Organizing idea:

> A config string picks an **architecture** from a **factory**; the architecture is built
> from **layers**; layers are built from **synapses**; synapses are built from
> **weight-transforms** + **sparse-selection** rules.

Read in that order: scaffolding → synapses → dendritic layer → cores → families on top.

### 1. Scaffolding — string → network
- `architectures/factory.py` — central switchboard. `get_architecture(type, parameters,
  input_dim)` routes a type string (`"einet"`, `"population_network"`, `"unified_ei"`,
  `"mlp"`, …) to one of 100+ `_build_*` functions that instantiate an `nn.Module`. Where
  `config.model.core.type` becomes a real object.
- `architectures/registry.py` — extension mechanism: `register_architecture(spec)` adds
  custom builders without editing the factory; built-in names reserved against shadowing.
- `networks/base.py` — `BaseNetwork` abstract parent (`forward`, `get_output_dim`); lets the
  rest of the codebase treat all networks interchangeably.

### 2. Synapse layer (`excitation_inhibition/synapse/`) — the true leaf
Sparse linear connections; where biological sparsity lives. All build on `TopKLinear`.
- `topk.py` — `TopKLinear`: keep K strongest weights per output. Base of all sparse conn.
- `dense_to_sparse.py` — `DenseToSparseLinear`: start dense, anneal to K (developmental pruning).
- `stochastic.py` — `StochasticTopKLinear`: probabilistic top-K (weak weights sometimes survive) + fast mode.
- `variance.py` — `VarianceTopKLinear`: select by `variance × weight` via EMA of input stats.
- `indexed.py` — `IndexedSparseLinear` + rewiring variants: store K weights + indices, not a dense matrix.
- `deepst.py` — `DeepstLinear`: dynamic sparse training, Brownian weight evolution + per-step rewiring.
- `structured_mask.py` — anatomical connectivity constraints (allowed-connection masks, fixed in-degree, distance kernels).

`utils/sparse_ops.py::get_sparse_layer()` (Layer 0) maps config strings → these classes.

### 3. Dendritic core (`excitation_inhibition/dendritic/`) — the signature idea
Biology: a neuron is a **tree** of dendritic branches integrating inputs distal→soma.
- `branch_layer.py` — `DendriticBranchLayer`: ONE tree level. Per branch: excitatory TopK
  synapses → optional inhibitory gating (shunting=divisive / additive) → reactivation gate.
- `blocklinear.py` — `BlockLinear`: the convergence op (several distal branches → one proximal),
  block-structured weights, positive conductances via weight transforms.
- `dendrinet.py` — `DendriNet`: **the atom.** Stacks branch layers into a full tree; `branch_factors`
  like `[4,4,2]` defines geometry (each factor = one level), ending at soma. A single dendritic "neuron".
- `initialize.py` — analytical weight/reactivation init (truncated-normal moments, scipy) so
  expected conductances match biological targets.
- `gradient_scaling.py` — `GradientScaler`: optional backward hooks reshaping gradient flow
  (distal upweighting, conductance scaling).

Vision extensions:
- `spatial_dendritic/spatial_conv.py` — `HierarchicalDendriticConv`: replace a Conv2d with a
  shared DendriNet over unfolded patches (spatially local, weight-shared).
- `spatial_dendritic/tensor_map.py` — `HierarchicalDendriticTensorMap`: control variant (flatten,
  one DendriNet, no locality) for isolating the effect of locality.
- `vision_factories/` — hand-designed RFs (Gabor, center-surround, motion, end-stopped) injected into conv weights pre-training.

### 4. Feedforward E/I (`excitation_inhibition/`)
- `ei_layer.py` — `ExcitationInhibitionLayer`: pairs excitatory DendriNet + optional inhibitory,
  wiring E→E/E→I/I→E/I→I synapses, shunting or additive.
- `ei_network.py` — `ExcitationInhibitionNetwork` stacks layers; `ConfigurableEINetwork` is the
  unified config-driven variant (reused by the vision cores).
- `input_transform/` — split raw input into two pathways: `identity.py` (both see all),
  `index.py` (partition channels), `transfer.py` (`TransferLayer`, `input_mode` 0=shared/1=independent).

This family = the `einet` / `dendritic_shunting` / `dendritic_additive` / `flat_*` cores.

### 5. Recurrent E/I (`recurrent/`) — adds time + named populations
The atom gains a clock:
- `stateful_dendrinet.py` — `StatefulDendriNet`: a DendriNet population with temporal state.
  Four synapse compartments per level (FF-E, FF-I, REC-E, REC-I), exponential traces with
  geometric time constants, optional spiking soma + soma→dendrite feedback.
- `spiking_soma.py` — `LIFSoma` + `SurrogateSpike` (fast-sigmoid surrogate gradient → backprop through spikes).
- `synapse_types.py` — receptor conductances (AMPA/NMDA/GABA) with kinetics, Mg-block, STP; replaces binary E/I trace when enabled.
- `ei_state.py` — per-timestep state containers (`DendriNetState`, `EIState`).
- `ei_config.py` / `population_common.py` — declarative specs: `PopulationConfig` (one population)
  and `PopulationDefinitionConfig` (name populations: `"pyr"`=E, `"PV"`/`"SST"`=I).

Built on those:
- `ei_layer.py` + `ei_network.py` — `EILayer`/`EINetwork`: minimal 2-population (1 E + 1 I)
  recurrent backbone. The `unified_ei` / `ei_unified` core.
- `population_network.py` — `PopulationNetwork`: the GENERAL form. Directed graph of arbitrarily
  many named populations; same-step edges form a DAG, delayed edges form cycles (recurrence).
  README's flagship `population_network` core (pyr/PV/SST circuits in `outputs/`).
- `baseline_rnn.py` — `BaselineRNN`: GRU/LSTM/vanilla controls.
- `protocol.py` — `RecurrentCore` / `SteppableRecurrentCore` typing protocols. The glue: EINetwork,
  PopulationNetwork, BaselineRNN all satisfy them, so trainers/analyzers treat them identically.

### 6. Supporting families
- `activations/` — `ActivationFactory.create()` dispatches activation strings. `parametric.py`:
  `ParametricTanh` etc. with learnable slope/midpoint — the "reactivation" gates inside branches.
  `resolution.py` maps paper aliases → `(reactivate?, type)`.
- `architectures/classical/` — baselines that matter: `matched_param.py` / `point_mlp.py` /
  `effective_matched.py` binary-search MLPs to match total/effective param counts of a dendritic
  net (apples-to-apples). Plus CNN, AlexNet, autoencoder, VAE.
- `architectures/transformer/dendritic_ffn.py` — `DendriticFFNReplacement` swaps a transformer MLP
  for a DendriNet; `LayerwiseEIStackReplacement` shares one E/I stack across layers; loaders swap a
  pretrained model's MLPs (the `jobs/transformer_replacement_benchmarks/` work).
- `convert/cnn_to_dendrite.py` — `conv2d_to_dendrite()`: retrofit a pretrained Conv2d into a
  `DendriteConv2d` via push-pull decomposition (positive→E, negative→I).
- `layers/dendritic_conv2d.py` — `DendriteConv2d`: conv primitive, output `E/(E+I+eps)` (shunting in conv form).
- `input/`, `pathway_router.py`, `utils/weight_transforms.py` — input wrappers (sigmoid bounding,
  dual-pathway routing) + positive-weight transforms (`exp`/`softplus`/`relu`) every synapse uses.

### The one diagram
```
config string ─► factory.get_architecture() ─► feedforward EI | recurrent EI | classical baseline
                                                      │              │
                                            ExcitationInhibition-  PopulationNetwork / EINetwork
                                                 Network          (+time, LIFSoma, synapse_types, feedback)
                                                      │              │
                                          stack of EILayer    stack of PopulationLayer
                                                      └──────┬───────┘
                                                   each wraps DendriNet (the atom)
                                                   = stack of DendriticBranchLayer (one tree level)
                                                       ├─ TopKLinear synapses (+ variants)
                                                       ├─ BlockLinear convergence
                                                       └─ ParametricTanh reactivation gate
                                                   weight_transforms keep conductances ≥ 0
```
Feedforward and recurrent families share the same `DendriNet` atom; recurrence just adds
time, spiking, typed synapses, and feedback.

---

## Layer 2 — `models/` (task heads over a core)

Tiny but pivotal: it turns a Layer-1 *network* into a *trainable task model* by composing
three networks — **encoder → core → decoder** — behind one interface.

- `base.py` — `BaseModel(nn.Module)`: holds `encoder_network`, `core_network`,
  `decoder_network`. `forward` = `decoder(core(encoder(x)))`. Crucially, it checks
  `core.is_recurrent`: for `[B,T,D]` sequence inputs it applies the encoder *per-timestep*
  and threads `seq_lengths` into the recurrent core. Also owns the shared plumbing every
  trainer relies on: `decay_weights` (model-specific weight decay), `apply_rewiring`, and
  `get_param_groups` (splits params into per-primitive optimizer groups: topk / blocklinear /
  reactivation / decoder / encoder / catch-all).
- `classifier.py` — `Classifier(BaseModel)`: non-recurrent, returns raw logits (CE applied
  externally); optional learned `log_output_scale` when the decoder is `Identity`; `predict = argmax`.
- `recurrent_classifier.py` — `RecurrentClassifier(BaseModel)`: inherits the recurrent-aware
  `forward`; `predict` argmaxes the last dim (handles `[B,C]` and `[B,T,C]`); `eval_logits`
  collapses 3-D output to the last timestep so analyzers get a clean 2-D tensor.
- `regressor.py` — `Regressor(BaseModel)`: continuous output unchanged; works recurrent or not.

**No factory lives here.** Assembly happens in `scripts/script_utils/setup_utils.py::initialize_model`:
it calls `networks.get_architecture(...)` three times (wiring dims so `core.input_dim =
encoder.output_dim`, `decoder.input_dim = core.output_dim`), then picks the wrapper from
`model_config.task` + a recurrent-core check. **This is the seam between Layer 1 and everything above:**
higher layers only ever see a `BaseModel`.

---

## Layer 3 — `datasets/` (name → (train, valid, test))

Config says `data.dataset_name: "cifar10"`; this folder resolves that string to three datasets.

- `standard_datasets.py` — the dispatcher. `get_unified_datasets(task_cfg)` resolves a name in
  order: (1) the plugin **registry**, (2) torchvision standards (MNIST / Fashion-MNIST /
  CIFAR-10/100, with optional label noise), (3) a `_SPECIAL_DATASET_LOADERS` table
  (poisson, imagenet, modulo-10 contextual variants) + synthetic + sequence, (4) an optional
  external `experiments` package. Always returns `(train, valid, test)`.
- `registry.py` — `DatasetSpec` + `register_dataset()`: add a dataset config-only, no edits to
  the dispatcher (mirrors the architecture/analyzer registries — a recurring pattern).
- `synthetic_datasets.py` — programmatic datasets with *controlled* E/I correlation structure
  and tunable linear↔nonlinear separability (Gaussian, XOR/quadratic, block-correlated,
  population-code, contextual gain-shift…). These are the "theory" datasets for probing what
  dendritic nonlinearity buys you.
- `sequence_datasets.py` — compatibility shim re-exporting the real temporal tasks from the
  `datasets.sequence` subpackage (adding problem, copy task, sequential-MNIST, Romo delay
  comparison, Lorenz, switching-LDS, Stringer V1, variable-delay DMS…). These feed the recurrent cores.
- `augmentation/noise.py` — pure tensor perturbations (Gaussian / uniform / Poisson / Fourier
  frequency), orthogonal to selection; used for robustness tests and on-the-fly augmentation.

**Pattern to notice:** registry → standard → special → external is the same "try increasingly
general fallbacks" shape as the architecture factory.

---

## Layer 4 — `training/` (turn a model + data into a trained model)

The engine room. Selection mirrors Layer 1: a **strategy string** picks a trainer from a factory.

**The spine**
- `factory.py` — `get_trainer(strategy, optimizer, config, analysis_manager)`: a
  `TRAINER_STRATEGY_REGISTRY` maps a string (`standard`, `freeze_layers`, `fa`, `dfa`,
  `local_ca`, `voltage_stabilization`, `homeostatic_control`, …) to a `Trainer` subclass;
  bespoke ones (`shunting_fa/dfa`, `recurrent`, `multi_stage`) route to `_build_*` helpers.
  `register_trainer_strategy` allows runtime extension.
- `strategies/standard.py` — **`Trainer`, the base class every strategy subclasses.** `train()`
  → `_run_training_loop` over epochs → `_epoch_train` (forward → loss + `RegularizationManager`
  → grad-clip → optimizer step) + `_epoch_valid`, wrapped with checkpointing, early stopping,
  best-state reload, AMP, DDP, pruning (`PruningManager`), and W&B logging. Understand this file
  and you understand every trainer.

**The standard family (all subclass `Trainer`)**
- `recurrent_trainer.py` — `RecurrentTrainer`: unrolls `[B,T,D]` sequences, sums per-timestep loss, cosine-LR+warmup.
- `multi_stage.py` — `MultiStageTrainer`: runs a configured pipeline of stages, each a different sub-strategy/optimizer/param-group/epoch budget, chaining checkpoints.
- `layerwise.py` — `LayerWiseTrainer`: trains one layer at a time (soma→input or reverse), freezing the rest.
- `two_step.py` — `TwoStepTrainer(WithKL)`: local per-layer pretraining, then global fine-tune.
- `specialized.py` — swap in a biophysical auxiliary loss: `VoltageStabilizationTrainer`,
  `ExcitationInhibitionEquilibriumTrainer`, `HomeostaticControlTrainer`, `TrainOnlyMReactivation`.
- `fsdp_standard.py` — `FSDPTrainer`: same loop, sharded across GPUs for models too big for one card.

**The biologically-plausible learning family (the research payload)**
Real neurons can't do backprop — there's no mechanism to ship the exact transposed-weight
gradient backward, and a dendrite only "sees" local voltages + a broadcast neuromodulatory
signal. This family asks: *can local rules still learn on dendritic hardware?*
- `feedback_alignment.py` — `FeedbackAlignmentTrainer` (FA / DFA): fixed **random** feedback
  matrices replace `Wᵀ` in the backward pass. `ShuntingFeedbackAlignmentTrainer` for dendritic layers.
- `local_learning.py` — `LocalCreditAssignment`: composed from ~9 mixins implementing the
  **"N-factor" ladder** of Hebbian-like rules — 3-factor (pre × post × broadcast-error),
  4-factor (+ `rho` activity modulator), 5-factor (+ `phi` statistical modulator) — plus
  **STDP** (spike-timing traces), **HSIC** (statistical-dependence objective), **homeostasis**
  (keep firing/voltage in range), and **morphology** factors (dendritic path attenuation,
  apical vs basal). The heavy logic lives in the `local_learning_parts/` subpackage; config
  comes from `config/local_learning.py`. This is the point of a bio project: compare backprop
  vs biologically-plausible rules *on the same dendritic architecture*.

**Losses, optimizers, distributed, support**
- `loss/functions.py` — registry of loss modules: standard (MSE/CE/BCE/NLL/VAE-ELBO/cosine) +
  biophysical (`VoltageStabilizationLoss`, `ExcitationInhibitionBalanceLoss`,
  `HomeostaticControlLoss`, `EIWeightRatioLoss`); several read internal voltages via forward hooks.
- `loss/bio_losses.py` — probe-driven V1-tuning objectives: push units toward measured cortex
  properties (orientation/direction selectivity, DoG center-surround, phase invariance,
  end-stopping, sparsity). `compute_bio_losses()` returns a weighted total + per-component breakdown.
- `optimizers/factory.py` — `create_optimizer`: splits params into ordered groups (default/topk/
  blocklinear/reactivation/decoder), each its own LR; dispatches adam/adamw/sgd/rmsprop.
- `optimizers/custom.py` — `CustomWeightDecayOptimizer`: decays via `model.decay_weights`
  (respecting conductance transforms) instead of vanilla weight decay; cleans state for rewired modules.
- `fsdp_utils.py` — build sharding/mixed-precision config, auto-select which dendritic/large-linear
  modules to wrap, gradient checkpointing, sharded checkpoint save/load, memory estimation.
- `distributed_samplers.py` — `PaddedDistributedEvalSampler`: `-1`-padding so every rank gets equal batch counts on eval.
- `regularization.py` — `RegularizationManager` (L1/L2, E/I ratio) + `PruningManager` (structured pruning).
- `utils/evaluation.py` — metric fns (accuracy/AUC/MI/MSE/cosine) supporting materialize/stream/auto modes.
- `utils/profiling.py` — env-gated torch profiler lifecycle + section timing.
- `transformer_replacement.py` — facade for the transformer-MLP → dendritic distillation subsystem.

---

## Layer 5 — `analysis/` + `plotting/` (what did the trained network actually do?)

Post-hoc science. Same registry-driven shape as the rest of the repo, now with four levels.

**`analysis/` — a four-layer, registry-driven pipeline**
1. `registry.py` — declarative metadata: each analyzer is an `AnalyzerSpec` (config field →
   lazy class path → attribute → save subdir → constructor style). ~22 builtins auto-register;
   third parties can `register_analyzer`.
2. `managers/analysis_plan.py` — turns specs into construction + execution runners and hard-codes
   the scientific ordering (`_ORDERED_ANALYSIS_SEQUENCE`: performance → information → turnover →
   noise/contribution/ablation → weight → reactivation → compartment/gain/RF/activation →
   local-rule → correlation → routing → inhibitory/jacobian/timetraces → spike → adversarial).
3. `managers/analysis_manager.py` — `AnalysisManager`: instantiates only *enabled* analyzers,
   gates each by training-vs-final phase, resolves the runtime profile, dispatches via shared
   call-signature helpers. This is what `train_experiments.py` calls to produce a run's analysis.
4. Analyzers — all subclass `core/base.py::AbstractAnalyzer` (uniform `analyze(...)` → dict + saved JSON/plots).

**The analyzers, by scientific question**
- Structure / weights: `weight`, `compartment_statistics`, `ei_connectivity_motifs`, `inhibitory_specialization`, `jacobian_spectrum`.
- Function / signal: `compartment_snr`, `correlation_analysis`, `receptive_fields` (~2400 lines), `multiplicative_gain`, `linearity_analysis`, `routing_analysis`, `core/information.py` (MI between E/I inputs, branch outputs, labels; PID; null baselines).
- Dynamics / temporal: `dendritic_timetraces`, `reactivation_dynamics`, `spike_trains`, `synaptic_activation`, `core/branch_activation.py`.
- Plasticity / learning: `synaptic_turnover`, `synaptic_pruning`, `local_rule_components`.
- Robustness / performance: `metrics/adversarial_robustness` (FGSM), `metrics/noise_perturbation`, `metrics/layer_contribution`, `core/ablation.py`, `metrics/performance.py`.

**`analysis/utils/runtime.py` — the stream-vs-materialize engine.** An `EvaluationRuntimeConfig`
(chosen per phase) drives `should_materialize_dataset`: small sets are materialized into one
in-memory batch (fast for hook-based analyzers); large sets stream through deterministic loaders.
`analysis_device_context` moves the model to the analysis device in eval mode and restores it
after. This is what the README's `analysis.runtime` `mode: stream/materialize/auto` controls.

**`plotting/` — analyzers depend on it, not vice versa.** `visualizations/base.py`
(`PlotConfig` + `BasePlotter`/`TimeSeries`/`Distribution`/`Comparison`) + `plotting_utils.py`
(save/figure/heatmap) form the base; topic modules mirror the analyzer categories
(`information_plots` ~3000 lines, `correlation_plots`, `performance_plots`,
`synaptic_weight_plots`, `general_analysis_plots`). Analyzers import `plot_*` and call them
inside `analyze()`. `core/utils.py` converts raw activations into plot-ready dicts.

---

## Layer 6 — `scripts/` + `configs/` + `jobs/` (orchestration)

Where a human (or Slurm) actually starts a run.

**`scripts/` — the CLI entry points**
- `training/train_experiments.py` — **THE main driver** (console script `dendritic-train`).
  End-to-end `main(config_path, args)`:
  1. distributed setup (torchrun env → rank/world; `(0,0,1)` single-GPU)
  2. load + validate config (`load_config` → `validate_training_stability` → `apply_automatic_safeguards` → CLI overrides)
  3. build env (`setup_utils.setup_environment`: datasets + model + encoder; DDP-wrap; dump init-gate diagnostics; save `config.json`)
  4. encoder pretrain/load + encode datasets; build `AnalysisManager`; optional reactivation-gate calibration
  5. train (`create_optimizer` → `CustomWeightDecayOptimizer` → `get_trainer(strategy)` → `trainer.train()`)
  6. finalize (optional pruning step → save `final_model.pt` → `run_analysis()`)
  7. cleanup (wandb finish, destroy process group)
- `training/train_experiments_fsdp.py` — same flow, FSDP-sharded for very large models.
- `training/train_encoder_network.py` — build/pretrain (or load cached) encoder; config-hash reuse.
- `training/{train,benchmark,aggregate,compare}_transformer_replacement.py` — the transformer-MLP → dendritic distillation track.
- `sweeps/sweep_manager.py` — read one sweep YAML → `unified/unified_sweep.py` expands
  `sweep_config × filter_config × seeds` into many per-run configs → generate (splittable) Slurm
  script → submit. Each generated config is just a normal `train_experiments.py` run.
- `sweeps/result_analyzer.py` — auto-detect sweep type → `collectors → analyzers → plotters` → summary figures. `rerun_missing_configs.py` fills gaps.
- `script_utils/` — `setup_utils.py` (`setup_environment`/`initialize_model` — the model-assembly seam), `config_utils.py` (trainer-config prep, legacy canonicalization), `init_diagnostics.py` (gate-stats dump).
- `tools/` — resource planning: `calculate_ei_parameters`, `estimate_gpu_requirements`, `generate_slurm_job`, `submit_fsdp_job`.
- `analysis/` — standalone post-hoc studies on saved checkpoints (routing-by-condition, OOD-delay, lesion, rec-inhibition impact, inhibitory structure, stability, trace decomposition).
- `transformer_replacement/` — OLMo config generators (all-layer / EI-stack / joint) delegating to `_olmo_generators.py`.

**`configs/` — the experiments themselves**
- Top-level: `config_exp.yaml` (single-GPU reference), `config_exp_fsdp.yaml`, `sweep_config.yaml`.
- `examples/` — 15 documented configs (feedforward E/I, CIFAR-10, recurrent pyr/PV/SST, spiking,
  soma feedback, dense-to-sparse pruning, indexed rewiring, local learning, AMPA/NMDA/GABA kinetics, transformer replacement).
- `vision/` — AlexNet / backbone-replacement configs. `distributed/` — FSDP settings.
  `templates/` — full `analysis:` block to copy. `transformer_replacement_benchmarks/` — 10 benchmark configs (pythia, olmo2, deepseek, synthetic).
- **Config shape** = the Layer-0 `Config` tree in YAML: `experiment / data / model / training / analysis / wandb / outputs / distributed`.

**`jobs/` + `setup/` + `outputs/`**
- `jobs/` — Slurm profiling batch scripts (`profile_nsys.sbatch`, `profile_ncu.sbatch`) + README.
- `setup/` — `environment.yml` (conda), `setup_cuda129.sh` (cluster CUDA 12.9).
- `outputs/` — timestamped run dirs. Each produces: `config.json`, `nparams`,
  `init_gate_stats.json`, `train.log`, `final_model.pt`, `main_network/` (best model +
  loss-curve PNG), `performance/final.json`, and per-analyzer subdirs (e.g. `routing_analysis/final/`).
- `pyproject.toml` console scripts: `dendritic-train`, `dendritic-train-transformer-replacement`,
  `dendritic-benchmark-transformer-replacement`, `dendritic-sweep`, `dendritic-analyze`.

---

## Scaling up — the whole repo as one flow

```
configs/config_exp.yaml
        │  (Layer 0) config/load_config → validated Config tree
        ▼
scripts/train_experiments.py  ── the conductor (Layer 6)
        │
        ├─ datasets/get_unified_datasets(name)      → (train, valid, test)      [Layer 3]
        ├─ setup_utils.initialize_model:
        │     networks.get_architecture(encoder) ┐
        │     networks.get_architecture(core)    ├─ wired by dims →  BaseModel  [Layers 1→2]
        │     networks.get_architecture(decoder) ┘   (Classifier / RecurrentClassifier / Regressor)
        │        core = DendriNet atoms inside EI / PopulationNetwork
        │
        ├─ training/get_trainer(strategy) → Trainer subclass                    [Layer 4]
        │     optimizer = create_optimizer + CustomWeightDecayOptimizer
        │     loss = get_loss_function(...)  (+ bio losses / local rules)
        │     trainer.train(model, train, valid)  → epochs → checkpoints
        │
        ├─ analysis/AnalysisManager.run_analysis(model, data)                   [Layer 5]
        │     registry → plan (ordered) → analyzers → plotting/ → JSON + PNG
        │
        └─ outputs/<run_timestamp>/ : final_model.pt, performance/final.json,
                                       per-analyzer subdirs, logs                [Layer 6]
```

**The three ideas that unlock the whole codebase:**
1. **Registry + factory everywhere.** Architectures, datasets, trainers, losses, optimizers,
   and analyzers are all "string → class" registries with an extension hook. Learn one, you know all six.
2. **`DendriNet` is the atom; protocols are the glue.** Every core is built from dendritic
   trees; `BaseNetwork` / `RecurrentCore` / `BaseModel` / `AbstractAnalyzer` are the interfaces
   that let each layer stay ignorant of the layer below's concrete type.
3. **One `Config`, one conductor.** `config/` defines the whole experiment as a typed tree;
   `train_experiments.py` reads it top-to-bottom and calls each layer in turn. Follow that file
   and you traverse the entire stack once.

---

# Deep Dives

Below the map: the actual mechanics, read from source. Each dive names the file+line so you
can jump straight in.

## Deep Dive 1 — The dendritic core computation

Files: `dendritic/branch_dynamics.py` (the math), `dendritic/dendrinet.py` (the geometry),
`dendritic/blocklinear.py` (convergence), `dendritic/branch_layer.py` (config/wiring shell).

**Refactor note:** `branch_layer.py` (803 lines) is now almost entirely config-and-wiring; the
computation was extracted into sibling modules (`branch_dynamics.py`, `branch_synapses.py`,
`branch_grad_scales.py`, `branch_maintenance.py`, `branch_options.py`, `branch_config.py`) that
were NOT in the original flat listing. The math lives in `branch_dynamics.py`.

### 1. The shunting equation (`branch_dynamics.py:10` `forward_branch_dynamics`)
Shunting mode reduces to:
```
                E  +  E_rec  +  branch
   V  =  ─────────────────────────────────────────────
          1  +  E  +  E_rec  +  I  +  I_rec  +  g_branch  +  ε
```
`E`=excitatory current, `I`=inhibitory, `_rec`=recurrent, `branch`=aggregated child-branch
current, `g_branch`=branch coupling conductance. Four deliberate properties:
- **Inhibition is denominator-only → it divides, never subtracts.** This is shunting (divisive)
  inhibition: it scales gain down instead of cancelling. The headline behavior real dendrites
  show and standard artificial neurons don't.
- **Excitation is in BOTH numerator and denominator** (`:29-30`): opening excitatory channels
  also raises total membrane conductance, so `V=E/(1+E)` **self-saturates toward 1** — no runaway.
- **The `1+` is the leak/resting conductance** — bounds output to `[0,1)` for positive currents,
  which is why inputs are sigmoid-bounded before the core (`input/sigmoid.py`). Designed to fit.
- **`ε` guards the divide.**

The `else` branch (`:71-104`) is **additive**: `V = E + E_rec − I − I_rec` (inhibition subtracts;
README `dendritic_additive`), optionally z-scored by `normalize_branch_additive_voltage` (`:161`)
→ `dendritic_normalized_additive`. **The three README core types are three branches of one function.**
Then `V → owner.reactivation(voltage)` (`:112`), the learnable ParametricTanh gate.

### 2. Tree geometry (`dendrinet.py:124` `_plan_dendrinet_branch_layout`, `:436` forward)
Example `n_soma=10, branch_factors=[4,4]`:
```
layer_sizes: [10] → [10,40] → [10,40,160]
build order (reversed): [160, 40, 10]    input_branch_factors: [None, 4, 4]
  160 distal ──BlockLinear 4→1──► 40 ──4→1──► 10 soma
  layer_idx=2                     idx=1        idx=0
```
Forward loop threads the previous level's output into the next as `branch_input`:
```python
output = None
for i in range(self.n_branch_layers + 1):
    output = self.branch_layers[i](x, inhibitory_input, output)
```
Every level integrates BOTH (a) fresh synapses on the same external `x` AND (b) the aggregated
child-branch `output` — hierarchical dendritic integration. Tree narrows by `branch_factor` per
level to the soma. `layer_idx` counts down distal(high)→soma(0); used by init + gradient scaling
to up-weight electrotonically-distant distal branches.

### 3. Sign & positivity — Dale's law, enforced structurally
- **Positivity:** params stored as `log_weight`, passed through `apply_weight_transform`
  (`exp`/`softplus`/`relu`) → conductances always ≥ 0 (`blocklinear.py:114`). Raw param
  unconstrained (good for optimization); transform guarantees validity.
- **Sign:** excitatory current → numerator, inhibitory → denominator. Sign is set by which
  population a synapse belongs to, NOT by weight value — a neuron can't flip E↔I during training.
- **Weight decay respects the transform** (`blocklinear.py:162`): in exp-space, multiplicative
  decay `w·(1−λ)` becomes additive `log_weight += log(1−λ)`.

### 4. BlockLinear convergence (`blocklinear.py:280` `EfficientBlockLinear.forward`)
160 branches → 40 means `block_size=4`. Stores only `(40,4)` weights (one conductance per
converging child), not a 160×40 matrix. Forward: reshape to `(B,40,4)`, multiply, sum last axis.
In shunting, branch current → numerator while `sum_conductances()` → denominator
(`branch_dynamics.py:42-45`): branch coupling is itself a conductance (cable-model consistent).

**Takeaway:** a DendriNet neuron is a divisive (shunting) rational function of sparse,
sign-constrained, non-negative synaptic currents, evaluated hierarchically up a converging tree,
with a learnable saturating gate at each node.
