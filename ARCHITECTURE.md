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
