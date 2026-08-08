# Configuration Layout

This directory is the configuration surface for the codebase.

Keep this directory small, documented, and runnable. It should contain
reference configs and reusable examples.

## Top-level files

- `config_exp.yaml`: single-GPU reference config using the
  `population_network` schema.
- `config_exp_fsdp.yaml`: FSDP / multi-GPU reference config using the
  `population_network` schema.
- `sweep_config.yaml`: sweep template using
  population-network dot paths.
- `templates/analysis_full_template.yaml`: copy/paste reference for the full
  preferred `analysis:` block, including SNR/RF options and inline comments.
- `examples/`: small, documented examples for common model families and features.
- `vision/`: AlexNet / vision-backbone replacement examples.
- `<project>/`: project-owned, versioned experiment contracts. Generated
  expansions belong under that project's ignored `generated/` directory.
- `local/`: optional untracked private configurations; never place experiment
  YAMLs under editor, log, output, or repository-root directories.

## Recommended examples

| Goal | Config |
| --- | --- |
| Feedforward E/I MNIST | `examples/population_network_feedforward.yaml` |
| CIFAR-10 E/I training recipe | `examples/population_network_cifar10.yaml` |
| Morphology-aligned CIFAR-10 image routing | `examples/population_network_spatial_morphology_cifar10.yaml` |
| Normalized additive inhibition | `examples/population_network_normalized_additive_mnist.yaml` |
| Heterogeneous E/I populations in one layer | `examples/population_network_heterogeneous_feedforward.yaml` |
| Explicit two-layer population routing | `examples/population_network_two_layer.yaml` |
| Recurrent Pyr/PV/SST microcircuit | `examples/population_network_recurrent_pyr_pv_sst.yaml` |
| Dendritic taus and delayed edges | `examples/population_network_taus_and_delays.yaml` |
| AMPA/NMDA/GABA kinetics | `examples/population_network_synapse_types_ampa_nmda_gaba.yaml` |
| Spiking soma | `examples/population_network_spiking.yaml` |
| Soma-to-dendrite feedback | `examples/population_network_soma_feedback.yaml` |
| Dense-to-sparse developmental pruning | `examples/population_network_dense_to_sparse_cifar10.yaml` |
| Indexed adaptive rewiring | `examples/population_network_indexed_rewire_mnist.yaml` |
| Raw ImageNet spatial DendriNet, backpropagation | `vision/imagenet_raw_spatial_dendrinet_alexnet_matched_bp.yaml` |
| Raw ImageNet spatial DendriNet, local credit assignment | `vision/imagenet_raw_spatial_dendrinet_alexnet_matched_localca.yaml` |
| End-to-end dendritic AlexNet canary | `vision/dendritic_alexnet_imagenet_canary.yaml` |
| Full-model shunting/additive learning pilot | `vision/dendritic_alexnet_imagenet_learning_pilot.yaml` |
| Local credit assignment | `examples/population_network_local_learning_mnist.yaml` |
| Frozen-checkpoint representation analysis | `examples/representation_accessibility_analysis.yaml` |
| Transformer FFN replacement | `examples/transformer_replacement_synthetic_distillation.yaml` |
| OLMo single/multiple/all FFN replacement | `examples/transformer_replacement/README.md` |

Population-network examples should use:

- `model.core.type: population_network`
- explicit `population_network.layers[].populations`
- explicit `population_network.layers[].connections`
- `reactivation_init_policy: occupancy_quantile`
- `weight_transform: softplus`
- `topk_type: dense_to_sparse` when the experiment is about learned pruning
- `topk_type: indexed_rewire` when the experiment needs memory-efficient
  same-dendrite turnover of weak synapses without a dense score matrix
- `topk_type: indexed` with `structured_connectivity.method:
  spatial_morphology` when each image branch should receive fixed contacts from
  its own nested spatial region. The default `spatial.region_mode: partition`
  uses disjoint child regions. High-fan-out trees can instead use
  `spatial.region_mode: overlapping_grid` with an explicit
  `spatial.minimum_window_shape`, which places child regions on nested grids
  and expands undersized terminal regions into localized overlapping windows
- `topk_type: indexed_dynamic` only when the experiment specifically needs a
  fixed candidate pool for comparison
- `efficient_blocklinear: true` when populations have multi-level branch
  factors, so branch aggregation avoids materializing dense block matrices.
  Transformer and vision replacement examples use the same setting through
  their population-network core blocks
- `weight_decay_rate: 0.01` and `weight_boosting: true` for examples matching
  the E/I training recipe

Large fixed indexed populations can additionally control storage and temporary
memory entirely from the population YAML:

```yaml
topk_type: indexed
indexed_seed: 17
indexed_index_dtype: auto
indexed_output_chunk_size: 65536
indexed_workspace_mb: 128
indexed_cache_transformed_weights: true
indexed_projection_backend: recompute
indexed_persistent_indices: false
indexed_init_mode: rank0_broadcast
reactivation_memory_efficient: true
```

- `indexed_index_dtype: auto` uses 32-bit indices whenever the input dimension
  fits, halving topology-buffer memory relative to the legacy 64-bit default.
- `indexed_workspace_mb` caps the gathered-input tensor for each indexed
  projection; it can only reduce `indexed_output_chunk_size`.
- `indexed_cache_transformed_weights` reuses deterministic fixed weights within
  one recurrent unroll. It is disabled by default.
- `indexed_projection_backend` is `eager`, `recompute`, `triton`, `cuda`, or
  `auto`. `recompute` avoids retaining contact-sized gathered inputs at every
  recurrent timestep. The legacy `indexed_recompute_backward: true` spelling
  still maps to it. Use `recompute` for small-batch recurrent workloads unless
  the repeated native launches are beneficial for the intended sequence and
  batch shape.
- `reactivation_memory_efficient` opts `param_tanh` gates into the
  memory-saving backward and fuses the shunt divide when diagnostics do not
  need its intermediate voltage. It is disabled by default.
- `indexed_persistent_indices: false` regenerates topology from
  `indexed_seed` instead of saving it in checkpoints. An explicit seed is
  required, and adaptive `indexed_rewire` topology must remain persistent.
- `indexed_init_mode: rank0_broadcast` generates topology on global rank 0.
  Use it with FSDP `sync_module_states: true`.
- `allow_self_recurrence: false` is compatible with indexed layers without a
  dense connectivity mask; self-indices are excluded directly by the sampler.

Keep these outside `configs/examples/`:

- input-mode equivalence configs (`input_mode0/1`, one-layer/two-layer);
  express the streams with explicit `population_network.layers[].connections`
- combination examples that only stack already-demonstrated features, such as
  multilayer recurrent plus spiking
- training packs that belong under job- or project-owned directories

Transformer replacement benchmark configs are under
`configs/transformer_replacement_benchmarks/`. Project-specific sweep packs
belong under a named `configs/<project>/` directory. Cluster job scripts and
generated sweep expansions remain ignored artifacts rather than source
configurations.

Ignored local/private configuration files may live only under `configs/local/`.
