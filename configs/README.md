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

## Recommended examples

| Goal | Config |
| --- | --- |
| Feedforward E/I MNIST | `examples/population_network_feedforward.yaml` |
| CIFAR-10 E/I training recipe | `examples/population_network_cifar10.yaml` |
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
| Local credit assignment | `examples/population_network_local_learning_mnist.yaml` |
| Frozen-checkpoint representation analysis | `examples/representation_accessibility_analysis.yaml` |
| Transformer FFN replacement | `examples/transformer_replacement_synthetic_distillation.yaml` |

Population-network examples should use:

- `model.core.type: population_network`
- explicit `population_network.layers[].populations`
- explicit `population_network.layers[].connections`
- `reactivation_init_policy: occupancy_quantile`
- `weight_transform: softplus`
- `topk_type: dense_to_sparse` when the experiment is about learned pruning
- `topk_type: indexed_rewire` when the experiment needs memory-efficient
  same-dendrite turnover of weak synapses without a dense score matrix
- `topk_type: indexed_dynamic` only when the experiment specifically needs a
  fixed candidate pool for comparison
- `efficient_blocklinear: true` when populations have multi-level branch
  factors, so branch aggregation avoids materializing dense block matrices.
  Transformer and vision replacement examples use the same setting through
  their population-network core blocks
- `weight_decay_rate: 0.01` and `weight_boosting: true` for examples matching
  the E/I training recipe

Keep these outside `configs/examples/`:

- input-mode equivalence configs (`input_mode0/1`, one-layer/two-layer);
  express the streams with explicit `population_network.layers[].connections`
- combination examples that only stack already-demonstrated features, such as
  multilayer recurrent plus spiking
- training packs that belong under job- or project-owned directories

Transformer replacement benchmark configs are under
`configs/transformer_replacement_benchmarks/`. Project-specific sweep packs and
cluster job scripts should live outside the source tree or under ignored local
directories.

Ignored local/private config archives may live under `.vscode/archived_configs/`
or `projects/private_configs/`.
