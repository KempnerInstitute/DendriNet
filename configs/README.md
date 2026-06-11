# Configuration Layout

This directory is the public configuration surface for the codebase.

Keep this directory small, documented, and runnable. It should contain
reference configs and reusable examples, not one-off personal runs.

## Top-level files

- `config_exp.yaml`: complete single-GPU reference config with documented fields.
- `config_exp_fsdp.yaml`: complete FSDP / multi-GPU reference config.
- `sweep_config.yaml`: maintained generic sweep template using current
  population-network dot paths.
- `examples/`: small, documented examples for common model families and features.
- `vision/`: AlexNet / vision-backbone replacement examples.

## Recommended examples

| Goal | Config |
| --- | --- |
| Original feedforward E/I MNIST | `examples/standard_mnist.yaml` |
| Same E/I model in canonical notation | `examples/standard_mnist_population_network.yaml` |
| Heterogeneous E/I populations in one layer | `examples/population_network_heterogeneous_feedforward.yaml` |
| Explicit two-layer population routing | `examples/population_network_two_layer.yaml` |
| Recurrent Pyr/PV/SST microcircuit | `examples/population_network_recurrent_pyr_pv_sst.yaml` |
| AMPA/NMDA/GABA kinetics | `examples/population_network_synapse_types_ampa_nmda_gaba.yaml` |
| Spiking soma | `examples/population_network_spiking.yaml` |
| Soma-to-dendrite feedback | `examples/population_network_soma_feedback.yaml` |
| Dense-to-sparse developmental pruning | `examples/standard_cifar10_dense_to_sparse_population_network.yaml` |

New architecture examples should prefer:

- `model.core.type: population_network`
- explicit `population_network.layers[].populations`
- explicit `population_network.layers[].connections`
- `reactivation_init_policy: occupancy_quantile`
- `weight_transform: softplus`
- `topk_type: dense_to_sparse` when the experiment is about learned pruning
- `weight_decay_rate: 0.01` and `weight_boosting: true` for examples meant to
  reproduce the maintained E/I training recipe

Operational benchmark packs live with their job owners. Transformer replacement
benchmark configs are under `jobs/transformer_replacement_benchmarks/configs/`.
Project- or draft-specific sweep packs should live under the relevant
`drafts/*/configs/` tree or another project-owned directory, not here.

Ignored local/private config archives may live under `.vscode/archived_configs/`
or `projects/private_configs/`.
