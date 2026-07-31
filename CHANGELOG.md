# Changelog

## 2024-11

- Initial EI-layer and DendriNet codebase structure.

## 2025-02

- Added analysis, branch ablation, and information tools.

## 2025-03

- Pre-factor version tagged as `v0.1.0`.

## 2025-07

- Refactored codebase structure.
- Added SLURM generation for FSDP and distributed training.
- Added the newer sweep workflow.

## 2025-08

- Added regularization and pruning support.
- Expanded information analysis and training optimization options.
- Added computation-level controls and multivariate aggregation options.
- Added encoder-network caching and split encoder/main trainer flow.

## 2026-06

- Added population-network configs for feedforward, recurrent, spiking, synapse-type, and heterogeneous-cell examples.
- Standardized additive aliases and occupancy-quantile reactivation initialization.
- Added recurrent TopK weight caching and dense-to-sparse sparsity examples.
- Switched the public replacement/population examples to `indexed_rewire` for
  memory-efficient sparse storage with adaptive same-dendrite synapse turnover;
  older sparse modes remain available by setting `topk_type` explicitly.
- Enabled `train_target: full_model` in joint transformer replacement training
  to train the whole patched student instead of only replacement modules.
- Moved benchmark configuration examples under
  `configs/transformer_replacement_benchmarks/` and removed tracked
  cluster-specific `jobs/` artifacts; generated jobs/manifests are local output.
- Consolidated population-network config validation and warning-only legacy alias handling.
- Refactored local-learning and information-analysis helper modules.
- Verified feedforward and population-format numerical behavior against develop and pre-cleanup baselines.

## 2026-07

- Scaled indexed recurrent DendriNets to one million neurons with FSDP and
  added reproducible single-node and strong-scaling benchmark controls.
- Expanded checkpoint-bound empirical analysis for representation access,
  reactivation operating states, synaptic support, perturbations, and
  provenance-checked result collection.
- Added deterministic `spatial_morphology` connectivity for image DendriNets.
  Dendritic paths recursively partition a `(C, H, W)` input and store compact
  fixed sparse indices without a dense branch-by-pixel mask or convolutional
  weight sharing.
- Added a raw-ImageNet `[3, 3, 3, 3]` DendriNet experiment with no learned
  vision backbone. Its 61,103,392 trainable parameters closely match AlexNet,
  and its fixed connectivity can be regenerated instead of stored in
  checkpoints.
