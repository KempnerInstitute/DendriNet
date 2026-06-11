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
- Consolidated population-network config validation and warning-only legacy alias handling.
- Refactored local-learning and information-analysis helper modules.
- Verified feedforward and population-format numerical behavior against develop and pre-cleanup baselines.
