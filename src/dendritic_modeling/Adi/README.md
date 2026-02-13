# Adi: vision baseline and dendritic ablation

All new code for the vision baseline and depth-indexed dendritic swaps lives here. No modifications to the rest of `dendritic_modeling` except a single factory branch that delegates to Adi.

## Contents

- **vision_encoder.py**: `VisionEncoderWrapper`, `get_baseline_encoder` (CNNDownsample wrapper with verified `output_dim`).
- **test_vision_baseline.py**: Sanity test (build baseline via config path, one forward pass, assert output shape `[batch, 10]`).
- **BASELINE.md**: Smoke test and determinism instructions.

## Running

- Vision baseline: `python src/dendritic_modeling/scripts/training/train_experiments.py configs/vision/alexnet_cifar10_baseline.yaml`
- Sanity test: from repo root, `PYTHONPATH=src python -m dendritic_modeling.Adi.test_vision_baseline`

## Config

New YAMLs live under `configs/vision/`. Baseline uses encoder type `alexnet_cifar10_baseline`; core `identity`; decoder `MLP`. Ablation variants: conv1_only .. fc7_only, early_only, late_only, all_swapped (see ablation_configs.ABLATION_VARIANTS). Run each with >=3 seeds.

## Representation probes and robustness

- **representation_probes.py**: Per-layer activations, linear probe accuracy, CKA (add implementation; run after performance stable).
- **robustness_eval.py**: CIFAR-10-C evaluation, mCE; cite Hendrycks and Dietterich 2019. Create this module with evaluate_cifar10c(checkpoint_path, results_path) that loads a trained checkpoint, runs inference on CIFAR-10-C (download from standard artifact), aggregates over corruption types/severities, and writes mCE and clean-vs-corrupted accuracy drop to results_path.
