# Adi baseline and determinism

## Smoke test (reproducible baseline)

1. Install per repo [INSTALL.md](../../../INSTALL.md) (conda env with Python 3.9+).
2. From repo root, run the **existing** single-GPU entry point **twice** with the **same** seed and **unmodified** config:
   ```bash
   python src/dendritic_modeling/scripts/training/train_experiments.py configs/config_exp.yaml
   ```
   Run again (same command).
3. **Confirm:** Same final metrics within a **tight numeric tolerance** across the two runs:
   - Test accuracy within **0.1% absolute**
   - Loss within **1e-3**
   Logs do not need to match line-by-line (avoids GPU nondeterminism issues).
4. **Seed used:** `experiment.seed: 42` in config.

## Vision baseline (CIFAR-10, classical CNN)

- Config: `configs/vision/alexnet_cifar10_baseline.yaml`
- Encoder: CNNDownsample (locked hyperparams), wrapped in `VisionEncoderWrapper`.
- Core: Identity. Decoder: MLP → 10 classes.
- Run: `python src/dendritic_modeling/scripts/training/train_experiments.py configs/vision/alexnet_cifar10_baseline.yaml`
- Sanity test: `PYTHONPATH=src python -m dendritic_modeling.Adi.test_vision_baseline` (from repo root).

## Tolerance note

If two runs with the same seed do not meet the tolerance (0.1% accuracy, 1e-3 loss), document the observed difference and environment (Python, CUDA, GPU). Do not block on exact bit-level reproducibility.
