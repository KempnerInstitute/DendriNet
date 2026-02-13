"""
Adi: Vision baseline and dendritic ablation extensions.

All new code for the vision baseline and depth-indexed dendritic swaps
lives here. No modifications to the rest of dendritic_modeling.
"""

from dendritic_modeling.Adi.vision_encoder import (
    VisionEncoderWrapper,
    get_baseline_encoder,
    get_hybrid_encoder,
)

# robustness_eval: create Adi/robustness_eval.py with evaluate_cifar10c(checkpoint_path, results_path)
# for CIFAR-10-C mCE; cite Hendrycks & Dietterich 2019.

__all__ = [
    "VisionEncoderWrapper",
    "get_baseline_encoder",
    "get_hybrid_encoder",
]
