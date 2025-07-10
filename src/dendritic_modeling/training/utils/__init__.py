"""
Training utilities module.

This module provides utilities for training dendritic models including:
- Backward hooks for gradient manipulation
- Evaluation functions for model performance
"""

from .evaluation import (
    accuracy_score,
    categorical_loglikelihood_score,
    evaluate_accuracy,
    evaluate_auc,
    evaluate_categorical_loglikelihood,
    evaluate_cosine_similarity,
    evaluate_mse,
    mse_score,
)
from .hooks import safe_backward_hook

__all__ = [
    # Hooks
    "safe_backward_hook",
    # Evaluation
    "accuracy_score",
    "evaluate_accuracy",
    "evaluate_auc",
    "categorical_loglikelihood_score",
    "evaluate_categorical_loglikelihood",
    "mse_score",
    "evaluate_mse",
    "evaluate_cosine_similarity",
]
