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
    # Evaluation
    "accuracy_score"
    "categorical_loglikelihood_score"
    "evaluate_accuracy"
    "evaluate_auc"
    "evaluate_categorical_loglikelihood"
    "evaluate_cosine_similarity"
    "evaluate_mse"
    "mse_score"
    # Hooks
    "safe_backward_hook"
]
