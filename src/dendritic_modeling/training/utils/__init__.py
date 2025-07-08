"""
Training utilities module.

This module provides utilities for training dendritic models including:
- Backward hooks for gradient manipulation
- Evaluation functions for model performance
"""

from .hooks import safe_backward_hook
from .evaluation import (
    accuracy_score,
    evaluate_accuracy,
    categorical_loglikelihood_score,
    evaluate_categorical_loglikelihood,
    mse_score,
    evaluate_mse
)

__all__ = [
    # Hooks
    "safe_backward_hook",
    # Evaluation
    "accuracy_score",
    "evaluate_accuracy",
    "categorical_loglikelihood_score",
    "evaluate_categorical_loglikelihood",
    "mse_score",
    "evaluate_mse",
] 