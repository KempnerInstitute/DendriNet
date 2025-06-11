"""
Utilities package for dendritic modeling.

Contains utility functions for data handling, evaluation, noise addition, and other common operations.
"""

# Data utilities
from .data import (
    Shaper,
    convert_to_serializable,
    set_seed,
    save_dict,
)

# Evaluation utilities
from .evaluation import (
    accuracy_score,
    evaluate_accuracy,
    categorical_loglikelihood_score,
    evaluate_categorical_loglikelihood,
    mse_score,
    evaluate_mse,
)

# Noise utilities
from .noise import (
    add_gaussian_noise,
    add_uniform_noise,
    add_poisson_noise,
    add_frequency_noise,
)

__all__ = [
    # Data utilities
    "Shaper",
    "convert_to_serializable",
    "set_seed",
    "save_dict",
    
    # Evaluation utilities
    "accuracy_score",
    "evaluate_accuracy",
    "categorical_loglikelihood_score",
    "evaluate_categorical_loglikelihood",
    "mse_score",
    "evaluate_mse",
    
    # Noise utilities
    "add_gaussian_noise",
    "add_uniform_noise",
    "add_poisson_noise",
    "add_frequency_noise",
] 