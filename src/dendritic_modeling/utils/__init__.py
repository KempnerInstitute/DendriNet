"""
General utilities for dendritic modeling.

This module contains core utility functions for:
- Data handling and serialization
- Logging configuration
- Random seed management
- Data reshaping utilities
"""

# Core utilities
from .data import (
    Shaper,
    convert_to_serializable,
    set_seed,
    save_dict
)

from .logging_config import (
    setup_logging,
    get_logger
)

# Import evaluation utilities from training module
from ..training.utils.evaluation import (
    accuracy_score,
    evaluate_accuracy,
    auc_score,
    evaluate_auc,
    categorical_loglikelihood_score,
    evaluate_categorical_loglikelihood,
    mse_score,
    evaluate_mse,
    cosine_similarity_score,
    evaluate_cosine_similarity
)

# Import hooks from training utils
from ..training.utils.hooks import (
    safe_backward_hook,
    safe_register_hooks,
    register_safe_hook,
    safe_hook_fn,
    SafeHookModule
)

# Import noise utilities from datasets
from ..datasets.augmentation.noise import (
    add_gaussian_noise,
    add_uniform_noise,
    add_poisson_noise,
    add_frequency_noise
)

__all__ = [
    # Core utilities
    "Shaper",
    "convert_to_serializable",
    "set_seed",
    "save_dict",
    "setup_logging",
    "get_logger",
    # Evaluation (from training)
    "accuracy_score",
    "evaluate_accuracy",
    "auc_score",
    "evaluate_auc",
    "categorical_loglikelihood_score",
    "evaluate_categorical_loglikelihood",
    "mse_score",
    "evaluate_mse",
    "cosine_similarity_score",
    "evaluate_cosine_similarity",
    
    # Noise utilities
    "add_gaussian_noise",
    "add_uniform_noise",
    "add_poisson_noise",
    "add_frequency_noise",
    
    # Hook utilities
    # Hooks (from training)
    "safe_backward_hook",
    "safe_register_hooks",
    "register_safe_hook",
    "safe_hook_fn",
    "SafeHookModule",
    # Noise (from datasets)
    "add_gaussian_noise",
    "add_uniform_noise",
    "add_poisson_noise",
    "add_frequency_noise"
] 