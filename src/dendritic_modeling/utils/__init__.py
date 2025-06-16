"""
Utilities package for dendritic modeling.

Contains utility functions for data handling, evaluation, noise addition, and other common operations.
"""

# Data utilities
from dendritic_modeling.utils.data import (
    Shaper,
    convert_to_serializable,
    set_seed,
    save_dict,
)

# Evaluation utilities
from dendritic_modeling.utils.evaluation import (
    accuracy_score,
    evaluate_accuracy,
    categorical_loglikelihood_score,
    evaluate_categorical_loglikelihood,
    mse_score,
    evaluate_mse,
)

# Noise utilities
from dendritic_modeling.utils.noise import (
    add_gaussian_noise,
    add_uniform_noise,
    add_poisson_noise,
    add_frequency_noise,
)

# Hook utilities
from dendritic_modeling.utils.hooks import (
    safe_backward_hook,
    safe_register_hooks,
    register_safe_hook,
    safe_hook_fn,
    SafeHookModule,
)

# Classification utilities
from dendritic_modeling.utils.classification import (
    MLPClassifier,
    train_classifier,
    evaluate_classifier,
    get_layer_activations,
    get_ablated_activations,
    analyze_layerwise_ablation,
    train_and_evaluate_classifier,
)

# Analysis utilities
from dendritic_modeling.utils.analysis import (
    record_branch_activations,
    compute_MI,
    analyze_branch_information,
    measure_performance,
    ablate_branch,
    analyze_branch_ablation,
    compute_entropy,
    compute_mutual_information,
    estimate_hidden_output_MI,
    analyze_dendritic_information,
)

# Logging utilities
from dendritic_modeling.utils.logging_config import (
    LoggerManager,
    LOGGER_NAME,
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
    
    # Hook utilities
    "safe_backward_hook",
    "safe_register_hooks",
    "register_safe_hook",
    "safe_hook_fn",
    "SafeHookModule",
    
    # Classification utilities
    "MLPClassifier",
    "train_classifier",
    "evaluate_classifier",
    "get_layer_activations",
    "get_ablated_activations",
    "analyze_layerwise_ablation",
    "train_and_evaluate_classifier",
    
    # Analysis utilities
    "record_branch_activations",
    "compute_MI",
    "analyze_branch_information",
    "measure_performance",
    "ablate_branch",
    "analyze_branch_ablation",
    "compute_entropy",
    "compute_mutual_information",
    "estimate_hidden_output_MI",
    "analyze_dendritic_information",
    
    # Logging utilities
    "LoggerManager",
    "LOGGER_NAME",
] 