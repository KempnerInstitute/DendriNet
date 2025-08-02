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
from .logging_config import get_logger, setup_logging
# New validation modules
from .config_validation import (
    validate_full_config,
    validate_architecture_only,
    validate_training_only,
    get_validation_summary,
)
from .model_architecture_validator import (
    ModelArchitectureValidator,
    validate_model_architecture,
    validate_config,  # Legacy compatibility
)
from .training_stability_validator import (
    validate_training_stability,
    validate_optimization_combinations,  # Legacy compatibility
    suggest_optimal_config_for_model_size,
    apply_automatic_safeguards,
)

# Legacy imports for backward compatibility
ConfigValidator = ModelArchitectureValidator
from .sparse_ops import (
    get_sparse_layer,
    get_available_sparse_types,
    calculate_effective_sparsity,
    count_active_parameters,
    estimate_memory_usage,
    parallel_topk_selection,
)

# Math utilities
from .math import roc_auc_score

# Import from other modules for backward compatibility
# Note: Commented out due to circular import issues - import these directly when needed
# from dendritic_modeling.training.utils.evaluation import accuracy_score
# from dendritic_modeling.plotting.core.utils import split_MNIST_inputs
# from dendritic_modeling.datasets import load_MNIST_modulo10

__all__ = [
    # Configuration and validation
    "convert_to_serializable",
    "get_logger",
    "save_dict",
    "set_seed",
    "setup_logging",
    # Core utilities
    "Shaper",
    # New validation functions
    "validate_full_config",
    "validate_architecture_only", 
    "validate_training_only",
    "validate_model_architecture",
    "validate_training_stability",
    "get_validation_summary",
    "ModelArchitectureValidator",
    # Legacy compatibility
    "ConfigValidator",
    "validate_config",
    "validate_optimization_combinations",
    "suggest_optimal_config_for_model_size", 
    "apply_automatic_safeguards",
    # Math utilities
    "roc_auc_score",
    # Note: Backward compatibility imports commented out due to circular imports
    # "accuracy_score",
    # "split_MNIST_inputs",
    # "load_MNIST_modulo10"
]
