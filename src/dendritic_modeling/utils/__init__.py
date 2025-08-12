"""
General utilities for dendritic modeling.

Common utility functions for data serialization, logging, seed management,
and configuration validation.
"""

# New validation modules
from .config_validation import (
    get_validation_summary,
    validate_architecture_only,
    validate_full_config,
    validate_training_only,
)

# Core utilities
from .general import Shaper, convert_to_serializable, save_dict, set_seed
from .logging_config import get_logger, setup_logging

# Math utilities
from .math import roc_auc_score
from .model_architecture_validator import (
    ModelArchitectureValidator,
    validate_model_architecture,
)
from .sparse_ops import (
    calculate_effective_sparsity,
    count_active_parameters,
    estimate_memory_usage,
    get_available_sparse_types,
    get_sparse_layer,
    parallel_topk_selection,
)
from .training_stability_validator import (
    apply_automatic_safeguards,
    suggest_optimal_config_for_model_size,
    validate_training_stability,
)

__all__ = [
    "ModelArchitectureValidator",
    # Core utilities
    "Shaper",
    "apply_automatic_safeguards",
    # Configuration and validation
    "convert_to_serializable",
    "get_logger",
    "get_validation_summary",
    # Math utilities
    "roc_auc_score",
    "save_dict",
    "set_seed",
    "setup_logging",
    "suggest_optimal_config_for_model_size",
    "validate_architecture_only",
    # New validation functions
    "validate_full_config",
    "validate_model_architecture",
    "validate_training_only",
    "validate_training_stability",
    # Note: Backward compatibility imports commented out due to circular imports
    # "accuracy_score",
    # "split_MNIST_inputs",
    # "load_MNIST_modulo10"
]
