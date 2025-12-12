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
from .lda import (
    apply_nonnegative_lda,
    apply_random_weights,
    compute_lda_importance_scores,
    fit_nonnegative_lda,
    generate_random_nonnegative_weights,
    select_topk_features_by_lda,
)
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
    "Shaper",
    "apply_automatic_safeguards",
    "apply_nonnegative_lda",
    "apply_random_weights",
    "compute_lda_importance_scores",
    "convert_to_serializable",
    "fit_nonnegative_lda",
    "generate_random_nonnegative_weights",
    "get_logger",
    "get_validation_summary",
    "roc_auc_score",
    "save_dict",
    "select_topk_features_by_lda",
    "set_seed",
    "setup_logging",
    "suggest_optimal_config_for_model_size",
    "validate_architecture_only",
    "validate_full_config",
    "validate_model_architecture",
    "validate_training_only",
    "validate_training_stability",
]
