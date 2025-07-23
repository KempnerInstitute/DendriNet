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
    save_dict,
    Data,
    load_MNIST
)
from .logging_config import get_logger, setup_logging
from dendritic_modeling.utils.config_validator import ConfigValidator, validate_config
from .sparse_ops import (
    get_sparse_layer,
    get_available_sparse_types,
    calculate_effective_sparsity,
    count_active_parameters,
    estimate_memory_usage,
    parallel_topk_selection,
)
from .clean_outputs import *

# Math utilities
from .math import roc_auc_score

# Import from other modules for backward compatibility
from dendritic_modeling.training.utils.evaluation import accuracy_score
from dendritic_modeling.plotting.utils import split_MNIST_inputs
from dendritic_modeling.datasets import load_MNIST_modulo10

__all__ = [
    "convert_to_serializable",
    "get_logger",
    "save_dict",
    "set_seed",
    "setup_logging",
    # Core utilities
    "Shaper",
    "ConfigValidator",
    "validate_config",
    # Math utilities
    "roc_auc_score",
    # Backward compatibility imports
    "accuracy_score",
    "split_MNIST_inputs",
    "load_MNIST_modulo10"
]
