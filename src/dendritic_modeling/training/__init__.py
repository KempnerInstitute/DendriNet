"""
Training package for dendritic models.

This package provides various training strategies, optimizers, and utilities.
"""

from . import strategies
from . import optimizers
from .factory import (
    get_trainer,
    train_simple_mlp,
    train_mlp_with_classification_head,
    transfer_mlp_weights_to_input_net,
    extract_weights_from_simple_mlp,
)

__all__ = [
    "strategies",
    "optimizers", 
    "get_trainer",
    "train_simple_mlp",
    "train_mlp_with_classification_head",
    "transfer_mlp_weights_to_input_net",
    "extract_weights_from_simple_mlp",
] 