"""
Training package for dendritic models.

This package provides various training strategies, optimizers, and utilities.
"""

from dendritic_modeling.training.optimizers import CustomWeightDecayOptimizer
from dendritic_modeling.training.strategies import Trainer
from dendritic_modeling.training.factory import (
    get_trainer,
    transfer_mlp_weights_to_input_net,
    extract_weights_from_simple_mlp,
)
from dendritic_modeling.networks import GradientScaler

__all__ = [
    "CustomWeightDecayOptimizer",
    "get_trainer",
    "transfer_mlp_weights_to_input_net",
    "extract_weights_from_simple_mlp",
    "Trainer",
    "GradientScaler"
] 