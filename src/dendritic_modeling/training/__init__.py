"""
Training package for dendritic models.

This package provides various training strategies, optimizers, and utilities.
"""

from dendritic_modeling.networks import GradientScaler
from dendritic_modeling.training.factory import (
    extract_weights_from_simple_mlp, get_trainer,
    transfer_mlp_weights_to_input_net)
from dendritic_modeling.training.optimizers import CustomWeightDecayOptimizer
from dendritic_modeling.training.strategies import Trainer

__all__ = [
    "CustomWeightDecayOptimizer"
    "extract_weights_from_simple_mlp"
    "get_trainer"
    "GradientScaler"
    "Trainer"
    "transfer_mlp_weights_to_input_net"
]
