"""
Training package for dendritic models.

This package provides various training strategies, optimizers, and utilities.
"""

from dendritic_modeling.networks import GradientScaler
from dendritic_modeling.training.factory import (
    get_available_trainer_strategies,
    get_trainer,
    register_trainer_strategy,
)
from dendritic_modeling.training.optimizers import CustomWeightDecayOptimizer
from dendritic_modeling.training.strategies import Trainer

__all__ = [
    "CustomWeightDecayOptimizer",
    "GradientScaler",
    "Trainer",
    "get_available_trainer_strategies",
    "get_trainer",
    "register_trainer_strategy",
]
