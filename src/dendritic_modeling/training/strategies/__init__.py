"""
Training strategies for dendritic models.

This module contains various training strategies and algorithms
for dendritic network models.
"""

from dendritic_modeling.training.strategies.layerwise import LayerWiseTrainer
from dendritic_modeling.training.strategies.local_learning import LocalCreditAssignment
from dendritic_modeling.training.strategies.specialized import (
    TrainOnlyMReactivation,
    VoltageStabilizationTrainer,
)
from dendritic_modeling.training.strategies.standard import Trainer
from dendritic_modeling.training.strategies.two_step import (
    TwoStepTrainer,
    TwoStepTrainerWithKL,
)

__all__ = [
    "LayerWiseTrainer",
    "LocalCreditAssignment",
    "TrainOnlyMReactivation",
    "Trainer",
    "TwoStepTrainer",
    "TwoStepTrainerWithKL",
    "VoltageStabilizationTrainer",
]
