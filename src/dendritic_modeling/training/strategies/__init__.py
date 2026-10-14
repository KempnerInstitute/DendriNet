"""
Training strategies for dendritic models.

This module contains various training strategies and algorithms
for dendritic network models.
"""

from dendritic_modeling.training.strategies.feedback_alignment import (
    FeedbackAlignmentTrainer,
    ShuntingFeedbackAlignmentTrainer,
)
from dendritic_modeling.training.strategies.layerwise import LayerWiseTrainer
from dendritic_modeling.training.strategies.local_learning import LocalCreditAssignment
from dendritic_modeling.training.strategies.multi_stage import MultiStageTrainer
from dendritic_modeling.training.strategies.presentation_stream import (
    PresentationStreamTrainer,
)
from dendritic_modeling.training.strategies.recurrent_trainer import RecurrentTrainer
from dendritic_modeling.training.strategies.soma_dfa import SomaDFATrainer
from dendritic_modeling.training.strategies.specialized import (
    ExcitationInhibitionEquilibriumTrainer,
    HomeostaticControlTrainer,
    TrainOnlyMReactivation,
    VoltageStabilizationTrainer,
)
from dendritic_modeling.training.strategies.standard import Trainer
from dendritic_modeling.training.strategies.two_step import (
    TwoStepTrainer,
    TwoStepTrainerWithKL,
)

__all__ = [
    "ExcitationInhibitionEquilibriumTrainer",
    "FeedbackAlignmentTrainer",
    "HomeostaticControlTrainer",
    "LayerWiseTrainer",
    "LocalCreditAssignment",
    "MultiStageTrainer",
    "PresentationStreamTrainer",
    "RecurrentTrainer",
    "ShuntingFeedbackAlignmentTrainer",
    "SomaDFATrainer",
    "TrainOnlyMReactivation",
    "Trainer",
    "TwoStepTrainer",
    "TwoStepTrainerWithKL",
    "VoltageStabilizationTrainer",
]
