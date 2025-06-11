"""
Training strategies for dendritic models.

This module contains various training strategies and algorithms
for dendritic network models.
"""

from .base import BaseTrainer, compute_metrics_classification
from .mle import TrainerMLE
from .regression import TrainerRegression
from .local_learning import LocalCreditAssignment
from .two_step import TwoStepTrainer, TwoStepTrainerWithKL
from .specialized import (
    VoltageStabilizationTrainer, 
    TrainOnlyMReactivation, 
    InhibitoryHeadTrainer,
    voltage_warmup_loss,
    generate_warmup_report
)

__all__ = [
    # Base classes and utilities
    "BaseTrainer", 
    "compute_metrics_classification", 
    
    # Main training strategies
    "TrainerMLE", 
    "TrainerRegression", 
    "LocalCreditAssignment",
    "TwoStepTrainer",
    "TwoStepTrainerWithKL",
    
    # Specialized training strategies
    "VoltageStabilizationTrainer",
    "TrainOnlyMReactivation",
    "InhibitoryHeadTrainer",
    
    # Utility functions
    "voltage_warmup_loss",
    "generate_warmup_report"
] 