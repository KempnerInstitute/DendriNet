"""
Training strategies for dendritic models.

This module contains various training strategies and algorithms
for dendritic network models.
"""

from .base import BaseTrainer, compute_metrics_classification
from .mle import TrainerMLE
from .regression import TrainerRegression
from .local_learning import LocalCreditAssignment

__all__ = ["BaseTrainer", "compute_metrics_classification", "TrainerMLE", "TrainerRegression", "LocalCreditAssignment"] 