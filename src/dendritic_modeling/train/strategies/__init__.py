"""
Training strategies package.

This package contains various training strategies for dendritic models:
- MLE: Classic maximum likelihood estimation training
- Regression: Training for regression tasks with learning rate scheduling
- Local Credit Assignment: Local learning using feedback alignment
- Two-Step: Layer-wise pretraining followed by end-to-end training
- Voltage Stabilization: Training with voltage warm-up phase
"""

from .base import BaseTrainer, CustomWeightDecayOptimizer, compute_metrics_classification
from .mle import TrainerMLE, TrainOnlyMReactivation
from .regression import TrainerRegression
from .local_credit import LocalCreditAssignment
from .two_step import TwoStepTrainer, TwoStepTrainerWithKL
from .voltage_stabilization import VoltageStabilizationTrainer, voltage_warmup_loss, generate_warmup_report
from .factory import get_trainer

__all__ = [
    # Base classes
    "BaseTrainer",
    "CustomWeightDecayOptimizer", 
    "compute_metrics_classification",
    
    # MLE strategies
    "TrainerMLE",
    "TrainOnlyMReactivation",
    
    # Regression training
    "TrainerRegression",
    
    # Local credit assignment
    "LocalCreditAssignment",
    
    # Two-step training
    "TwoStepTrainer",
    "TwoStepTrainerWithKL",
    
    # Voltage stabilization
    "VoltageStabilizationTrainer",
    "voltage_warmup_loss",
    "generate_warmup_report",
    
    # Factory function
    "get_trainer",
] 