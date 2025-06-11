"""
Training package for dendritic models.

This package provides various training strategies and utilities:
- strategies: Different training approaches (MLE, regression, local credit, etc.)
- gradient_scaling: Gradient scaling utilities
- Factory function for easy trainer instantiation
"""

from .strategies import (
    # Base classes
    BaseTrainer,
    CustomWeightDecayOptimizer,
    compute_metrics_classification,
    
    # Training strategies
    TrainerMLE,
    TrainOnlyMReactivation,
    TrainerRegression,
    LocalCreditAssignment,
    TwoStepTrainer,
    TwoStepTrainerWithKL,
    VoltageStabilizationTrainer,
    
    # Utility functions
    voltage_warmup_loss,
    generate_warmup_report,
    
    # Factory function
    get_trainer,
)

# Import gradient scaling if it exists
try:
    from . import gradient_scaling
except ImportError:
    gradient_scaling = None

__all__ = [
    # Base classes
    "BaseTrainer",
    "CustomWeightDecayOptimizer",
    "compute_metrics_classification",
    
    # Training strategies
    "TrainerMLE",
    "TrainOnlyMReactivation", 
    "TrainerRegression",
    "LocalCreditAssignment",
    "TwoStepTrainer",
    "TwoStepTrainerWithKL",
    "VoltageStabilizationTrainer",
    
    # Utility functions
    "voltage_warmup_loss",
    "generate_warmup_report",
    
    # Factory function
    "get_trainer",
    
    # Submodules
    "strategies",
]

if gradient_scaling is not None:
    __all__.append("gradient_scaling") 