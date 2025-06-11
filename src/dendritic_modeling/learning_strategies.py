"""
learning_strategies.py
======================
This module maintains backward compatibility by importing all training strategies
and factory functions from the new organized structure in the training/ directory.

The training strategies have been organized into specialized modules:
- training.strategies.base: BaseTrainer, compute_metrics_classification
- training.strategies.mle: TrainerMLE (classic NLL backprop)
- training.strategies.regression: TrainerRegression
- training.strategies.local_learning: LocalCreditAssignment (advanced local learning)
- training.strategies.two_step: TwoStepTrainer, TwoStepTrainerWithKL (local supervision)
- training.strategies.specialized: VoltageStabilizationTrainer, TrainOnlyMReactivation, InhibitoryHeadTrainer
- training.optimizers: CustomWeightDecayOptimizer
- training.factory: get_trainer, train_simple_mlp, weight transfer utilities

For new code, consider importing directly from the specific modules for better clarity.
"""

# Import all training strategies from the organized structure
from dendritic_modeling.training.strategies import (
    BaseTrainer, 
    compute_metrics_classification, 
    TrainerMLE, 
    TrainerRegression, 
    LocalCreditAssignment, 
    TwoStepTrainer, 
    TwoStepTrainerWithKL,
    VoltageStabilizationTrainer,
    TrainOnlyMReactivation,
    InhibitoryHeadTrainer,
    voltage_warmup_loss,
    generate_warmup_report
)

# Import factory functions and utilities
from dendritic_modeling.training import (
    get_trainer,
    train_simple_mlp,
    train_mlp_with_classification_head,
    transfer_mlp_weights_to_input_net,
    extract_weights_from_simple_mlp,
)

# Import optimizer
from dendritic_modeling.training.optimizers import CustomWeightDecayOptimizer

# This maintains full backward compatibility with existing code that imports from learning_strategies.py
# while providing access to the new organized structure via training.strategies.*, training.factory, etc. 

# Training strategy classes
from .training.strategies import (
    LocalCreditAssignment,
    TrainOnlyMReactivation, 
    TwoStepTrainer,
    TwoStepTrainerWithKL,
    VoltageStabilizationTrainer,
    InhibitoryHeadTrainer,
)

# Factory functions and training utilities
from .training.factory import (
    get_trainer,
    train_simple_mlp, 
    train_mlp_with_classification_head,
)

# Transfer learning utilities
from .training.utils import (
    transfer_mlp_weights_to_input_net,
    extract_weights_from_simple_mlp,
    voltage_warmup_loss,
    generate_warmup_report,
)

# Base trainer methods (including recovered layer training methods)
from .training.strategies.base import BaseTrainer

# Re-export recovered layer training methods for backward compatibility
def train_freeze_layers_by_layer(trainer_instance, *args, **kwargs):
    """Layer-by-layer training from soma to input. Use trainer.train_freeze_layers_by_layer()"""
    return trainer_instance.train_freeze_layers_by_layer(*args, **kwargs)

def train_freeze_layers_reversed_by_layer(trainer_instance, *args, **kwargs):
    """Layer-by-layer training from input to soma. Use trainer.train_freeze_layers_reversed_by_layer()"""
    return trainer_instance.train_freeze_layers_reversed_by_layer(*args, **kwargs)

def train_freeze_layers(trainer_instance, *args, **kwargs):
    """Alias for train_freeze_layers_by_layer. Use trainer.train_freeze_layers()"""
    return trainer_instance.train_freeze_layers(*args, **kwargs)

def train_freeze_layers_reversed(trainer_instance, *args, **kwargs):
    """Alias for train_freeze_layers_reversed_by_layer. Use trainer.train_freeze_layers_reversed()"""
    return trainer_instance.train_freeze_layers_reversed(*args, **kwargs)


__all__ = [
    # Training strategy classes
    'LocalCreditAssignment',
    'TrainOnlyMReactivation', 
    'TwoStepTrainer',
    'TwoStepTrainerWithKL',
    'VoltageStabilizationTrainer',
    'InhibitoryHeadTrainer',
    # Factory functions
    'get_trainer',
    'train_simple_mlp', 
    'train_mlp_with_classification_head',
    # Transfer learning utilities
    'transfer_mlp_weights_to_input_net',
    'extract_weights_from_simple_mlp',
    'voltage_warmup_loss',
    'generate_warmup_report',
    # Base trainer class
    'BaseTrainer',
    # Recovered layer training methods
    'train_freeze_layers_by_layer',
    'train_freeze_layers_reversed_by_layer',
    'train_freeze_layers',
    'train_freeze_layers_reversed',
] 