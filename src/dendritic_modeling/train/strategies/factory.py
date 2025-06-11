"""
Factory function for creating training strategy instances.
"""

from dendritic_modeling import logger
from .mle import TrainerMLE, TrainOnlyMReactivation
from .regression import TrainerRegression
from .local_credit import LocalCreditAssignment
from .two_step import TwoStepTrainer, TwoStepTrainerWithKL
from .voltage_stabilization import VoltageStabilizationTrainer


def get_trainer(strategy, optimizer, suppress_prints=False, print_every=10, **kwargs):
    """
    Factory function to get the appropriate trainer based on strategy string.
    
    Args:
        strategy: String identifying the training strategy
        optimizer: PyTorch optimizer instance
        suppress_prints: Whether to suppress training prints
        print_every: Print frequency for training logs
        **kwargs: Additional arguments passed to specific trainers
    
    Returns:
        Trainer instance
    """
    strategy = strategy.lower()
    
    if strategy == "mle":
        return TrainerMLE(optimizer, suppress_prints, print_every)
    elif strategy == "freeze_branches":
        return TrainerMLE(optimizer, suppress_prints, print_every)
    elif strategy == "freeze_branches_reversed":
        return TrainerMLE(optimizer, suppress_prints, print_every)
    elif strategy == "freeze_branches_with_pretraining":
        return TrainerMLE(optimizer, suppress_prints, print_every)
    elif strategy == "freeze_branches_reversed_with_pretraining":
        return TrainerMLE(optimizer, suppress_prints, print_every)
    elif strategy == "train_only_m_reactivation":
        return TrainOnlyMReactivation(optimizer, suppress_prints, print_every)
    elif strategy == "local_credit_assignment":
        return LocalCreditAssignment(optimizer, suppress_prints, print_every)
    elif strategy == "two_step":
        return TwoStepTrainer(optimizer, suppress_prints, print_every)
    elif strategy == "two_step_kl":
        return TwoStepTrainerWithKL(optimizer, suppress_prints, print_every)
    elif strategy == "voltage_stabilization":
        warmup_epochs = kwargs.get("warmup_epochs", 1)
        stability_target_voltage = kwargs.get("stability_target_voltage", 0.5)
        return VoltageStabilizationTrainer(
            optimizer, suppress_prints, print_every, 
            warmup_epochs, stability_target_voltage
        )
    elif strategy == "mse" or strategy == "regression":
        use_scheduler = kwargs.get("use_scheduler", True)
        min_lr = kwargs.get("min_lr", 1e-6)
        return TrainerRegression(optimizer, suppress_prints, print_every, use_scheduler, min_lr)
    else:
        logger.info(f"Unrecognized strategy '{strategy}', defaulting to 'mle'.")
        return TrainerMLE(optimizer, suppress_prints, print_every) 