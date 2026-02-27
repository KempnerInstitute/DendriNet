"""
Custom optimizer implementations.

This module contains custom optimizer wrappers and implementations for
dendritic network training.
"""

from torch.optim import Optimizer

from dendritic_modeling.models import BaseModel


class CustomWeightDecayOptimizer:
    """
    Custom weight decay optimizer wrapper.

    This optimizer wrapper applies weight decay directly to the model parameters
    by calling the model's decay_weights method, rather than using PyTorch's
    built-in weight decay. This allows for more flexible weight decay strategies
    that are model-specific.

    Args:
        model: The model to apply weight decay to
        optimizer: The underlying PyTorch optimizer
        weight_decay: Weight decay coefficient
    """

    def __init__(
        self,
        model: BaseModel,
        optimizer: Optimizer,
        weight_decay: float = 0.1,
        weight_boosting: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.weight_boosting = weight_boosting

    @property
    def param_groups(self):
        """Expose the underlying optimizer's param_groups."""
        return self.optimizer.param_groups

    @property
    def defaults(self):
        """Expose the underlying optimizer's defaults."""
        return self.optimizer.defaults

    def add_param_group(self, param_group: dict):
        """Delegate add_param_group to the underlying optimizer."""
        self.optimizer.add_param_group(param_group)

    def zero_grad(self):
        """Zero gradients of the underlying optimizer."""
        self.optimizer.zero_grad()

    def step(self):
        """
        Perform optimization step with custom weight decay.

        This method first applies model-specific weight decay using the current
        learning rate, then performs the standard optimizer step.
        """
        lr = self.optimizer.param_groups[0]["lr"]
        if self.weight_decay > 0:
            self.model.decay_weights(
                weight_decay=lr * self.weight_decay,
                weight_boosting=self.weight_boosting,
            )
        self.model.apply_rewiring()
        self.optimizer.step()


__all__ = ["CustomWeightDecayOptimizer"]
