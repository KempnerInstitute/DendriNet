"""
Custom optimizer implementations.

This module contains custom optimizer wrappers and implementations for
dendritic network training.
"""

from collections.abc import Iterator

import torch
from torch.optim import Optimizer

from dendritic_modeling.models import BaseModel


def _iter_rewired_optimizer_modules(
    model,
) -> Iterator[tuple[object, torch.nn.Parameter]]:
    """Yield modules that expose rewiring masks and optimizer-backed weights."""
    for module in model.modules():
        consume = getattr(module, "consume_rewired_mask", None)
        pre_w = getattr(module, "pre_w", None)
        if callable(consume) and pre_w is not None:
            yield module, pre_w


def _clear_rewired_optimizer_state(model, optimizer: Optimizer) -> None:
    """Clear optimizer moments for sparse slots replaced during rewiring."""
    for module, pre_w in _iter_rewired_optimizer_modules(model):
        consume = module.consume_rewired_mask
        mask = consume()
        if mask is None:
            continue
        state = optimizer.state.get(pre_w)
        if not state:
            continue
        mask = mask.to(device=pre_w.device)
        for value in state.values():
            if getattr(value, "shape", None) == pre_w.shape:
                value[mask] = 0


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
        self.optimizer.step()
        self.model.apply_rewiring()
        _clear_rewired_optimizer_state(self.model, self.optimizer)


__all__ = ["CustomWeightDecayOptimizer"]
