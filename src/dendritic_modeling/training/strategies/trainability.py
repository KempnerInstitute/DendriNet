"""Train a subset of parameters for one phase and restore the rest exactly.

Some parameters are fixed on purpose (the slope of fixed ``linear_sigmoid`` or
``linear_tanh`` gates, frozen encoders), so a phase that trains one layer must
leave them fixed, and the model's trainability must return to its prior state
afterwards rather than to "everything trainable".
"""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn

TrainabilitySnapshot = dict[nn.Parameter, bool]


def snapshot_requires_grad(model: nn.Module) -> TrainabilitySnapshot:
    """Record ``requires_grad`` of every parameter of ``model``."""
    return {param: bool(param.requires_grad) for param in model.parameters()}


def train_only(
    model: nn.Module,
    modules: Iterable[nn.Module],
    snapshot: TrainabilitySnapshot,
) -> None:
    """Train the parameters of ``modules`` that ``snapshot`` records as trainable.

    Every other parameter of ``model`` is frozen. Parameters of ``modules``
    outside ``model`` (a phase's own heads) are trained.
    """
    selected = [param for module in modules for param in module.parameters()]
    for param in model.parameters():
        param.requires_grad = False
    for param in selected:
        param.requires_grad = snapshot.get(param, True)


def restore_requires_grad(snapshot: TrainabilitySnapshot) -> None:
    """Return every recorded parameter to its recorded ``requires_grad``."""
    for param, requires_grad in snapshot.items():
        param.requires_grad = requires_grad


__all__ = [
    "TrainabilitySnapshot",
    "restore_requires_grad",
    "snapshot_requires_grad",
    "train_only",
]
