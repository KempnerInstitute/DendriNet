"""Gradient clipping and non-finite gradient handling shared by the trainers."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from dendritic_modeling.config.training import GRAD_CLIP_MODES


def clip_gradients(
    parameters: Iterable[torch.nn.Parameter],
    max_value: float | None,
    mode: str,
) -> None:
    """Clip gradients in place; a ``max_value`` of 0 or None disables clipping.

    ``mode="value"`` clamps every gradient element to ``[-max_value,
    max_value]``; ``mode="norm"`` rescales all gradients together so their
    global L2 norm is at most ``max_value``.
    """
    if not max_value:
        return
    if mode == "norm":
        torch.nn.utils.clip_grad_norm_(parameters, float(max_value))
    elif mode == "value":
        torch.nn.utils.clip_grad_value_(parameters, float(max_value))
    else:
        raise ValueError(
            f"grad_clip_mode must be one of {GRAD_CLIP_MODES}, got {mode!r}"
        )


def zero_nonfinite_gradients(parameters: Iterable[torch.nn.Parameter]) -> bool:
    """Set NaN and infinite gradient entries to zero; return whether any existed.

    The check is one fused reduction with a single host synchronization. Under
    AMP it must run after ``GradScaler.unscale_``, which has already recorded
    the overflow, so the scaler still skips the step and lowers its scale.
    """
    grads = [param.grad for param in parameters if param.grad is not None]
    if not grads:
        return False
    device = grads[0].device
    nonfinite = torch.stack(
        [(~torch.isfinite(grad)).any().to(device) for grad in grads]
    ).any()
    if not bool(nonfinite):
        return False
    for grad in grads:
        torch.nan_to_num_(grad, nan=0.0, posinf=0.0, neginf=0.0)
    return True


__all__ = ["clip_gradients", "zero_nonfinite_gradients"]
