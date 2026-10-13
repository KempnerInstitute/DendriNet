"""Keep reactivation-gate state identical across distributed ranks.

Data-driven calibration writes gate parameters, buffers and, for gates with a
calibrated ``fixed_b``, a Python float the forward pass reads. DDP broadcasts
parameters only when it wraps the model, and FSDP's ``sync_module_states``
broadcasts tensors but not Python attributes, so calibration on one rank, or
on several ranks that disagree in the last bits, must be propagated explicitly.
"""

from __future__ import annotations

import torch.distributed as dist
from torch import nn

from dendritic_modeling.networks.activations.endpoint_tanh import EndpointNormalizedTanh
from dendritic_modeling.networks.activations.parametric import (
    ParametricActivation,
    _PersistentFixedBActivation,
)
from dendritic_modeling.utils.hooks import iter_modules_of_type


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def broadcast_reactivation_state(model: nn.Module, *, src: int = 0) -> None:
    """Give every rank the reactivation-gate state of rank ``src``.

    Each gate's parameters and buffers are broadcast in module order and
    loaded with ``load_state_dict``, so the calibrated ``fixed_b`` floats
    follow their buffers. Does nothing outside a multi-process run.
    """
    if not _distributed():
        return
    for module in iter_modules_of_type(
        model, (ParametricActivation, EndpointNormalizedTanh)
    ):
        state = module.state_dict()
        for key, tensor in state.items():
            state[key] = tensor.clone()
            dist.broadcast(state[key], src=src)
        module.load_state_dict(state)


def refresh_fixed_b_from_buffers(model: nn.Module) -> None:
    """Set each gate's ``fixed_b`` float from its (broadcast) buffer."""
    for module in iter_modules_of_type(model, _PersistentFixedBActivation):
        module.fixed_b = float(module.fixed_b_buffer)


__all__ = ["broadcast_reactivation_state", "refresh_fixed_b_from_buffers"]
