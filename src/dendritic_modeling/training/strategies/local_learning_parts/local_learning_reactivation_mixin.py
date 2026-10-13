"""Reactivation-gradient helpers for local credit assignment."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from dendritic_modeling.training.strategies.local_learning_parts.local_learning_gradient_utils import (
    _accumulate_parameter_grad,
    _scale_reactivation_error,
)


def reactivation_parameter_grads(
    react_module: nn.Module,
    v_n: torch.Tensor,
    gate_error: torch.Tensor,
) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
    """Return the exact local gradients of a gate's trainable parameters.

    Every sample's gate output ``a = g(V)`` receives ``gate_error``; the
    gradients are the batch mean of ``gate_error * da/dtheta`` with the voltage
    ``V`` held fixed. Autograd covers every gate type and every trainable gate
    parameter, and frozen parameters receive nothing.
    """
    params = [param for param in react_module.parameters() if param.requires_grad]
    if not params:
        return []
    with torch.inference_mode(False), torch.enable_grad():
        voltage = v_n.detach().clone()
        output = react_module(voltage)
        grads = torch.autograd.grad(
            output,
            params,
            grad_outputs=(gate_error / v_n.shape[0]).to(output.dtype),
            allow_unused=True,
        )
    return [(param, grad) for param, grad in zip(params, grads) if grad is not None]


class LocalLearningReactivationMixin:
    """Gradient writeback for trainable reactivation gates."""

    def _apply_reactivation_local_grad(
        self,
        *,
        rec: dict[str, Any],
        e_n: torch.Tensor,
        v_n: torch.Tensor,
        rho: Any,
        phi: float,
        branch_scale: Any,
        update_reactivation: bool,
    ) -> None:
        react_module = getattr(rec.get("layer"), "reactivation", None)
        v_out = rec.get("v_out")
        if not (
            update_reactivation
            and isinstance(react_module, nn.Module)
            and v_n is not None
            and v_out is not None
        ):
            return

        gate_error = _scale_reactivation_error(
            e_n, rho=rho, phi=phi, branch_scale=branch_scale
        )
        for param, grad in reactivation_parameter_grads(react_module, v_n, gate_error):
            _accumulate_parameter_grad(param, grad)

        aux_grad_log_m, aux_grad_b = self._compute_gate_homeostasis_aux_grads(
            react_module=react_module,
            v_n=v_n,
            v_out=v_out,
        )
        log_m = getattr(react_module, "log_m", None)
        bias = getattr(react_module, "b", None)
        if aux_grad_log_m is not None and _trainable(log_m):
            _accumulate_parameter_grad(log_m, aux_grad_log_m)
        if aux_grad_b is not None and _trainable(bias):
            _accumulate_parameter_grad(bias, aux_grad_b)


def _trainable(param: Any) -> bool:
    return isinstance(param, torch.Tensor) and param.requires_grad


__all__ = ["LocalLearningReactivationMixin", "reactivation_parameter_grads"]
