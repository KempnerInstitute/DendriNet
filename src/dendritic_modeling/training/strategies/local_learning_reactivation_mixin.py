"""Reactivation-gradient helpers for local credit assignment."""

from __future__ import annotations

from typing import Any

import torch

from dendritic_modeling.networks import ParametricActivation
from dendritic_modeling.training.strategies.local_learning_gradient_utils import (
    _accumulate_parameter_grad,
    _reactivation_output_slope,
    _scale_reactivation_error,
)


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
            and isinstance(react_module, ParametricActivation)
            and v_n is not None
            and v_out is not None
        ):
            return

        one_minus_y2 = _reactivation_output_slope(v_out)
        if hasattr(react_module, "log_m"):
            if hasattr(react_module, "b"):
                elig_m = one_minus_y2 * (v_n - react_module.b.detach())
            else:
                elig_m = one_minus_y2 * v_n
            m_vec = react_module.log_m.detach().exp()
            react_factor = _scale_reactivation_error(
                e_n,
                rho=rho,
                phi=phi,
                branch_scale=branch_scale,
            )
            grad_m = (react_factor * elig_m).mean(dim=0)
            grad_log_m = grad_m * m_vec
            _accumulate_parameter_grad(react_module.log_m, grad_log_m)

        if hasattr(react_module, "b"):
            m_vec = (
                react_module.log_m.detach().exp()
                if hasattr(react_module, "log_m")
                else 1.0
            )
            elig_b = -one_minus_y2 * m_vec
            react_factor_b = _scale_reactivation_error(
                e_n,
                rho=rho,
                phi=phi,
                branch_scale=branch_scale,
            )
            grad_b = (react_factor_b * elig_b).mean(dim=0)
            _accumulate_parameter_grad(react_module.b, grad_b)

        aux_grad_log_m, aux_grad_b = self._compute_gate_homeostasis_aux_grads(
            react_module=react_module,
            v_n=v_n,
            v_out=v_out,
        )
        if aux_grad_log_m is not None and hasattr(react_module, "log_m"):
            _accumulate_parameter_grad(react_module.log_m, aux_grad_log_m)
        if (
            aux_grad_b is not None
            and hasattr(react_module, "b")
            and react_module.b is not None
        ):
            _accumulate_parameter_grad(react_module.b, aux_grad_b)


__all__ = ["LocalLearningReactivationMixin"]
