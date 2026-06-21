"""Mixin methods for :mod:`dendritic_modeling.training.strategies.local_learning`.

This module is a mechanical split of ``LocalCreditAssignment`` helpers. Method
bodies are kept behavior-compatible with the original trainer; the main trainer
class owns orchestration and mutable state.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from dendritic_modeling.training.strategies.local_learning_homeostasis import (
    compute_gate_homeostasis_aux_grads,
    compute_inhibitory_homeostasis_factor,
    compute_voltage_homeostasis_error,
)
from dendritic_modeling.training.strategies.local_learning_hsic import (
    compute_hsic_gradient,
    compute_kernel_matrix,
)
from dendritic_modeling.training.strategies.local_learning_rho_mixin import (
    LocalLearningRhoMixin,
)
from dendritic_modeling.training.strategies.local_learning_state import (
    _LayerModulators,
    _PostFactors,
)


class LocalLearningModulatorMixin(LocalLearningRhoMixin):
    def _compute_inhibitory_homeostasis_factor(
        self,
        rec: dict[str, Any],
        R_tot: torch.Tensor | float,
        v_n: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute an auxiliary local inhibitory-homeostasis gradient factor."""

        return compute_inhibitory_homeostasis_factor(self.local_cfg, rec, R_tot, v_n)

    def _compute_voltage_homeostasis_error(
        self, v_n: torch.Tensor
    ) -> torch.Tensor | None:
        """Return a local voltage-centering error signal."""

        return compute_voltage_homeostasis_error(self.local_cfg, v_n)

    def _compute_gate_homeostasis_aux_grads(
        self,
        react_module: nn.Module | None,
        v_n: torch.Tensor | None,
        v_out: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return strictly local auxiliary grads for reactivation parameters."""

        return compute_gate_homeostasis_aux_grads(
            self.local_cfg, react_module, v_n, v_out
        )

    def _compute_kernel_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute kernel matrix K(X, Y) based on configured kernel type."""

        return compute_kernel_matrix(self.local_cfg, X, Y)

    def _compute_hsic_gradient(
        self, z: torch.Tensor, y: torch.Tensor, weight: float, grad_type: str
    ) -> torch.Tensor:
        """Compute HSIC gradient with respect to z using specified kernel."""

        return compute_hsic_gradient(self.local_cfg, z, y, weight, grad_type)

    def _compute_local_layer_modulators(
        self,
        *,
        rec: dict[str, Any],
        v0: torch.Tensor,
        layer_depth: int,
    ) -> _LayerModulators:
        rho = 1.0
        if self.local_cfg.rule_variant in {"4f", "5f"}:
            rho = self._compute_layer_rho(rec, v0)
            if self.local_cfg.four_factor.layer_wise_rho_scale > 0:
                rho = rho / (
                    layer_depth**0.5 * self.local_cfg.four_factor.layer_wise_rho_scale
                )

            if self.local_cfg.morphology_aware.morphology_modulator_mode == "depth":
                rho = self._compute_branch_depth_modulator(rec, rho)
            elif (
                self.local_cfg.morphology_aware.morphology_modulator_mode
                == "centrality"
            ):
                rho = self._compute_branch_centrality_modulator(rec, rho)

        phi = 1.0
        if self.local_cfg.rule_variant == "5f":
            phi_mode = self.local_cfg.five_factor.phi_mode.lower()
            phi_estimator = self.local_cfg.five_factor.phi_estimator.lower()
            if phi_mode == "conditional":
                if phi_estimator == "variance_ema":
                    phi = self._compute_layer_phi(rec)
                else:
                    phi = self._compute_layer_phi_conditional(rec)
            else:
                phi = self._compute_layer_phi(rec)
            if self.local_cfg.five_factor.layer_wise_phi_scale > 0:
                phi = phi / (
                    layer_depth**0.5 * self.local_cfg.five_factor.layer_wise_phi_scale
                )

        branch_scale: torch.Tensor | float = 1.0
        dendritic_branch_scale: torch.Tensor | float = 1.0
        role_synaptic_alignment: torch.Tensor | None = None
        role_block_alignment: torch.Tensor | None = None
        using_branch_role_scale = self.local_cfg.morphology_aware.use_branch_role_rules
        if using_branch_role_scale:
            branch_scale, dendritic_branch_scale = self._get_branch_role_scales(rec)
            (
                role_synaptic_alignment,
                role_block_alignment,
            ) = self._get_branch_role_alignment_factors(rec)
        if not using_branch_role_scale and (
            self.local_cfg.morphology_aware.use_branch_type_rules
            or self.local_cfg.morphology_aware.use_branch_length_modulation
        ):
            branch_scale = self._get_branch_type_scale(rec)
            dendritic_branch_scale = branch_scale

        return _LayerModulators(
            rho=rho,
            phi=phi,
            branch_scale=branch_scale,
            dendritic_branch_scale=dendritic_branch_scale,
            role_synaptic_alignment=role_synaptic_alignment,
            role_block_alignment=role_block_alignment,
        )

    def _compute_local_post_factors(
        self,
        *,
        rec: dict[str, Any],
        e_n: torch.Tensor,
        v_n: torch.Tensor,
        r_tot: torch.Tensor | float,
        layer_dynamics_mode: str,
        layer_idx: int,
        modulators: _LayerModulators,
    ) -> _PostFactors:
        activation_derivative = self._get_layer_activation_derivative(
            rec=rec,
            v_n=v_n,
            v_out=rec.get("v_out"),
        )
        if not isinstance(activation_derivative, torch.Tensor):
            activation_derivative = torch.ones_like(v_n)
        activation_derivative = activation_derivative.to(
            device=v_n.device, dtype=v_n.dtype
        )
        e_v = e_n * activation_derivative

        if layer_dynamics_mode == "conductance":
            if self.local_cfg.three_factor.use_driving_force:
                post_mod = self.local_cfg.three_factor.e_rev_exc - v_n  # [B, out]
            else:
                post_mod = v_n - self.local_cfg.three_factor.theta
            post_factor = e_v * r_tot * post_mod  # [B, out]
        else:
            additive_mode = getattr(
                self.local_cfg.three_factor, "additive_gain_mode", "none"
            )
            if additive_mode == "learned_gain":
                gain_params = self._ensure_additive_gain_params(v_n.size(1), v_n.device)
                gain_n = torch.sigmoid(gain_params)  # [out]
                post_factor = e_v * gain_n.unsqueeze(0)  # [B, out]
            elif additive_mode == "input_dependent":
                pseudo_R, pseudo_drive = self._compute_additive_pseudo_signals(rec, v_n)
                post_factor = e_v * pseudo_R * pseudo_drive  # [B, out]
            elif additive_mode == "running_stats":
                inv_std = self._get_additive_running_inv_std(layer_idx, v_n)
                post_factor = e_v * inv_std  # [B, out]
            else:
                post_factor = e_v

        post_factor_raw = post_factor
        if isinstance(modulators.rho, torch.Tensor):
            post_factor = post_factor * modulators.rho.unsqueeze(0)
        else:
            post_factor = post_factor * float(modulators.rho)

        post_factor = post_factor * float(modulators.phi)

        if isinstance(modulators.branch_scale, torch.Tensor):
            post_factor = post_factor * modulators.branch_scale.unsqueeze(0)
        else:
            post_factor = post_factor * float(modulators.branch_scale)

        return _PostFactors(
            e_v=e_v,
            post_factor=post_factor,
            post_factor_raw=post_factor_raw,
        )


__all__ = ["LocalLearningModulatorMixin"]
