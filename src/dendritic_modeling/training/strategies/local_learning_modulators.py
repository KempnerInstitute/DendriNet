"""Mixin methods for :mod:`dendritic_modeling.training.strategies.local_learning`.

This module is a mechanical split of ``LocalCreditAssignment`` helpers. Method
bodies are kept behavior-compatible with the original trainer; the main trainer
class owns orchestration and mutable state.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from dendritic_modeling.training.strategies.local_learning_homeostasis import (
    compute_gate_homeostasis_aux_grads,
    compute_inhibitory_homeostasis_factor,
    compute_voltage_homeostasis_error,
)
from dendritic_modeling.training.strategies.local_learning_hsic import (
    clamp_phi,
    compute_hsic_gradient,
    compute_kernel_matrix,
)
from dendritic_modeling.training.strategies.local_learning_state import (
    _LayerModulators,
    _LayerStats,
    _PostFactors,
)

logger = logging.getLogger(__name__)


class LocalLearningModulatorMixin:
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

    def _compute_layer_rho(self, rec: dict[str, Any], v0: torch.Tensor) -> float:

        layer_id = id(rec.get("layer"))

        stats = self._layer_stats.setdefault(layer_id, _LayerStats())

        v_n: torch.Tensor = rec.get("v_n")

        if v_n is None:

            return 1.0

        batch_size = v_n.size(0)

        alpha = (
            self.local_cfg.four_factor.ema_alpha
        )  # Learning rate for EMA (e.g., 0.05)

        decay = 1.0 - alpha  # Decay factor

        # Compute per-sample average voltages (over neurons)

        a = v_n.mean(dim=1)  # [B]

        b = v0.mean(dim=1)  # [B]

        # Optional alternative: dot-product-like proxy that works with B=1

        rho_mode = self.local_cfg.four_factor.rho_mode.lower()

        if rho_mode == "dot":

            # Use batch mean of elementwise product as a scalar proxy with EMA smoothing

            rho_val = float((a * b).mean().item())

            if stats.rho_ema is None:

                stats.rho_ema = rho_val

            else:

                stats.rho_ema = decay * stats.rho_ema + alpha * rho_val

            return float(max(0.1, min(2.0, stats.rho_ema)))  # Clamp to stable range

        elif rho_mode == "none":

            return 1.0

        if batch_size >= 2:

            # Batch mode: direct correlation + EMA smoothing

            a_c = a - a.mean()

            b_c = b - b.mean()

            denom = a_c.std(unbiased=False) * b_c.std(unbiased=False) + 1e-8

            rho_batch = float((a_c * b_c).mean().item() / denom.item())

            if stats.rho_ema is None:

                stats.rho_ema = rho_batch

            else:

                stats.rho_ema = decay * stats.rho_ema + alpha * rho_batch

            return float(max(0.1, min(2.0, stats.rho_ema)))

        else:

            # Single-sample mode: Use configured estimator

            estimator = self.local_cfg.four_factor.rho_estimator.lower()

            if estimator in ["ema", "ema+augment"]:

                if estimator == "ema+augment":

                    # Combined EMA + augmentation: use augmented samples for EMA update

                    K = self.local_cfg.four_factor.augment_k

                    noise_sigma = self.local_cfg.four_factor.augment_noise_sigma

                    # Collect statistics from K noisy versions

                    a_samples = []

                    b_samples = []

                    # Original values (for adding noise)

                    a_orig = a.clone()

                    b_orig = b.clone()

                    for _ in range(K):

                        # Add Gaussian noise to create augmented samples

                        noise_a = torch.randn_like(a_orig) * noise_sigma

                        noise_b = torch.randn_like(b_orig) * noise_sigma

                        a_noisy = a_orig + noise_a

                        b_noisy = b_orig + noise_b

                        # Collect scalar statistics

                        a_samples.append(a_noisy.mean().item())

                        b_samples.append(b_noisy.mean().item())

                    # Use mean of augmented samples as the current sample

                    a_sample = sum(a_samples) / K

                    b_sample = sum(b_samples) / K

                else:

                    # Standard EMA: use original sample

                    a_sample = (
                        a.mean().item()
                    )  # Scalar average over batch (B=1) and neurons

                    b_sample = b.mean().item()

                # Initialize

                if stats.ema_mean_y is None:

                    stats.ema_mean_y = a_sample

                    stats.ema_mean_x = b_sample

                    stats.ema_var_y = 0.0

                    stats.ema_var_x = 0.0

                    stats.ema_cov_xy = 0.0

                # Update means

                old_mean_y = stats.ema_mean_y

                old_mean_x = stats.ema_mean_x

                stats.ema_mean_y = decay * stats.ema_mean_y + alpha * a_sample

                stats.ema_mean_x = decay * stats.ema_mean_x + alpha * b_sample

                # Update deltas

                delta_y = a_sample - old_mean_y

                delta_x = b_sample - old_mean_x

                # Update var/cov (incremental)

                stats.ema_var_y = decay * stats.ema_var_y + alpha * (delta_y**2)

                stats.ema_var_x = decay * stats.ema_var_x + alpha * (delta_x**2)

                stats.ema_cov_xy = decay * stats.ema_cov_xy + alpha * (
                    delta_y * delta_x
                )

                denom = (stats.ema_var_y * stats.ema_var_x) ** 0.5 + 1e-8

                rho_ema = stats.ema_cov_xy / denom

                stats.rho_ema = rho_ema

                return float(max(0.1, min(2.0, rho_ema)))  # Clamp for stability

            elif estimator == "augment":

                # Micro-ensemble augmentation: K noisy forward passes

                K = self.local_cfg.four_factor.augment_k

                noise_sigma = self.local_cfg.four_factor.augment_noise_sigma

                # Collect statistics from K noisy versions

                a_samples = []

                b_samples = []

                # Original values (for adding noise)

                a_orig = a.clone()

                b_orig = b.clone()

                for _ in range(K):

                    # Add Gaussian noise to create augmented samples

                    noise_a = torch.randn_like(a_orig) * noise_sigma

                    noise_b = torch.randn_like(b_orig) * noise_sigma

                    a_noisy = a_orig + noise_a

                    b_noisy = b_orig + noise_b

                    # Collect scalar statistics

                    a_samples.append(a_noisy.mean().item())

                    b_samples.append(b_noisy.mean().item())

                # Convert to tensors for easier computation

                a_samples = torch.tensor(a_samples)

                b_samples = torch.tensor(b_samples)

                # Compute correlation from augmented samples

                a_centered = a_samples - a_samples.mean()

                b_centered = b_samples - b_samples.mean()

                # Compute correlation coefficient

                denom = (
                    a_centered.std(unbiased=False) * b_centered.std(unbiased=False)
                    + 1e-8
                )

                rho_augment = float(
                    (a_centered * b_centered).mean().item() / denom.item()
                )

                # Update EMA if tracking

                if stats.rho_ema is None:

                    stats.rho_ema = rho_augment

                else:

                    stats.rho_ema = decay * stats.rho_ema + alpha * rho_augment

                return float(max(0.1, min(2.0, rho_augment)))  # Clamp for stability

            else:

                logger.warning(f"Unknown rho_estimator '{estimator}'; returning 1.0")

                return 1.0

    def _compute_layer_phi(self, rec: dict[str, Any]) -> float:

        layer_id = id(rec.get("layer"))

        stats = self._layer_stats.setdefault(layer_id, _LayerStats())

        v_n: torch.Tensor = rec.get("v_n")

        if v_n is None:

            return 1.0

        # If batch size < 2, fall back to neutral scaling

        if v_n.size(0) < 2:

            return 1.0

        var_batch = float(v_n.var(unbiased=False).item())

        alpha = self.local_cfg.four_factor.ema_alpha

        if stats.var_ema is None:

            stats.var_ema = var_batch

        else:

            stats.var_ema = (1 - alpha) * stats.var_ema + alpha * var_batch

        phi = var_batch / (stats.var_ema + 1e-8)

        phi = self._clamp_phi(phi)

        return float(phi)

    def _compute_layer_phi_conditional(self, rec: dict[str, Any]) -> float:
        """Compute conditional information proxy using parent voltages.



        Approximates I(S; V_n | V_parent) by the fraction of variance in V_n

        explained by its residual after regressing on a parent proxy (blk_out if present).

        """

        v_n: torch.Tensor | None = rec.get("v_n")

        if v_n is None:

            return 1.0

        # Parent proxy: use blocklinear output if available, else inhibitory/excitatory outputs

        parent = rec.get("blk_out")

        if parent is None:

            parent = rec.get("exc_out")

        if parent is None:

            parent = rec.get("inh_out")

        if parent is None:

            return self._compute_layer_phi(rec)

        # Handle single-sample case with EMA

        if (
            v_n.size(0) < 2
            and self.local_cfg.five_factor.phi_estimator == "conditional_ema"
        ):

            return self._compute_layer_phi_conditional_ema(rec, v_n, parent)

        # Center variables

        y = v_n - v_n.mean(dim=0, keepdim=True)  # [B, out]

        x = parent - parent.mean(dim=0, keepdim=True)  # [B, out]

        # RLS-like EMA update for single-sample friendly conditional variance

        lam = self.local_cfg.five_factor.phi_ridge_lambda

        alpha = 1.0 - self.local_cfg.four_factor.ema_alpha  # decay for moments

        # Update EMA moments

        layer_id = id(rec.get("layer"))

        stats_local = self._layer_stats.setdefault(layer_id, _LayerStats())

        stats_local.ema_var_y = (alpha * (stats_local.ema_var_y or 0.0)) + (
            1 - alpha
        ) * float(y.pow(2).mean().item())

        stats_local.ema_var_x = (alpha * (stats_local.ema_var_x or 0.0)) + (
            1 - alpha
        ) * float(x.pow(2).mean().item())

        stats_local.ema_cov_xy = (alpha * (stats_local.ema_cov_xy or 0.0)) + (
            1 - alpha
        ) * float((y * x).mean().item())

        var_y = stats_local.ema_var_y

        var_x = stats_local.ema_var_x

        cov_xy = stats_local.ema_cov_xy

        if var_x <= 1e-12:

            return 1.0

        beta = cov_xy / (var_x + lam)

        residual_var = var_y - beta * cov_xy

        if residual_var <= 1e-12:

            residual_var = 1e-12

        # Conditional SNR-like factor: higher when residual variance is small

        phi = var_y / residual_var

        return float(self._clamp_phi(phi))

    def _compute_layer_phi_conditional_ema(
        self, rec: dict[str, Any], v_n: torch.Tensor, parent: torch.Tensor
    ) -> float:
        """Compute conditional phi for single samples using online EMA.



        Uses Welford-like updates to maintain running statistics of conditional variance.

        """

        layer_id = id(rec.get("layer"))

        stats = self._layer_stats.setdefault(layer_id, _LayerStats())

        # Compute per-sample averages

        y_sample = v_n.mean().item()  # Scalar

        x_sample = parent.mean().item()  # Scalar

        alpha = self.local_cfg.four_factor.ema_alpha

        decay = 1.0 - alpha

        lam = self.local_cfg.five_factor.phi_ridge_lambda

        # Initialize if needed

        if stats.ema_mean_y is None:

            stats.ema_mean_y = y_sample

            stats.ema_mean_x = x_sample

            stats.ema_var_y = 0.0

            stats.ema_var_x = 0.0

            stats.ema_cov_xy = 0.0

            return 1.0  # Neutral on first sample

        # Update means

        old_mean_y = stats.ema_mean_y

        old_mean_x = stats.ema_mean_x

        stats.ema_mean_y = decay * stats.ema_mean_y + alpha * y_sample

        stats.ema_mean_x = decay * stats.ema_mean_x + alpha * x_sample

        # Compute deltas

        delta_y = y_sample - old_mean_y

        delta_x = x_sample - old_mean_x

        # Update var/cov using incremental formulation

        stats.ema_var_y = decay * stats.ema_var_y + alpha * (delta_y**2)

        stats.ema_var_x = decay * stats.ema_var_x + alpha * (delta_x**2)

        stats.ema_cov_xy = decay * stats.ema_cov_xy + alpha * (delta_y * delta_x)

        # Compute conditional variance

        if stats.ema_var_x <= 1e-12:

            return 1.0

        beta = stats.ema_cov_xy / (stats.ema_var_x + lam)

        residual_var = stats.ema_var_y - beta * stats.ema_cov_xy

        if residual_var <= 1e-12:

            residual_var = 1e-12

        # Conditional SNR proxy

        phi = stats.ema_var_y / residual_var

        return float(self._clamp_phi(phi))

    def _clamp_phi(self, phi: float) -> float:
        """Clamp the 5F confidence factor using configured stability bounds."""

        return clamp_phi(self.local_cfg, phi)


__all__ = ["LocalLearningModulatorMixin"]
