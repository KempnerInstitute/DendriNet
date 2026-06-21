"""Five-factor phi modulators for local credit assignment."""

from __future__ import annotations

from typing import Any

import torch

from dendritic_modeling.training.strategies.local_learning_hsic import clamp_phi
from dendritic_modeling.training.strategies.local_learning_state import _LayerStats


class LocalLearningPhiMixin:
    """Information-confidence modulators for 5-factor local rules."""

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


__all__ = ["LocalLearningPhiMixin"]
