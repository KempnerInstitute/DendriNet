"""Four-factor rho modulators for local credit assignment."""

from __future__ import annotations

import logging
from typing import Any

import torch

from dendritic_modeling.training.strategies.local_learning_phi_mixin import (
    LocalLearningPhiMixin,
)
from dendritic_modeling.training.strategies.local_learning_state import _LayerStats

logger = logging.getLogger(__name__)


class LocalLearningRhoMixin(LocalLearningPhiMixin):
    """Input-output correlation modulators for 4-factor and 5-factor rules."""

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


__all__ = ["LocalLearningRhoMixin"]
