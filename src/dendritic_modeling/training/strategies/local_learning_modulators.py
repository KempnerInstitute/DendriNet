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

from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.training.strategies.local_learning_homeostasis import (
    compute_gate_homeostasis_aux_grads,
    compute_inhibitory_homeostasis_factor,
    compute_voltage_homeostasis_error,
)
from dendritic_modeling.training.strategies.local_learning_hsic import (
    clamp_phi,
    compute_dendritic_normalization,
    compute_hsic_gradient,
    compute_kernel_matrix,
)
from dendritic_modeling.training.strategies.local_learning_state import _LayerStats

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

    def _compute_path_propagation_factor(
        self, layer_idx: int, rec: dict[str, Any], layer_records: list[dict[str, Any]]
    ) -> torch.Tensor:
        """Backward-compatible wrapper around the precomputed path factors."""

        path_factors = self._precompute_path_propagation_factors(layer_records)

        path_factor = path_factors[layer_idx]

        if isinstance(path_factor, torch.Tensor):

            return path_factor

        v_n = rec.get("v_n")

        if isinstance(v_n, torch.Tensor):

            return torch.ones(v_n.size(0), 1, device=v_n.device, dtype=v_n.dtype)

        return torch.ones(1, 1)

    def _compute_branch_depth_modulator(
        self, rec: dict[str, Any], rho_base: float
    ) -> torch.Tensor:
        """Compute branch-specific modulator based on depth.



        Modulates rho by branch depth: rho_j = rho_base / (depth_j + offset)



        Args:

            rec: Layer record

            rho_base: Base rho value to modulate



        Returns:

            Per-branch modulator tensor or scalar

        """

        layer_id = id(rec.get("layer"))

        stats = self._layer_stats.get(layer_id)

        if stats is None or stats.branch_depths is None:

            # No depth info: return uniform modulator

            return rho_base

        depths = stats.branch_depths  # [out_features]

        offset = self.local_cfg.morphology_aware.morphology_depth_offset

        # Depth-based scaling: deeper branches get smaller factors

        modulator = rho_base / (depths + offset)

        return modulator

    def _compute_branch_centrality_modulator(
        self, rec: dict[str, Any], rho_base: float
    ) -> torch.Tensor:
        """Compute branch-specific modulator based on centrality proxy.



        Without explicit tree metadata, we use depth-derived centrality:

        proximal branches (lower depth) receive higher centrality.

        """

        layer_id = id(rec.get("layer"))

        stats = self._layer_stats.get(layer_id)

        if stats is None or stats.branch_depths is None:

            return rho_base

        depths = stats.branch_depths

        offset = self.local_cfg.morphology_aware.morphology_depth_offset

        metric = self.local_cfg.morphology_aware.morphology_centrality_metric.lower()

        if metric in {"betweenness", "closeness", "degree"}:

            centrality = 1.0 / (depths + offset)

        else:

            centrality = 1.0 / (depths + offset)

        # Keep average scale near rho_base.

        centrality = centrality / (centrality.mean() + 1e-8)

        return rho_base * centrality

    def _compute_dendritic_normalization(
        self, rec: dict[str, Any], grad_g_den: torch.Tensor
    ) -> torch.Tensor:
        """Apply dendritic normalization to branch conductance updates."""

        return compute_dendritic_normalization(rec, grad_g_den)

    def _get_branch_type_scale(self, rec: dict[str, Any]) -> torch.Tensor | float:
        """Get scaling factors for apical vs basal branches.



        Args:

            rec: Layer record



        Returns:

            Scale factor per branch [out_features] or scalar 1.0

        """

        layer_id = id(rec.get("layer"))

        stats = self._layer_stats.get(layer_id)

        if stats is None:

            return 1.0

        scale: Any = 1.0

        if stats.branch_types is not None:

            branch_types = stats.branch_types  # [out_features], 1=apical, 0=basal

            apical_scale = self.local_cfg.morphology_aware.apical_branch_scale

            basal_scale = self.local_cfg.morphology_aware.basal_branch_scale

            # Linear interpolation based on branch type

            scale = basal_scale + branch_types * (apical_scale - basal_scale)

        if (
            self.local_cfg.morphology_aware.use_branch_length_modulation
            and stats.branch_depths is not None
        ):

            depth_offset = self.local_cfg.morphology_aware.morphology_depth_offset

            length_factor = 1.0 / (stats.branch_depths + depth_offset)

            length_factor = length_factor / (length_factor.mean() + 1e-8)

            if isinstance(scale, torch.Tensor):

                scale = scale * length_factor

            else:

                scale = float(scale) * length_factor

        return scale

    def _initialize_branch_morphology(
        self, model: BaseModel, layer_records: list[dict[str, Any]]
    ) -> None:
        """Initialize morphology metadata and infer pathway roles from connectivity."""

        base_input_roles = self._resolve_router_output_roles(model)

        prior_output_roles: list[torch.Tensor] = []

        for layer_idx, rec in enumerate(layer_records):

            layer_id = id(rec.get("layer"))

            stats = self._layer_stats.setdefault(layer_id, _LayerStats())

            v_n = rec.get("v_n")

            if v_n is None:

                continue

            out_features = v_n.size(1)

            stats.branch_role_profile = None

            stats.branch_role_selectivity = None

            stats.input_role_profile = None

            stats.block_input_role_profile = None

            stats.branch_block_roles = None

            stats.branch_block_selectivity = None

            # Heuristic depth: based on layer index (deeper layers = higher depth)

            # In future, this should come from actual dendritic tree graph

            if stats.branch_depths is None:

                # Assign uniform depth based on layer position

                base_depth = float(layer_idx + 1)

                stats.branch_depths = (
                    torch.ones(out_features, device=v_n.device, dtype=v_n.dtype)
                    * base_depth
                )

            # Heuristic branch types: assume first half are basal, second half apical

            # In future, this should come from actual branch annotations

            if stats.branch_types is None:

                branch_types = torch.zeros(
                    out_features, device=v_n.device, dtype=v_n.dtype
                )

                # Simple heuristic: deeper layers tend to have more apical character

                if layer_idx >= len(layer_records) // 2:

                    branch_types[:] = 1.0  # Mark as apical

                stats.branch_types = branch_types

            role_mass_terms: list[torch.Tensor] = []

            exc_layer: TopKLinear | None = rec.get("exc_module")

            x_exc = rec.get("x_exc")

            if exc_layer is not None and isinstance(x_exc, torch.Tensor):

                exc_input_roles = self._resolve_recent_role_profile(
                    x_exc.size(1), base_input_roles, prior_output_roles
                )

                if exc_input_roles is not None:

                    stats.input_role_profile = exc_input_roles.to(
                        device=v_n.device, dtype=v_n.dtype
                    )

                    exc_weight = self._effective_topk_weight(
                        exc_layer, rec.get("exc_mask")
                    )

                    exc_role_mass = self._role_mass_from_weight_matrix(
                        exc_weight, stats.input_role_profile
                    )

                    if exc_role_mass is not None:

                        role_mass_terms.append(
                            exc_role_mass.to(device=v_n.device, dtype=v_n.dtype)
                        )

            inh_layer: TopKLinear | None = rec.get("inh_module")

            x_inh = rec.get("x_inh")

            if inh_layer is not None and isinstance(x_inh, torch.Tensor):

                inh_input_roles = self._resolve_recent_role_profile(
                    x_inh.size(1), base_input_roles, prior_output_roles
                )

                if inh_input_roles is not None:

                    if stats.input_role_profile is None:

                        stats.input_role_profile = inh_input_roles.to(
                            device=v_n.device, dtype=v_n.dtype
                        )

                    inh_weight = self._effective_topk_weight(
                        inh_layer, rec.get("inh_mask")
                    )

                    inh_role_mass = self._role_mass_from_weight_matrix(
                        inh_weight, inh_input_roles
                    )

                    if inh_role_mass is not None:

                        role_mass_terms.append(
                            inh_role_mass.to(device=v_n.device, dtype=v_n.dtype)
                        )

            blk_layer = rec.get("blk_module")

            x_blk_raw = rec.get("x_blk_raw")

            if blk_layer is not None and isinstance(x_blk_raw, torch.Tensor):

                block_input_roles = self._resolve_recent_role_profile(
                    x_blk_raw.size(1), base_input_roles, prior_output_roles
                )

                if block_input_roles is not None:

                    stats.block_input_role_profile = block_input_roles.to(
                        device=v_n.device, dtype=v_n.dtype
                    )

                    block_size = int(getattr(blk_layer, "block_size", 1))

                    n_roles = stats.block_input_role_profile.size(1)

                    expected_in = out_features * block_size

                    if stats.block_input_role_profile.size(0) == expected_in:

                        block_roles = stats.block_input_role_profile.reshape(
                            out_features, block_size, n_roles
                        )

                        stats.branch_block_roles = block_roles

                        block_selectivity = self._compute_role_selectivity(
                            block_roles.reshape(-1, n_roles)
                        )

                        stats.branch_block_selectivity = block_selectivity.reshape(
                            out_features, block_size
                        )

                        blk_weight = (
                            blk_layer.weight()
                            .detach()
                            .to(device=v_n.device, dtype=v_n.dtype)
                        )

                        block_role_mass = (
                            blk_weight.unsqueeze(-1)
                            * block_roles.to(device=v_n.device, dtype=v_n.dtype)
                        ).sum(dim=1)

                        role_mass_terms.append(block_role_mass)

            if role_mass_terms:

                total_role_mass = role_mass_terms[0]

                for extra_mass in role_mass_terms[1:]:

                    total_role_mass = total_role_mass + extra_mass

                role_profile = self._normalize_role_mass(total_role_mass)

                stats.branch_role_profile = role_profile

                stats.branch_role_selectivity = self._compute_role_selectivity(
                    role_profile
                )

                prior_output_roles.append(role_profile.detach())

    def _get_branch_role_scales(
        self, rec: dict[str, Any]
    ) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        """Get plasticity scales from inferred pathway specialization."""

        layer_id = id(rec.get("layer"))

        stats = self._layer_stats.get(layer_id)

        if (
            stats is None
            or not isinstance(stats.branch_role_selectivity, torch.Tensor)
            or stats.branch_role_selectivity.numel() == 0
        ):

            return 1.0, 1.0

        cfg = self.local_cfg.morphology_aware

        power = max(float(getattr(cfg, "branch_role_power", 1.0)), 1e-6)

        specialized_scale = float(getattr(cfg, "specialized_branch_scale", 1.0))

        mixed_scale = float(getattr(cfg, "mixed_branch_scale", 1.0))

        neuron_selectivity = stats.branch_role_selectivity.to(
            dtype=stats.branch_role_selectivity.dtype
        )

        branch_scale = mixed_scale + neuron_selectivity.pow(power) * (
            specialized_scale - mixed_scale
        )

        dendritic_scale: torch.Tensor | float = branch_scale

        if isinstance(stats.branch_block_selectivity, torch.Tensor):

            block_selectivity = stats.branch_block_selectivity.to(
                dtype=branch_scale.dtype, device=branch_scale.device
            )

            dendritic_scale = mixed_scale + block_selectivity.pow(power) * (
                specialized_scale - mixed_scale
            )

        if cfg.use_branch_length_modulation and stats.branch_depths is not None:

            depth_offset = cfg.morphology_depth_offset

            length_factor = 1.0 / (stats.branch_depths + depth_offset)

            length_factor = length_factor / (length_factor.mean() + 1e-8)

            branch_scale = branch_scale * length_factor

            if isinstance(dendritic_scale, torch.Tensor):

                if dendritic_scale.dim() == 2:

                    dendritic_scale = dendritic_scale * length_factor.unsqueeze(-1)

                else:

                    dendritic_scale = dendritic_scale * length_factor

        return branch_scale, dendritic_scale

    def _get_branch_role_alignment_factors(
        self, rec: dict[str, Any]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Get synapse- and branch-specific alignment gains from inferred roles."""

        layer_id = id(rec.get("layer"))

        stats = self._layer_stats.get(layer_id)

        if (
            stats is None
            or not isinstance(stats.branch_role_profile, torch.Tensor)
            or stats.branch_role_profile.numel() == 0
        ):

            return None, None

        weight = float(
            getattr(
                self.local_cfg.morphology_aware, "branch_role_alignment_weight", 0.0
            )
        )

        if weight <= 0.0:

            return None, None

        role_profile = stats.branch_role_profile

        syn_factor: torch.Tensor | None = None

        if isinstance(stats.input_role_profile, torch.Tensor):

            alignment = role_profile @ stats.input_role_profile.t().to(
                device=role_profile.device, dtype=role_profile.dtype
            )

            alignment = alignment - alignment.mean(dim=1, keepdim=True)

            syn_factor = torch.clamp(1.0 + weight * alignment, min=0.0)

        block_factor: torch.Tensor | None = None

        if isinstance(stats.branch_block_roles, torch.Tensor):

            block_roles = stats.branch_block_roles.to(
                device=role_profile.device, dtype=role_profile.dtype
            )

            alignment = (block_roles * role_profile.unsqueeze(1)).sum(dim=-1)

            alignment = alignment - alignment.mean(dim=1, keepdim=True)

            block_factor = torch.clamp(1.0 + weight * alignment, min=0.0)

        return syn_factor, block_factor


__all__ = ["LocalLearningModulatorMixin"]
