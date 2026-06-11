"""State containers for local learning strategies."""

from __future__ import annotations

import torch


class _LayerStats:
    """Holds per-layer running statistics for morphology/info factors."""

    def __init__(self):
        self.rho_ema: float | None = None
        self.var_ema: float | None = None
        # Moments for conditional EMA (single-sample friendly)
        self.ema_var_y: float | None = None
        self.ema_var_x: float | None = None
        self.ema_cov_xy: float | None = None
        # Running means for proper online variance/covariance
        self.ema_mean_y: float | None = None
        self.ema_mean_x: float | None = None
        # Morphology-specific statistics
        self.path_factor: torch.Tensor | None = None  # Path attenuation factor
        self.branch_depths: torch.Tensor | None = None  # Branch depths from soma
        self.branch_types: torch.Tensor | None = None  # Apical (1) vs basal (0)
        self.branch_role_profile: torch.Tensor | None = None  # [out, roles]
        self.branch_role_selectivity: torch.Tensor | None = None  # [out]
        self.input_role_profile: torch.Tensor | None = None  # [in, roles]
        self.block_input_role_profile: torch.Tensor | None = None  # [in, roles]
        self.branch_block_roles: torch.Tensor | None = None  # [out, block, roles]
        self.branch_block_selectivity: torch.Tensor | None = None  # [out, block]


__all__ = ["_LayerStats"]
