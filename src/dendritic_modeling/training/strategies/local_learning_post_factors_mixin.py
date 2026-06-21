"""Post-synaptic factor assembly for local credit assignment."""

from __future__ import annotations

from typing import Any

import torch

from dendritic_modeling.training.strategies.local_learning_homeostasis_mixin import (
    LocalLearningHomeostasisMixin,
)
from dendritic_modeling.training.strategies.local_learning_state import (
    _LayerModulators,
    _PostFactors,
)


class LocalLearningPostFactorsMixin(LocalLearningHomeostasisMixin):
    """Build local post-synaptic factors from error, voltage, and modulators."""

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


__all__ = ["LocalLearningPostFactorsMixin"]
