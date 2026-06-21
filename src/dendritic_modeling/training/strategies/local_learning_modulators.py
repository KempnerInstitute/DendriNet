"""Mixin methods for :mod:`dendritic_modeling.training.strategies.local_learning`.

This module is a mechanical split of ``LocalCreditAssignment`` helpers. Method
bodies are kept behavior-compatible with the original trainer; the main trainer
class owns orchestration and mutable state.
"""

from __future__ import annotations

from typing import Any

import torch

from dendritic_modeling.training.strategies.local_learning_post_factors_mixin import (
    LocalLearningPostFactorsMixin,
)
from dendritic_modeling.training.strategies.local_learning_state import _LayerModulators


class LocalLearningModulatorMixin(LocalLearningPostFactorsMixin):
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


__all__ = ["LocalLearningModulatorMixin"]
