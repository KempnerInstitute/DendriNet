"""TopK synapse-gradient helpers for local credit assignment."""

from __future__ import annotations

from typing import Any

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.training.strategies.local_learning_gradient_utils import (
    _apply_optional_alignment,
    _factor_pre_weight_grad,
    _topk_pre_weight_grad,
)
from dendritic_modeling.training.strategies.local_learning_pathways import (
    EXCITATORY_TOPK_PATHS,
    INHIBITORY_TOPK_PATHS,
)
from dendritic_modeling.training.strategies.local_learning_state import (
    _LayerModulators,
    _PostFactors,
)


class LocalLearningTopKGradientMixin:
    """Gradient writeback for TopK feedforward and recurrent pathways."""

    def _apply_excitatory_topk_pathways(
        self,
        *,
        rec: dict[str, Any],
        v_n: torch.Tensor,
        r_tot: torch.Tensor | float,
        layer_dynamics_mode: str,
        batch_size: int,
        modulators: _LayerModulators,
        post_factors: _PostFactors,
    ) -> TopKLinear | None:
        """Apply local gradients to feedforward and recurrent excitatory paths."""
        exc_layer = None
        for path_idx, path in enumerate(EXCITATORY_TOPK_PATHS):
            layer = self._apply_excitatory_topk_local_grad(
                rec=rec,
                module_key=path.module_key,
                input_key=path.input_key,
                mask_key=path.mask_key,
                post_factor=post_factors.post_factor,
                v_n=v_n,
                r_tot=r_tot,
                layer_dynamics_mode=layer_dynamics_mode,
                batch_size=batch_size,
                role_synaptic_alignment=modulators.role_synaptic_alignment,
            )
            if path_idx == 0:
                exc_layer = layer
        return exc_layer

    def _apply_inhibitory_topk_pathways(
        self,
        *,
        rec: dict[str, Any],
        v_n: torch.Tensor,
        r_tot: torch.Tensor | float,
        layer_dynamics_mode: str,
        batch_size: int,
        modulators: _LayerModulators,
        post_factors: _PostFactors,
    ) -> None:
        """Apply local gradients to feedforward and recurrent inhibitory paths."""
        for path in INHIBITORY_TOPK_PATHS:
            self._apply_inhibitory_topk_local_grad(
                rec=rec,
                module_key=path.module_key,
                input_key=path.input_key,
                mask_key=path.mask_key,
                e_v=post_factors.e_v,
                post_factor_raw=post_factors.post_factor_raw,
                v_n=v_n,
                r_tot=r_tot,
                rho=modulators.rho,
                phi=modulators.phi,
                branch_scale=modulators.branch_scale,
                layer_dynamics_mode=layer_dynamics_mode,
                batch_size=batch_size,
                role_synaptic_alignment=modulators.role_synaptic_alignment,
            )

    def _apply_excitatory_topk_local_grad(
        self,
        *,
        rec: dict[str, Any],
        module_key: str,
        input_key: str,
        mask_key: str,
        post_factor: torch.Tensor,
        v_n: torch.Tensor,
        r_tot: Any,
        layer_dynamics_mode: str,
        batch_size: int,
        role_synaptic_alignment: torch.Tensor | None,
    ) -> TopKLinear | None:
        layer: TopKLinear | None = rec.get(module_key)
        if layer is None or rec.get(input_key) is None:
            return layer

        x_pre: torch.Tensor = rec[input_key]
        grad_g = _factor_pre_weight_grad(
            post_factor,
            x_pre,
            normalize_by_batch=self.local_cfg.normalize_by_batch,
            batch_size=batch_size,
        )

        voltage_homeo_error = self._compute_voltage_homeostasis_error(v_n)
        if voltage_homeo_error is not None:
            if layer_dynamics_mode == "conductance":
                homeo_factor = (
                    voltage_homeo_error
                    * r_tot
                    * (self.local_cfg.three_factor.e_rev_exc - v_n)
                )
            else:
                homeo_factor = voltage_homeo_error
            grad_homeo = _factor_pre_weight_grad(
                homeo_factor,
                x_pre,
                normalize_by_batch=self.local_cfg.normalize_by_batch,
                batch_size=batch_size,
            )
            grad_g = grad_g + grad_homeo

        grad_g = _apply_optional_alignment(grad_g, role_synaptic_alignment)

        self._add_topk_transformed_grad(layer, grad_g, rec.get(mask_key))
        return layer

    def _apply_inhibitory_topk_local_grad(
        self,
        *,
        rec: dict[str, Any],
        module_key: str,
        input_key: str,
        mask_key: str,
        e_v: torch.Tensor,
        post_factor_raw: torch.Tensor,
        v_n: torch.Tensor,
        r_tot: Any,
        rho: Any,
        phi: float,
        branch_scale: Any,
        layer_dynamics_mode: str,
        batch_size: int,
        role_synaptic_alignment: torch.Tensor | None,
    ) -> TopKLinear | None:
        layer: TopKLinear | None = rec.get(module_key)
        if layer is None or rec.get(input_key) is None:
            return layer

        x_pre: torch.Tensor = rec[input_key]
        inh_factor = self._compute_inhibitory_topk_factor(
            rec=rec,
            e_v=e_v,
            post_factor_raw=post_factor_raw,
            v_n=v_n,
            r_tot=r_tot,
            rho=rho,
            phi=phi,
            branch_scale=branch_scale,
            layer_dynamics_mode=layer_dynamics_mode,
        )

        grad_g = _factor_pre_weight_grad(
            inh_factor,
            x_pre,
            normalize_by_batch=self.local_cfg.normalize_by_batch,
            batch_size=batch_size,
        )

        grad_g = _apply_optional_alignment(grad_g, role_synaptic_alignment)

        voltage_homeo_error = self._compute_voltage_homeostasis_error(v_n)
        if voltage_homeo_error is not None:
            if layer_dynamics_mode == "conductance":
                homeo_factor = (
                    voltage_homeo_error
                    * r_tot
                    * (self.local_cfg.three_factor.e_rev_inh - v_n)
                )
            else:
                homeo_factor = -voltage_homeo_error
            grad_homeo = _factor_pre_weight_grad(
                homeo_factor,
                x_pre,
                normalize_by_batch=self.local_cfg.normalize_by_batch,
                batch_size=batch_size,
            )
            grad_g = grad_g + grad_homeo

        inhibitory_homeo_factor = self._compute_inhibitory_homeostasis_factor(
            rec=rec, R_tot=r_tot, v_n=v_n
        )
        if inhibitory_homeo_factor is not None:
            grad_homeo = _factor_pre_weight_grad(
                inhibitory_homeo_factor,
                x_pre,
                normalize_by_batch=self.local_cfg.normalize_by_batch,
                batch_size=batch_size,
            )
            grad_g = grad_g + grad_homeo

        self._add_topk_transformed_grad(layer, grad_g, rec.get(mask_key))
        return layer

    def _compute_inhibitory_topk_factor(
        self,
        *,
        rec: dict[str, Any],
        e_v: torch.Tensor,
        post_factor_raw: torch.Tensor,
        v_n: torch.Tensor,
        r_tot: Any,
        rho: Any,
        phi: float,
        branch_scale: Any,
        layer_dynamics_mode: str,
    ) -> torch.Tensor:
        if layer_dynamics_mode == "conductance":
            if self.local_cfg.three_factor.use_driving_force:
                inh_post = self.local_cfg.three_factor.e_rev_inh - v_n
            else:
                inh_post = self.local_cfg.three_factor.theta - v_n
            inh_factor = e_v * r_tot * inh_post
        else:
            additive_mode = getattr(
                self.local_cfg.three_factor, "additive_gain_mode", "none"
            )
            if additive_mode == "input_dependent":
                pseudo_R, _ = self._compute_additive_pseudo_signals(rec, v_n)
                inh_post = self.local_cfg.three_factor.theta - v_n
                inh_factor = e_v * pseudo_R * inh_post
            elif additive_mode in ("learned_gain", "running_stats"):
                inh_factor = -post_factor_raw
            else:
                inh_factor = -e_v

        if isinstance(rho, torch.Tensor):
            inh_factor = inh_factor * rho.unsqueeze(0)
        else:
            inh_factor = (
                inh_factor * float(rho)
                if self.local_cfg.rule_variant in {"4f", "5f"}
                else inh_factor
            )

        inh_factor = (
            inh_factor * float(phi)
            if self.local_cfg.rule_variant == "5f"
            else inh_factor
        )

        if isinstance(branch_scale, torch.Tensor):
            inh_factor = inh_factor * branch_scale.unsqueeze(0)
        else:
            inh_factor = inh_factor * float(branch_scale)
        return inh_factor

    def _add_topk_transformed_grad(
        self,
        layer: TopKLinear,
        local_grad: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> None:
        grad_param = _topk_pre_weight_grad(
            local_grad,
            pre_weight=layer.pre_w,
            weight_transform=getattr(layer, "weight_transform", "exp"),
            derivative_fn=self._weight_transform_derivative,
        )
        self._add_topk_grad(layer, grad_param, mask)


__all__ = ["LocalLearningTopKGradientMixin"]
