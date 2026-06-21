"""Local gradient-application methods for local credit assignment."""

from __future__ import annotations

from typing import Any

import torch

from dendritic_modeling.networks import ParametricActivation
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.training.strategies.local_learning_pathways import (
    EXCITATORY_TOPK_PATHS,
    INHIBITORY_TOPK_PATHS,
)
from dendritic_modeling.training.strategies.local_learning_state import (
    _LayerModulators,
    _PostFactors,
)


def _accumulate_parameter_grad(param: torch.Tensor, grad: torch.Tensor) -> None:
    """Accumulate a local-rule gradient using the parameter's dtype."""
    grad = grad.to(param.dtype)
    if param.grad is None:
        param.grad = grad
    else:
        param.grad = param.grad + grad


def _scale_reactivation_error(
    error_signal: torch.Tensor,
    *,
    rho: Any,
    phi: float,
    branch_scale: Any,
) -> torch.Tensor:
    """Apply local-rule modulation to a reactivation error signal."""
    scaled = error_signal
    if isinstance(rho, torch.Tensor):
        scaled = scaled * rho.unsqueeze(0)
    else:
        scaled = scaled * float(rho)
    scaled = scaled * float(phi)
    if isinstance(branch_scale, torch.Tensor):
        scaled = scaled * branch_scale.unsqueeze(0)
    else:
        scaled = scaled * float(branch_scale)
    return scaled


def _reactivation_output_slope(output: torch.Tensor) -> torch.Tensor:
    """Derivative factor for the normalized tanh-style reactivation output."""
    return 1 - (2 * output - 1) ** 2


def _apply_optional_alignment(
    grad: torch.Tensor,
    alignment: torch.Tensor | None,
) -> torch.Tensor:
    """Apply a same-shaped alignment tensor to a local gradient."""
    if isinstance(alignment, torch.Tensor) and alignment.size() == grad.size():
        return grad * alignment.to(device=grad.device, dtype=grad.dtype)
    return grad


def _normalize_batch_gradient(
    grad: torch.Tensor,
    *,
    normalize_by_batch: bool,
    batch_size: int,
) -> torch.Tensor:
    """Average a summed local gradient when the local rule requests it."""
    if normalize_by_batch:
        return grad / float(batch_size)
    return grad


def _reduce_batch_mean_gradient(
    factor: torch.Tensor,
    *,
    normalize_by_batch: bool,
    batch_size: int,
) -> torch.Tensor:
    """Reduce per-example local factors using the legacy batch scaling rule."""
    grad = factor.mean(dim=0)
    if not normalize_by_batch:
        grad = grad * float(batch_size)
    return grad


def _topk_pre_weight_grad(
    local_grad: torch.Tensor,
    *,
    pre_weight: torch.Tensor,
    weight_transform: str,
    derivative_fn: Any,
) -> torch.Tensor:
    """Map a local conductance gradient through the TopK weight transform."""
    chain = derivative_fn(pre_weight.detach(), weight_transform)
    return local_grad * chain


class LocalLearningGradientMixin:
    """Gradient writeback helpers used by ``LocalCreditAssignment``."""

    def _apply_layer_local_gradients(
        self,
        *,
        rec: dict[str, Any],
        layer_idx: int,
        num_layers: int,
        y_target: torch.Tensor | None,
        batch_size: int,
        e_n: torch.Tensor,
        stdp_error_signal: torch.Tensor,
        v_n: torch.Tensor,
        r_tot: torch.Tensor | float,
        layer_dynamics_mode: str,
        modulators: _LayerModulators,
        post_factors: _PostFactors,
        update_reactivation: bool,
    ) -> None:
        exc_layer = self._apply_excitatory_topk_pathways(
            rec=rec,
            v_n=v_n,
            r_tot=r_tot,
            layer_dynamics_mode=layer_dynamics_mode,
            batch_size=batch_size,
            modulators=modulators,
            post_factors=post_factors,
        )

        self._apply_inhibitory_topk_pathways(
            rec=rec,
            v_n=v_n,
            r_tot=r_tot,
            layer_dynamics_mode=layer_dynamics_mode,
            batch_size=batch_size,
            modulators=modulators,
            post_factors=post_factors,
        )

        self._apply_dendritic_block_local_grad(
            rec=rec,
            e_v=post_factors.e_v,
            post_factor_raw=post_factors.post_factor_raw,
            v_n=v_n,
            r_tot=r_tot,
            rho=modulators.rho,
            phi=modulators.phi,
            dendritic_branch_scale=modulators.dendritic_branch_scale,
            role_block_alignment=modulators.role_block_alignment,
            layer_dynamics_mode=layer_dynamics_mode,
            batch_size=batch_size,
        )

        self._apply_reactivation_local_grad(
            rec=rec,
            e_n=e_n,
            v_n=v_n,
            rho=modulators.rho,
            phi=modulators.phi,
            branch_scale=modulators.branch_scale,
            update_reactivation=update_reactivation,
        )

        if self._stdp_enabled():
            post_signal = rec.get("v_out")
            if not isinstance(post_signal, torch.Tensor):
                post_signal = v_n
            self._apply_stdp_pathways(
                rec=rec,
                post_signal=post_signal,
                error_signal=stdp_error_signal,
            )

        self._apply_hsic_auxiliary_gradients(
            rec=rec,
            exc_layer=exc_layer,
            layer_idx=layer_idx,
            num_layers=num_layers,
            y_target=y_target,
            batch_size=batch_size,
        )

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
        grad_g = post_factor.t() @ x_pre
        grad_g = _normalize_batch_gradient(
            grad_g,
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
            grad_homeo = homeo_factor.t() @ x_pre
            grad_homeo = _normalize_batch_gradient(
                grad_homeo,
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

        grad_g = inh_factor.t() @ x_pre
        grad_g = _normalize_batch_gradient(
            grad_g,
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
            grad_homeo = homeo_factor.t() @ x_pre
            grad_homeo = _normalize_batch_gradient(
                grad_homeo,
                normalize_by_batch=self.local_cfg.normalize_by_batch,
                batch_size=batch_size,
            )
            grad_g = grad_g + grad_homeo

        inhibitory_homeo_factor = self._compute_inhibitory_homeostasis_factor(
            rec=rec, R_tot=r_tot, v_n=v_n
        )
        if inhibitory_homeo_factor is not None:
            grad_homeo = inhibitory_homeo_factor.t() @ x_pre
            grad_homeo = _normalize_batch_gradient(
                grad_homeo,
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

    def _apply_dendritic_block_local_grad(
        self,
        *,
        rec: dict[str, Any],
        e_v: torch.Tensor,
        post_factor_raw: torch.Tensor,
        v_n: torch.Tensor,
        r_tot: Any,
        rho: Any,
        phi: float,
        dendritic_branch_scale: Any,
        role_block_alignment: torch.Tensor | None,
        layer_dynamics_mode: str,
        batch_size: int,
    ) -> None:
        blk_layer = rec.get("blk_module")
        x_blk_raw = rec.get("x_blk_raw")
        if blk_layer is None or x_blk_raw is None:
            return

        block_size = blk_layer.block_size
        x_blk = x_blk_raw.view(x_blk_raw.size(0), blk_layer.out_features, block_size)
        den_factor = self._compute_dendritic_block_factor(
            rec=rec,
            e_v=e_v,
            post_factor_raw=post_factor_raw,
            v_n=v_n,
            r_tot=r_tot,
            rho=rho,
            phi=phi,
            dendritic_branch_scale=dendritic_branch_scale,
            x_blk=x_blk,
            layer_dynamics_mode=layer_dynamics_mode,
        )

        grad_g_den = _reduce_batch_mean_gradient(
            den_factor,
            normalize_by_batch=self.local_cfg.normalize_by_batch,
            batch_size=batch_size,
        )

        voltage_homeo_error = self._compute_voltage_homeostasis_error(v_n)
        if voltage_homeo_error is not None:
            den_homeo_factor = self._compute_dendritic_block_homeostasis_factor(
                voltage_homeo_error=voltage_homeo_error,
                v_n=v_n,
                r_tot=r_tot,
                x_blk=x_blk,
                layer_dynamics_mode=layer_dynamics_mode,
            )
            grad_homeo_den = _reduce_batch_mean_gradient(
                den_homeo_factor,
                normalize_by_batch=self.local_cfg.normalize_by_batch,
                batch_size=batch_size,
            )
            grad_g_den = grad_g_den + grad_homeo_den

        grad_g_den = _apply_optional_alignment(grad_g_den, role_block_alignment)

        if self.local_cfg.morphology_aware.use_dendritic_normalization:
            grad_g_den = self._compute_dendritic_normalization(rec, grad_g_den)

        self._add_dendritic_block_transformed_grad(blk_layer, grad_g_den)

    def _add_dendritic_block_transformed_grad(
        self,
        blk_layer: Any,
        local_grad: torch.Tensor,
    ) -> None:
        den_chain = self._weight_transform_derivative(
            blk_layer.log_weight.detach(),
            getattr(blk_layer, "weight_transform", "exp"),
        )
        grad_param_den = local_grad * den_chain
        _accumulate_parameter_grad(blk_layer.log_weight, grad_param_den)

    def _compute_dendritic_block_factor(
        self,
        *,
        rec: dict[str, Any],
        e_v: torch.Tensor,
        post_factor_raw: torch.Tensor,
        v_n: torch.Tensor,
        r_tot: Any,
        rho: Any,
        phi: float,
        dendritic_branch_scale: Any,
        x_blk: torch.Tensor,
        layer_dynamics_mode: str,
    ) -> torch.Tensor:
        if layer_dynamics_mode == "conductance":
            diff = x_blk - v_n.unsqueeze(-1)
            if isinstance(r_tot, torch.Tensor):
                r_exp = r_tot.unsqueeze(-1)
            else:
                r_exp = torch.ones_like(v_n).unsqueeze(-1) * float(r_tot)
            den_factor = e_v.unsqueeze(-1) * r_exp * diff
        else:
            additive_mode = getattr(
                self.local_cfg.three_factor, "additive_gain_mode", "none"
            )
            if additive_mode == "input_dependent":
                pseudo_R, _ = self._compute_additive_pseudo_signals(rec, v_n)
                diff = x_blk - v_n.unsqueeze(-1)
                den_factor = e_v.unsqueeze(-1) * pseudo_R.unsqueeze(-1) * diff
            elif additive_mode in ("learned_gain", "running_stats"):
                den_factor = post_factor_raw.unsqueeze(-1) * x_blk
            else:
                den_factor = e_v.unsqueeze(-1) * x_blk

        if self.local_cfg.rule_variant in {"4f", "5f"}:
            if isinstance(rho, torch.Tensor):
                den_factor = den_factor * rho.unsqueeze(0).unsqueeze(-1)
            else:
                den_factor = den_factor * float(rho)

        if self.local_cfg.rule_variant == "5f":
            den_factor = den_factor * float(phi)

        if isinstance(dendritic_branch_scale, torch.Tensor):
            if dendritic_branch_scale.dim() == 2:
                den_factor = den_factor * dendritic_branch_scale.unsqueeze(0)
            else:
                den_factor = den_factor * dendritic_branch_scale.unsqueeze(0).unsqueeze(
                    -1
                )
        else:
            den_factor = den_factor * float(dendritic_branch_scale)
        return den_factor

    def _compute_dendritic_block_homeostasis_factor(
        self,
        *,
        voltage_homeo_error: torch.Tensor,
        v_n: torch.Tensor,
        r_tot: Any,
        x_blk: torch.Tensor,
        layer_dynamics_mode: str,
    ) -> torch.Tensor:
        if layer_dynamics_mode == "conductance":
            diff_homeo = x_blk - v_n.unsqueeze(-1)
            if isinstance(r_tot, torch.Tensor):
                r_homeo = r_tot.unsqueeze(-1)
            else:
                r_homeo = torch.ones_like(v_n).unsqueeze(-1) * float(r_tot)
            return voltage_homeo_error.unsqueeze(-1) * r_homeo * diff_homeo
        return voltage_homeo_error.unsqueeze(-1) * x_blk

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


__all__ = ["LocalLearningGradientMixin"]
