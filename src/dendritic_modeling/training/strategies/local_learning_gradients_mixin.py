"""Local gradient-application methods for local credit assignment."""

from __future__ import annotations

from typing import Any

import torch

from dendritic_modeling.networks import ParametricActivation
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.training.strategies.local_learning_state import (
    _LayerModulators,
    _PostFactors,
)


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
        exc_layer = self._apply_excitatory_topk_local_grad(
            rec=rec,
            module_key="exc_module",
            input_key="x_exc",
            mask_key="exc_mask",
            post_factor=post_factors.post_factor,
            v_n=v_n,
            r_tot=r_tot,
            layer_dynamics_mode=layer_dynamics_mode,
            batch_size=batch_size,
            role_synaptic_alignment=modulators.role_synaptic_alignment,
        )

        self._apply_excitatory_topk_local_grad(
            rec=rec,
            module_key="rec_exc_module",
            input_key="x_rec_exc",
            mask_key="rec_exc_mask",
            post_factor=post_factors.post_factor,
            v_n=v_n,
            r_tot=r_tot,
            layer_dynamics_mode=layer_dynamics_mode,
            batch_size=batch_size,
            role_synaptic_alignment=modulators.role_synaptic_alignment,
        )

        self._apply_inhibitory_topk_local_grad(
            rec=rec,
            module_key="inh_module",
            input_key="x_inh",
            mask_key="inh_mask",
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

        self._apply_inhibitory_topk_local_grad(
            rec=rec,
            module_key="rec_inh_module",
            input_key="x_rec_inh",
            mask_key="rec_inh_mask",
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
        if self.local_cfg.normalize_by_batch:
            grad_g = grad_g / float(batch_size)

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
            if self.local_cfg.normalize_by_batch:
                grad_homeo = grad_homeo / float(batch_size)
            grad_g = grad_g + grad_homeo

        if (
            isinstance(role_synaptic_alignment, torch.Tensor)
            and role_synaptic_alignment.size() == grad_g.size()
        ):
            grad_g = grad_g * role_synaptic_alignment.to(
                device=grad_g.device, dtype=grad_g.dtype
            )

        chain = self._weight_transform_derivative(
            layer.pre_w.detach(),
            getattr(layer, "weight_transform", "exp"),
        )
        grad_param = grad_g * chain
        self._add_topk_grad(layer, grad_param, rec.get(mask_key))
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

        grad_g = inh_factor.t() @ x_pre
        if self.local_cfg.normalize_by_batch:
            grad_g = grad_g / float(batch_size)

        if (
            isinstance(role_synaptic_alignment, torch.Tensor)
            and role_synaptic_alignment.size() == grad_g.size()
        ):
            grad_g = grad_g * role_synaptic_alignment.to(
                device=grad_g.device, dtype=grad_g.dtype
            )

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
            if self.local_cfg.normalize_by_batch:
                grad_homeo = grad_homeo / float(batch_size)
            grad_g = grad_g + grad_homeo

        inhibitory_homeo_factor = self._compute_inhibitory_homeostasis_factor(
            rec=rec, R_tot=r_tot, v_n=v_n
        )
        if inhibitory_homeo_factor is not None:
            grad_homeo = inhibitory_homeo_factor.t() @ x_pre
            if self.local_cfg.normalize_by_batch:
                grad_homeo = grad_homeo / float(batch_size)
            grad_g = grad_g + grad_homeo

        chain = self._weight_transform_derivative(
            layer.pre_w.detach(),
            getattr(layer, "weight_transform", "exp"),
        )
        grad_param = grad_g * chain
        self._add_topk_grad(layer, grad_param, rec.get(mask_key))
        return layer

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

        grad_g_den = den_factor.mean(dim=0)
        if not self.local_cfg.normalize_by_batch:
            grad_g_den = grad_g_den * float(batch_size)

        voltage_homeo_error = self._compute_voltage_homeostasis_error(v_n)
        if voltage_homeo_error is not None:
            if layer_dynamics_mode == "conductance":
                diff_homeo = x_blk - v_n.unsqueeze(-1)
                if isinstance(r_tot, torch.Tensor):
                    r_homeo = r_tot.unsqueeze(-1)
                else:
                    r_homeo = torch.ones_like(v_n).unsqueeze(-1) * float(r_tot)
                den_homeo_factor = (
                    voltage_homeo_error.unsqueeze(-1) * r_homeo * diff_homeo
                )
            else:
                den_homeo_factor = voltage_homeo_error.unsqueeze(-1) * x_blk
            grad_homeo_den = den_homeo_factor.mean(dim=0)
            if not self.local_cfg.normalize_by_batch:
                grad_homeo_den = grad_homeo_den * float(batch_size)
            grad_g_den = grad_g_den + grad_homeo_den

        if (
            isinstance(role_block_alignment, torch.Tensor)
            and role_block_alignment.size() == grad_g_den.size()
        ):
            grad_g_den = grad_g_den * role_block_alignment.to(
                device=grad_g_den.device, dtype=grad_g_den.dtype
            )

        if self.local_cfg.morphology_aware.use_dendritic_normalization:
            grad_g_den = self._compute_dendritic_normalization(rec, grad_g_den)

        den_chain = self._weight_transform_derivative(
            blk_layer.log_weight.detach(),
            getattr(blk_layer, "weight_transform", "exp"),
        )
        grad_param_den = grad_g_den * den_chain
        if blk_layer.log_weight.grad is None:
            blk_layer.log_weight.grad = grad_param_den.to(blk_layer.log_weight.dtype)
        else:
            blk_layer.log_weight.grad = blk_layer.log_weight.grad + grad_param_den.to(
                blk_layer.log_weight.dtype
            )

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

        if hasattr(react_module, "log_m"):
            y = v_out
            one_minus_y2 = 1 - (2 * y - 1) ** 2
            if hasattr(react_module, "b"):
                elig_m = one_minus_y2 * (v_n - react_module.b.detach())
            else:
                elig_m = one_minus_y2 * v_n
            m_vec = react_module.log_m.detach().exp()
            react_factor = e_n
            if isinstance(rho, torch.Tensor):
                react_factor = react_factor * rho.unsqueeze(0)
            else:
                react_factor = react_factor * float(rho)
            react_factor = react_factor * float(phi)
            if isinstance(branch_scale, torch.Tensor):
                react_factor = react_factor * branch_scale.unsqueeze(0)
            else:
                react_factor = react_factor * float(branch_scale)
            grad_m = (react_factor * elig_m).mean(dim=0)
            grad_log_m = grad_m * m_vec
            if react_module.log_m.grad is None:
                react_module.log_m.grad = grad_log_m.to(react_module.log_m.dtype)
            else:
                react_module.log_m.grad = react_module.log_m.grad + grad_log_m.to(
                    react_module.log_m.dtype
                )

        if hasattr(react_module, "b"):
            y = v_out
            one_minus_y2 = 1 - (2 * y - 1) ** 2
            m_vec = (
                react_module.log_m.detach().exp()
                if hasattr(react_module, "log_m")
                else 1.0
            )
            elig_b = -one_minus_y2 * m_vec
            react_factor_b = e_n
            if isinstance(rho, torch.Tensor):
                react_factor_b = react_factor_b * rho.unsqueeze(0)
            else:
                react_factor_b = react_factor_b * float(rho)
            react_factor_b = react_factor_b * float(phi)
            if isinstance(branch_scale, torch.Tensor):
                react_factor_b = react_factor_b * branch_scale.unsqueeze(0)
            else:
                react_factor_b = react_factor_b * float(branch_scale)
            grad_b = (react_factor_b * elig_b).mean(dim=0)
            if react_module.b.grad is None:
                react_module.b.grad = grad_b.to(react_module.b.dtype)
            else:
                react_module.b.grad = react_module.b.grad + grad_b.to(
                    react_module.b.dtype
                )

        aux_grad_log_m, aux_grad_b = self._compute_gate_homeostasis_aux_grads(
            react_module=react_module,
            v_n=v_n,
            v_out=v_out,
        )
        if aux_grad_log_m is not None and hasattr(react_module, "log_m"):
            if react_module.log_m.grad is None:
                react_module.log_m.grad = aux_grad_log_m.to(react_module.log_m.dtype)
            else:
                react_module.log_m.grad = react_module.log_m.grad + aux_grad_log_m.to(
                    react_module.log_m.dtype
                )
        if (
            aux_grad_b is not None
            and hasattr(react_module, "b")
            and react_module.b is not None
        ):
            if react_module.b.grad is None:
                react_module.b.grad = aux_grad_b.to(react_module.b.dtype)
            else:
                react_module.b.grad = react_module.b.grad + aux_grad_b.to(
                    react_module.b.dtype
                )


__all__ = ["LocalLearningGradientMixin"]
