"""
Local learning training strategies.

This module implements multi-factor local learning rules for dendritic networks.

It provides biologically-inspired 3-/4-/5-factor Hebbian-like rules that use
locally available signals (pre-synaptic input, compartment voltages) modulated
by a broadcast soma error and optional morphology/information factors.
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from dendritic_modeling.config.local_learning import (
    build_local_rule_config,
    coerce_legacy_local_rule_config,
    section_to_dict,
)
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    DendriticBranchLayer,
    ExcitationInhibitionNetwork,
    ParametricActivation,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.training.strategies.local_learning_broadcast_mixin import (
    LocalLearningBroadcastMixin,
)
from dendritic_modeling.training.strategies.local_learning_modulators import (
    LocalLearningModulatorMixin,
)
from dendritic_modeling.training.strategies.local_learning_recorders import (
    LocalLearningRecorderMixin,
)
from dendritic_modeling.training.strategies.local_learning_signals import (
    LocalLearningSignalMixin,
)
from dendritic_modeling.training.strategies.local_learning_state import _LayerStats
from dendritic_modeling.training.strategies.local_learning_stdp_mixin import (
    LocalLearningSTDPMixin,
)
from dendritic_modeling.training.strategies.standard import Trainer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _BroadcastState:
    mode: str
    transported_errors: list[Optional[torch.Tensor]]
    feedback_seeds: list[Optional[torch.Tensor]]


@dataclass(frozen=True)
class _LayerModulators:
    rho: Any
    phi: Any
    branch_scale: torch.Tensor | float
    dendritic_branch_scale: torch.Tensor | float
    role_synaptic_alignment: Optional[torch.Tensor]
    role_block_alignment: Optional[torch.Tensor]


@dataclass(frozen=True)
class _PostFactors:
    e_v: torch.Tensor
    post_factor: torch.Tensor
    post_factor_raw: torch.Tensor


class LocalCreditAssignment(
    LocalLearningSignalMixin,
    LocalLearningRecorderMixin,
    LocalLearningSTDPMixin,
    LocalLearningBroadcastMixin,
    LocalLearningModulatorMixin,
    Trainer,
):
    """
    Local multi-factor learning strategy for dendritic networks.

    Implements 3-/4-/5-factor Hebbian-like local rules for both synaptic
    conductances (TopKLinear) and dendritic branch conductances (BlockLinear)
    using locally available signals plus a broadcast error from the soma.
    """

    def __init__(self, *args, **kwargs):
        config_dict = kwargs.pop("local_rule_config", {}) or {}
        super().__init__(*args, **kwargs)
        # Build LocalRuleConfig with proper nested config handling
        self.local_cfg = self._build_local_rule_config(config_dict)
        self._layer_stats: dict[int, _LayerStats] = {}
        self._decoder_cache: dict[str, Any] = {}
        self._additive_gain_cache: dict[str, Any] = {}
        self._additive_running_var: dict[int, torch.Tensor] = {}
        self._broadcast_cache: dict[tuple[Any, ...], Any] = {}
        self._stdp_traces: dict[int, dict[str, torch.Tensor]] = {}
        self._warned_decoder_soma_fallback = False
        self.filename_prefix = "local_learning_"
        self.logger_info_prefix = "[Local] "

    @staticmethod
    def _section_to_dict(section: Any) -> dict[str, Any]:
        """Convert a config section object to a plain dict."""
        return section_to_dict(section)

    def _coerce_legacy_local_rule_config(
        self, config_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """Normalize legacy flat local learning config to nested LocalRuleConfig layout."""
        return coerce_legacy_local_rule_config(config_dict)

    def _build_local_rule_config(self, config_dict: dict):
        """Build LocalRuleConfig from dict with proper nested config handling."""
        return build_local_rule_config(config_dict)

    def _should_update_local_record(self, rec: dict[str, Any]) -> bool:
        """Return whether LocalCA should write gradients for a recorded branch.

        This only changes explicit inhibitory-cell DendriNet populations. I-to-E
        synapses that live on excitatory dendrites remain ordinary excitatory
        population records and are still updated by LocalCA.
        """
        if rec.get("population") != "explicit_inhibitory":
            return True

        mode = getattr(self.local_cfg, "explicit_inhibitory_update_mode", None)
        if mode is None:
            # Compatibility for older LocalRuleConfig-like objects in tests.
            return bool(
                getattr(self.local_cfg, "update_explicit_inhibitory_cells", True)
            )
        return str(mode).lower() == "local_ca"

    # Override: do not run extra pretrain/final phases here
    def _pretrain(self, *args, **kwargs):  # type: ignore[override]
        return [], []

    def _resolve_local_reactivation_update_flag(
        self, update_reactivation: bool
    ) -> bool:
        """Disable local reactivation updates when the shared trainer mode does not use learned b,m."""
        if self._normalize_reactivation_update_mode() == "backprop":
            return bool(update_reactivation)
        return False

    # ----------------------- Core training overrides -----------------------
    def _epoch_train(
        self, model: BaseModel, train_loader
    ) -> tuple[float, float, float]:  # type: ignore[override]
        model.train()
        train_loss = 0.0
        train_loss_base = 0.0
        train_loss_reg = 0.0

        # Fallback: if model has no dendritic branch layers, use standard epoch_train
        try:
            has_branch_layers = any(
                isinstance(m, DendriticBranchLayer) for m in model.modules()
            )
        except Exception:
            has_branch_layers = False
        if not has_branch_layers:
            # Defer to base implementation
            return super()._epoch_train(model, train_loader)

        for batch in train_loader:
            x_batch, y_batch = batch[0], batch[1]
            # Resolve per-epoch schedule for reactivation/decoder behavior
            update_reactivation = self.local_cfg.update_reactivation
            encoder_update_mode = self.local_cfg.encoder_update_mode
            decoder_update_mode = self.local_cfg.decoder_update_mode
            # Freeze phases by epoch index
            if self.epoch_counter <= max(0, int(self.local_cfg.freeze_encoder_epochs)):
                encoder_update_mode = "none"
            if self.epoch_counter <= max(
                0, int(self.local_cfg.freeze_reactivation_epochs)
            ):
                update_reactivation = False
            if self.epoch_counter <= max(0, int(self.local_cfg.freeze_decoder_epochs)):
                decoder_update_mode = "none"
            # Optional schedule entries override
            if self.local_cfg.training_schedule:
                for sched in self.local_cfg.training_schedule:
                    try:
                        s = int(sched.get("start_epoch", 1))
                        e = int(sched.get("end_epoch", 10**9))
                    except Exception:
                        s, e = 1, 10**9
                    if s <= self.epoch_counter <= e:
                        if "update_reactivation" in sched:
                            update_reactivation = bool(sched["update_reactivation"])
                        if "encoder_update_mode" in sched:
                            encoder_update_mode = str(
                                sched["encoder_update_mode"]
                            ).lower()
                        if "decoder_update_mode" in sched:
                            decoder_update_mode = str(
                                sched["decoder_update_mode"]
                            ).lower()
                        break
            update_reactivation = self._resolve_local_reactivation_update_flag(
                update_reactivation
            )
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Capture layer-wise local signals via forward hooks
            layer_records, handles = self._attach_local_recorders(model)
            topk_modules = [m for m in model.modules() if isinstance(m, TopKLinear)]
            for m in topk_modules:
                m.cache_mask = True
            try:
                # Forward once to obtain outputs and local signals
                y_hat: torch.Tensor = model(x_batch)
            finally:
                for h in handles:
                    h.remove()
                for m in topk_modules:
                    m.cache_mask = False
                    m._last_forward_weight_mask = None

            # Compute scalar loss for logging (no backward)
            loss = self._compute_logging_loss(model, x_batch, y_batch)
            loss_item = loss.item()
            train_loss += loss_item
            train_loss_base += loss_item  # Local learning doesn't separate base/reg
            # train_loss_reg remains 0

            # Compute output-space error (dL/dy_hat) then resolve local soma-space
            # signals for local dendritic updates.
            delta_out: torch.Tensor = self._compute_soma_error(y_hat, y_batch)
            v0_local, delta_local = self._resolve_local_soma_signals(
                model=model,
                y_hat=y_hat,
                delta_out=delta_out.detach(),
            )

            # Prepare HSIC targets if needed
            y_target: Optional[torch.Tensor] = None
            if self.local_cfg.hsic.enabled:
                if self.local_cfg.hsic.target_source == "labels":
                    if y_batch.dim() == 1 or (
                        y_batch.dim() == 2 and y_batch.size(1) == 1
                    ):
                        num_classes = y_hat.size(-1)
                        y_target = torch.nn.functional.one_hot(
                            y_batch.view(-1), num_classes=num_classes
                        ).to(y_hat.dtype)
                    else:
                        y_target = y_batch.to(y_hat.dtype)
                else:
                    y_target = y_hat.detach()

            # Zero gradients (we will set param.grads directly)
            try:
                self.optimizer.zero_grad(set_to_none=True)
            except TypeError:
                self.optimizer.zero_grad()

            # Apply local learning updates as gradients on parameters
            self._apply_local_rule_gradients(
                model,
                layer_records,
                v0=v0_local,
                delta=delta_local,
                y_target=y_target,
                update_reactivation=update_reactivation,
            )

            backprop_param_groups: list[list[torch.nn.Parameter]] = []
            if (
                encoder_update_mode == "backprop"
                and hasattr(model, "encoder_network")
                and model.encoder_network is not None
            ):
                enc_params = [
                    p for p in model.encoder_network.parameters() if p.requires_grad
                ]
                if enc_params:
                    backprop_param_groups.append(enc_params)

            if hasattr(model, "decoder_network") and decoder_update_mode == "backprop":
                dec_params = [
                    p for p in model.decoder_network.parameters() if p.requires_grad
                ]
                if dec_params:
                    backprop_param_groups.append(dec_params)

            for group_idx, params in enumerate(backprop_param_groups):
                grads = torch.autograd.grad(
                    loss,
                    params,
                    retain_graph=group_idx < len(backprop_param_groups) - 1,
                    allow_unused=True,
                )
                for p, g in zip(params, grads):
                    if g is None:
                        continue
                    grad = g.detach()
                    if p.grad is None:
                        p.grad = grad.clone()
                    else:
                        p.grad = p.grad + grad

            # Decoder update mode
            if hasattr(model, "decoder_network") and decoder_update_mode == "local":
                # Simple local 3-factor on last Linear: grad_W = e_out^T @ h_in; grad_b = sum(e_out)
                lin_mod = self._decoder_cache.get("module")
                h_in = self._decoder_cache.get("input")
                if isinstance(lin_mod, nn.Linear) and isinstance(h_in, torch.Tensor):
                    e_out = delta_out.detach()  # [B, out]
                    if self.local_cfg.normalize_by_batch and e_out.size(0) > 0:
                        scale = 1.0 / float(e_out.size(0))
                    else:
                        scale = 1.0
                    grad_w = scale * (e_out.t() @ h_in.detach())  # [out, in]
                    if lin_mod.weight.grad is None:
                        lin_mod.weight.grad = grad_w.to(lin_mod.weight.dtype)
                    else:
                        lin_mod.weight.grad = lin_mod.weight.grad + grad_w.to(
                            lin_mod.weight.dtype
                        )
                    if lin_mod.bias is not None:
                        grad_b = scale * e_out.sum(dim=0)
                        if lin_mod.bias.grad is None:
                            lin_mod.bias.grad = grad_b.to(lin_mod.bias.dtype)
                        else:
                            lin_mod.bias.grad = lin_mod.bias.grad + grad_b.to(
                                lin_mod.bias.dtype
                            )

            # Optional gradient clipping (value-wise)
            if self.local_cfg.clip_grad_value and self.local_cfg.clip_grad_value > 0:
                torch.nn.utils.clip_grad_value_(
                    model.parameters(), self.local_cfg.clip_grad_value
                )

            # Step optimizer
            self.optimizer.step()

        train_loss /= len(train_loader)
        train_loss_base /= len(train_loader)
        train_loss_reg /= len(train_loader)
        return train_loss, train_loss_base, train_loss_reg

    # ------------------------ Helper: local rule grads -----------------------
    def _apply_local_rule_gradients(
        self,
        model: BaseModel,
        layer_records: list[dict[str, Any]],
        v0: torch.Tensor,
        delta: torch.Tensor,
        y_target: Optional[torch.Tensor] = None,
        update_reactivation: bool = True,
    ) -> None:
        batch_size = v0.size(0)
        delta_scalar: torch.Tensor = self._reduce_error_to_scalar(delta)  # [B, 1]
        broadcast_state = self._prepare_local_rule_broadcast_state(
            model=model,
            layer_records=layer_records,
            delta=delta,
            delta_scalar=delta_scalar,
        )

        # Iterate in the order records were captured
        for layer_idx, rec in enumerate(layer_records):
            if not self._should_update_local_record(rec):
                # In freeze mode, explicit inhibitory-cell dendrites are held
                # fixed. The E-cell I-to-E synapses still learn locally because
                # they are part of excitatory dendrite records.
                continue

            v_n: Optional[torch.Tensor] = rec.get("v_n")  # [B, out]
            if v_n is None:
                continue

            # Layer depth scaling (1-indexed)
            layer_depth = layer_idx + 1
            out_features = v_n.size(1)

            e_n = self._compute_local_broadcast_error(
                rec=rec,
                layer_idx=layer_idx,
                out_features=out_features,
                v_n=v_n,
                delta=delta,
                delta_scalar=delta_scalar,
                broadcast_state=broadcast_state,
            )
            stdp_error_signal = e_n
            if str(getattr(self.local_cfg, "rule_variant", "")).lower() == "stdp":
                e_n = torch.zeros_like(e_n)

            layer_dynamics_mode = self._resolve_layer_dynamics_mode(rec)
            R_tot = self._compute_local_input_resistance(
                rec=rec,
                v_n=v_n,
                layer_dynamics_mode=layer_dynamics_mode,
            )

            # === PATH PROPAGATION FACTOR ===
            # Apply path-integrated attenuation if enabled
            if (
                self.local_cfg.morphology_aware.use_path_propagation
                and broadcast_state.mode != "path_transport"
            ):
                path_factor = rec.get("path_factor", 1.0)
                # Modulate error by path attenuation
                if isinstance(path_factor, torch.Tensor):
                    e_n = e_n * path_factor

            modulators = self._compute_local_layer_modulators(
                rec=rec,
                v0=v0,
                layer_depth=layer_depth,
            )
            post_factors = self._compute_local_post_factors(
                rec=rec,
                e_n=e_n,
                v_n=v_n,
                r_tot=R_tot,
                layer_dynamics_mode=layer_dynamics_mode,
                layer_idx=layer_idx,
                modulators=modulators,
            )
            self._apply_layer_local_gradients(
                rec=rec,
                layer_idx=layer_idx,
                num_layers=len(layer_records),
                y_target=y_target,
                batch_size=batch_size,
                e_n=e_n,
                stdp_error_signal=stdp_error_signal,
                v_n=v_n,
                r_tot=R_tot,
                layer_dynamics_mode=layer_dynamics_mode,
                modulators=modulators,
                post_factors=post_factors,
                update_reactivation=update_reactivation,
            )

    def _prepare_local_rule_broadcast_state(
        self,
        *,
        model: BaseModel,
        layer_records: list[dict[str, Any]],
        delta: torch.Tensor,
        delta_scalar: torch.Tensor,
    ) -> _BroadcastState:
        bmode = (self.local_cfg.error_broadcast_mode or "scalar").lower()

        if (
            bmode == "pathway_vector"
            or bmode == "path_transport"
            or self.local_cfg.morphology_aware.use_path_propagation
            or self.local_cfg.morphology_aware.morphology_modulator_mode != "none"
            or self.local_cfg.morphology_aware.use_branch_type_rules
            or self.local_cfg.morphology_aware.use_branch_role_rules
            or self.local_cfg.morphology_aware.use_branch_length_modulation
        ):
            self._initialize_branch_morphology(model, layer_records)

        if self.local_cfg.morphology_aware.use_path_propagation:
            path_factors = self._precompute_path_propagation_factors(
                layer_records,
                include_parent_activation_derivative=True,
            )
            for rec, path_factor in zip(layer_records, path_factors):
                rec["path_factor"] = path_factor

        transported_errors: list[Optional[torch.Tensor]] = []
        if bmode == "path_transport":
            transported_errors = self._precompute_path_transport_errors(
                layer_records=layer_records,
                delta=delta,
                delta_scalar=delta_scalar,
            )

        feedback_seeds: list[Optional[torch.Tensor]] = []
        if bmode == "pathway_vector":
            feedback_seeds = self._precompute_feedback_seeds(
                layer_records=layer_records,
                delta=delta,
                delta_scalar=delta_scalar,
            )

        return _BroadcastState(
            mode=bmode,
            transported_errors=transported_errors,
            feedback_seeds=feedback_seeds,
        )

    def _compute_local_broadcast_error(
        self,
        *,
        rec: dict[str, Any],
        layer_idx: int,
        out_features: int,
        v_n: torch.Tensor,
        delta: torch.Tensor,
        delta_scalar: torch.Tensor,
        broadcast_state: _BroadcastState,
    ) -> torch.Tensor:
        # Error term per compartment.
        # Modes:
        # - 'scalar': broadcast per-sample scalar to all somas in layer (default)
        # - 'per_soma': map output-dim error to soma-dim if sizes match, else scalar
        # - 'local_mismatch': modulate scalar error by parent-vs-soma mismatch
        bmode = broadcast_state.mode
        if bmode == "per_soma" and delta.dim() == 2 and delta.size(1) == out_features:
            e_n = delta  # [B, out]
        elif bmode == "low_rank":
            e_n = self._compute_low_rank_broadcast(
                delta=delta,
                layer_idx=layer_idx,
                out_features=out_features,
                device=v_n.device,
                dtype=v_n.dtype,
            )
        elif bmode == "pathway_vector":
            e_n = self._compute_pathway_vector_broadcast(
                seed_error=(
                    broadcast_state.feedback_seeds[layer_idx]
                    if layer_idx < len(broadcast_state.feedback_seeds)
                    else None
                ),
                delta=delta,
                delta_scalar=delta_scalar,
                rec=rec,
                v_n=v_n,
            )
        elif bmode == "path_transport":
            transported = (
                broadcast_state.transported_errors[layer_idx]
                if layer_idx < len(broadcast_state.transported_errors)
                else None
            )
            if isinstance(transported, torch.Tensor):
                e_n = transported.to(device=v_n.device, dtype=v_n.dtype)
            else:
                e_n = delta_scalar.expand(-1, out_features)
        elif bmode == "local_mismatch":
            parent = rec.get("blk_out")
            if parent is None:
                parent = rec.get("exc_out")
            if parent is None:
                parent = rec.get("inh_out")
            e_n = self._compute_local_mismatch_broadcast(
                delta_scalar=delta_scalar,
                v_n=v_n,
                parent=parent,
            )
        else:
            e_n = delta_scalar.expand(-1, out_features)  # [B, out]

        noise_sigma = getattr(self.local_cfg, "error_noise_sigma", 0.0)
        if noise_sigma > 0.0:
            e_n = e_n + noise_sigma * torch.randn_like(e_n)

        bandwidth = getattr(self.local_cfg, "broadcast_bandwidth", "full")
        if bandwidth != "full":
            e_n = self._apply_broadcast_bandwidth(e_n, bandwidth)
        return e_n

    def _compute_local_input_resistance(
        self,
        *,
        rec: dict[str, Any],
        v_n: torch.Tensor,
        layer_dynamics_mode: str,
    ) -> torch.Tensor | float:
        if layer_dynamics_mode != "conductance":
            # Additive dendritic integration does not include divisive input resistance.
            return 1.0
        if not self.local_cfg.three_factor.use_conductance_scaling:
            return 1.0
        g_tot = self._compute_layer_total_conductance(rec, v_n)
        return 1.0 / (g_tot + 1e-8)

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
        role_synaptic_alignment: Optional[torch.Tensor] = None
        role_block_alignment: Optional[torch.Tensor] = None
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

    def _apply_layer_local_gradients(
        self,
        *,
        rec: dict[str, Any],
        layer_idx: int,
        num_layers: int,
        y_target: Optional[torch.Tensor],
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
        role_synaptic_alignment: Optional[torch.Tensor],
    ) -> Optional[TopKLinear]:
        layer: Optional[TopKLinear] = rec.get(module_key)
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
        role_synaptic_alignment: Optional[torch.Tensor],
    ) -> Optional[TopKLinear]:
        layer: Optional[TopKLinear] = rec.get(module_key)
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
        role_block_alignment: Optional[torch.Tensor],
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

    def _apply_hsic_auxiliary_gradients(
        self,
        *,
        rec: dict[str, Any],
        exc_layer: Optional[TopKLinear],
        layer_idx: int,
        num_layers: int,
        y_target: Optional[torch.Tensor],
        batch_size: int,
    ) -> None:
        if not self.local_cfg.hsic.enabled:
            return

        v_out = rec.get("v_out")
        if v_out is None:
            return

        z = v_out - v_out.mean(dim=0, keepdim=True)
        hsic_warm = 1.0
        if (
            self.local_cfg.hsic.warmup_epochs > 0
            and self.epoch_counter < self.local_cfg.hsic.warmup_epochs
        ):
            hsic_warm = max(
                0.0,
                float(self.epoch_counter)
                / float(max(1, self.local_cfg.hsic.warmup_epochs)),
            )

        if self.local_cfg.hsic.apply_last_layer_only:
            is_last = layer_idx == num_layers - 1
            if not is_last:
                hsic_warm = 0.0

        if self.local_cfg.hsic.self_weight != 0.0:
            grad_z_self = self._compute_hsic_gradient(
                z, z, self.local_cfg.hsic.self_weight * hsic_warm, "self"
            )
            if self.local_cfg.hsic.grad_clip_value > 0:
                grad_z_self = torch.clamp(
                    grad_z_self,
                    min=-self.local_cfg.hsic.grad_clip_value,
                    max=self.local_cfg.hsic.grad_clip_value,
                )
            if exc_layer is not None and rec.get("x_exc") is not None:
                x_exc = rec["x_exc"]
                grad_g_self = grad_z_self.t() @ x_exc
                if self.local_cfg.normalize_by_batch:
                    grad_g_self = grad_g_self / float(batch_size)
                g_curr = exc_layer.weight().detach()
                grad_param_self = grad_g_self * g_curr
                self._add_topk_grad(exc_layer, grad_param_self, rec.get("exc_mask"))

        if self.local_cfg.hsic.target_weight != 0.0 and y_target is not None:
            y_centered = y_target - y_target.mean(dim=0, keepdim=True)
            grad_z_tgt = self._compute_hsic_gradient(
                z,
                y_centered,
                self.local_cfg.hsic.target_weight * hsic_warm,
                "target",
            )
            if self.local_cfg.hsic.grad_clip_value > 0:
                grad_z_tgt = torch.clamp(
                    grad_z_tgt,
                    min=-self.local_cfg.hsic.grad_clip_value,
                    max=self.local_cfg.hsic.grad_clip_value,
                )
            if exc_layer is not None and rec.get("x_exc") is not None:
                x_exc = rec["x_exc"]
                grad_g_tgt = grad_z_tgt.t() @ x_exc
                if self.local_cfg.normalize_by_batch:
                    grad_g_tgt = grad_g_tgt / float(batch_size)
                g_curr = exc_layer.weight().detach()
                grad_param_tgt = grad_g_tgt * g_curr
                self._add_topk_grad(exc_layer, grad_param_tgt, rec.get("exc_mask"))

    # ----------------------- Helper: HSIC kernel functions ------------------

    # ----------------------- Helper: additive gain modes --------------------

    # ----------------------- Helper: rho and phi factors --------------------

    # -------------------- Morphology-Aware Helper Methods --------------------

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ):
        # Standard training loop but with local rule gradient computation
        has_ei = hasattr(model, "core_network") and isinstance(
            model.core_network, ExcitationInhibitionNetwork
        )
        has_branches = any(isinstance(m, DendriticBranchLayer) for m in model.modules())
        if not (has_ei or has_branches):
            logger.info(
                "No EI/dendritic branch layers detected; falling back to standard training."
            )
            return super().train(model, train_data, valid_data)

        train_loader, valid_loader = self._dataloaders(train_data, valid_data)
        self._initialize_attributes(model, self.epochs)
        self._save_checkpoint(model)

        # Use the standard loop to ensure epoch counters and early stopping work
        train_losses: list[float] = []
        valid_losses: list[float] = []
        train_losses_base: list[float] = []
        train_losses_reg: list[float] = []
        train_losses, valid_losses, train_losses_base, train_losses_reg = (
            self._run_training_loop(
                n_epochs=self.epochs,
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                train_losses=train_losses,
                valid_losses=valid_losses,
                train_losses_base=train_losses_base,
                train_losses_reg=train_losses_reg,
                logger_info=self.logger_info_prefix,
                wandb_info={},
            )
        )

        if self.load_best_state_dict:
            model.load_state_dict(self.best_state_dict)

        # Log best model information like standard training
        logger.info(f"Best epoch: {self.best_epoch}, Best loss: {self.best_loss:.4f}")
        logger.info("Final model loaded from best checkpoint")

        results = {
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "best_state_dict": self.best_state_dict,
            "train_losses": train_losses,
            "valid_losses": valid_losses,
        }
        self._loss_plotting(train_losses, valid_losses)
        return results


__all__ = ["LocalCreditAssignment"]
