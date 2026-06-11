"""
Local learning training strategies.

This module implements multi-factor local learning rules for dendritic networks.

It provides biologically-inspired 3-/4-/5-factor Hebbian-like rules that use
locally available signals (pre-synaptic input, compartment voltages) modulated
by a broadcast soma error and optional morphology/information factors.
"""

import logging
import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as functional

from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    DendriticBranchLayer,
    ExcitationInhibitionNetwork,
    ParametricActivation,
    PathwayRouter,
)
from dendritic_modeling.networks.activations.parametric import (
    ParametricTanh,
    ParametricTanhOnlyM,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.training.strategies.local_learning_config import (
    build_local_rule_config,
    coerce_legacy_local_rule_config,
    section_to_dict,
)
from dendritic_modeling.training.strategies.local_learning_state import _LayerStats
from dendritic_modeling.training.strategies.local_learning_stdp import (
    stdp_activity,
    stdp_decay,
    stdp_enabled,
    stdp_pathway_enabled,
)
from dendritic_modeling.training.strategies.standard import Trainer

logger = logging.getLogger(__name__)


class LocalCreditAssignment(Trainer):
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

    # -------------------------- Helper: loss/error --------------------------
    def _compute_logging_loss(
        self, model: BaseModel, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Compute scalar loss for logging using configured loss function.

        Uses the existing loss_function for compatibility with reporting.
        """
        return self.loss_function(model, x, y)

    def _compute_soma_error(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute output-space error signal: dL/dy_hat per sample.

        Supports common losses. Falls back to MSE-like residual if unknown.
        Returns tensor of shape [batch, output_dim].
        """
        mode = self.local_cfg.error_mode.lower()

        # If explicit mode is selected
        if mode in {"mse", "bce", "ce"}:
            return self._error_by_mode(y_hat, y, mode)

        # Auto-detect based on loss name or shapes
        loss_name = getattr(self.loss_function, "_loss_name", "").lower()
        if (
            "cross entropy" in loss_name
            or "negative log likelihood" in loss_name
            or "nll" in loss_name
            or "categorical" in loss_name
        ):
            return self._error_by_mode(y_hat, y, "ce")
        if "binary cross entropy" in loss_name:
            return self._error_by_mode(y_hat, y, "bce")
        # Heuristic: if y are class indices (Long) and y_hat are logits
        if (
            y_hat.dim() >= 2
            and (y.dim() == 1 or (y.dim() == 2 and y.size(1) == 1))
            and y.dtype in (torch.long, torch.int64)
        ):
            return self._error_by_mode(y_hat, y, "ce")
        # Default: MSE-like
        return self._error_by_mode(y_hat, y, "mse")

    def _resolve_local_soma_signals(
        self,
        model: BaseModel,
        y_hat: torch.Tensor,
        delta_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve soma-space signals for local rules.

        Local dendritic rules ideally use error at the core soma output (pre-decoder).
        When the decoder input is cached with graph information intact, we can map
        output-space error back to core space through the exact decoder Jacobian:

            delta_core = J_decoder(h_core)^T delta_out

        This keeps per-soma and path-transport modes meaningful even when the
        decoder is nonlinear. If that mapping is unavailable, we fall back to the
        legacy output-space behavior for backward compatibility.
        """
        # Backward-compatible fallback
        v0_local = y_hat.detach()
        delta_local = delta_out.detach()

        if not hasattr(model, "decoder_network"):
            return v0_local, delta_local

        decoder_input = self._decoder_cache.get("decoder_input")
        if isinstance(decoder_input, torch.Tensor):
            try:
                delta_core = torch.autograd.grad(
                    outputs=y_hat,
                    inputs=decoder_input,
                    grad_outputs=delta_out,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
            except RuntimeError:
                delta_core = None
            if isinstance(delta_core, torch.Tensor):
                return decoder_input.detach(), delta_core.detach()

        try:
            linear_layers = [
                module
                for module in model.decoder_network.modules()
                if isinstance(module, nn.Linear)
            ]
        except Exception:
            linear_layers = []

        # We only apply explicit mapping when decoder is a single linear readout.
        if len(linear_layers) != 1:
            if not self._warned_decoder_soma_fallback:
                logger.info(
                    "Local soma-error mapping fallback: decoder is not a single linear layer."
                )
                self._warned_decoder_soma_fallback = True
            return v0_local, delta_local

        lin_mod = self._decoder_cache.get("module")
        h_in = self._decoder_cache.get("input")
        if not (isinstance(lin_mod, nn.Linear) and isinstance(h_in, torch.Tensor)):
            if not self._warned_decoder_soma_fallback:
                logger.info(
                    "Local soma-error mapping fallback: missing cached decoder input."
                )
                self._warned_decoder_soma_fallback = True
            return v0_local, delta_local

        # Ensure dimensions are aligned before mapping.
        if (
            delta_out.dim() != 2
            or h_in.dim() != 2
            or delta_out.size(0) != h_in.size(0)
            or delta_out.size(1) != lin_mod.weight.size(0)
            or h_in.size(1) != lin_mod.weight.size(1)
        ):
            if not self._warned_decoder_soma_fallback:
                logger.info(
                    "Local soma-error mapping fallback: incompatible decoder shapes."
                )
                self._warned_decoder_soma_fallback = True
            return v0_local, delta_local

        delta_core = delta_out @ lin_mod.weight.detach()
        return h_in.detach(), delta_core.detach()

    @staticmethod
    def _reduce_error_to_scalar(delta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Reduce output error to a per-sample scalar for broadcast modes.

        For CE-like errors, per-sample mean can be near zero due to probability
        simplex constraints. In that case we fall back to the max-magnitude
        component to keep a meaningful signed modulatory signal.
        """
        if delta.dim() == 1:
            return delta.unsqueeze(-1)
        if delta.dim() != 2:
            delta = delta.view(delta.size(0), -1)

        delta_scalar = delta.mean(dim=1, keepdim=True)
        near_zero = delta_scalar.abs() <= eps
        if near_zero.any():
            max_idx = delta.abs().argmax(dim=1, keepdim=True)
            fallback = delta.gather(dim=1, index=max_idx)
            delta_scalar = torch.where(near_zero, fallback, delta_scalar)
        return delta_scalar

    @staticmethod
    def _compute_local_mismatch_broadcast(
        delta_scalar: torch.Tensor,
        v_n: torch.Tensor,
        parent: Optional[torch.Tensor],
        residual_fraction: float = 0.2,
        clip_value: float = 3.0,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Compute a stabilized local-mismatch broadcast signal.

        The raw local mismatch signal can collapse when parent and soma voltages are
        close or poorly scaled. We stabilize it by:
        1) centering mismatch across batch,
        2) normalizing mismatch per sample by RMS magnitude,
        3) clipping outliers,
        4) blending with a residual scalar broadcast path.
        """
        base = delta_scalar.expand(-1, v_n.size(1))
        if parent is None:
            return base

        p = parent
        if p.size() != v_n.size():
            try:
                p = p.view_as(v_n)
            except Exception:
                return base

        mismatch = p - v_n
        mismatch = mismatch - mismatch.mean(dim=0, keepdim=True)
        rms = mismatch.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(eps)
        mismatch_norm = mismatch / rms

        if clip_value > 0.0:
            mismatch_norm = torch.clamp(mismatch_norm, -clip_value, clip_value)

        modulated = base * mismatch_norm
        blend = float(max(0.0, min(1.0, residual_fraction)))
        return (1.0 - blend) * modulated + blend * base

    @staticmethod
    def _weight_transform_derivative(
        raw_param: torch.Tensor, transform_type: str
    ) -> torch.Tensor:
        """Derivative of transformed conductance w.r.t. raw parameter."""
        t = (transform_type or "exp").lower()
        if t == "exp":
            return raw_param.exp()
        if t == "relu":
            return (raw_param > 0).to(raw_param.dtype)
        if t == "softplus":
            return torch.sigmoid(raw_param)
        if t == "identity":
            return torch.ones_like(raw_param)
        logger.warning(
            f"Unknown weight transform '{transform_type}', falling back to exp derivative."
        )
        return raw_param.exp()

    def _stdp_enabled(self) -> bool:
        """Return whether trace-based STDP updates should be applied."""
        return stdp_enabled(self.local_cfg)

    def _stdp_pathway_enabled(self, pathway: str) -> bool:
        """Return whether STDP is configured for a recorded synaptic pathway."""
        return stdp_pathway_enabled(self.local_cfg, pathway)

    def _stdp_activity(
        self, values: torch.Tensor, threshold: float, mode: str
    ) -> torch.Tensor:
        """Convert raw local values to non-backpropagating STDP activities."""
        return stdp_activity(values, threshold, mode)

    @staticmethod
    def _stdp_decay(tau: float) -> float:
        """Single-step exponential decay for an STDP eligibility trace."""
        return stdp_decay(tau)

    def _add_topk_grad(
        self,
        layer: TopKLinear,
        grad_param: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> None:
        """Accumulate a raw-parameter gradient while respecting connectivity masks."""
        if hasattr(layer, "_connection_mask_like"):
            grad_param = grad_param * layer._connection_mask_like(grad_param)
        if mask is not None and not self.local_cfg.update_inactive_weights:
            grad_param = grad_param * mask.to(
                device=grad_param.device, dtype=grad_param.dtype
            )
        if layer.pre_w.grad is None:
            layer.pre_w.grad = grad_param.to(layer.pre_w.dtype)
        else:
            layer.pre_w.grad = layer.pre_w.grad + grad_param.to(layer.pre_w.dtype)

    def _apply_stdp_gradient(
        self,
        layer: TopKLinear,
        pre: torch.Tensor,
        post: torch.Tensor,
        mask: torch.Tensor | None,
        pathway: str,
        error_signal: torch.Tensor | None = None,
    ) -> None:
        """Apply one trace-based STDP gradient to a TopK synapse module.

        Positive STDP potentiation is represented as a negative optimizer
        gradient because PyTorch optimizers perform ``param -= lr * grad``.
        """
        if not self._stdp_pathway_enabled(pathway):
            return

        stdp_cfg = self.local_cfg.stdp
        if not hasattr(self, "_stdp_traces"):
            self._stdp_traces = {}
        pre_activity = self._stdp_activity(
            pre,
            threshold=float(getattr(stdp_cfg, "pre_threshold", 0.0)),
            mode=str(getattr(stdp_cfg, "activity_mode", "relu")),
        )
        post_activity = self._stdp_activity(
            post,
            threshold=float(getattr(stdp_cfg, "post_threshold", 0.0)),
            mode=str(getattr(stdp_cfg, "activity_mode", "relu")),
        )
        if (
            pre_activity.size(1) != layer.in_features
            or post_activity.size(1) != layer.out_features
        ):
            logger.debug(
                "Skipping STDP for %s: activity shape %s -> %s does not match %s -> %s.",
                pathway,
                tuple(pre_activity.shape),
                tuple(post_activity.shape),
                layer.in_features,
                layer.out_features,
            )
            return

        trace_state = self._stdp_traces.setdefault(id(layer), {})
        prev_pre = trace_state.get("pre")
        prev_post = trace_state.get("post")
        if (
            prev_pre is None
            or prev_pre.shape != (layer.in_features,)
            or prev_pre.device != pre_activity.device
            or prev_pre.dtype != pre_activity.dtype
        ):
            prev_pre = torch.zeros(
                layer.in_features, device=pre_activity.device, dtype=pre_activity.dtype
            )
        if (
            prev_post is None
            or prev_post.shape != (layer.out_features,)
            or prev_post.device != post_activity.device
            or prev_post.dtype != post_activity.dtype
        ):
            prev_post = torch.zeros(
                layer.out_features,
                device=post_activity.device,
                dtype=post_activity.dtype,
            )

        pre_batch = pre_activity.mean(dim=0)
        post_batch = post_activity.mean(dim=0)

        potentiation = post_batch[:, None] * prev_pre[None, :]
        depression = prev_post[:, None] * pre_batch[None, :]
        delta_w = (
            float(getattr(stdp_cfg, "a_plus", 1.0)) * potentiation
            - float(getattr(stdp_cfg, "a_minus", 0.5)) * depression
        )

        if pathway in {"inh", "rec_inh"}:
            delta_w = delta_w * float(getattr(stdp_cfg, "inhibitory_update_sign", -1.0))

        if bool(getattr(stdp_cfg, "use_error_modulation", False)):
            if isinstance(error_signal, torch.Tensor):
                mode = str(
                    getattr(stdp_cfg, "error_modulation_mode", "scalar_abs")
                ).lower()
                if mode == "scalar_signed":
                    error_scale = error_signal.detach().mean()
                else:
                    error_scale = error_signal.detach().abs().mean()
                delta_w = delta_w * error_scale.to(
                    device=delta_w.device, dtype=delta_w.dtype
                )

        clamp_update = float(getattr(stdp_cfg, "clamp_update", 0.0))
        if clamp_update > 0.0:
            delta_w = torch.clamp(delta_w, -clamp_update, clamp_update)

        chain = self._weight_transform_derivative(
            layer.pre_w.detach(),
            getattr(layer, "weight_transform", "exp"),
        )
        grad_param = -float(getattr(stdp_cfg, "learning_rate_scale", 1.0)) * delta_w
        grad_param = grad_param * chain
        self._add_topk_grad(layer, grad_param, mask)

        pre_decay = self._stdp_decay(float(getattr(stdp_cfg, "tau_pre", 20.0)))
        post_decay = self._stdp_decay(float(getattr(stdp_cfg, "tau_post", 20.0)))
        next_pre = pre_decay * prev_pre + (1.0 - pre_decay) * pre_batch
        next_post = post_decay * prev_post + (1.0 - post_decay) * post_batch
        if bool(getattr(stdp_cfg, "detach_traces", True)):
            next_pre = next_pre.detach()
            next_post = next_post.detach()
        trace_state["pre"] = next_pre
        trace_state["post"] = next_post

    def _apply_stdp_pathways(
        self,
        rec: dict[str, Any],
        post_signal: torch.Tensor,
        error_signal: torch.Tensor | None = None,
    ) -> None:
        """Apply configured STDP updates to all recorded TopK pathways."""
        pathway_keys = {
            "exc": ("exc_module", "x_exc", "exc_mask"),
            "inh": ("inh_module", "x_inh", "inh_mask"),
            "rec_exc": ("rec_exc_module", "x_rec_exc", "rec_exc_mask"),
            "rec_inh": ("rec_inh_module", "x_rec_inh", "rec_inh_mask"),
        }
        for pathway, (layer_key, input_key, mask_key) in pathway_keys.items():
            layer = rec.get(layer_key)
            pre = rec.get(input_key)
            if isinstance(layer, TopKLinear) and isinstance(pre, torch.Tensor):
                self._apply_stdp_gradient(
                    layer=layer,
                    pre=pre,
                    post=post_signal,
                    mask=rec.get(mask_key),
                    pathway=pathway,
                    error_signal=error_signal,
                )

    def _compute_reactivation_derivative(
        self,
        react_module: Optional[nn.Module],
        v_n: torch.Tensor,
        v_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the elementwise local slope d a / d V for a branch reactivation."""
        if react_module is None or isinstance(react_module, nn.Identity):
            return torch.ones_like(v_n)

        if isinstance(react_module, nn.ReLU):
            return (v_n > 0).to(device=v_n.device, dtype=v_n.dtype)

        if isinstance(react_module, nn.Sigmoid):
            y = (
                v_out.to(device=v_n.device, dtype=v_n.dtype)
                if isinstance(v_out, torch.Tensor)
                else torch.sigmoid(v_n)
            )
            return y * (1.0 - y)

        if isinstance(react_module, nn.Tanh):
            y = (
                v_out.to(device=v_n.device, dtype=v_n.dtype)
                if isinstance(v_out, torch.Tensor)
                else torch.tanh(v_n)
            )
            return 1.0 - y.square()

        if isinstance(react_module, ParametricTanh):
            y = (
                v_out.to(device=v_n.device, dtype=v_n.dtype)
                if isinstance(v_out, torch.Tensor)
                else react_module(v_n)
            )
            m = react_module.log_m.detach().exp().to(device=v_n.device, dtype=v_n.dtype)
            return 0.5 * m.unsqueeze(0) * (1.0 - (2.0 * y - 1.0).square())

        if isinstance(react_module, ParametricTanhOnlyM):
            y = (
                v_out.to(device=v_n.device, dtype=v_n.dtype)
                if isinstance(v_out, torch.Tensor)
                else react_module(v_n)
            )
            m = react_module.log_m.detach().exp().to(device=v_n.device, dtype=v_n.dtype)
            return 0.5 * m.unsqueeze(0) * (1.0 - (2.0 * y - 1.0).square())

        try:
            with torch.enable_grad():
                x = v_n.detach().clone().requires_grad_(True)
                y = react_module(x)
                grad = torch.autograd.grad(
                    y,
                    x,
                    grad_outputs=torch.ones_like(y),
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )[0]
            return grad.detach().to(device=v_n.device, dtype=v_n.dtype)
        except Exception:
            logger.warning(
                "Falling back to unit reactivation derivative for module %s.",
                react_module.__class__.__name__,
            )
            return torch.ones_like(v_n)

    def _get_layer_activation_derivative(
        self,
        rec: dict[str, Any],
        v_n: Optional[torch.Tensor] = None,
        v_out: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        cached = rec.get("_activation_derivative")
        if isinstance(cached, torch.Tensor):
            return cached
        if v_n is None:
            v_n = rec.get("v_n")
        if not isinstance(v_n, torch.Tensor):
            return None
        if v_out is None:
            v_out = rec.get("v_out")
        react_module = getattr(rec.get("layer"), "reactivation", None)
        deriv = self._compute_reactivation_derivative(
            react_module=react_module, v_n=v_n, v_out=v_out
        )
        rec["_activation_derivative"] = deriv
        return deriv

    def _apply_broadcast_bandwidth(
        self, e_n: torch.Tensor, bandwidth: str
    ) -> torch.Tensor:
        """Apply bandwidth reduction to the broadcast error signal.

        Simulates biologically plausible low-bandwidth feedback channels.

        Args:
            e_n: Error signal tensor [B, out_features].
            bandwidth: One of "sign_only", "quantized", "sparse_topk".

        Returns:
            Bandwidth-reduced error signal (same shape as input).
        """
        if bandwidth == "sign_only":
            # Only transmit the sign of the error (1-bit)
            magnitude = e_n.abs().mean()
            return torch.sign(e_n) * magnitude

        elif bandwidth == "quantized":
            # Uniform quantization to N bits
            bits = getattr(self.local_cfg, "broadcast_bits", 8)
            n_levels = 2**bits
            e_min = e_n.min()
            e_max = e_n.max()
            e_range = e_max - e_min + 1e-8
            # Quantize to [0, n_levels-1] then rescale back
            quantized = torch.round((e_n - e_min) / e_range * (n_levels - 1))
            return quantized / (n_levels - 1) * e_range + e_min

        elif bandwidth == "sparse_topk":
            # Only broadcast top-k% of neurons (by magnitude), zero the rest
            frac = getattr(self.local_cfg, "broadcast_topk_fraction", 0.3)
            k = max(1, int(e_n.shape[-1] * frac))
            # Per-sample top-k
            _, topk_idx = torch.topk(e_n.abs(), k, dim=-1)
            mask = torch.zeros_like(e_n)
            mask.scatter_(-1, topk_idx, 1.0)
            return e_n * mask

        else:
            return e_n

    @staticmethod
    def _compute_layer_total_conductance(
        rec: dict[str, Any], v_n: torch.Tensor
    ) -> torch.Tensor:
        """Compute total conductance used by conductance-aware local rules."""
        g_tot = torch.ones_like(v_n)

        exc_out = rec.get("exc_out")
        if exc_out is not None:
            g_tot = g_tot + functional.relu(exc_out)

        inh_out = rec.get("inh_out")
        if inh_out is not None:
            g_tot = g_tot + functional.relu(inh_out)

        blk_module = rec.get("blk_module")
        if blk_module is not None and hasattr(blk_module, "sum_conductances"):
            g_blk = blk_module.sum_conductances().detach()[None, :].expand_as(v_n)
            g_tot = g_tot + functional.relu(g_blk)

        return g_tot

    @staticmethod
    def _expand_parent_signal_to_children(
        signal: torch.Tensor, block_size: int, child_out_features: int
    ) -> torch.Tensor:
        """Broadcast a parent-layer signal to its grouped child branches."""
        expanded = signal.repeat_interleave(block_size, dim=1)
        if expanded.size(1) == child_out_features:
            return expanded

        if expanded.size(1) > child_out_features:
            return expanded[:, :child_out_features]

        repeats = math.ceil(float(child_out_features) / float(max(1, expanded.size(1))))
        tiled = expanded.repeat(1, repeats)
        return tiled[:, :child_out_features]

    @staticmethod
    def _normalize_role_mass(
        role_mass: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        """Normalize non-negative role mass into a role profile."""
        denom = role_mass.sum(dim=-1, keepdim=True).clamp_min(eps)
        return role_mass / denom

    @staticmethod
    def _compute_role_selectivity(
        role_profile: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        """Measure pathway selectivity from normalized role profiles."""
        n_roles = role_profile.size(-1)
        if n_roles <= 1:
            return torch.ones(
                role_profile.size(0),
                device=role_profile.device,
                dtype=role_profile.dtype,
            )
        probs = role_profile.clamp_min(eps)
        entropy = -(probs * probs.log()).sum(dim=-1)
        max_entropy = math.log(float(n_roles))
        return 1.0 - entropy / max(max_entropy, eps)

    @staticmethod
    def _resolve_recent_role_profile(
        input_dim: int,
        base_input_roles: Optional[torch.Tensor],
        prior_output_roles: list[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Resolve the most plausible role basis for a layer input size."""
        if (
            isinstance(base_input_roles, torch.Tensor)
            and base_input_roles.dim() == 2
            and base_input_roles.size(0) == input_dim
        ):
            return base_input_roles

        for role_profile in reversed(prior_output_roles):
            if role_profile.dim() == 2 and role_profile.size(0) == input_dim:
                return role_profile
        return None

    @staticmethod
    def _effective_topk_weight(
        layer: TopKLinear, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Return masked effective synaptic weights."""
        eff_weight = layer.weight().detach()
        if mask is None:
            mask = layer.weight_mask().detach()
        return eff_weight * mask.to(device=eff_weight.device, dtype=eff_weight.dtype)

    @staticmethod
    def _role_mass_from_weight_matrix(
        weight_matrix: torch.Tensor,
        input_roles: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Aggregate pathway-role mass through a synaptic weight matrix."""
        if input_roles is None or input_roles.dim() != 2:
            return None
        if weight_matrix.dim() != 2 or weight_matrix.size(1) != input_roles.size(0):
            return None
        roles = input_roles.to(device=weight_matrix.device, dtype=weight_matrix.dtype)
        return weight_matrix.abs() @ roles

    def _resolve_router_output_roles(self, model: BaseModel) -> Optional[torch.Tensor]:
        """Return pathway ownership for encoder outputs when available."""
        source = str(
            getattr(
                self.local_cfg.morphology_aware, "branch_role_source", "router_pathways"
            )
        ).lower()
        if source != "router_pathways":
            logger.warning(
                "Unknown branch_role_source='%s'; disabling pathway-role inference.",
                source,
            )
            return None

        encoder = getattr(model, "encoder_network", None)
        if not isinstance(encoder, PathwayRouter):
            return None
        return encoder.get_output_role_matrix().detach()

    def _compute_pathway_activity(
        self,
        rec: dict[str, Any],
        stats: _LayerStats,
        v_n: torch.Tensor,
        eps: float = 1e-8,
    ) -> Optional[torch.Tensor]:
        """Compute a sample-wise local pathway activity gate."""
        candidates: list[torch.Tensor] = []

        x_exc = rec.get("x_exc")
        if isinstance(x_exc, torch.Tensor) and isinstance(
            stats.input_role_profile, torch.Tensor
        ):
            if x_exc.size(1) == stats.input_role_profile.size(0):
                act_exc = x_exc.abs() @ stats.input_role_profile.to(
                    device=x_exc.device, dtype=x_exc.dtype
                )
                candidates.append(act_exc)

        x_blk_raw = rec.get("x_blk_raw")
        if (
            isinstance(x_blk_raw, torch.Tensor)
            and isinstance(stats.block_input_role_profile, torch.Tensor)
            and x_blk_raw.size(1) == stats.block_input_role_profile.size(0)
        ):
            act_blk = x_blk_raw.abs() @ stats.block_input_role_profile.to(
                device=x_blk_raw.device, dtype=x_blk_raw.dtype
            )
            candidates.append(act_blk)

        if isinstance(stats.branch_role_profile, torch.Tensor):
            denom = stats.branch_role_profile.sum(dim=0, keepdim=True).clamp_min(eps)
            act_v = functional.relu(v_n) @ stats.branch_role_profile.to(
                device=v_n.device, dtype=v_n.dtype
            )
            candidates.append(act_v / denom.to(device=v_n.device, dtype=v_n.dtype))

        if not candidates:
            return None

        normalized: list[torch.Tensor] = []
        for candidate in candidates:
            denom = candidate.mean(dim=1, keepdim=True).clamp_min(eps)
            normalized.append(candidate / denom)
        return torch.stack(normalized, dim=0).mean(dim=0)

    def _compute_pathway_vector_broadcast(
        self,
        seed_error: Optional[torch.Tensor],
        delta: torch.Tensor,
        delta_scalar: torch.Tensor,
        rec: dict[str, Any],
        v_n: torch.Tensor,
    ) -> torch.Tensor:
        """Project soma error into a pathway-aligned apical feedback vector."""
        stats = self._layer_stats.get(id(rec.get("layer")))
        if stats is None or not isinstance(stats.branch_role_profile, torch.Tensor):
            return delta_scalar.expand(-1, v_n.size(1))

        role_profile = stats.branch_role_profile.to(device=v_n.device, dtype=v_n.dtype)
        if role_profile.dim() != 2 or role_profile.size(0) != v_n.size(1):
            return delta_scalar.expand(-1, v_n.size(1))

        role_mass = role_profile.sum(dim=0, keepdim=True).clamp_min(1e-6)
        seed = seed_error
        if isinstance(seed, torch.Tensor):
            if seed.dim() == 1:
                seed = seed.unsqueeze(-1)
            elif seed.dim() != 2:
                seed = seed.view(seed.size(0), -1)

        if isinstance(seed, torch.Tensor) and seed.size(1) == v_n.size(1):
            seed = seed.to(device=v_n.device, dtype=v_n.dtype)
            pathway_error = seed @ role_profile
            pathway_error = pathway_error / role_mass
            base = seed
        else:
            pathway_error = delta_scalar.to(device=v_n.device, dtype=v_n.dtype).expand(
                -1, role_profile.size(1)
            )
            base = delta_scalar.to(device=v_n.device, dtype=v_n.dtype).expand(
                -1, v_n.size(1)
            )

        pathway_activity = self._compute_pathway_activity(rec, stats, v_n)
        gate_strength = float(
            getattr(self.local_cfg, "pathway_activity_gate_strength", 1.0)
        )
        if pathway_activity is not None and gate_strength > 0.0:
            pathway_activity = pathway_activity.to(device=v_n.device, dtype=v_n.dtype)
            pathway_gate = 1.0 + gate_strength * (pathway_activity - 1.0)
            pathway_error = pathway_error * pathway_gate

        projected = pathway_error @ role_profile.t()
        residual = float(
            max(
                0.0,
                min(1.0, getattr(self.local_cfg, "pathway_broadcast_residual", 0.25)),
            )
        )
        return (1.0 - residual) * projected + residual * base

    def _get_low_rank_broadcast_factors(
        self,
        layer_idx: int,
        in_features: int,
        out_features: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached low-rank random broadcast factors for a layer."""
        rank = max(
            1,
            min(
                int(getattr(self.local_cfg, "broadcast_rank", 4)),
                int(in_features),
                int(out_features),
            ),
        )
        cache_key = (
            "low_rank_broadcast",
            layer_idx,
            in_features,
            out_features,
            rank,
            str(device),
            str(dtype),
        )
        cached = self._broadcast_cache.get(cache_key)
        if cached is not None:
            return cached

        generator = torch.Generator()
        seed = (
            15485863
            + 7919 * int(layer_idx)
            + 104729 * int(in_features)
            + 53 * int(out_features)
        )
        generator.manual_seed(seed)

        left = torch.randn(in_features, rank, generator=generator, dtype=dtype)
        right = torch.randn(rank, out_features, generator=generator, dtype=dtype)

        left = left / left.norm(dim=0, keepdim=True).clamp_min(1e-6)
        right = right / right.norm(dim=1, keepdim=True).clamp_min(1e-6)

        left = left.to(device=device)
        right = right.to(device=device)
        self._broadcast_cache[cache_key] = (left, right)
        return left, right

    def _compute_low_rank_broadcast(
        self,
        delta: torch.Tensor,
        layer_idx: int,
        out_features: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Project soma error into a layer-specific vector broadcast."""
        if delta.dim() == 1:
            delta = delta.unsqueeze(-1)
        elif delta.dim() != 2:
            delta = delta.view(delta.size(0), -1)

        in_features = delta.size(1)
        if in_features <= 1:
            return delta.mean(dim=1, keepdim=True).expand(-1, out_features).to(dtype)

        left, right = self._get_low_rank_broadcast_factors(
            layer_idx=layer_idx,
            in_features=in_features,
            out_features=out_features,
            device=device,
            dtype=dtype,
        )
        scale = float(getattr(self.local_cfg, "broadcast_init_scale", 1.0)) / (
            float(left.size(1)) ** 0.5
        )
        return (delta.to(dtype=dtype, device=device) @ left @ right) * scale

    def _precompute_path_propagation_factors(
        self,
        layer_records: list[dict[str, Any]],
        include_parent_activation_derivative: bool = False,
    ) -> list[torch.Tensor | float]:
        """Approximate path attenuation from each compartment to the soma.

        DendriNet records layers from distal to proximal. The attenuation for a
        child layer therefore depends on the *next* (more proximal) layer in the
        record list, not the previous one.
        """
        if not layer_records:
            return []

        mode = str(
            getattr(self.local_cfg.morphology_aware, "path_factor_mode", "per_branch")
        ).lower()
        path_factors: list[torch.Tensor | float] = [1.0] * len(layer_records)

        soma_v = layer_records[-1].get("v_n")
        if isinstance(soma_v, torch.Tensor):
            if mode == "scalar_mean":
                path_factors[-1] = torch.ones(
                    soma_v.size(0), 1, device=soma_v.device, dtype=soma_v.dtype
                )
            else:
                path_factors[-1] = torch.ones_like(soma_v)

        for layer_idx in range(len(layer_records) - 2, -1, -1):
            child_rec = layer_records[layer_idx]
            parent_rec = layer_records[layer_idx + 1]

            child_v = child_rec.get("v_n")
            parent_v = parent_rec.get("v_n")
            blk_module = parent_rec.get("blk_module")
            if not (
                isinstance(child_v, torch.Tensor)
                and isinstance(parent_v, torch.Tensor)
                and blk_module is not None
                and hasattr(blk_module, "weight")
            ):
                path_factors[layer_idx] = (
                    torch.ones(
                        child_v.size(0), 1, device=child_v.device, dtype=child_v.dtype
                    )
                    if isinstance(child_v, torch.Tensor) and mode == "scalar_mean"
                    else (
                        torch.ones_like(child_v)
                        if isinstance(child_v, torch.Tensor)
                        else 1.0
                    )
                )
                continue

            parent_path = path_factors[layer_idx + 1]
            if not isinstance(parent_path, torch.Tensor):
                parent_path = torch.ones_like(parent_v)
            elif parent_path.size(1) == 1 and parent_v.size(1) > 1:
                parent_path = parent_path.expand(-1, parent_v.size(1))

            parent_act_deriv = torch.ones_like(parent_v)
            if include_parent_activation_derivative:
                parent_act_deriv_candidate = self._get_layer_activation_derivative(
                    parent_rec, parent_v
                )
                if isinstance(parent_act_deriv_candidate, torch.Tensor):
                    parent_act_deriv = parent_act_deriv_candidate.to(
                        device=parent_v.device, dtype=parent_v.dtype
                    )

            parent_g_tot = self._compute_layer_total_conductance(parent_rec, parent_v)
            parent_r_tot = 1.0 / (parent_g_tot + 1e-8)

            block_size = int(getattr(blk_module, "block_size", 1))
            edge_weights = (
                blk_module.weight()
                .detach()
                .to(device=child_v.device, dtype=child_v.dtype)
            )
            expected_child_out = edge_weights.numel()
            if child_v.size(1) != expected_child_out:
                logger.warning(
                    "Path propagation shape mismatch: child_out=%s, expected=%s. "
                    "Falling back to scalar averaging.",
                    child_v.size(1),
                    expected_child_out,
                )
                scalar_factor = (parent_path * parent_act_deriv * parent_r_tot).mean(
                    dim=1, keepdim=True
                ) * edge_weights.mean()
                path_factors[layer_idx] = scalar_factor
                continue

            expanded_parent_path = self._expand_parent_signal_to_children(
                parent_path, block_size, child_v.size(1)
            )
            expanded_parent_act = self._expand_parent_signal_to_children(
                parent_act_deriv, block_size, child_v.size(1)
            )
            expanded_parent_r_tot = self._expand_parent_signal_to_children(
                parent_r_tot, block_size, child_v.size(1)
            )
            edge_gain = edge_weights.reshape(1, -1)
            branch_factor = (
                expanded_parent_path
                * expanded_parent_act
                * expanded_parent_r_tot
                * edge_gain
            )
            if mode == "scalar_mean":
                branch_factor = branch_factor.mean(dim=1, keepdim=True)
            path_factors[layer_idx] = branch_factor

        return path_factors

    def _precompute_path_transport_errors(
        self,
        layer_records: list[dict[str, Any]],
        delta: torch.Tensor,
        delta_scalar: torch.Tensor,
    ) -> list[Optional[torch.Tensor]]:
        """Recursively transport soma-space activation errors through the tree.

        This is the operational form of the tree recursion:

            dL/da_child = dL/da_parent * f'_parent(V_parent) * R_parent_tot * g_child->parent

        It preserves vector-valued soma errors where dimensions remain aligned,
        rather than collapsing transport to a scalar gain. The resulting tensor
        approximates activation-space error at each layer; conductance updates
        convert it locally to voltage-space by multiplying by f'(V_n).
        """
        if not layer_records:
            return []

        transported: list[Optional[torch.Tensor]] = [None] * len(layer_records)
        soma_v = layer_records[-1].get("v_n")
        if isinstance(soma_v, torch.Tensor):
            if delta.dim() == 1:
                delta = delta.unsqueeze(-1)
            elif delta.dim() != 2:
                delta = delta.view(delta.size(0), -1)

            if delta.size(1) == soma_v.size(1):
                transported[-1] = delta.to(device=soma_v.device, dtype=soma_v.dtype)
            else:
                transported[-1] = delta_scalar.to(
                    device=soma_v.device, dtype=soma_v.dtype
                ).expand(-1, soma_v.size(1))

        for layer_idx in range(len(layer_records) - 2, -1, -1):
            child_rec = layer_records[layer_idx]
            parent_rec = layer_records[layer_idx + 1]
            child_v = child_rec.get("v_n")
            parent_v = parent_rec.get("v_n")
            parent_error = transported[layer_idx + 1]
            blk_module = parent_rec.get("blk_module")

            if not (
                isinstance(child_v, torch.Tensor)
                and isinstance(parent_v, torch.Tensor)
                and isinstance(parent_error, torch.Tensor)
                and blk_module is not None
                and hasattr(blk_module, "weight")
            ):
                if isinstance(child_v, torch.Tensor):
                    transported[layer_idx] = delta_scalar.to(
                        device=child_v.device, dtype=child_v.dtype
                    ).expand(-1, child_v.size(1))
                continue

            edge_weights = (
                blk_module.weight()
                .detach()
                .to(device=child_v.device, dtype=child_v.dtype)
            )
            expected_child_out = edge_weights.numel()
            if child_v.size(1) != expected_child_out:
                logger.warning(
                    "Path transport shape mismatch: child_out=%s, expected=%s. "
                    "Falling back to scalar averaging.",
                    child_v.size(1),
                    expected_child_out,
                )
                parent_act_deriv = self._get_layer_activation_derivative(
                    parent_rec, parent_v
                )
                if not isinstance(parent_act_deriv, torch.Tensor):
                    parent_act_deriv = torch.ones_like(parent_v)
                parent_r_tot = 1.0 / (
                    self._compute_layer_total_conductance(parent_rec, parent_v) + 1e-8
                )
                transported[layer_idx] = (
                    (parent_error * parent_act_deriv * parent_r_tot).mean(
                        dim=1, keepdim=True
                    )
                    * edge_weights.mean()
                ).expand(-1, child_v.size(1))
                continue

            block_size = int(getattr(blk_module, "block_size", 1))
            parent_act_deriv = self._get_layer_activation_derivative(
                parent_rec, parent_v
            )
            if not isinstance(parent_act_deriv, torch.Tensor):
                parent_act_deriv = torch.ones_like(parent_v)
            parent_r_tot = 1.0 / (
                self._compute_layer_total_conductance(parent_rec, parent_v) + 1e-8
            )
            expanded_parent_error = self._expand_parent_signal_to_children(
                (
                    parent_error.to(device=child_v.device, dtype=child_v.dtype)
                    * parent_act_deriv.to(device=child_v.device, dtype=child_v.dtype)
                ),
                block_size,
                child_v.size(1),
            )
            expanded_parent_r_tot = self._expand_parent_signal_to_children(
                parent_r_tot.to(device=child_v.device, dtype=child_v.dtype),
                block_size,
                child_v.size(1),
            )
            transported[layer_idx] = (
                expanded_parent_error
                * expanded_parent_r_tot
                * edge_weights.reshape(1, -1)
            )

        return transported

    def _precompute_feedback_seeds(
        self,
        layer_records: list[dict[str, Any]],
        delta: torch.Tensor,
        delta_scalar: torch.Tensor,
    ) -> list[Optional[torch.Tensor]]:
        """Precompute hierarchical feedback vectors from proximal to distal layers."""
        if not layer_records:
            return []

        feedback: list[Optional[torch.Tensor]] = [None] * len(layer_records)

        soma_v = layer_records[-1].get("v_n")
        if isinstance(soma_v, torch.Tensor):
            if delta.dim() == 1:
                delta = delta.unsqueeze(-1)
            elif delta.dim() != 2:
                delta = delta.view(delta.size(0), -1)

            if delta.size(1) == soma_v.size(1):
                feedback[-1] = delta.to(device=soma_v.device, dtype=soma_v.dtype)
            else:
                feedback[-1] = delta_scalar.to(
                    device=soma_v.device, dtype=soma_v.dtype
                ).expand(-1, soma_v.size(1))

        for layer_idx in range(len(layer_records) - 2, -1, -1):
            child_rec = layer_records[layer_idx]
            parent_rec = layer_records[layer_idx + 1]
            child_v = child_rec.get("v_n")
            parent_feedback = feedback[layer_idx + 1]
            blk_module = parent_rec.get("blk_module")
            if not (
                isinstance(child_v, torch.Tensor)
                and isinstance(parent_feedback, torch.Tensor)
                and blk_module is not None
                and hasattr(blk_module, "weight")
            ):
                if isinstance(child_v, torch.Tensor):
                    feedback[layer_idx] = delta_scalar.to(
                        device=child_v.device, dtype=child_v.dtype
                    ).expand(-1, child_v.size(1))
                continue

            edge_weights = (
                blk_module.weight()
                .detach()
                .to(device=child_v.device, dtype=child_v.dtype)
            )
            expanded_parent = self._expand_parent_signal_to_children(
                parent_feedback.to(device=child_v.device, dtype=child_v.dtype),
                int(getattr(blk_module, "block_size", 1)),
                child_v.size(1),
            )
            feedback[layer_idx] = expanded_parent * edge_weights.reshape(1, -1)

        return feedback

    def _compute_inhibitory_homeostasis_factor(
        self,
        rec: dict[str, Any],
        R_tot: torch.Tensor | float,
        v_n: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Compute an auxiliary local inhibitory-homeostasis gradient factor."""
        cfg = getattr(self.local_cfg, "inhibitory_homeostasis", None)
        if cfg is None or not getattr(cfg, "enabled", False) or cfg.weight <= 0.0:
            return None

        layer = rec.get("layer")
        if layer is not None and not getattr(layer, "use_shunting", True):
            return None

        mode = str(getattr(cfg, "mode", "r_tot")).lower()
        weight = float(cfg.weight)

        if mode == "r_tot":
            if not isinstance(R_tot, torch.Tensor):
                return None
            target_r = float(getattr(cfg, "target_r_tot", 0.35))
            return -weight * (R_tot - target_r) * (R_tot**2)

        if mode == "voltage":
            r_tensor = (
                R_tot
                if isinstance(R_tot, torch.Tensor)
                else torch.ones_like(v_n) * float(R_tot)
            )
            target_v = float(getattr(cfg, "target_voltage", 0.15))
            e_inh = float(getattr(self.local_cfg.three_factor, "e_rev_inh", 0.0))
            return weight * (v_n - target_v) * r_tensor * (e_inh - v_n)

        logger.warning(
            "Unknown inhibitory_homeostasis.mode='%s'; disabling auxiliary term.",
            mode,
        )
        return None

    def _compute_voltage_homeostasis_error(
        self, v_n: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Return a local voltage-centering error signal.

        This term is intentionally local: it depends only on the compartment
        voltage and a fixed target midpoint, and it is applied through the
        same local sensitivity factors used by the task-driven updates.
        """
        cfg = getattr(self.local_cfg, "voltage_homeostasis", None)
        if cfg is None or not getattr(cfg, "enabled", False) or cfg.weight <= 0.0:
            return None

        target_v = float(getattr(cfg, "target_voltage", 0.5))
        weight = float(cfg.weight)
        return weight * (target_v - v_n)

    def _compute_gate_homeostasis_aux_grads(
        self,
        react_module: Optional[nn.Module],
        v_n: Optional[torch.Tensor],
        v_out: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return strictly local auxiliary grads for reactivation parameters.

        This is the local counterpart to gate-occupancy warmup: it uses only
        each layer's own gate outputs and local derivatives, never a global
        backpropagated signal. The returned tensors already live in parameter
        coordinates (`log_m`, `b`) and can be added directly to `.grad`.
        """
        cfg = getattr(self.local_cfg, "gate_homeostasis", None)
        if (
            cfg is None
            or not getattr(cfg, "enabled", False)
            or v_n is None
            or v_out is None
            or not isinstance(react_module, ParametricActivation)
            or not hasattr(react_module, "log_m")
        ):
            return None, None

        if v_n.dim() != 2 or v_out.dim() != 2:
            return None, None

        if not isinstance(react_module, (ParametricTanh, ParametricTanhOnlyM)):
            return None, None

        target_mean = float(getattr(cfg, "target_mean", 0.5))
        target_saturation = float(getattr(cfg, "target_saturation", 0.2))
        low_thr = float(getattr(cfg, "low_threshold", 0.1))
        high_thr = float(getattr(cfg, "high_threshold", 0.9))
        sat_k = float(getattr(cfg, "saturation_slope", 20.0))
        center_weight = float(getattr(cfg, "center_weight", 0.0))
        saturation_weight = float(getattr(cfg, "saturation_weight", 0.0))

        if center_weight <= 0.0 and saturation_weight <= 0.0:
            return None, None

        y = v_out
        base = 0.5 * (1.0 - (2.0 * y - 1.0).square())
        m_vec = react_module.log_m.detach().exp().to(device=v_n.device, dtype=v_n.dtype)

        if hasattr(react_module, "b"):
            b_vec = react_module.b.detach().to(device=v_n.device, dtype=v_n.dtype)
        else:
            b_vec = torch.full_like(m_vec, float(getattr(react_module, "fixed_b", 0.5)))

        dr_dlogm = base * (v_n - b_vec.unsqueeze(0)) * m_vec.unsqueeze(0)
        dr_db: Optional[torch.Tensor]
        if hasattr(react_module, "b"):
            dr_db = -base * m_vec.unsqueeze(0)
        else:
            dr_db = None

        grad_log_m = torch.zeros_like(m_vec)
        grad_b = torch.zeros_like(b_vec) if dr_db is not None else None

        if center_weight > 0.0:
            mean_r = y.mean(dim=0)
            mean_err = mean_r - target_mean
            dmean_dlogm = dr_dlogm.mean(dim=0)
            grad_log_m = grad_log_m + (2.0 * center_weight) * mean_err * dmean_dlogm
            if dr_db is not None and grad_b is not None:
                dmean_db = dr_db.mean(dim=0)
                grad_b = grad_b + (2.0 * center_weight) * mean_err * dmean_db

        if saturation_weight > 0.0:
            soft_lo = torch.sigmoid(sat_k * (low_thr - y))
            soft_hi = torch.sigmoid(sat_k * (y - high_thr))
            total_sat = soft_lo.mean(dim=0) + soft_hi.mean(dim=0)
            sat_err = total_sat - target_saturation
            dsat_dr = -sat_k * soft_lo * (1.0 - soft_lo) + sat_k * soft_hi * (
                1.0 - soft_hi
            )
            dsat_dlogm = (dsat_dr * dr_dlogm).mean(dim=0)
            grad_log_m = grad_log_m + (2.0 * saturation_weight) * sat_err * dsat_dlogm
            if dr_db is not None and grad_b is not None:
                dsat_db = (dsat_dr * dr_db).mean(dim=0)
                grad_b = grad_b + (2.0 * saturation_weight) * sat_err * dsat_db

        return grad_log_m, grad_b

    def _resolve_layer_dynamics_mode(self, rec: dict[str, Any]) -> str:
        """Resolve local update dynamics for the current layer.

        Returns:
            "conductance" for shunting-style updates or "additive" for
            additive-consistent updates.
        """
        cfg_mode = str(
            getattr(self.local_cfg.three_factor, "dynamics_mode", "auto")
        ).lower()
        if cfg_mode in {"conductance", "legacy"}:
            return "conductance"
        if cfg_mode == "additive":
            return "additive"
        if cfg_mode != "auto":
            logger.warning(
                f"Unknown three_factor.dynamics_mode='{cfg_mode}', falling back to auto."
            )

        layer = rec.get("layer")
        uses_shunting = bool(getattr(layer, "use_shunting", True))
        return "conductance" if uses_shunting else "additive"

    @staticmethod
    def _error_by_mode(y_hat: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "mse":
            # d/dy_hat (1/2 ||y_hat - y||^2) = (y_hat - y)
            return y_hat - y
        elif mode == "bce":
            # Support both logits (BCEWithLogits-style) and probability outputs (BCELoss).
            y_hat_detached = y_hat.detach()
            if y_hat_detached.min() < 0 or y_hat_detached.max() > 1:
                pred = torch.sigmoid(y_hat)
            else:
                pred = torch.clamp(y_hat, min=1e-6, max=1.0 - 1e-6)
            return pred - y
        elif mode == "ce":
            # Gradient of CrossEntropy w.r.t logits: softmax(y_hat) - one_hot(y)
            if y.dim() == 1 or (y.dim() == 2 and y.size(1) == 1):
                num_classes = y_hat.size(-1)
                y_onehot = functional.one_hot(y.view(-1), num_classes=num_classes).to(
                    y_hat.dtype
                )
            else:
                # assume already one-hot
                y_onehot = y
            probs = functional.softmax(y_hat, dim=-1)
            return probs - y_onehot
        else:
            # Fallback
            return y_hat - y

    # -------------------------- Helper: recorders ---------------------------
    def _attach_local_recorders(
        self, model: BaseModel
    ) -> tuple[list[dict[str, Any]], list[torch.utils.hooks.RemovableHandle]]:
        """Attach forward hooks to collect local signals per DendriticBranchLayer.

        Returns a tuple of (records, handles). Each record contains:
            - layer: module reference
            - v_n: pre-activation compartment voltage (inputs to reactivation)
            - v_out: post-reactivation output
            - x_exc, exc_out, exc_module, exc_mask (if present)
            - x_inh, inh_out, inh_module, inh_mask (if present)
            - x_rec_exc, rec_exc_out, rec_exc_module, rec_exc_mask (if present)
            - x_rec_inh, rec_inh_out, rec_inh_module, rec_inh_mask (if present)
            - x_blk (reshaped input), blk_out, blk_module (if present)
        """
        records: list[dict[str, Any]] = []
        handles: list[torch.utils.hooks.RemovableHandle] = []

        # Cache the decoder input so local rules can map output error back to
        # soma space through the decoder Jacobian when the decoder is nonlinear.
        self._decoder_cache.clear()
        if hasattr(model, "decoder_network"):

            def _decoder_pre_hook(mod, inputs):
                if inputs and isinstance(inputs[0], torch.Tensor):
                    self._decoder_cache["decoder_input"] = inputs[0]

            handles.append(
                model.decoder_network.register_forward_pre_hook(_decoder_pre_hook)
            )

            # Hook decoder input for local decoder updates (capture input to
            # the last Linear specifically).
            for m in model.decoder_network.modules():
                if isinstance(m, nn.Linear):

                    def _dec_in_hook(mod, inputs, outputs):
                        self._decoder_cache["module"] = mod
                        self._decoder_cache["input"] = inputs[0].detach()

                    handle = m.register_forward_hook(_dec_in_hook)
                    handles.append(handle)

        for module_name, module in model.named_modules():
            if not isinstance(module, DendriticBranchLayer):
                continue
            population = (
                "explicit_inhibitory"
                if ".inhibitory_cells." in module_name
                else "excitatory"
            )
            rec: dict[str, Any] = {
                "layer": module,
                "module_name": module_name,
                "population": population,
            }

            # Reactivation: capture pre-activation (voltage) and output
            def _react_hook(m, inputs, outputs, rec=rec):
                try:
                    rec["v_n"] = inputs[0].detach()
                except Exception:
                    rec["v_n"] = outputs.detach() if torch.is_tensor(outputs) else None
                rec["v_out"] = outputs.detach() if torch.is_tensor(outputs) else None

            handles.append(module.reactivation.register_forward_hook(_react_hook))

            # Excitatory TopK
            if getattr(module, "branch_excitation", None) is not None:
                exc_layer: TopKLinear = module.branch_excitation

                def _exc_hook(m, inputs, outputs, rec=rec, layer=exc_layer):
                    rec["x_exc"] = inputs[0].detach()
                    rec["exc_out"] = outputs.detach()
                    rec["exc_module"] = layer
                    try:
                        cached_mask = getattr(layer, "_last_forward_weight_mask", None)
                        if isinstance(cached_mask, torch.Tensor):
                            rec["exc_mask"] = cached_mask.detach()
                        else:
                            rec["exc_mask"] = layer.weight_mask().detach()
                    except Exception:
                        rec["exc_mask"] = None

                handles.append(exc_layer.register_forward_hook(_exc_hook))

            # Inhibitory TopK
            if getattr(module, "branch_inhibition", None) is not None:
                inh_layer: TopKLinear = module.branch_inhibition

                def _inh_hook(m, inputs, outputs, rec=rec, layer=inh_layer):
                    rec["x_inh"] = inputs[0].detach()
                    rec["inh_out"] = outputs.detach()
                    rec["inh_module"] = layer
                    try:
                        cached_mask = getattr(layer, "_last_forward_weight_mask", None)
                        if isinstance(cached_mask, torch.Tensor):
                            rec["inh_mask"] = cached_mask.detach()
                        else:
                            rec["inh_mask"] = layer.weight_mask().detach()
                    except Exception:
                        rec["inh_mask"] = None

                handles.append(inh_layer.register_forward_hook(_inh_hook))

            # Recurrent excitatory TopK
            if getattr(module, "branch_recurrent", None) is not None:
                rec_exc_layer: TopKLinear = module.branch_recurrent

                def _rec_exc_hook(m, inputs, outputs, rec=rec, layer=rec_exc_layer):
                    rec["x_rec_exc"] = inputs[0].detach()
                    rec["rec_exc_out"] = outputs.detach()
                    rec["rec_exc_module"] = layer
                    try:
                        cached_mask = getattr(layer, "_last_forward_weight_mask", None)
                        if isinstance(cached_mask, torch.Tensor):
                            rec["rec_exc_mask"] = cached_mask.detach()
                        else:
                            rec["rec_exc_mask"] = layer.weight_mask().detach()
                    except Exception:
                        rec["rec_exc_mask"] = None

                handles.append(rec_exc_layer.register_forward_hook(_rec_exc_hook))

            # Recurrent inhibitory TopK
            if getattr(module, "branch_rec_inhibition", None) is not None:
                rec_inh_layer: TopKLinear = module.branch_rec_inhibition

                def _rec_inh_hook(m, inputs, outputs, rec=rec, layer=rec_inh_layer):
                    rec["x_rec_inh"] = inputs[0].detach()
                    rec["rec_inh_out"] = outputs.detach()
                    rec["rec_inh_module"] = layer
                    try:
                        cached_mask = getattr(layer, "_last_forward_weight_mask", None)
                        if isinstance(cached_mask, torch.Tensor):
                            rec["rec_inh_mask"] = cached_mask.detach()
                        else:
                            rec["rec_inh_mask"] = layer.weight_mask().detach()
                    except Exception:
                        rec["rec_inh_mask"] = None

                handles.append(rec_inh_layer.register_forward_hook(_rec_inh_hook))

            # Dendritic BlockLinear
            if getattr(module, "branches_to_output", None) is not None:
                blk_layer = module.branches_to_output

                def _blk_hook(m, inputs, outputs, rec=rec, layer=blk_layer):
                    rec["x_blk_raw"] = inputs[0].detach()
                    rec["blk_out"] = outputs.detach()
                    rec["blk_module"] = layer

                handles.append(blk_layer.register_forward_hook(_blk_hook))

            records.append(rec)

        return records, handles

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

        # Initialize branch morphology if using morphology-aware features
        if (
            (self.local_cfg.error_broadcast_mode or "scalar").lower()
            == "pathway_vector"
            or (self.local_cfg.error_broadcast_mode or "scalar").lower()
            == "path_transport"
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
        if (
            self.local_cfg.error_broadcast_mode or "scalar"
        ).lower() == "path_transport":
            transported_errors = self._precompute_path_transport_errors(
                layer_records=layer_records,
                delta=delta,
                delta_scalar=delta_scalar,
            )

        feedback_seeds: list[Optional[torch.Tensor]] = []
        if (
            self.local_cfg.error_broadcast_mode or "scalar"
        ).lower() == "pathway_vector":
            feedback_seeds = self._precompute_feedback_seeds(
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

            # Error term per compartment
            # Modes:
            # - 'scalar': broadcast per-sample scalar to all somas in layer (default)
            # - 'per_soma': map output-dim error to soma-dim if sizes match, else fallback to scalar
            # - 'local_mismatch': modulate scalar error by local parent-vs-soma voltage mismatch
            bmode = (self.local_cfg.error_broadcast_mode or "scalar").lower()
            if (
                bmode == "per_soma"
                and delta.dim() == 2
                and delta.size(1) == out_features
            ):
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
                        feedback_seeds[layer_idx]
                        if layer_idx < len(feedback_seeds)
                        else None
                    ),
                    delta=delta,
                    delta_scalar=delta_scalar,
                    rec=rec,
                    v_n=v_n,
                )
            elif bmode == "path_transport":
                transported = (
                    transported_errors[layer_idx]
                    if layer_idx < len(transported_errors)
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
                # Scalar broadcast
                e_n = delta_scalar.expand(-1, out_features)  # [B, out]

            # Additive Gaussian noise on broadcast error (for robustness testing)
            noise_sigma = getattr(self.local_cfg, "error_noise_sigma", 0.0)
            if noise_sigma > 0.0:
                e_n = e_n + noise_sigma * torch.randn_like(e_n)

            # Low-bandwidth broadcast compression
            bandwidth = getattr(self.local_cfg, "broadcast_bandwidth", "full")
            if bandwidth != "full":
                e_n = self._apply_broadcast_bandwidth(e_n, bandwidth)

            stdp_error_signal = e_n
            if str(getattr(self.local_cfg, "rule_variant", "")).lower() == "stdp":
                e_n = torch.zeros_like(e_n)

            layer_dynamics_mode = self._resolve_layer_dynamics_mode(rec)
            if layer_dynamics_mode == "conductance":
                # Conductance scaling: R_tot = 1 / g_tot
                if self.local_cfg.three_factor.use_conductance_scaling:
                    g_tot = self._compute_layer_total_conductance(rec, v_n)
                    R_tot = 1.0 / (g_tot + 1e-8)
                else:
                    R_tot = 1.0
            else:
                # Additive dendritic integration does not include divisive input resistance.
                R_tot = 1.0

            # === PATH PROPAGATION FACTOR ===
            # Apply path-integrated attenuation if enabled
            if (
                self.local_cfg.morphology_aware.use_path_propagation
                and bmode != "path_transport"
            ):
                path_factor = rec.get("path_factor", 1.0)
                # Modulate error by path attenuation
                if isinstance(path_factor, torch.Tensor):
                    e_n = e_n * path_factor

            # === MORPHOLOGY CORRELATION RHO (scalar or per-branch) ===
            rho = 1.0
            if self.local_cfg.rule_variant in {"4f", "5f"}:
                rho = self._compute_layer_rho(rec, v0)
                # Apply layer-wise scaling if enabled
                if self.local_cfg.four_factor.layer_wise_rho_scale > 0:
                    rho = rho / (
                        layer_depth**0.5
                        * self.local_cfg.four_factor.layer_wise_rho_scale
                    )

                # Branch-specific morphology modulation
                if self.local_cfg.morphology_aware.morphology_modulator_mode == "depth":
                    rho = self._compute_branch_depth_modulator(rec, rho)
                elif (
                    self.local_cfg.morphology_aware.morphology_modulator_mode
                    == "centrality"
                ):
                    rho = self._compute_branch_centrality_modulator(rec, rho)

            # === INFORMATION PROXY PHI (scalar per layer) ===
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
                # Apply layer-wise scaling if enabled
                if self.local_cfg.five_factor.layer_wise_phi_scale > 0:
                    phi = phi / (
                        layer_depth**0.5
                        * self.local_cfg.five_factor.layer_wise_phi_scale
                    )

            # === BRANCH TYPE SCALING ===
            branch_scale: torch.Tensor | float = 1.0
            dendritic_branch_scale: torch.Tensor | float = 1.0
            role_synaptic_alignment: Optional[torch.Tensor] = None
            role_block_alignment: Optional[torch.Tensor] = None
            using_branch_role_scale = (
                self.local_cfg.morphology_aware.use_branch_role_rules
            )
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

            # Post-synaptic modulation per-sample, per-neuron.
            if layer_dynamics_mode == "conductance":
                if self.local_cfg.three_factor.use_driving_force:
                    post_mod = self.local_cfg.three_factor.e_rev_exc - v_n  # [B, out]
                else:
                    post_mod = v_n - self.local_cfg.three_factor.theta
                post_factor = e_v * R_tot * post_mod  # [B, out]
            else:
                # Additive rule -- with optional gain control modes.
                additive_mode = getattr(
                    self.local_cfg.three_factor, "additive_gain_mode", "none"
                )
                if additive_mode == "learned_gain":
                    gain_params = self._ensure_additive_gain_params(
                        v_n.size(1), v_n.device
                    )
                    gain_n = torch.sigmoid(gain_params)  # [out]
                    post_factor = e_v * gain_n.unsqueeze(0)  # [B, out]
                elif additive_mode == "input_dependent":
                    pseudo_R, pseudo_drive = self._compute_additive_pseudo_signals(
                        rec, v_n
                    )
                    post_factor = e_v * pseudo_R * pseudo_drive  # [B, out]
                elif additive_mode == "running_stats":
                    inv_std = self._get_additive_running_inv_std(layer_idx, v_n)
                    post_factor = e_v * inv_std  # [B, out]
                else:
                    post_factor = e_v  # Original behavior

            # Save raw post_factor before morphology scaling so inhibitory/dendritic
            # sections can reuse it without double-applying rho/phi/branch_scale.
            post_factor_raw = post_factor

            # Apply morphology/information factors
            if isinstance(rho, torch.Tensor):
                # Per-branch rho (depth modulation)
                post_factor = post_factor * rho.unsqueeze(
                    0
                )  # Broadcast [out] to [B, out]
            else:
                post_factor = post_factor * float(rho)

            post_factor = post_factor * float(phi)

            # Apply branch type scaling
            if isinstance(branch_scale, torch.Tensor):
                post_factor = post_factor * branch_scale.unsqueeze(0)  # [B, out]
            else:
                post_factor = post_factor * float(branch_scale)

            # Synaptic (TopKLinear) - excitatory
            exc_layer: Optional[TopKLinear] = rec.get("exc_module")
            if exc_layer is not None and rec.get("x_exc") is not None:
                x_exc: torch.Tensor = rec["x_exc"]  # [B, in]
                # dL/dg ≈ E_batch[ e_n * R_tot * (E_rev - V_n) * x ]
                grad_g = post_factor.t() @ x_exc
                if self.local_cfg.normalize_by_batch:
                    grad_g = grad_g / float(batch_size)
                voltage_homeo_error = self._compute_voltage_homeostasis_error(v_n)
                if voltage_homeo_error is not None:
                    if layer_dynamics_mode == "conductance":
                        exc_homeo_factor = (
                            voltage_homeo_error
                            * R_tot
                            * (self.local_cfg.three_factor.e_rev_exc - v_n)
                        )
                    else:
                        exc_homeo_factor = voltage_homeo_error
                    grad_homeo_exc = exc_homeo_factor.t() @ x_exc
                    if self.local_cfg.normalize_by_batch:
                        grad_homeo_exc = grad_homeo_exc / float(batch_size)
                    grad_g = grad_g + grad_homeo_exc
                if (
                    isinstance(role_synaptic_alignment, torch.Tensor)
                    and role_synaptic_alignment.size() == grad_g.size()
                ):
                    grad_g = grad_g * role_synaptic_alignment.to(
                        device=grad_g.device, dtype=grad_g.dtype
                    )
                exc_chain = self._weight_transform_derivative(
                    exc_layer.pre_w.detach(),
                    getattr(exc_layer, "weight_transform", "exp"),
                )
                grad_param = grad_g * exc_chain
                self._add_topk_grad(exc_layer, grad_param, rec.get("exc_mask"))

            # Synaptic (TopKLinear) - recurrent excitatory (if present)
            rec_exc_layer: Optional[TopKLinear] = rec.get("rec_exc_module")
            if rec_exc_layer is not None and rec.get("x_rec_exc") is not None:
                x_rec_exc: torch.Tensor = rec["x_rec_exc"]  # [B, in]
                grad_g_rec_exc = post_factor.t() @ x_rec_exc
                if self.local_cfg.normalize_by_batch:
                    grad_g_rec_exc = grad_g_rec_exc / float(batch_size)
                voltage_homeo_error = self._compute_voltage_homeostasis_error(v_n)
                if voltage_homeo_error is not None:
                    if layer_dynamics_mode == "conductance":
                        rec_exc_homeo_factor = (
                            voltage_homeo_error
                            * R_tot
                            * (self.local_cfg.three_factor.e_rev_exc - v_n)
                        )
                    else:
                        rec_exc_homeo_factor = voltage_homeo_error
                    grad_homeo_rec_exc = rec_exc_homeo_factor.t() @ x_rec_exc
                    if self.local_cfg.normalize_by_batch:
                        grad_homeo_rec_exc = grad_homeo_rec_exc / float(batch_size)
                    grad_g_rec_exc = grad_g_rec_exc + grad_homeo_rec_exc
                if (
                    isinstance(role_synaptic_alignment, torch.Tensor)
                    and role_synaptic_alignment.size() == grad_g_rec_exc.size()
                ):
                    grad_g_rec_exc = grad_g_rec_exc * role_synaptic_alignment.to(
                        device=grad_g_rec_exc.device, dtype=grad_g_rec_exc.dtype
                    )
                rec_exc_chain = self._weight_transform_derivative(
                    rec_exc_layer.pre_w.detach(),
                    getattr(rec_exc_layer, "weight_transform", "exp"),
                )
                grad_param_rec_exc = grad_g_rec_exc * rec_exc_chain
                self._add_topk_grad(
                    rec_exc_layer, grad_param_rec_exc, rec.get("rec_exc_mask")
                )

            # Synaptic (TopKLinear) - inhibitory (if present)
            inh_layer: Optional[TopKLinear] = rec.get("inh_module")
            if inh_layer is not None and rec.get("x_inh") is not None:
                x_inh: torch.Tensor = rec["x_inh"]  # [B, in]
                if layer_dynamics_mode == "conductance":
                    if self.local_cfg.three_factor.use_driving_force:
                        inh_post = self.local_cfg.three_factor.e_rev_inh - v_n
                    else:
                        inh_post = self.local_cfg.three_factor.theta - v_n
                    inh_factor = e_v * R_tot * inh_post  # [B, out]
                else:
                    # Additive inhibitory rule -- with optional gain control.
                    additive_mode = getattr(
                        self.local_cfg.three_factor, "additive_gain_mode", "none"
                    )
                    if additive_mode == "input_dependent":
                        pseudo_R, _ = self._compute_additive_pseudo_signals(rec, v_n)
                        inh_post = self.local_cfg.three_factor.theta - v_n
                        inh_factor = e_v * pseudo_R * inh_post  # [B, out]
                    elif additive_mode in ("learned_gain", "running_stats"):
                        # Reuse raw (pre-morphology) post_factor, negate for inhibitory.
                        # Morphology factors (rho, phi, branch_scale) are applied below.
                        inh_factor = -post_factor_raw
                    else:
                        inh_factor = -e_v  # Original behavior

                # Apply morphology/information factors (same as excitatory)
                if isinstance(rho, torch.Tensor):
                    # Per-branch rho (depth modulation)
                    inh_factor = inh_factor * rho.unsqueeze(
                        0
                    )  # Broadcast [out] to [B, out]
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

                # Apply branch type scaling (same as excitatory)
                if isinstance(branch_scale, torch.Tensor):
                    inh_factor = inh_factor * branch_scale.unsqueeze(0)  # [B, out]
                else:
                    inh_factor = inh_factor * float(branch_scale)

                # dL/dg_inh ≈ E_batch[ e_n * R_tot * inh_post * x ]
                grad_g_inh = inh_factor.t() @ x_inh
                if self.local_cfg.normalize_by_batch:
                    grad_g_inh = grad_g_inh / float(batch_size)
                if (
                    isinstance(role_synaptic_alignment, torch.Tensor)
                    and role_synaptic_alignment.size() == grad_g_inh.size()
                ):
                    grad_g_inh = grad_g_inh * role_synaptic_alignment.to(
                        device=grad_g_inh.device, dtype=grad_g_inh.dtype
                    )
                voltage_homeo_error = self._compute_voltage_homeostasis_error(v_n)
                if voltage_homeo_error is not None:
                    if layer_dynamics_mode == "conductance":
                        inh_homeo_factor = (
                            voltage_homeo_error
                            * R_tot
                            * (self.local_cfg.three_factor.e_rev_inh - v_n)
                        )
                    else:
                        inh_homeo_factor = -voltage_homeo_error
                    grad_homeo_inh = inh_homeo_factor.t() @ x_inh
                    if self.local_cfg.normalize_by_batch:
                        grad_homeo_inh = grad_homeo_inh / float(batch_size)
                    grad_g_inh = grad_g_inh + grad_homeo_inh
                homeo_factor = self._compute_inhibitory_homeostasis_factor(
                    rec=rec, R_tot=R_tot, v_n=v_n
                )
                if homeo_factor is not None:
                    grad_homeo = homeo_factor.t() @ x_inh
                    if self.local_cfg.normalize_by_batch:
                        grad_homeo = grad_homeo / float(batch_size)
                    grad_g_inh = grad_g_inh + grad_homeo
                inh_chain = self._weight_transform_derivative(
                    inh_layer.pre_w.detach(),
                    getattr(inh_layer, "weight_transform", "exp"),
                )
                grad_param_inh = grad_g_inh * inh_chain
                self._add_topk_grad(inh_layer, grad_param_inh, rec.get("inh_mask"))

            # Synaptic (TopKLinear) - recurrent inhibitory (if present)
            rec_inh_layer: Optional[TopKLinear] = rec.get("rec_inh_module")
            if rec_inh_layer is not None and rec.get("x_rec_inh") is not None:
                x_rec_inh: torch.Tensor = rec["x_rec_inh"]  # [B, in]
                if layer_dynamics_mode == "conductance":
                    if self.local_cfg.three_factor.use_driving_force:
                        rec_inh_post = self.local_cfg.three_factor.e_rev_inh - v_n
                    else:
                        rec_inh_post = self.local_cfg.three_factor.theta - v_n
                    rec_inh_factor = e_v * R_tot * rec_inh_post  # [B, out]
                else:
                    additive_mode = getattr(
                        self.local_cfg.three_factor, "additive_gain_mode", "none"
                    )
                    if additive_mode == "input_dependent":
                        pseudo_R, _ = self._compute_additive_pseudo_signals(rec, v_n)
                        rec_inh_post = self.local_cfg.three_factor.theta - v_n
                        rec_inh_factor = e_v * pseudo_R * rec_inh_post
                    elif additive_mode in ("learned_gain", "running_stats"):
                        rec_inh_factor = -post_factor_raw
                    else:
                        rec_inh_factor = -e_v

                if isinstance(rho, torch.Tensor):
                    rec_inh_factor = rec_inh_factor * rho.unsqueeze(0)
                else:
                    rec_inh_factor = (
                        rec_inh_factor * float(rho)
                        if self.local_cfg.rule_variant in {"4f", "5f"}
                        else rec_inh_factor
                    )

                rec_inh_factor = (
                    rec_inh_factor * float(phi)
                    if self.local_cfg.rule_variant == "5f"
                    else rec_inh_factor
                )

                if isinstance(branch_scale, torch.Tensor):
                    rec_inh_factor = rec_inh_factor * branch_scale.unsqueeze(0)
                else:
                    rec_inh_factor = rec_inh_factor * float(branch_scale)

                grad_g_rec_inh = rec_inh_factor.t() @ x_rec_inh
                if self.local_cfg.normalize_by_batch:
                    grad_g_rec_inh = grad_g_rec_inh / float(batch_size)
                if (
                    isinstance(role_synaptic_alignment, torch.Tensor)
                    and role_synaptic_alignment.size() == grad_g_rec_inh.size()
                ):
                    grad_g_rec_inh = grad_g_rec_inh * role_synaptic_alignment.to(
                        device=grad_g_rec_inh.device, dtype=grad_g_rec_inh.dtype
                    )
                voltage_homeo_error = self._compute_voltage_homeostasis_error(v_n)
                if voltage_homeo_error is not None:
                    if layer_dynamics_mode == "conductance":
                        rec_inh_homeo_factor = (
                            voltage_homeo_error
                            * R_tot
                            * (self.local_cfg.three_factor.e_rev_inh - v_n)
                        )
                    else:
                        rec_inh_homeo_factor = -voltage_homeo_error
                    grad_homeo_rec_inh = rec_inh_homeo_factor.t() @ x_rec_inh
                    if self.local_cfg.normalize_by_batch:
                        grad_homeo_rec_inh = grad_homeo_rec_inh / float(batch_size)
                    grad_g_rec_inh = grad_g_rec_inh + grad_homeo_rec_inh
                homeo_factor = self._compute_inhibitory_homeostasis_factor(
                    rec=rec, R_tot=R_tot, v_n=v_n
                )
                if homeo_factor is not None:
                    grad_homeo = homeo_factor.t() @ x_rec_inh
                    if self.local_cfg.normalize_by_batch:
                        grad_homeo = grad_homeo / float(batch_size)
                    grad_g_rec_inh = grad_g_rec_inh + grad_homeo
                rec_inh_chain = self._weight_transform_derivative(
                    rec_inh_layer.pre_w.detach(),
                    getattr(rec_inh_layer, "weight_transform", "exp"),
                )
                grad_param_rec_inh = grad_g_rec_inh * rec_inh_chain
                self._add_topk_grad(
                    rec_inh_layer, grad_param_rec_inh, rec.get("rec_inh_mask")
                )

            # Dendritic (BlockLinear)
            blk_layer = rec.get("blk_module")
            x_blk_raw = rec.get("x_blk_raw")
            if blk_layer is not None and x_blk_raw is not None:
                block_size = blk_layer.block_size
                x_blk = x_blk_raw.view(
                    x_blk_raw.size(0), blk_layer.out_features, block_size
                )
                if layer_dynamics_mode == "conductance":
                    # dL/dg_den ≈ E_batch[ e_n * R_tot * (V_j - V_n) * ... ]
                    diff = x_blk - v_n.unsqueeze(-1)  # [B, out, block]
                    # Ensure R_tot is a tensor for broadcasting
                    if isinstance(R_tot, torch.Tensor):
                        R_exp = R_tot.unsqueeze(-1)
                    else:
                        R_exp = torch.ones_like(v_n).unsqueeze(-1) * float(R_tot)
                    den_factor = e_v.unsqueeze(-1) * R_exp * diff
                else:
                    # Additive dendritic rule -- with optional gain control.
                    additive_mode = getattr(
                        self.local_cfg.three_factor, "additive_gain_mode", "none"
                    )
                    if additive_mode == "input_dependent":
                        pseudo_R, pseudo_drive = self._compute_additive_pseudo_signals(
                            rec, v_n
                        )
                        diff = x_blk - v_n.unsqueeze(-1)
                        den_factor = e_v.unsqueeze(-1) * pseudo_R.unsqueeze(-1) * diff
                    elif additive_mode in ("learned_gain", "running_stats"):
                        # Use raw (pre-morphology) post_factor; rho/phi/branch_scale
                        # are applied below to avoid double-counting.
                        den_factor = post_factor_raw.unsqueeze(-1) * x_blk
                    else:
                        den_factor = e_v.unsqueeze(-1) * x_blk

                # Apply rho (may be tensor if using depth modulation)
                if self.local_cfg.rule_variant in {"4f", "5f"}:
                    if isinstance(rho, torch.Tensor):
                        den_factor = den_factor * rho.unsqueeze(0).unsqueeze(
                            -1
                        )  # [1, out, 1]
                    else:
                        den_factor = den_factor * float(rho)

                # Apply phi
                if self.local_cfg.rule_variant == "5f":
                    den_factor = den_factor * float(phi)

                # Apply branch type scaling
                if isinstance(dendritic_branch_scale, torch.Tensor):
                    if dendritic_branch_scale.dim() == 2:
                        den_factor = den_factor * dendritic_branch_scale.unsqueeze(0)
                    else:
                        den_factor = den_factor * dendritic_branch_scale.unsqueeze(
                            0
                        ).unsqueeze(
                            -1
                        )  # [1, out, 1]
                else:
                    den_factor = den_factor * float(dendritic_branch_scale)

                grad_g_den = den_factor.mean(dim=0)  # [out, block]
                if not self.local_cfg.normalize_by_batch:
                    grad_g_den = grad_g_den * float(batch_size)
                voltage_homeo_error = self._compute_voltage_homeostasis_error(v_n)
                if voltage_homeo_error is not None:
                    if layer_dynamics_mode == "conductance":
                        diff_homeo = x_blk - v_n.unsqueeze(-1)
                        if isinstance(R_tot, torch.Tensor):
                            R_homeo = R_tot.unsqueeze(-1)
                        else:
                            R_homeo = torch.ones_like(v_n).unsqueeze(-1) * float(R_tot)
                        den_homeo_factor = (
                            voltage_homeo_error.unsqueeze(-1) * R_homeo * diff_homeo
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

                # Apply dendritic normalization if enabled
                if self.local_cfg.morphology_aware.use_dendritic_normalization:
                    grad_g_den = self._compute_dendritic_normalization(rec, grad_g_den)

                den_chain = self._weight_transform_derivative(
                    blk_layer.log_weight.detach(),
                    getattr(blk_layer, "weight_transform", "exp"),
                )
                grad_param_den = grad_g_den * den_chain
                if blk_layer.log_weight.grad is None:
                    blk_layer.log_weight.grad = grad_param_den.to(
                        blk_layer.log_weight.dtype
                    )
                else:
                    blk_layer.log_weight.grad = (
                        blk_layer.log_weight.grad
                        + grad_param_den.to(blk_layer.log_weight.dtype)
                    )

            # Reactivation parameters (ParametricActivation): local eligibility times factors
            react_module = getattr(rec.get("layer"), "reactivation", None)
            v_out = rec.get("v_out")
            if (
                update_reactivation
                and isinstance(react_module, ParametricActivation)
                and v_n is not None
                and v_out is not None
            ):
                # Generic eligibility via autograd: d f(V;θ)/dθ approximated by local derivative on outputs
                # Here we use analytic tanh-based eligibilities when available
                if hasattr(react_module, "log_m"):
                    # y = f(V) in [0,1] for ParametricTanh variants
                    y = v_out
                    one_minus_y2 = (
                        1 - (2 * y - 1) ** 2
                    )  # since y = (tanh(z)+1)/2 => 2y-1 = tanh(z)
                    # For ParametricTanh: z = m (V - b), m = exp(log_m)
                    if hasattr(react_module, "b"):
                        # d y / d m ∝ (1 - tanh(z)^2) * (V - b)
                        elig_m = one_minus_y2 * (v_n - react_module.b.detach())
                    else:
                        elig_m = one_minus_y2 * v_n
                    # Map to raw parameter log_m via chain rule: dm/dlog_m = m = exp(log_m)
                    m_vec = react_module.log_m.detach().exp()
                    # Use post error factor without dendritic driving for activation params
                    react_factor = e_n
                    # Apply rho
                    if isinstance(rho, torch.Tensor):
                        react_factor = react_factor * rho.unsqueeze(0)
                    else:
                        react_factor = react_factor * float(rho)
                    # Apply phi
                    react_factor = react_factor * float(phi)
                    # Apply branch scale
                    if isinstance(branch_scale, torch.Tensor):
                        react_factor = react_factor * branch_scale.unsqueeze(0)
                    else:
                        react_factor = react_factor * float(branch_scale)
                    # Aggregate over batch to per-neuron grads
                    grad_m = (react_factor * elig_m).mean(dim=0)
                    grad_log_m = grad_m * m_vec
                    if react_module.log_m.grad is None:
                        react_module.log_m.grad = grad_log_m.to(
                            react_module.log_m.dtype
                        )
                    else:
                        react_module.log_m.grad = (
                            react_module.log_m.grad
                            + grad_log_m.to(react_module.log_m.dtype)
                        )
                if hasattr(react_module, "b"):
                    # d y / d b ∝ -(1 - tanh(z)^2) * m
                    y = v_out
                    one_minus_y2 = 1 - (2 * y - 1) ** 2
                    m_vec = (
                        react_module.log_m.detach().exp()
                        if hasattr(react_module, "log_m")
                        else 1.0
                    )
                    elig_b = -one_minus_y2 * m_vec
                    # Reuse react_factor from above
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
                        react_module.log_m.grad = aux_grad_log_m.to(
                            react_module.log_m.dtype
                        )
                    else:
                        react_module.log_m.grad = (
                            react_module.log_m.grad
                            + aux_grad_log_m.to(react_module.log_m.dtype)
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

            if self._stdp_enabled():
                post_signal = rec.get("v_out")
                if not isinstance(post_signal, torch.Tensor):
                    post_signal = v_n
                self._apply_stdp_pathways(
                    rec=rec,
                    post_signal=post_signal,
                    error_signal=stdp_error_signal,
                )

            # HSIC/pHSIC auxiliary: add kernel-based gradients
            if self.local_cfg.hsic.enabled:
                v_out = rec.get("v_out")
                if v_out is not None:
                    z = v_out - v_out.mean(dim=0, keepdim=True)  # [B, out]
                    # Optional warmup for HSIC weights over initial epochs
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
                    # Optionally restrict HSIC to last dendritic layer only
                    if self.local_cfg.hsic.apply_last_layer_only:
                        is_last = layer_idx == len(layer_records) - 1
                        if not is_last:
                            hsic_warm = 0.0
                    if self.local_cfg.hsic.self_weight != 0.0:
                        grad_z_self = self._compute_hsic_gradient(
                            z, z, self.local_cfg.hsic.self_weight * hsic_warm, "self"
                        )
                        # Optional HSIC grad clipping
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
                            self._add_topk_grad(
                                exc_layer, grad_param_self, rec.get("exc_mask")
                            )
                    if (
                        self.local_cfg.hsic.target_weight != 0.0
                        and y_target is not None
                    ):
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
                            self._add_topk_grad(
                                exc_layer, grad_param_tgt, rec.get("exc_mask")
                            )

    # ----------------------- Helper: HSIC kernel functions ------------------
    def _compute_kernel_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute kernel matrix K(X, Y) based on configured kernel type.

        Args:
            X: Input tensor [B, d1]
            Y: Input tensor [B, d2]

        Returns:
            Kernel matrix [B, B]
        """
        kernel_type = self.local_cfg.hsic.kernel.lower()

        if kernel_type == "linear":
            # Linear kernel: K(x, y) = x^T y
            return X @ Y.t()
        elif kernel_type == "rbf" or kernel_type == "gaussian":
            # RBF kernel: K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
            gamma = (
                1.0 / (2.0 * self.local_cfg.hsic.sigma**2)
                if self.local_cfg.hsic.sigma > 0
                else 1.0
            )
            # Compute pairwise distances efficiently
            X_norm = (X**2).sum(dim=1, keepdim=True)  # [B, 1]
            Y_norm = (Y**2).sum(dim=1, keepdim=True)  # [B, 1]
            dist_sq = X_norm + Y_norm.t() - 2.0 * (X @ Y.t())  # [B, B]
            return torch.exp(-gamma * dist_sq)
        elif kernel_type == "polynomial" or kernel_type == "poly":
            # Polynomial kernel: K(x, y) = (coef0 + x^T y)^degree
            dot_product = X @ Y.t()
            return (
                self.local_cfg.hsic.coef0 + dot_product
            ) ** self.local_cfg.hsic.degree
        else:
            # Fallback to linear
            return X @ Y.t()

    def _compute_hsic_gradient(
        self, z: torch.Tensor, y: torch.Tensor, weight: float, grad_type: str
    ) -> torch.Tensor:
        """Compute HSIC gradient with respect to z using specified kernel.

        Args:
            z: Layer activations [B, d_z]
            y: Target/reference [B, d_y]
            weight: Gradient scaling weight
            grad_type: "self" for self-dependence, "target" for target dependence

        Returns:
            Gradient w.r.t. z [B, d_z]
        """
        batch_size = z.size(0)
        kernel_type = self.local_cfg.hsic.kernel.lower()

        if kernel_type == "linear":
            # Linear kernel gradients (original implementation)
            if grad_type == "self":
                # NOTE: `hsic.weight` is a global scale intended to gate the entire HSIC auxiliary.
                # Keep behavior consistent across self/target terms.
                return (
                    (2.0 / float(batch_size)) * z * self.local_cfg.hsic.weight * weight
                )
            else:  # target
                cov_zy = (z.t() @ y) / float(batch_size)
                grad_z = -2.0 * (cov_zy @ y.t()).t() / float(batch_size)
                return grad_z * self.local_cfg.hsic.weight * weight

        elif kernel_type in ["rbf", "gaussian"]:
            # RBF kernel gradients
            gamma = (
                1.0 / (2.0 * self.local_cfg.hsic.sigma**2)
                if self.local_cfg.hsic.sigma > 0
                else 1.0
            )
            if grad_type == "self":
                # Self-dependence: encourage diversity (negative correlation)
                K_zz = self._compute_kernel_matrix(z, z)  # [B, B]
                z_expanded = z.unsqueeze(1)  # [B, 1, d]
                z_diff = z_expanded - z.unsqueeze(0)  # [B, B, d]
                K_expanded = K_zz.unsqueeze(-1)  # [B, B, 1]
                grad_z = -2.0 * gamma * (K_expanded * z_diff).sum(dim=1)  # [B, d]
                return grad_z * self.local_cfg.hsic.weight * weight / float(batch_size)
            else:  # target
                # Use Z-Z kernel and Y-Y kernel; weight Z pairwise diffs by centered Y kernel
                K_zz = self._compute_kernel_matrix(z, z)  # [B, B]
                L_yy = self._compute_kernel_matrix(y, y)  # [B, B]
                # Center L_yy
                Lc = (
                    L_yy
                    - L_yy.mean(dim=0, keepdim=True)
                    - L_yy.mean(dim=1, keepdim=True)
                    + L_yy.mean()
                )
                z_expanded = z.unsqueeze(1)  # [B, 1, d_z]
                z_diff = z_expanded - z.unsqueeze(0)  # [B, B, d_z]
                weight_mat = (K_zz * Lc).unsqueeze(-1)  # [B, B, 1]
                grad_z = 2.0 * gamma * (weight_mat * z_diff).sum(dim=1)  # [B, d_z]
                return grad_z * self.local_cfg.hsic.weight * weight / float(batch_size)

        elif kernel_type in ["polynomial", "poly"]:
            # Polynomial kernel gradients
            degree = self.local_cfg.hsic.degree
            coef0 = self.local_cfg.hsic.coef0
            if grad_type == "self":
                # Self-dependence gradient
                dot_zz = z @ z.t()  # [B, B]
                poly_base = coef0 + dot_zz  # [B, B]
                poly_grad_coef = degree * (poly_base ** (degree - 1))  # [B, B]
                grad_z = (poly_grad_coef @ z) * 2.0  # [B, d] (factor 2 for symmetry)
                return grad_z * self.local_cfg.hsic.weight * weight / float(batch_size)
            else:  # target
                # Use Z-Z polynomial kernel and Y-Y polynomial kernel; weight by centered Y kernel
                dot_zz = z @ z.t()  # [B, B]
                K_base = coef0 + dot_zz
                K_coef = degree * (K_base ** (degree - 1))  # [B, B]
                dot_yy = y @ y.t()  # [B, B]
                L_base = coef0 + dot_yy
                L_yy = L_base**degree  # kernel values (only for centering/weighting)
                Lc = (
                    L_yy
                    - L_yy.mean(dim=0, keepdim=True)
                    - L_yy.mean(dim=1, keepdim=True)
                    + L_yy.mean()
                )
                weight_mat = K_coef * Lc  # [B, B]
                grad_z = (weight_mat @ z) * 2.0  # symmetry factor
                return grad_z * self.local_cfg.hsic.weight * weight / float(batch_size)

        else:
            # Fallback to linear
            if grad_type == "self":
                return (
                    (2.0 / float(batch_size)) * z * self.local_cfg.hsic.weight * weight
                )
            else:
                cov_zy = (z.t() @ y) / float(batch_size)
                grad_z = -2.0 * (cov_zy @ y.t()).t() / float(batch_size)
                return grad_z * self.local_cfg.hsic.weight * weight

    # ----------------------- Helper: additive gain modes --------------------

    def _ensure_additive_gain_params(
        self, out_features: int, device: torch.device
    ) -> torch.Tensor:
        """Get or create per-neuron learnable gain params for additive mode.

        Uses ``_additive_gain_cache`` (persists across steps) rather than
        ``_decoder_cache`` (cleared every forward pass).
        """
        key = f"additive_gain_{out_features}"
        if key not in self._additive_gain_cache:
            # Initialize so sigmoid(param) ≈ 0.5
            param = torch.nn.Parameter(torch.zeros(out_features, device=device))
            self.optimizer.add_param_group(
                {"params": [param], "lr": self.optimizer.param_groups[0]["lr"]}
            )
            self._additive_gain_cache[key] = param
        return self._additive_gain_cache[key]

    def _compute_additive_pseudo_signals(
        self, rec: dict[str, Any], v_n: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute pseudo-R_tot and pseudo-driving-force for additive networks.

        pseudo_R: analogous to 1/g_tot; normalizes by total activation magnitude
        pseudo_drive: analogous to (E_rev - V); uses voltage deviation from mean
        """
        # Compute pseudo input resistance from absolute activations
        total_activation = torch.ones_like(v_n)
        exc_out = rec.get("exc_out")
        if exc_out is not None:
            total_activation = total_activation + exc_out.abs()
        inh_out = rec.get("inh_out")
        if inh_out is not None:
            total_activation = total_activation + inh_out.abs()
        pseudo_R = 1.0 / (total_activation + 1e-8)  # [B, out]

        # Compute pseudo driving force from voltage deviation
        v_mean = v_n.mean(dim=0, keepdim=True)  # [1, out]
        pseudo_drive = v_mean - v_n + 1.0  # Shift to ensure non-zero baseline
        return pseudo_R, pseudo_drive

    def _get_additive_running_inv_std(
        self, layer_idx: int, v_n: torch.Tensor
    ) -> torch.Tensor:
        """Return inverse running std for additive normalization scaling."""
        stats = self._layer_stats.setdefault(layer_idx, _LayerStats())
        alpha = getattr(self.local_cfg.three_factor, "additive_stats_ema_alpha", 0.01)
        with torch.no_grad():
            batch_var = v_n.var(dim=0)  # [out]
            if stats.var_ema is None:
                stats.var_ema = batch_var.detach()
            else:
                stats.var_ema = (1 - alpha) * stats.var_ema + alpha * batch_var.detach()
        inv_std = 1.0 / (stats.var_ema.sqrt() + 1e-6)  # [out]
        return inv_std.unsqueeze(0)  # [1, out]

    # ----------------------- Helper: rho and phi factors --------------------
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
        v_n: Optional[torch.Tensor] = rec.get("v_n")
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
        phi_cfg = getattr(self.local_cfg, "five_factor", None)
        phi_min = float(getattr(phi_cfg, "phi_clamp_min", 0.25))
        phi_max = float(getattr(phi_cfg, "phi_clamp_max", 4.0))
        if phi_min > phi_max:
            phi_min, phi_max = phi_max, phi_min
        return max(phi_min, min(phi_max, phi))

    # -------------------- Morphology-Aware Helper Methods --------------------
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
        """Apply dendritic normalization to branch conductance updates.

        Normalizes by total branch conductance: Δg_j ← Δg_j / (∑_k g_k + ε)

        Args:
            rec: Layer record
            grad_g_den: Gradient w.r.t. dendritic conductances [out, block]

        Returns:
            Normalized gradient
        """
        blk_layer = rec.get("blk_module")
        if blk_layer is None or not hasattr(blk_layer, "sum_conductances"):
            return grad_g_den

        # Total conductance per output neuron
        g_den_total = blk_layer.sum_conductances().detach()  # [out]
        norm_factor = 1.0 / (g_den_total.unsqueeze(-1) + 1e-8)  # [out, 1]

        # Broadcast to gradient shape
        if grad_g_den.dim() == 2:
            norm_factor = norm_factor  # Already [out, 1]
        elif grad_g_den.dim() == 3:
            norm_factor = norm_factor.unsqueeze(0)  # [1, out, 1] for [B, out, block]

        return grad_g_den * norm_factor

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

            exc_layer: Optional[TopKLinear] = rec.get("exc_module")
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

            inh_layer: Optional[TopKLinear] = rec.get("inh_module")
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
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
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
        syn_factor: Optional[torch.Tensor] = None
        if isinstance(stats.input_role_profile, torch.Tensor):
            alignment = role_profile @ stats.input_role_profile.t().to(
                device=role_profile.device, dtype=role_profile.dtype
            )
            alignment = alignment - alignment.mean(dim=1, keepdim=True)
            syn_factor = torch.clamp(1.0 + weight * alignment, min=0.0)

        block_factor: Optional[torch.Tensor] = None
        if isinstance(stats.branch_block_roles, torch.Tensor):
            block_roles = stats.branch_block_roles.to(
                device=role_profile.device, dtype=role_profile.dtype
            )
            alignment = (block_roles * role_profile.unsqueeze(1)).sum(dim=-1)
            alignment = alignment - alignment.mean(dim=1, keepdim=True)
            block_factor = torch.clamp(1.0 + weight * alignment, min=0.0)

        return syn_factor, block_factor

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
