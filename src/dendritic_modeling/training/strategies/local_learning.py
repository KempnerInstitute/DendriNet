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
import torch.nn.functional as functional

from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    DendriticBranchLayer,
    ExcitationInhibitionNetwork,
    ParametricActivation,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.training.strategies.standard import Trainer

logger = logging.getLogger(__name__)


@dataclass
class LocalRuleConfig:
    """Configuration for local learning rules.

    Attributes
    ----------
    rule_variant: str
        '3f' for three-factor, '4f' to include morphology correlation, '5f' to
        include both morphology and information proxy.
    error_mode: str
        'auto', 'mse', 'ce', or 'bce' for deriving the soma error signal.
    error_broadcast_mode: str
        How to broadcast the soma error to compartments: 'scalar' (default, per-sample
        scalar broadcast to all somas), 'per_soma' (map output-dim error to per-soma),
        or 'local_mismatch' (modulate scalar error by local parent-vs-soma mismatch).
    ema_alpha: float
        EMA smoothing for morphology/information statistics.
    use_conductance_scaling: bool
        If True, scale updates by R_tot = 1 / (g_tot) per compartment.
    use_driving_force: bool
        If True, include (E_rev - V_n) multiplier for synapses.
    theta: float
        Leak reversal potential (post term uses V_n - theta if driving_force=False).
    e_rev_exc: float
        Excitatory reversal potential used for driving force.
    e_rev_inh: float
        Inhibitory reversal potential proxy (typically 0 for shunting).
    clip_grad_value: float
        Clip absolute gradient values before optimizer.step().
    normalize_by_batch: bool
        If True, average updates over batch.
    """

    rule_variant: str = "3f"
    error_mode: str = "auto"
    error_broadcast_mode: str = "scalar"
    ema_alpha: float = 0.05
    use_conductance_scaling: bool = True
    use_driving_force: bool = True
    theta: float = 0.0
    e_rev_exc: float = 1.0
    e_rev_inh: float = 0.0
    clip_grad_value: float = 5.0
    normalize_by_batch: bool = True
    # If True, update inactive TopK weights too (mask only applied in forward)
    update_inactive_weights: bool = False
    # phi (information proxy) configuration
    phi_mode: str = "conditional"  # "conditional" uses parent voltages; "variance" uses layer variance
    phi_ridge_lambda: float = 1e-3  # Regularization for conditional regression
    # Single-sample estimators
    rho_estimator: str = "ema"  # "ema", "augment", "ema+augment"
    phi_estimator: str = "conditional_ema"  # "conditional_ema", "variance_ema"
    augment_k: int = 8
    augment_noise_sigma: float = 0.01
    rls_forgetting: float = 0.99
    # Rho (morphology factor) mode: "pearson" uses correlation; "dot" uses dot-product proxy; "none" disables
    rho_mode: str = "pearson"
    # HSIC/pHSIC auxiliary loss configuration
    hsic_enabled: bool = False
    hsic_weight: float = 0.0
    hsic_self_weight: float = 1.0
    hsic_target_weight: float = 1.0
    hsic_target_source: str = "labels"  # "labels" or "logits"
    hsic_kernel: str = "linear"  # "linear", "rbf", "polynomial"
    hsic_sigma: float = 1.0  # RBF kernel bandwidth (gamma = 1/(2*sigma^2))
    hsic_degree: int = 2  # Polynomial kernel degree
    hsic_coef0: float = 1.0  # Polynomial kernel coefficient
    # HSIC stability options
    hsic_grad_clip_value: float = 0.0  # Clip HSIC gradients elementwise if > 0
    hsic_warmup_epochs: int = 0  # Linearly ramp HSIC weights over first N epochs
    hsic_apply_last_layer_only: bool = False  # Apply HSIC only on last dendritic layer
    # Decoder update strategy: "backprop" or "none"
    decoder_update_mode: str = "backprop"
    # Gate reactivation parameter updates
    update_reactivation: bool = True
    # Optional epoch-based freezing and phased training
    freeze_reactivation_epochs: int = (
        0  # Freeze reactivation params for initial N epochs
    )
    freeze_decoder_epochs: int = 0  # Freeze decoder for initial N epochs
    # Optional per-epoch schedule overriding the above. Each entry may include:
    # { start_epoch: int, end_epoch: int, update_reactivation: bool, decoder_update_mode: str }
    training_schedule: Optional[list[dict[str, Any]]] = None
    # Layer-wise scaling for morphology factors
    layer_wise_rho_scale: float = 1.0  # Scale rho by 1/sqrt(layer_depth) if > 0
    layer_wise_phi_scale: float = 1.0  # Scale phi by 1/sqrt(layer_depth) if > 0


class _LayerStats:
    """Holds per-layer running statistics for morphology/info factors."""

    def __init__(self):
        self.rho_ema: Optional[float] = None
        self.var_ema: Optional[float] = None
        # Moments for conditional EMA (single-sample friendly)
        self.ema_var_y: Optional[float] = None
        self.ema_var_x: Optional[float] = None
        self.ema_cov_xy: Optional[float] = None
        # NEW: Running means for proper online variance/covariance
        self.ema_mean_y: Optional[float] = None
        self.ema_mean_x: Optional[float] = None


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
        self.local_cfg = LocalRuleConfig(**config_dict)
        self._layer_stats: dict[int, _LayerStats] = {}
        self._decoder_cache: dict[str, Any] = {}
        self.filename_prefix = "local_learning_"
        self.logger_info_prefix = "[Local] "

    # Override: do not run extra pretrain/final phases here
    def _pretrain(self, *args, **kwargs):  # type: ignore[override]
        return [], []

    # ----------------------- Core training overrides -----------------------
    def _epoch_train(self, model: BaseModel, train_loader) -> float:  # type: ignore[override]
        model.train()
        train_loss = 0.0

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

        for x_batch, y_batch in train_loader:
            # Resolve per-epoch schedule for reactivation/decoder behavior
            update_reactivation = self.local_cfg.update_reactivation
            decoder_update_mode = self.local_cfg.decoder_update_mode
            # Freeze phases by epoch index
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
                        if "decoder_update_mode" in sched:
                            decoder_update_mode = str(
                                sched["decoder_update_mode"]
                            ).lower()
                        break
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Capture layer-wise local signals via forward hooks
            layer_records, handles = self._attach_local_recorders(model)
            try:
                # Forward once to obtain outputs and local signals
                y_hat: torch.Tensor = model(x_batch)
            finally:
                for h in handles:
                    h.remove()

            # Compute scalar loss for logging (no backward)
            loss = self._compute_logging_loss(model, x_batch, y_batch)
            train_loss += loss.item()

            # Compute broadcast soma error dL/dV0 per-sample
            delta: torch.Tensor = self._compute_soma_error(y_hat, y_batch)

            # Prepare HSIC targets if needed
            y_target: Optional[torch.Tensor] = None
            if self.local_cfg.hsic_enabled:
                if self.local_cfg.hsic_target_source == "labels":
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
                v0=y_hat.detach(),
                delta=delta.detach(),
                y_target=y_target,
                update_reactivation=update_reactivation,
            )

            # Decoder update mode
            if hasattr(model, "decoder_network"):
                if decoder_update_mode == "backprop":
                    # Backprop only decoder parameters using global loss
                    dec_params = [
                        p for p in model.decoder_network.parameters() if p.requires_grad
                    ]
                    if dec_params:
                        grads = torch.autograd.grad(
                            loss, dec_params, retain_graph=True, allow_unused=True
                        )
                        for p, g in zip(dec_params, grads):
                            if g is not None:
                                if p.grad is None:
                                    p.grad = g.detach().clone()
                                else:
                                    p.grad = p.grad + g.detach()
                elif decoder_update_mode == "local":
                    # Simple local 3-factor on last Linear: grad_W = e_out^T @ h_in; grad_b = sum(e_out)
                    lin_mod = self._decoder_cache.get("module")
                    h_in = self._decoder_cache.get("input")
                    if isinstance(lin_mod, nn.Linear) and isinstance(
                        h_in, torch.Tensor
                    ):
                        e_out = delta.detach()  # [B, out]
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
        return train_loss

    # -------------------------- Helper: loss/error --------------------------
    def _compute_logging_loss(
        self, model: BaseModel, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Compute scalar loss for logging using configured loss function.

        Uses the existing loss_function for compatibility with reporting.
        """
        return self.loss_function(model, x, y)

    def _compute_soma_error(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute broadcast error signal at the soma: dL/dV0 per sample.

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

    @staticmethod
    def _error_by_mode(y_hat: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "mse":
            # d/dy_hat (1/2 ||y_hat - y||^2) = (y_hat - y)
            return y_hat - y
        elif mode == "bce":
            # Assume y_hat are logits; gradient of BCEWithLogits ≈ sigmoid(y_hat) - y
            return torch.sigmoid(y_hat) - y
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
            - x_blk (reshaped input), blk_out, blk_module (if present)
        """
        records: list[dict[str, Any]] = []
        handles: list[torch.utils.hooks.RemovableHandle] = []

        # Hook decoder input for local mode (capture input to last Linear)
        self._decoder_cache.clear()
        if hasattr(model, "decoder_network"):
            for m in model.decoder_network.modules():
                if isinstance(m, nn.Linear):

                    def _dec_in_hook(mod, inputs, outputs):
                        self._decoder_cache["module"] = mod
                        self._decoder_cache["input"] = inputs[0].detach()

                    handle = m.register_forward_hook(_dec_in_hook)
                    handles.append(handle)

        for module in model.modules():
            if not isinstance(module, DendriticBranchLayer):
                continue
            rec: dict[str, Any] = {"layer": module}

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
                        rec["inh_mask"] = layer.weight_mask().detach()
                    except Exception:
                        rec["inh_mask"] = None

                handles.append(inh_layer.register_forward_hook(_inh_hook))

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
        delta_mean: torch.Tensor = delta.mean(dim=1, keepdim=True)  # [B, 1]

        # Iterate in the order records were captured
        for layer_idx, rec in enumerate(layer_records):
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
            elif bmode == "local_mismatch":
                parent = rec.get("blk_out")
                if parent is None:
                    parent = rec.get("exc_out")
                if parent is None:
                    parent = rec.get("inh_out")
                if parent is None:
                    # Fallback to scalar broadcast if no parent signal is available
                    e_n = delta_mean.expand(-1, out_features)
                else:
                    # Centered mismatch as a simple local error proxy
                    p = parent
                    if p.size() != v_n.size():
                        # Try to align shapes; if mismatch persists, fallback to scalar
                        try:
                            p = p.view_as(v_n)
                        except Exception:
                            p = None
                    if p is None:
                        e_n = delta_mean.expand(-1, out_features)
                    else:
                        mismatch = p - v_n  # [B, out]
                        # Optionally center mismatch per-batch to avoid bias
                        mismatch = mismatch - mismatch.mean(dim=0, keepdim=True)
                        e_n = delta_mean.expand(-1, out_features) * mismatch
            else:
                # Scalar broadcast
                e_n = delta_mean.expand(-1, out_features)  # [B, out]

            # Conductance scaling: R_tot = 1 / g_tot
            if self.local_cfg.use_conductance_scaling:
                g_tot = torch.ones_like(v_n)
                exc_out = rec.get("exc_out")
                if exc_out is not None:
                    g_tot = g_tot + functional.relu(exc_out)
                inh_out = rec.get("inh_out")
                if inh_out is not None:
                    g_tot = g_tot + functional.relu(inh_out)
                blk_module = rec.get("blk_module")
                if blk_module is not None and hasattr(blk_module, "sum_conductances"):
                    g_blk = (
                        blk_module.sum_conductances().detach()[None, :].expand_as(v_n)
                    )
                    g_tot = g_tot + functional.relu(g_blk)
                R_tot = 1.0 / (g_tot + 1e-8)
            else:
                R_tot = 1.0

            # Morphology correlation rho (scalar per layer)
            rho = 1.0
            if self.local_cfg.rule_variant in {"4f", "5f"}:
                rho = self._compute_layer_rho(rec, v0)
                # Apply layer-wise scaling if enabled
                if self.local_cfg.layer_wise_rho_scale > 0:
                    rho = rho / (layer_depth**0.5 * self.local_cfg.layer_wise_rho_scale)

            # Information proxy phi (scalar per layer)
            phi = 1.0
            if self.local_cfg.rule_variant == "5f":
                if self.local_cfg.phi_mode == "conditional":
                    phi = self._compute_layer_phi_conditional(rec)
                else:
                    phi = self._compute_layer_phi(rec)
                # Apply layer-wise scaling if enabled
                if self.local_cfg.layer_wise_phi_scale > 0:
                    phi = phi / (layer_depth**0.5 * self.local_cfg.layer_wise_phi_scale)

            # Post-synaptic modulation per-sample, per-neuron
            if self.local_cfg.use_driving_force:
                post_mod = self.local_cfg.e_rev_exc - v_n  # [B, out]
            else:
                post_mod = v_n - self.local_cfg.theta

            post_factor = e_n * R_tot * post_mod  # [B, out]
            post_factor = post_factor * float(rho) * float(phi)

            # Synaptic (TopKLinear) - excitatory
            exc_layer: Optional[TopKLinear] = rec.get("exc_module")
            if exc_layer is not None and rec.get("x_exc") is not None:
                x_exc: torch.Tensor = rec["x_exc"]  # [B, in]
                # dL/dg ≈ E_batch[ e_n * R_tot * (E_rev - V_n) * x ]
                grad_g = post_factor.t() @ x_exc
                if self.local_cfg.normalize_by_batch:
                    grad_g = grad_g / float(batch_size)
                g_current = exc_layer.weight().detach()
                grad_param = grad_g * g_current
                mask = rec.get("exc_mask")
                if mask is not None and not self.local_cfg.update_inactive_weights:
                    grad_param = grad_param * mask
                if exc_layer.pre_w.grad is None:
                    exc_layer.pre_w.grad = grad_param.to(exc_layer.pre_w.dtype)
                else:
                    exc_layer.pre_w.grad = exc_layer.pre_w.grad + grad_param.to(
                        exc_layer.pre_w.dtype
                    )

            # Synaptic (TopKLinear) - inhibitory (if present)
            inh_layer: Optional[TopKLinear] = rec.get("inh_module")
            if inh_layer is not None and rec.get("x_inh") is not None:
                x_inh: torch.Tensor = rec["x_inh"]  # [B, in]
                if self.local_cfg.use_driving_force:
                    inh_post = -v_n
                else:
                    inh_post = self.local_cfg.theta - v_n
                inh_factor = e_n * R_tot * inh_post
                inh_factor = inh_factor * float(
                    rho if self.local_cfg.rule_variant in {"4f", "5f"} else 1.0
                )
                inh_factor = inh_factor * float(
                    phi if self.local_cfg.rule_variant == "5f" else 1.0
                )
                # dL/dg_inh ≈ E_batch[ e_n * R_tot * inh_post * x ]
                grad_g_inh = inh_factor.t() @ x_inh
                if self.local_cfg.normalize_by_batch:
                    grad_g_inh = grad_g_inh / float(batch_size)
                g_current_inh = inh_layer.weight().detach()
                grad_param_inh = grad_g_inh * g_current_inh
                mask_inh = rec.get("inh_mask")
                if mask_inh is not None and not self.local_cfg.update_inactive_weights:
                    grad_param_inh = grad_param_inh * mask_inh
                if inh_layer.pre_w.grad is None:
                    inh_layer.pre_w.grad = grad_param_inh.to(inh_layer.pre_w.dtype)
                else:
                    inh_layer.pre_w.grad = inh_layer.pre_w.grad + grad_param_inh.to(
                        inh_layer.pre_w.dtype
                    )

            # Dendritic (BlockLinear)
            blk_layer = rec.get("blk_module")
            x_blk_raw = rec.get("x_blk_raw")
            if blk_layer is not None and x_blk_raw is not None:
                block_size = blk_layer.block_size
                x_blk = x_blk_raw.view(
                    x_blk_raw.size(0), blk_layer.out_features, block_size
                )
                # dL/dg_den ≈ E_batch[ e_n * R_tot * (V_j - V_n) * rho * phi ]
                diff = x_blk - v_n.unsqueeze(-1)  # [B, out, block]
                # Ensure R_tot is a tensor for broadcasting
                if isinstance(R_tot, torch.Tensor):
                    R_exp = R_tot.unsqueeze(-1)
                else:
                    R_exp = torch.ones_like(v_n).unsqueeze(-1) * float(R_tot)
                # Apply morphology and information factors
                den_factor = e_n.unsqueeze(-1) * R_exp * diff
                if self.local_cfg.rule_variant in {"4f", "5f"}:
                    den_factor = den_factor * float(rho)
                if self.local_cfg.rule_variant == "5f":
                    den_factor = den_factor * float(phi)
                grad_g_den = den_factor.mean(dim=0)
                if not self.local_cfg.normalize_by_batch:
                    grad_g_den = grad_g_den * float(batch_size)
                g_current_den = blk_layer.log_weight.detach().exp()
                grad_param_den = grad_g_den * g_current_den
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
                    post_factor = e_n * float(rho) * float(phi)
                    # Aggregate over batch to per-neuron grads
                    grad_m = (post_factor * elig_m).mean(dim=0)
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
                    post_factor = e_n * float(rho) * float(phi)
                    grad_b = (post_factor * elig_b).mean(dim=0)
                    if react_module.b.grad is None:
                        react_module.b.grad = grad_b.to(react_module.b.dtype)
                    else:
                        react_module.b.grad = react_module.b.grad + grad_b.to(
                            react_module.b.dtype
                        )

            # HSIC/pHSIC auxiliary: add kernel-based gradients
            if self.local_cfg.hsic_enabled:
                v_out = rec.get("v_out")
                if v_out is not None:
                    z = v_out - v_out.mean(dim=0, keepdim=True)  # [B, out]
                    # Optional warmup for HSIC weights over initial epochs
                    hsic_warm = 1.0
                    if (
                        getattr(self.local_cfg, "hsic_warmup_epochs", 0)
                        and self.epoch_counter < self.local_cfg.hsic_warmup_epochs
                    ):
                        hsic_warm = max(
                            0.0,
                            float(self.epoch_counter)
                            / float(max(1, self.local_cfg.hsic_warmup_epochs)),
                        )
                    # Optionally restrict HSIC to last dendritic layer only
                    if getattr(self.local_cfg, "hsic_apply_last_layer_only", False):
                        is_last = layer_idx == len(layer_records) - 1
                        if not is_last:
                            hsic_warm = 0.0
                    if self.local_cfg.hsic_self_weight != 0.0:
                        grad_z_self = self._compute_hsic_gradient(
                            z, z, self.local_cfg.hsic_self_weight * hsic_warm, "self"
                        )
                        # Optional HSIC grad clipping
                        if getattr(self.local_cfg, "hsic_grad_clip_value", 0.0) > 0:
                            grad_z_self = torch.clamp(
                                grad_z_self,
                                min=-self.local_cfg.hsic_grad_clip_value,
                                max=self.local_cfg.hsic_grad_clip_value,
                            )
                        if exc_layer is not None and rec.get("x_exc") is not None:
                            x_exc = rec["x_exc"]
                            grad_g_self = grad_z_self.t() @ x_exc
                            if self.local_cfg.normalize_by_batch:
                                grad_g_self = grad_g_self / float(batch_size)
                            g_curr = exc_layer.weight().detach()
                            grad_param_self = grad_g_self * g_curr
                            mask = rec.get("exc_mask")
                            if mask is not None:
                                grad_param_self = grad_param_self * mask
                            if exc_layer.pre_w.grad is None:
                                exc_layer.pre_w.grad = grad_param_self.to(
                                    exc_layer.pre_w.dtype
                                )
                            else:
                                exc_layer.pre_w.grad = (
                                    exc_layer.pre_w.grad
                                    + grad_param_self.to(exc_layer.pre_w.dtype)
                                )
                    if (
                        self.local_cfg.hsic_target_weight != 0.0
                        and y_target is not None
                    ):
                        y_centered = y_target - y_target.mean(dim=0, keepdim=True)
                        grad_z_tgt = self._compute_hsic_gradient(
                            z,
                            y_centered,
                            self.local_cfg.hsic_target_weight * hsic_warm,
                            "target",
                        )
                        if getattr(self.local_cfg, "hsic_grad_clip_value", 0.0) > 0:
                            grad_z_tgt = torch.clamp(
                                grad_z_tgt,
                                min=-self.local_cfg.hsic_grad_clip_value,
                                max=self.local_cfg.hsic_grad_clip_value,
                            )
                        if exc_layer is not None and rec.get("x_exc") is not None:
                            x_exc = rec["x_exc"]
                            grad_g_tgt = grad_z_tgt.t() @ x_exc
                            if self.local_cfg.normalize_by_batch:
                                grad_g_tgt = grad_g_tgt / float(batch_size)
                            g_curr = exc_layer.weight().detach()
                            grad_param_tgt = grad_g_tgt * g_curr
                            mask = rec.get("exc_mask")
                            if mask is not None:
                                grad_param_tgt = grad_param_tgt * mask
                            if exc_layer.pre_w.grad is None:
                                exc_layer.pre_w.grad = grad_param_tgt.to(
                                    exc_layer.pre_w.dtype
                                )
                            else:
                                exc_layer.pre_w.grad = (
                                    exc_layer.pre_w.grad
                                    + grad_param_tgt.to(exc_layer.pre_w.dtype)
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
        kernel_type = self.local_cfg.hsic_kernel.lower()

        if kernel_type == "linear":
            # Linear kernel: K(x, y) = x^T y
            return X @ Y.t()
        elif kernel_type == "rbf" or kernel_type == "gaussian":
            # RBF kernel: K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
            gamma = (
                1.0 / (2.0 * self.local_cfg.hsic_sigma**2)
                if self.local_cfg.hsic_sigma > 0
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
                self.local_cfg.hsic_coef0 + dot_product
            ) ** self.local_cfg.hsic_degree
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
        kernel_type = self.local_cfg.hsic_kernel.lower()

        if kernel_type == "linear":
            # Linear kernel gradients (original implementation)
            if grad_type == "self":
                return (2.0 / float(batch_size)) * z * weight
            else:  # target
                cov_zy = (z.t() @ y) / float(batch_size)
                grad_z = -2.0 * (cov_zy @ y.t()).t() / float(batch_size)
                return grad_z * self.local_cfg.hsic_weight * weight

        elif kernel_type in ["rbf", "gaussian"]:
            # RBF kernel gradients
            gamma = (
                1.0 / (2.0 * self.local_cfg.hsic_sigma**2)
                if self.local_cfg.hsic_sigma > 0
                else 1.0
            )
            if grad_type == "self":
                # Self-dependence: encourage diversity (negative correlation)
                K_zz = self._compute_kernel_matrix(z, z)  # [B, B]
                z_expanded = z.unsqueeze(1)  # [B, 1, d]
                z_diff = z_expanded - z.unsqueeze(0)  # [B, B, d]
                K_expanded = K_zz.unsqueeze(-1)  # [B, B, 1]
                grad_z = -2.0 * gamma * (K_expanded * z_diff).sum(dim=1)  # [B, d]
                return grad_z * weight / float(batch_size)
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
                return grad_z * self.local_cfg.hsic_weight * weight / float(batch_size)

        elif kernel_type in ["polynomial", "poly"]:
            # Polynomial kernel gradients
            degree = self.local_cfg.hsic_degree
            coef0 = self.local_cfg.hsic_coef0
            if grad_type == "self":
                # Self-dependence gradient
                dot_zz = z @ z.t()  # [B, B]
                poly_base = coef0 + dot_zz  # [B, B]
                poly_grad_coef = degree * (poly_base ** (degree - 1))  # [B, B]
                grad_z = (poly_grad_coef @ z) * 2.0  # [B, d] (factor 2 for symmetry)
                return grad_z * weight / float(batch_size)
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
                return grad_z * self.local_cfg.hsic_weight * weight / float(batch_size)

        else:
            # Fallback to linear
            if grad_type == "self":
                return (2.0 / float(batch_size)) * z * weight
            else:
                cov_zy = (z.t() @ y) / float(batch_size)
                grad_z = -2.0 * (cov_zy @ y.t()).t() / float(batch_size)
                return grad_z * self.local_cfg.hsic_weight * weight

    # ----------------------- Helper: rho and phi factors --------------------
    def _compute_layer_rho(self, rec: dict[str, Any], v0: torch.Tensor) -> float:
        layer_id = id(rec.get("layer"))
        stats = self._layer_stats.setdefault(layer_id, _LayerStats())
        v_n: torch.Tensor = rec.get("v_n")
        if v_n is None:
            return 1.0

        batch_size = v_n.size(0)
        alpha = self.local_cfg.ema_alpha  # Learning rate for EMA (e.g., 0.05)
        decay = 1.0 - alpha  # Decay factor

        # Compute per-sample average voltages (over neurons)
        a = v_n.mean(dim=1)  # [B]
        b = v0.mean(dim=1)  # [B]
        # Optional alternative: dot-product-like proxy that works with B=1
        rho_mode = getattr(self.local_cfg, "rho_mode", "pearson").lower()
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
            return float(stats.rho_ema)

        else:
            # Single-sample mode: Use configured estimator
            estimator = self.local_cfg.rho_estimator.lower()

            if estimator in ["ema", "ema+augment"]:
                # Proper online EMA with running means (Welford-like)
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
                # TODO: Implement micro-ensemble augmentation (K noisy forwards)
                logger.warning(
                    "Augment estimator not yet implemented; using EMA fallback."
                )
                # Recursive call to EMA path by changing estimator temporarily
                old_estimator = self.local_cfg.rho_estimator
                self.local_cfg.rho_estimator = "ema"
                result = self._compute_layer_rho(rec, v0)
                self.local_cfg.rho_estimator = old_estimator
                return result

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
        alpha = self.local_cfg.ema_alpha
        if stats.var_ema is None:
            stats.var_ema = var_batch
        else:
            stats.var_ema = (1 - alpha) * stats.var_ema + alpha * var_batch
        phi = var_batch / (stats.var_ema + 1e-8)
        phi = max(0.25, min(4.0, phi))
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
        if v_n.size(0) < 2 and self.local_cfg.phi_estimator == "conditional_ema":
            return self._compute_layer_phi_conditional_ema(rec, v_n, parent)
        # Center variables
        y = v_n - v_n.mean(dim=0, keepdim=True)  # [B, out]
        x = parent - parent.mean(dim=0, keepdim=True)  # [B, out]

        # RLS-like EMA update for single-sample friendly conditional variance
        lam = self.local_cfg.phi_ridge_lambda
        alpha = 1.0 - self.local_cfg.ema_alpha  # decay for moments
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
        return float(max(0.25, min(4.0, phi)))

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

        alpha = self.local_cfg.ema_alpha
        decay = 1.0 - alpha
        lam = self.local_cfg.phi_ridge_lambda

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
        return float(max(0.25, min(4.0, phi)))

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
        train_losses, valid_losses = self._run_training_loop(
            n_epochs=self.epochs,
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            train_losses=train_losses,
            valid_losses=valid_losses,
            logger_info=self.logger_info_prefix,
            wandb_info={},
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
