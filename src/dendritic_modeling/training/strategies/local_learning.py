"""
Local learning training strategies.

This module implements multi-factor local learning rules for dendritic networks.

It provides biologically-inspired 3-/4-/5-factor Hebbian-like rules that use
locally available signals (pre-synaptic input, compartment voltages) modulated
by a broadcast soma error and optional morphology/information factors.
"""

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as functional

from dendritic_modeling.config.training import LocalRuleConfig
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


class _LayerStats:
    """Holds per-layer running statistics for morphology/info factors."""

    def __init__(self):
        self.rho_ema: Optional[float] = None
        self.var_ema: Optional[float] = None
        # Moments for conditional EMA (single-sample friendly)
        self.ema_var_y: Optional[float] = None
        self.ema_var_x: Optional[float] = None
        self.ema_cov_xy: Optional[float] = None
        # Running means for proper online variance/covariance
        self.ema_mean_y: Optional[float] = None
        self.ema_mean_x: Optional[float] = None
        # Morphology-specific statistics
        self.path_factor: Optional[torch.Tensor] = None  # Path attenuation factor
        self.branch_depths: Optional[torch.Tensor] = None  # Branch depths from soma
        self.branch_types: Optional[torch.Tensor] = (
            None  # Apical (1) vs basal (0) flags
        )


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
        self._warned_decoder_soma_fallback = False
        self.filename_prefix = "local_learning_"
        self.logger_info_prefix = "[Local] "

    @staticmethod
    def _section_to_dict(section: Any) -> dict[str, Any]:
        """Convert a config section object to a plain dict."""
        if section is None:
            return {}
        if isinstance(section, dict):
            return dict(section)
        if hasattr(section, "asdict") and callable(section.asdict):
            return section.asdict()
        if hasattr(section, "__dict__"):
            return {
                key: value
                for key, value in vars(section).items()
                if not key.startswith("_")
            }
        return {}

    def _coerce_legacy_local_rule_config(
        self, config_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """Normalize legacy flat local learning config to nested LocalRuleConfig layout."""
        cfg = dict(config_dict)

        def ensure_section(section_name: str) -> dict[str, Any]:
            section = self._section_to_dict(cfg.get(section_name))
            cfg[section_name] = section
            return section

        legacy_group_fields = {
            "three_factor": [
                "dynamics_mode",
                "use_conductance_scaling",
                "use_driving_force",
                "theta",
                "e_rev_exc",
                "e_rev_inh",
            ],
            "four_factor": [
                "rho_mode",
                "rho_estimator",
                "ema_alpha",
                "layer_wise_rho_scale",
                "augment_k",
                "augment_noise_sigma",
            ],
            "five_factor": [
                "phi_mode",
                "phi_estimator",
                "phi_ridge_lambda",
                "layer_wise_phi_scale",
                "rls_forgetting",
            ],
            "morphology_aware": [
                "use_path_propagation",
                "morphology_modulator_mode",
                "morphology_depth_offset",
                "morphology_centrality_metric",
                "use_dendritic_normalization",
                "use_branch_type_rules",
                "apical_branch_scale",
                "basal_branch_scale",
                "use_branch_length_modulation",
            ],
        }

        for section_name, field_names in legacy_group_fields.items():
            section = ensure_section(section_name)
            for field_name in field_names:
                if field_name in cfg and field_name not in section:
                    section[field_name] = cfg.pop(field_name)

        hsic_section = ensure_section("hsic")
        hsic_field_map = {
            "hsic_enabled": "enabled",
            "hsic_weight": "weight",
            "hsic_self_weight": "self_weight",
            "hsic_target_weight": "target_weight",
            "hsic_target_source": "target_source",
            "hsic_kernel": "kernel",
            "hsic_sigma": "sigma",
            "hsic_degree": "degree",
            "hsic_coef0": "coef0",
            "hsic_grad_clip_value": "grad_clip_value",
            "hsic_warmup_epochs": "warmup_epochs",
            "hsic_apply_last_layer_only": "apply_last_layer_only",
        }
        for legacy_name, nested_name in hsic_field_map.items():
            if legacy_name in cfg and nested_name not in hsic_section:
                hsic_section[nested_name] = cfg.pop(legacy_name)

        # Drop known legacy-only keys that are not part of LocalRuleConfig.
        for legacy_only_key in [
            "rho_value",
            "weight_lr_multiplier",
            "compute_gradient_stats",
        ]:
            cfg.pop(legacy_only_key, None)

        return cfg

    def _build_local_rule_config(self, config_dict: dict) -> LocalRuleConfig:
        """Build LocalRuleConfig from dict with proper nested config handling."""
        from ...config.training import (
            FiveFactorConfig,
            FourFactorConfig,
            HSICConfig,
            MorphologyAwareConfig,
            ThreeFactorConfig,
        )

        cfg_kwargs = self._coerce_legacy_local_rule_config(dict(config_dict))

        # Handle nested configs - convert dicts to proper config objects
        if "three_factor" in cfg_kwargs and isinstance(
            cfg_kwargs["three_factor"], dict
        ):
            cfg_kwargs["three_factor"] = ThreeFactorConfig(**cfg_kwargs["three_factor"])

        if "four_factor" in cfg_kwargs and isinstance(cfg_kwargs["four_factor"], dict):
            cfg_kwargs["four_factor"] = FourFactorConfig(**cfg_kwargs["four_factor"])

        if "five_factor" in cfg_kwargs and isinstance(cfg_kwargs["five_factor"], dict):
            cfg_kwargs["five_factor"] = FiveFactorConfig(**cfg_kwargs["five_factor"])

        if "morphology_aware" in cfg_kwargs and isinstance(
            cfg_kwargs["morphology_aware"], dict
        ):
            cfg_kwargs["morphology_aware"] = MorphologyAwareConfig(
                **cfg_kwargs["morphology_aware"]
            )

        if "hsic" in cfg_kwargs and isinstance(cfg_kwargs["hsic"], dict):
            cfg_kwargs["hsic"] = HSICConfig(**cfg_kwargs["hsic"])

        return LocalRuleConfig(**cfg_kwargs)

    # Override: do not run extra pretrain/final phases here
    def _pretrain(self, *args, **kwargs):  # type: ignore[override]
        return [], []

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
                y_hat=y_hat.detach(),
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
        If decoder is a single linear layer, map output-space error back to core space:
            delta_core = delta_out @ W_dec
        and use the cached decoder input as v0.

        Otherwise fall back to output-space behavior for backward compatibility.
        """
        # Backward-compatible fallback
        v0_local = y_hat
        delta_local = delta_out

        if not hasattr(model, "decoder_network"):
            return v0_local, delta_local

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
            self.local_cfg.morphology_aware.use_path_propagation
            or self.local_cfg.morphology_aware.morphology_modulator_mode != "none"
            or self.local_cfg.morphology_aware.use_branch_type_rules
            or self.local_cfg.morphology_aware.use_branch_length_modulation
        ):
            self._initialize_branch_morphology(layer_records)

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

            layer_dynamics_mode = self._resolve_layer_dynamics_mode(rec)
            if layer_dynamics_mode == "conductance":
                # Conductance scaling: R_tot = 1 / g_tot
                if self.local_cfg.three_factor.use_conductance_scaling:
                    g_tot = torch.ones_like(v_n)
                    exc_out = rec.get("exc_out")
                    if exc_out is not None:
                        g_tot = g_tot + functional.relu(exc_out)
                    inh_out = rec.get("inh_out")
                    if inh_out is not None:
                        g_tot = g_tot + functional.relu(inh_out)
                    blk_module = rec.get("blk_module")
                    if blk_module is not None and hasattr(
                        blk_module, "sum_conductances"
                    ):
                        g_blk = (
                            blk_module.sum_conductances()
                            .detach()[None, :]
                            .expand_as(v_n)
                        )
                        g_tot = g_tot + functional.relu(g_blk)
                    R_tot = 1.0 / (g_tot + 1e-8)
                else:
                    R_tot = 1.0
            else:
                # Additive dendritic integration does not include divisive input resistance.
                R_tot = 1.0

            # === PATH PROPAGATION FACTOR ===
            # Apply path-integrated attenuation if enabled
            if self.local_cfg.morphology_aware.use_path_propagation:
                path_factor = self._compute_path_propagation_factor(
                    layer_idx, rec, layer_records
                )
                rec["path_factor"] = path_factor
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
            branch_scale = 1.0
            if (
                self.local_cfg.morphology_aware.use_branch_type_rules
                or self.local_cfg.morphology_aware.use_branch_length_modulation
            ):
                branch_scale = self._get_branch_type_scale(rec)

            # Post-synaptic modulation per-sample, per-neuron.
            if layer_dynamics_mode == "conductance":
                if self.local_cfg.three_factor.use_driving_force:
                    post_mod = self.local_cfg.three_factor.e_rev_exc - v_n  # [B, out]
                else:
                    post_mod = v_n - self.local_cfg.three_factor.theta
                post_factor = e_n * R_tot * post_mod  # [B, out]
            else:
                # Additive rule: dV/dg_exc = x (no driving-force or R_tot factor).
                post_factor = e_n

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
                exc_chain = self._weight_transform_derivative(
                    exc_layer.pre_w.detach(),
                    getattr(exc_layer, "weight_transform", "exp"),
                )
                grad_param = grad_g * exc_chain
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
                if layer_dynamics_mode == "conductance":
                    if self.local_cfg.three_factor.use_driving_force:
                        inh_post = self.local_cfg.three_factor.e_rev_inh - v_n
                    else:
                        inh_post = self.local_cfg.three_factor.theta - v_n
                    inh_factor = e_n * R_tot * inh_post  # [B, out]
                else:
                    # Additive rule: V = exc + den - inh, so dV/dg_inh = -x.
                    inh_factor = -e_n

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
                inh_chain = self._weight_transform_derivative(
                    inh_layer.pre_w.detach(),
                    getattr(inh_layer, "weight_transform", "exp"),
                )
                grad_param_inh = grad_g_inh * inh_chain
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
                if layer_dynamics_mode == "conductance":
                    # dL/dg_den ≈ E_batch[ e_n * R_tot * (V_j - V_n) * ... ]
                    diff = x_blk - v_n.unsqueeze(-1)  # [B, out, block]
                    # Ensure R_tot is a tensor for broadcasting
                    if isinstance(R_tot, torch.Tensor):
                        R_exp = R_tot.unsqueeze(-1)
                    else:
                        R_exp = torch.ones_like(v_n).unsqueeze(-1) * float(R_tot)
                    den_factor = e_n.unsqueeze(-1) * R_exp * diff
                else:
                    # Additive rule: V = ... + sum_j g_j * V_j, so dV/dg_j = V_j.
                    den_factor = e_n.unsqueeze(-1) * x_blk

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
                if isinstance(branch_scale, torch.Tensor):
                    den_factor = den_factor * branch_scale.unsqueeze(0).unsqueeze(
                        -1
                    )  # [1, out, 1]
                else:
                    den_factor = den_factor * float(branch_scale)

                grad_g_den = den_factor.mean(dim=0)  # [out, block]
                if not self.local_cfg.normalize_by_batch:
                    grad_g_den = grad_g_den * float(batch_size)

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
            return float(stats.rho_ema)

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
        return float(max(0.25, min(4.0, phi)))

    # -------------------- Morphology-Aware Helper Methods --------------------
    def _compute_path_propagation_factor(
        self, layer_idx: int, rec: dict[str, Any], layer_records: list[dict[str, Any]]
    ) -> torch.Tensor:
        """Compute path-integrated propagation factor: ∏_{k=1}^n R_k^tot g_k^den.

        This approximates the full dendritic tree attenuation from soma to layer n.

        Note: For stability and shape consistency across layers with different
        dimensionalities, we return a per-sample scalar path factor [B, 1]
        (sample-wise attenuation), which is then broadcast to the layer size.

        Args:
            layer_idx: Current layer index
            rec: Current layer record
            layer_records: All layer records

        Returns:
            Path factor tensor [B, 1]
        """
        if layer_idx == 0:
            # First layer: no path attenuation
            v_n = rec.get("v_n")
            if v_n is None:
                return 1.0
            # Per-sample scalar
            return torch.ones(v_n.size(0), 1, device=v_n.device, dtype=v_n.dtype)

        # Get previous layer's path factor and conductances
        prev_rec = layer_records[layer_idx - 1]
        prev_path = prev_rec.get("path_factor", 1.0)

        # Compute R_tot for previous layer
        v_prev = prev_rec.get("v_n")
        if v_prev is None:
            return 1.0

        g_tot_prev = torch.ones_like(v_prev)
        if prev_rec.get("exc_out") is not None:
            g_tot_prev = g_tot_prev + functional.relu(prev_rec["exc_out"])
        if prev_rec.get("inh_out") is not None:
            g_tot_prev = g_tot_prev + functional.relu(prev_rec["inh_out"])

        blk_module = prev_rec.get("blk_module")
        if blk_module is not None and hasattr(blk_module, "sum_conductances"):
            g_blk = blk_module.sum_conductances().detach()[None, :].expand_as(v_prev)
            g_tot_prev = g_tot_prev + functional.relu(g_blk)

        R_tot_prev = 1.0 / (g_tot_prev + 1e-8)

        # Dendritic conductance (approximate as mean of block conductances)
        if blk_module is not None and hasattr(blk_module, "sum_conductances"):
            g_den = blk_module.sum_conductances().detach().mean()
        else:
            g_den = 1.0

        # Update path factor: multiply by R_prev * g_den
        if isinstance(prev_path, torch.Tensor):
            # prev_path is expected to be [B, 1]
            if prev_path.dim() == 2 and prev_path.size(1) == 1:
                # Reduce R_tot_prev over neurons to a per-sample scalar and multiply
                r_scalar = R_tot_prev.mean(dim=1, keepdim=True)
                path_factor = prev_path * r_scalar * g_den
            else:
                # Fallback: reduce everything to per-sample scalar
                r_scalar = R_tot_prev.mean(dim=1, keepdim=True)
                path_factor = r_scalar * g_den
        else:
            # No previous path: start with per-sample scalar from previous layer
            r_scalar = R_tot_prev.mean(dim=1, keepdim=True)
            path_factor = r_scalar * g_den

        # Ensure [B, 1]
        if path_factor.dim() == 1:
            path_factor = path_factor.view(-1, 1)
        elif path_factor.dim() == 2 and path_factor.size(1) != 1:
            path_factor = path_factor.mean(dim=1, keepdim=True)

        return path_factor

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

    def _get_branch_type_scale(self, rec: dict[str, Any]) -> torch.Tensor:
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
        self, layer_records: list[dict[str, Any]]
    ) -> None:
        """Initialize branch morphology information (depths, types).

        This is a placeholder for future integration with actual dendritic tree structure.
        Currently assigns heuristic values based on layer topology.

        Args:
            layer_records: All layer records
        """
        for layer_idx, rec in enumerate(layer_records):
            layer_id = id(rec.get("layer"))
            stats = self._layer_stats.setdefault(layer_id, _LayerStats())
            v_n = rec.get("v_n")
            if v_n is None:
                continue

            out_features = v_n.size(1)

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
