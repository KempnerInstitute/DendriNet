"""Signal construction helpers for local credit assignment."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks.activations.parametric import (
    ParametricTanh,
    ParametricTanhOnlyM,
)
from dendritic_modeling.training.strategies.local_learning_hsic import (
    compute_additive_pseudo_signals,
)
from dendritic_modeling.training.strategies.local_learning_state import _LayerStats

logger = logging.getLogger(__name__)


class LocalLearningSignalMixin:
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

    def _compute_reactivation_derivative(
        self,
        react_module: nn.Module | None,
        v_n: torch.Tensor,
        v_out: torch.Tensor | None = None,
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
        v_n: torch.Tensor | None = None,
        v_out: torch.Tensor | None = None,
    ) -> torch.Tensor | None:

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
        """Compute pseudo-R_tot and pseudo-driving-force for additive networks."""

        return compute_additive_pseudo_signals(rec, v_n)

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


__all__ = ["LocalLearningSignalMixin"]
