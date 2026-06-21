"""HSIC auxiliary-gradient methods for local credit assignment."""

from __future__ import annotations

from typing import Any

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)


class LocalLearningHSICMixin:
    """Stateful HSIC gradient helpers used by ``LocalCreditAssignment``."""

    def _apply_hsic_auxiliary_gradients(
        self,
        *,
        rec: dict[str, Any],
        exc_layer: TopKLinear | None,
        layer_idx: int,
        num_layers: int,
        y_target: torch.Tensor | None,
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


__all__ = ["LocalLearningHSICMixin"]
