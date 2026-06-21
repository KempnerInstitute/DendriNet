"""Mixin methods for :mod:`dendritic_modeling.training.strategies.local_learning`.

This module is a mechanical split of ``LocalCreditAssignment`` helpers. Method
bodies are kept behavior-compatible with the original trainer; the main trainer
class owns orchestration and mutable state.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.training.strategies.local_learning_pathways import (
    STDP_TOPK_PATHS,
)
from dendritic_modeling.training.strategies.local_learning_stdp import (
    stdp_activity,
    stdp_decay,
    stdp_enabled,
    stdp_pathway_enabled,
)

logger = logging.getLogger(__name__)


class LocalLearningSTDPMixin:
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

        for path in STDP_TOPK_PATHS:

            layer = rec.get(path.module_key)

            pre = rec.get(path.input_key)

            if isinstance(layer, TopKLinear) and isinstance(pre, torch.Tensor):

                self._apply_stdp_gradient(
                    layer=layer,
                    pre=pre,
                    post=post_signal,
                    mask=rec.get(path.mask_key),
                    pathway=path.pathway,
                    error_signal=error_signal,
                )


__all__ = ["LocalLearningSTDPMixin"]
