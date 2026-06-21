"""Path propagation and transport helpers for local credit assignment."""

from __future__ import annotations

import logging
from typing import Any

import torch

from dendritic_modeling.training.strategies.local_learning_broadcast import (
    expand_parent_signal_to_children,
)

logger = logging.getLogger(__name__)


def _expand_scalar_like(reference: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
    """Expand a batch scalar to a reference layer's feature dimension."""

    return scalar.to(device=reference.device, dtype=reference.dtype).expand(
        -1, reference.size(1)
    )


def _edge_weights_like_child(blk_module: Any, child_v: torch.Tensor) -> torch.Tensor:
    """Return detached edge weights on the child tensor's device and dtype."""

    return blk_module.weight().detach().to(device=child_v.device, dtype=child_v.dtype)


def _block_size(blk_module: Any) -> int:
    """Return a BlockLinear-style module's branch block size."""

    return int(getattr(blk_module, "block_size", 1))


def _soma_seed_from_delta(
    soma_v: Any,
    delta: torch.Tensor,
    delta_scalar: torch.Tensor,
) -> torch.Tensor | None:
    """Resolve the soma-level vector or scalar-expanded feedback seed."""

    if not isinstance(soma_v, torch.Tensor):
        return None

    if delta.dim() == 1:
        delta = delta.unsqueeze(-1)
    elif delta.dim() != 2:
        delta = delta.view(delta.size(0), -1)

    if delta.size(1) == soma_v.size(1):
        return delta.to(device=soma_v.device, dtype=soma_v.dtype)

    return _expand_scalar_like(soma_v, delta_scalar)


def _align_parent_path(
    parent_path: torch.Tensor | float,
    parent_v: torch.Tensor,
) -> torch.Tensor:
    """Align a cached parent path factor to parent voltage features."""

    if not isinstance(parent_path, torch.Tensor):
        return torch.ones_like(parent_v)

    if parent_path.size(1) == 1 and parent_v.size(1) > 1:
        return parent_path.expand(-1, parent_v.size(1))

    return parent_path


def _scalar_edge_mean_factor(
    parent_signal: torch.Tensor,
    parent_act_deriv: torch.Tensor,
    parent_r_tot: torch.Tensor,
    edge_weights: torch.Tensor,
) -> torch.Tensor:
    """Return scalar fallback transport through averaged parent and edge terms."""

    return (parent_signal * parent_act_deriv * parent_r_tot).mean(
        dim=1, keepdim=True
    ) * edge_weights.mean()


def _expected_child_out_features(edge_weights: torch.Tensor) -> int:
    """Return the child feature count implied by edge weights."""

    return edge_weights.numel()


def _warn_child_out_mismatch(
    context: str,
    *,
    child_v: torch.Tensor,
    expected_child_out: int,
) -> None:
    """Log a path transport shape mismatch with legacy wording."""

    logger.warning(
        "%s shape mismatch: child_out=%s, expected=%s. "
        "Falling back to scalar averaging.",
        context,
        child_v.size(1),
        expected_child_out,
    )


def _default_path_factor(child_v: Any, mode: str) -> torch.Tensor | float:
    """Return the missing-link path factor with legacy scalar-mode behavior."""

    if isinstance(child_v, torch.Tensor) and mode == "scalar_mean":
        return torch.ones(
            child_v.size(0),
            1,
            device=child_v.device,
            dtype=child_v.dtype,
        )
    if isinstance(child_v, torch.Tensor):
        return torch.ones_like(child_v)
    return 1.0


class LocalLearningPathTransportBroadcastMixin:
    """Recursive transport of local-rule feedback through dendritic trees."""

    @staticmethod
    def _expand_parent_signal_to_children(
        signal: torch.Tensor, block_size: int, child_out_features: int
    ) -> torch.Tensor:
        """Broadcast a parent-layer signal to its grouped child branches."""

        return expand_parent_signal_to_children(signal, block_size, child_out_features)

    def _parent_activation_derivative_or_ones(
        self,
        parent_rec: dict[str, Any],
        parent_v: torch.Tensor,
        *,
        enabled: bool = True,
    ) -> torch.Tensor:
        """Resolve parent activation derivative, falling back to unit slope."""

        if not enabled:
            return torch.ones_like(parent_v)

        parent_act_deriv = self._get_layer_activation_derivative(parent_rec, parent_v)
        if isinstance(parent_act_deriv, torch.Tensor):
            return parent_act_deriv.to(device=parent_v.device, dtype=parent_v.dtype)
        return torch.ones_like(parent_v)

    def _parent_resistance_total(
        self,
        parent_rec: dict[str, Any],
        parent_v: torch.Tensor,
    ) -> torch.Tensor:
        """Return stabilized parent total resistance for path transport."""

        parent_g_tot = self._compute_layer_total_conductance(parent_rec, parent_v)
        return 1.0 / (parent_g_tot + 1e-8)

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

                path_factors[layer_idx] = _default_path_factor(child_v, mode)

                continue

            parent_path = _align_parent_path(path_factors[layer_idx + 1], parent_v)

            parent_act_deriv = self._parent_activation_derivative_or_ones(
                parent_rec,
                parent_v,
                enabled=include_parent_activation_derivative,
            )

            parent_r_tot = self._parent_resistance_total(parent_rec, parent_v)

            block_size = _block_size(blk_module)

            edge_weights = _edge_weights_like_child(blk_module, child_v)

            expected_child_out = _expected_child_out_features(edge_weights)

            if child_v.size(1) != expected_child_out:

                _warn_child_out_mismatch(
                    "Path propagation",
                    child_v=child_v,
                    expected_child_out=expected_child_out,
                )

                scalar_factor = _scalar_edge_mean_factor(
                    parent_path,
                    parent_act_deriv,
                    parent_r_tot,
                    edge_weights,
                )

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
    ) -> list[torch.Tensor | None]:
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

        transported: list[torch.Tensor | None] = [None] * len(layer_records)

        soma_v = layer_records[-1].get("v_n")

        transported[-1] = _soma_seed_from_delta(soma_v, delta, delta_scalar)

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

                    transported[layer_idx] = _expand_scalar_like(child_v, delta_scalar)

                continue

            edge_weights = _edge_weights_like_child(blk_module, child_v)

            expected_child_out = _expected_child_out_features(edge_weights)

            if child_v.size(1) != expected_child_out:

                _warn_child_out_mismatch(
                    "Path transport",
                    child_v=child_v,
                    expected_child_out=expected_child_out,
                )

                parent_act_deriv = self._parent_activation_derivative_or_ones(
                    parent_rec,
                    parent_v,
                )

                parent_r_tot = self._parent_resistance_total(parent_rec, parent_v)

                transported[layer_idx] = (
                    _scalar_edge_mean_factor(
                        parent_error,
                        parent_act_deriv,
                        parent_r_tot,
                        edge_weights,
                    )
                ).expand(-1, child_v.size(1))

                continue

            block_size = _block_size(blk_module)

            parent_act_deriv = self._parent_activation_derivative_or_ones(
                parent_rec,
                parent_v,
            )

            parent_r_tot = self._parent_resistance_total(parent_rec, parent_v)

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
    ) -> list[torch.Tensor | None]:
        """Precompute hierarchical feedback vectors from proximal to distal layers."""

        if not layer_records:

            return []

        feedback: list[torch.Tensor | None] = [None] * len(layer_records)

        soma_v = layer_records[-1].get("v_n")

        feedback[-1] = _soma_seed_from_delta(soma_v, delta, delta_scalar)

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

                    feedback[layer_idx] = _expand_scalar_like(child_v, delta_scalar)

                continue

            edge_weights = _edge_weights_like_child(blk_module, child_v)

            expanded_parent = self._expand_parent_signal_to_children(
                parent_feedback.to(device=child_v.device, dtype=child_v.dtype),
                _block_size(blk_module),
                child_v.size(1),
            )

            feedback[layer_idx] = expanded_parent * edge_weights.reshape(1, -1)

        return feedback


__all__ = [
    "LocalLearningPathTransportBroadcastMixin",
    "_align_parent_path",
    "_block_size",
    "_default_path_factor",
    "_edge_weights_like_child",
    "_expand_scalar_like",
    "_expected_child_out_features",
    "_scalar_edge_mean_factor",
    "_soma_seed_from_delta",
    "_warn_child_out_mismatch",
]
