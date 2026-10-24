"""Path propagation and transport helpers for local credit assignment."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import torch

from dendritic_modeling.training.strategies.local_learning_parts.local_learning_broadcast import (
    additive_zscore_vjp,
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


def _require_vector_soma_seed(mode: str, soma_v: Any, delta: torch.Tensor) -> None:
    """Fail when the readout error cannot seed the soma vector of the tree."""
    width = int(delta.reshape(delta.size(0), -1).size(1)) if delta.dim() > 1 else 1
    if not isinstance(soma_v, torch.Tensor) or int(soma_v.size(1)) != width:
        soma_width = int(soma_v.size(1)) if isinstance(soma_v, torch.Tensor) else None
        raise ValueError(
            f"error_broadcast_mode={mode!r} seeds the last recorded soma with the "
            f"readout error, but its width ({soma_width}) differs from the error "
            f"width ({width}); the mode needs one dendritic tree whose soma feeds "
            "the readout. Use per_soma_shared or scalar for this model."
        )


def _reject_scalar_fallback(option: str, layer_idx: int) -> None:
    """Fail when a recorded layer has no parent block to transport through."""
    raise ValueError(
        f"{option} follows the path from each compartment to one soma, but "
        f"recorded layer {layer_idx} has no parent branch block (the model has "
        "several trees, explicit inhibitory cells or populations), so it would "
        "silently fall back to a scalar. Use a single-tree model, or "
        "error_broadcast_mode per_soma_shared or scalar, or set "
        "learning_strategy_config.allow_scalar_fallback: true to use the scalar "
        "error where no path exists."
    )


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


def _path_branch_factor(
    *,
    parent_path: torch.Tensor,
    parent_act_deriv: torch.Tensor,
    parent_r_tot: torch.Tensor,
    edge_weights: torch.Tensor,
    block_size: int,
    child_out_features: int,
    mode: str,
) -> torch.Tensor:
    """Compute per-child path propagation factors for a parent-child edge."""
    expanded_parent_path = expand_parent_signal_to_children(
        parent_path, block_size, child_out_features
    )
    expanded_parent_act = expand_parent_signal_to_children(
        parent_act_deriv, block_size, child_out_features
    )
    expanded_parent_r_tot = expand_parent_signal_to_children(
        parent_r_tot, block_size, child_out_features
    )
    branch_factor = (
        expanded_parent_path
        * expanded_parent_act
        * expanded_parent_r_tot
        * edge_weights.reshape(1, -1)
    )
    if mode == "scalar_mean":
        return branch_factor.mean(dim=1, keepdim=True)
    return branch_factor


def _transport_child_error(
    *,
    parent_error: torch.Tensor,
    parent_act_deriv: torch.Tensor,
    parent_r_tot: torch.Tensor,
    edge_weights: torch.Tensor,
    block_size: int,
    child_v: torch.Tensor,
) -> torch.Tensor:
    """Transport parent activation error through a parent-child edge."""
    child_dtype = child_v.dtype
    child_device = child_v.device
    expanded_parent_error = expand_parent_signal_to_children(
        (
            parent_error.to(device=child_device, dtype=child_dtype)
            * parent_act_deriv.to(device=child_device, dtype=child_dtype)
        ),
        block_size,
        child_v.size(1),
    )
    expanded_parent_r_tot = expand_parent_signal_to_children(
        parent_r_tot.to(device=child_device, dtype=child_dtype),
        block_size,
        child_v.size(1),
    )
    return expanded_parent_error * expanded_parent_r_tot * edge_weights.reshape(1, -1)


def _feedback_child_seed(
    *,
    parent_feedback: torch.Tensor,
    edge_weights: torch.Tensor,
    block_size: int,
    child_v: torch.Tensor,
) -> torch.Tensor:
    """Transport parent feedback seed through a parent-child edge."""
    expanded_parent = expand_parent_signal_to_children(
        parent_feedback.to(device=child_v.device, dtype=child_v.dtype),
        block_size,
        child_v.size(1),
    )
    return expanded_parent * edge_weights.reshape(1, -1)


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


def _iter_distal_parent_records(
    layer_records: list[dict[str, Any]],
) -> Iterator[tuple[int, dict[str, Any], dict[str, Any]]]:
    """Yield recorded child/parent pairs from distal layers toward the soma."""

    for layer_idx in range(len(layer_records) - 2, -1, -1):
        yield layer_idx, layer_records[layer_idx], layer_records[layer_idx + 1]


def _project_transported_errors(
    layer_records: list[dict[str, Any]],
    transported: list[torch.Tensor | None],
    projection: str,
) -> list[torch.Tensor | None]:
    """Replace each nonsomatic transported error by the mean of its group.

    Records run from the most distal layer to the soma, and child ``c`` of a
    parent block of size ``s`` belongs to parent ``c // s``. ``arbor_mean``
    groups all nonsomatic compartments of one neuron; ``subtree_mean`` groups
    each child of the soma with its descendants. Replacing errors by group
    means is the Euclidean projection onto the groups' indicator vectors, so
    each group keeps its summed error. The soma error is returned unchanged.
    """
    option = f"path_transport_projection={projection!r}"
    depth = len(layer_records)
    if depth < 2:
        raise ValueError(
            f"{option} averages errors over dendritic compartments, but the "
            "readout tree records only its soma"
        )
    for layer_idx in range(depth - 1):
        child = transported[layer_idx]
        parent = transported[layer_idx + 1]
        blk_module = layer_records[layer_idx + 1].get("blk_module")
        if not (
            isinstance(child, torch.Tensor)
            and isinstance(parent, torch.Tensor)
            and blk_module is not None
        ):
            raise ValueError(
                f"{option} needs the exact transported error at every "
                f"compartment, but recorded layer {layer_idx} has none"
            )
        if child.size(1) != parent.size(1) * _block_size(blk_module):
            raise ValueError(
                f"{option} needs {_block_size(blk_module)} children per parent "
                f"compartment, but recorded layer {layer_idx} has {child.size(1)} "
                f"compartments under {parent.size(1)} parents"
            )
    root = depth - 1 if projection == "arbor_mean" else depth - 2
    n_groups = transported[root].size(1)
    total = None
    count = 0
    for layer_idx in range(depth - 1):
        error = transported[layer_idx]
        members = error.size(1) // n_groups
        summed = error.reshape(error.size(0), n_groups, members).sum(dim=-1)
        total = summed if total is None else total + summed
        count += members
    mean = total / count
    projected = list(transported)
    for layer_idx in range(depth - 1):
        width = transported[layer_idx].size(1)
        projected[layer_idx] = (
            mean.unsqueeze(-1)
            .expand(-1, n_groups, width // n_groups)
            .reshape(-1, width)
        )
    return projected


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
        """Return the parent voltage gain used by exact path transport.

        Conductance dynamics contribute the input resistance ``1 / G_tot``.
        Raw additive dynamics have unit voltage gain and must not inherit the
        conductance denominator merely because their records expose the same
        excitatory, inhibitory, and coupling fields.
        """

        resolve_mode = getattr(self, "_resolve_layer_dynamics_mode", None)
        if callable(resolve_mode) and resolve_mode(parent_rec) != "conductance":
            return torch.ones_like(parent_v)

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

        path_factors[-1] = _default_path_factor(soma_v, mode)

        for layer_idx, child_rec, parent_rec in _iter_distal_parent_records(
            layer_records
        ):

            child_v = child_rec.get("v_n")

            parent_v = parent_rec.get("v_n")

            blk_module = parent_rec.get("blk_module")

            if not (
                isinstance(child_v, torch.Tensor)
                and isinstance(parent_v, torch.Tensor)
                and blk_module is not None
                and hasattr(blk_module, "weight")
            ):

                if not getattr(self.local_cfg, "allow_scalar_fallback", False):
                    _reject_scalar_fallback(
                        "morphology_aware.use_path_propagation", layer_idx
                    )
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

            path_factors[layer_idx] = _path_branch_factor(
                parent_path=parent_path,
                parent_act_deriv=parent_act_deriv,
                parent_r_tot=parent_r_tot,
                edge_weights=edge_weights,
                block_size=block_size,
                child_out_features=child_v.size(1),
                mode=mode,
            )

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
        if not getattr(self.local_cfg, "allow_scalar_fallback", False):
            _require_vector_soma_seed("path_transport", soma_v, delta)

        transported[-1] = _soma_seed_from_delta(soma_v, delta, delta_scalar)

        for layer_idx, child_rec, parent_rec in _iter_distal_parent_records(
            layer_records
        ):

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

                if not getattr(self.local_cfg, "allow_scalar_fallback", False):
                    _reject_scalar_fallback(
                        "error_broadcast_mode='path_transport'", layer_idx
                    )
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

            scale = parent_rec.get("additive_scale")
            if scale is not None:
                # Through a z-scored additive parent the voltage error couples
                # the parent's outputs (additive_gain_mode: normalization_jacobian).
                parent_error = additive_zscore_vjp(
                    parent_error.to(device=parent_v.device, dtype=parent_v.dtype)
                    * parent_act_deriv,
                    parent_v,
                    scale,
                    parent_rec["additive_epsilon"],
                )
                parent_act_deriv = torch.ones_like(parent_v)

            transported[layer_idx] = _transport_child_error(
                parent_error=parent_error,
                parent_act_deriv=parent_act_deriv,
                parent_r_tot=parent_r_tot,
                edge_weights=edge_weights,
                block_size=block_size,
                child_v=child_v,
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
        if not getattr(self.local_cfg, "allow_scalar_fallback", False):
            _require_vector_soma_seed("pathway_vector", soma_v, delta)

        feedback[-1] = _soma_seed_from_delta(soma_v, delta, delta_scalar)

        for layer_idx, child_rec, parent_rec in _iter_distal_parent_records(
            layer_records
        ):

            child_v = child_rec.get("v_n")

            parent_feedback = feedback[layer_idx + 1]

            blk_module = parent_rec.get("blk_module")

            if not (
                isinstance(child_v, torch.Tensor)
                and isinstance(parent_feedback, torch.Tensor)
                and blk_module is not None
                and hasattr(blk_module, "weight")
            ):

                if not getattr(self.local_cfg, "allow_scalar_fallback", False):
                    _reject_scalar_fallback(
                        "error_broadcast_mode='pathway_vector'", layer_idx
                    )
                if isinstance(child_v, torch.Tensor):
                    feedback[layer_idx] = _expand_scalar_like(child_v, delta_scalar)
                continue

            edge_weights = _edge_weights_like_child(blk_module, child_v)

            feedback[layer_idx] = _feedback_child_seed(
                parent_feedback=parent_feedback,
                edge_weights=edge_weights,
                block_size=_block_size(blk_module),
                child_v=child_v,
            )

        return feedback


__all__ = [
    "LocalLearningPathTransportBroadcastMixin",
    "_align_parent_path",
    "_block_size",
    "_default_path_factor",
    "_edge_weights_like_child",
    "_expand_scalar_like",
    "_expected_child_out_features",
    "_feedback_child_seed",
    "_iter_distal_parent_records",
    "_path_branch_factor",
    "_project_transported_errors",
    "_scalar_edge_mean_factor",
    "_soma_seed_from_delta",
    "_transport_child_error",
    "_warn_child_out_mismatch",
]
