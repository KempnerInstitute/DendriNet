"""Mixin methods for :mod:`dendritic_modeling.training.strategies.local_learning`.

This module is a mechanical split of ``LocalCreditAssignment`` helpers. Method
bodies are kept behavior-compatible with the original trainer; the main trainer
class owns orchestration and mutable state.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.training.strategies.local_learning_broadcast import (
    apply_broadcast_bandwidth,
    compute_layer_total_conductance,
    compute_pathway_activity,
    compute_role_selectivity,
    effective_topk_weight,
    expand_parent_signal_to_children,
    normalize_role_mass,
    resolve_recent_role_profile,
    resolve_router_output_roles,
    role_mass_from_weight_matrix,
)
from dendritic_modeling.training.strategies.local_learning_state import (
    _BroadcastState,
    _LayerStats,
)

logger = logging.getLogger(__name__)


class LocalLearningBroadcastMixin:
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
        parent: torch.Tensor | None,
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

    def _apply_broadcast_bandwidth(
        self, e_n: torch.Tensor, bandwidth: str
    ) -> torch.Tensor:
        """Apply bandwidth reduction to the broadcast error signal."""

        return apply_broadcast_bandwidth(self.local_cfg, e_n, bandwidth)

    @staticmethod
    def _compute_layer_total_conductance(
        rec: dict[str, Any], v_n: torch.Tensor
    ) -> torch.Tensor:
        """Compute total conductance used by conductance-aware local rules."""

        return compute_layer_total_conductance(rec, v_n)

    def _prepare_local_rule_broadcast_state(
        self,
        *,
        model: BaseModel,
        layer_records: list[dict[str, Any]],
        delta: torch.Tensor,
        delta_scalar: torch.Tensor,
    ) -> _BroadcastState:
        """Prepare per-batch broadcast state shared across local-rule layers."""

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

        transported_errors: list[torch.Tensor | None] = []
        if bmode == "path_transport":
            transported_errors = self._precompute_path_transport_errors(
                layer_records=layer_records,
                delta=delta,
                delta_scalar=delta_scalar,
            )

        feedback_seeds: list[torch.Tensor | None] = []
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
        """Compute the compartment-level local broadcast error for one layer."""

        bmode = broadcast_state.mode
        if bmode == "per_soma" and delta.dim() == 2 and delta.size(1) == out_features:
            e_n = delta
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
            e_n = delta_scalar.expand(-1, out_features)

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
        """Compute conductance-mode input resistance for local-rule updates."""

        if layer_dynamics_mode != "conductance":
            return 1.0
        if not self.local_cfg.three_factor.use_conductance_scaling:
            return 1.0
        g_tot = self._compute_layer_total_conductance(rec, v_n)
        return 1.0 / (g_tot + 1e-8)

    @staticmethod
    def _expand_parent_signal_to_children(
        signal: torch.Tensor, block_size: int, child_out_features: int
    ) -> torch.Tensor:
        """Broadcast a parent-layer signal to its grouped child branches."""

        return expand_parent_signal_to_children(signal, block_size, child_out_features)

    @staticmethod
    def _normalize_role_mass(
        role_mass: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        """Normalize non-negative role mass into a role profile."""

        return normalize_role_mass(role_mass, eps)

    @staticmethod
    def _compute_role_selectivity(
        role_profile: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        """Measure pathway selectivity from normalized role profiles."""

        return compute_role_selectivity(role_profile, eps)

    @staticmethod
    def _resolve_recent_role_profile(
        input_dim: int,
        base_input_roles: torch.Tensor | None,
        prior_output_roles: list[torch.Tensor],
    ) -> torch.Tensor | None:
        """Resolve the most plausible role basis for a layer input size."""

        return resolve_recent_role_profile(
            input_dim, base_input_roles, prior_output_roles
        )

    @staticmethod
    def _effective_topk_weight(
        layer: TopKLinear, mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Return masked effective synaptic weights."""

        return effective_topk_weight(layer, mask)

    @staticmethod
    def _role_mass_from_weight_matrix(
        weight_matrix: torch.Tensor,
        input_roles: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Aggregate pathway-role mass through a synaptic weight matrix."""

        return role_mass_from_weight_matrix(weight_matrix, input_roles)

    def _resolve_router_output_roles(self, model: BaseModel) -> torch.Tensor | None:
        """Return pathway ownership for encoder outputs when available."""

        return resolve_router_output_roles(self.local_cfg, model)

    def _compute_pathway_activity(
        self,
        rec: dict[str, Any],
        stats: _LayerStats,
        v_n: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor | None:
        """Compute a sample-wise local pathway activity gate."""

        return compute_pathway_activity(rec, stats, v_n, eps)

    def _compute_pathway_vector_broadcast(
        self,
        seed_error: torch.Tensor | None,
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
    ) -> list[torch.Tensor | None]:
        """Precompute hierarchical feedback vectors from proximal to distal layers."""

        if not layer_records:

            return []

        feedback: list[torch.Tensor | None] = [None] * len(layer_records)

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


__all__ = ["LocalLearningBroadcastMixin"]
