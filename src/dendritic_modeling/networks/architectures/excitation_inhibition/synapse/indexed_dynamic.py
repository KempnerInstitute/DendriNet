"""Candidate-pool indexed dynamic TopK synapses."""

from __future__ import annotations

from math import log

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_common import (
    _allowed_mask_with_forbidden,
    _connection_generator,
    _sample_indices_from_mask,
    _scale_for_sparse_target,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TOPK_INIT_METHODS,
)
from dendritic_modeling.networks.utils.weight_transforms import (
    WeightTransformType,
    apply_weight_transform,
    inverse_softplus,
)


class IndexedDynamicTopKLinear(nn.Module):
    """
    Candidate-pool TopK layer with low-parameter dynamic synapse selection.

    ``IndexedSparseLinear`` fixes exactly ``K`` synapses per output.  This
    variant fixes a larger candidate pool of size ``candidate_size`` and
    dynamically selects ``K`` active candidates on each forward pass.  It keeps
    the old TopK idea that inactive candidate synapses can later become active,
    while avoiding a full dense ``out_features x in_features`` parameter matrix.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        K: int,
        *,
        candidate_size: int | None = None,
        selection: str = "standard",
        param_space: str = "log",
        init_method: str = "xavier_normal",
        init_gain: float = 1.0,
        noise_level: float = 0.0,
        temperature: float = 0.5,
        ultrafast: bool = False,
        weight_transform: WeightTransformType = "exp",
        weight_norm_order: int | None = None,
        gamma: float = 1.0,
        connection_indices: torch.Tensor | None = None,
        connection_mask: torch.Tensor | None = None,
        forbidden_input_index_per_output: torch.Tensor | None = None,
        seed: int | None = None,
        output_chunk_size: int = 2048,
        **_: object,
    ):
        super().__init__()

        if not isinstance(K, int):
            K = int(K)
        if K < 1:
            raise ValueError("K must be >= 1")
        if K > in_features:
            raise ValueError(
                f"K must be <= number of input features. "
                f"(K={K}, in_features={in_features})"
            )

        allowed_connection_mask = _allowed_mask_with_forbidden(
            connection_mask=connection_mask,
            forbidden_input_index_per_output=forbidden_input_index_per_output,
            out_features=int(out_features),
            in_features=int(in_features),
        )

        if candidate_size is None:
            candidate_size = min(in_features, max(K, 4 * K))
            if allowed_connection_mask is not None:
                min_allowed = int(allowed_connection_mask.sum(dim=1).min().item())
                candidate_size = min(candidate_size, min_allowed)
        candidate_size = int(candidate_size)
        if candidate_size < K:
            raise ValueError("candidate_size must be >= K")
        if candidate_size > in_features:
            raise ValueError("candidate_size must be <= in_features")
        if output_chunk_size < 1:
            raise ValueError("output_chunk_size must be >= 1")

        selection = str(selection).lower()
        if selection not in {"standard", "stochastic", "rank_probabilistic"}:
            raise ValueError(
                "selection must be one of: standard, stochastic, rank_probabilistic"
            )

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.K = K
        self.candidate_size = candidate_size
        self.selection = (
            "stochastic" if selection == "rank_probabilistic" else selection
        )
        self.param_space = param_space
        self.init_method = init_method
        self.init_gain = init_gain
        self.noise_level = float(noise_level)
        self.temperature = float(temperature)
        self.ultrafast = bool(ultrafast)
        self.weight_transform = weight_transform
        self.weight_norm_order = weight_norm_order
        self.gamma = float(gamma)
        self.output_chunk_size = int(output_chunk_size)
        self.input_scale_vec = None
        self.param_scale_vec = None
        self.cache_mask = False
        self._last_forward_weight_mask: torch.Tensor | None = None
        self._last_forward_active_sparse_mask: torch.Tensor | None = None
        self._last_forward_param_tensor: torch.Tensor | None = None

        if connection_indices is None:
            generator = _connection_generator(seed)
            connection_indices = _sample_indices_from_mask(
                out_features=self.out_features,
                in_features=self.in_features,
                K=self.candidate_size,
                connection_mask=allowed_connection_mask,
                generator=generator,
            )
        else:
            connection_indices = torch.as_tensor(connection_indices, dtype=torch.long)
            if tuple(connection_indices.shape) != (
                self.out_features,
                self.candidate_size,
            ):
                raise ValueError(
                    "connection_indices must have shape "
                    f"({self.out_features}, {self.candidate_size}), got "
                    f"{tuple(connection_indices.shape)}"
                )
            if bool((connection_indices < 0).any()) or bool(
                (connection_indices >= self.in_features).any()
            ):
                raise ValueError(
                    f"connection_indices entries must be in [0, {self.in_features - 1}]"
                )
            sorted_indices = connection_indices.sort(dim=1).values
            if bool((sorted_indices[:, 1:] == sorted_indices[:, :-1]).any()):
                raise ValueError("connection_indices must be unique within each row")
            if allowed_connection_mask is not None:
                row = torch.arange(self.out_features)[:, None]
                if not bool(allowed_connection_mask[row, connection_indices].all()):
                    raise ValueError(
                        "connection_indices contains entries disallowed by "
                        "connection_mask or forbidden_input_index_per_output"
                    )

        self.register_buffer(
            "connection_indices", connection_indices.contiguous(), persistent=True
        )
        self.pre_w = nn.Parameter(torch.empty(self.out_features, self.candidate_size))
        self.initialize()

    def initialize(self) -> None:
        """Initialize all candidate synaptic weights."""
        if self.init_method not in TOPK_INIT_METHODS:
            raise ValueError(
                f"Invalid initialization method: {self.init_method}. "
                f"Choose from {list(TOPK_INIT_METHODS.keys())}"
            )
        TOPK_INIT_METHODS[self.init_method](self.pre_w)
        self.pre_w.data.mul_(self.init_gain)

    def sparse_weight(self) -> torch.Tensor:
        """Return candidate weights with shape ``[out_features, candidate_size]``."""
        return apply_weight_transform(self.pre_w, self.weight_transform)

    def _scores_for_selection(self, *, transformed: bool = False) -> torch.Tensor:
        scores = self.sparse_weight() if transformed else self.pre_w
        if self.weight_transform == "identity":
            scores = scores.abs()
        return scores

    def _standard_candidate_mask(self, scores: torch.Tensor) -> torch.Tensor:
        topk_indices = torch.topk(scores, self.K, dim=-1, largest=True, sorted=False)[1]
        mask = torch.zeros_like(self.pre_w)
        row = torch.arange(self.out_features, device=self.pre_w.device)[:, None]
        mask[row, topk_indices] = 1
        return mask

    def _stochastic_candidate_mask(self) -> torch.Tensor:
        if self.temperature <= 0.0:
            return self._standard_candidate_mask(self._scores_for_selection())

        if self.ultrafast:
            scores = self._scores_for_selection()
            noise_scale = max(0.01, self.temperature) * (1.0 + self.noise_level)
            scores = scores + torch.randn_like(scores) * noise_scale
            return self._standard_candidate_mask(scores)

        scores = self._scores_for_selection(transformed=True)
        if self.noise_level > 0:
            scores = scores + torch.randn_like(scores) * self.noise_level

        _, sort_indices = torch.sort(scores, dim=1, descending=True)
        arange_indices = torch.arange(self.candidate_size, device=scores.device).expand(
            self.out_features, -1
        )
        ranks = torch.zeros_like(scores, dtype=arange_indices.dtype)
        row = torch.arange(self.out_features, device=scores.device)[:, None]
        ranks[row, sort_indices] = arange_indices
        ranks = ranks.to(torch.float)

        temperature = max(0.01, self.temperature)
        probs = torch.sigmoid((self.K - ranks) / (self.K * temperature))
        mask = torch.bernoulli(probs)

        counts = mask.sum(dim=1)
        too_few = counts < (self.K // 2)
        if bool(too_few.any()):
            for row_idx in too_few.nonzero(as_tuple=False).view(-1):
                top_indices = torch.topk(scores[row_idx], self.K, largest=True)[1]
                mask[row_idx, top_indices] = 1

        counts = mask.sum(dim=1)
        too_many = counts > (2 * self.K)
        if bool(too_many.any()):
            for row_idx in too_many.nonzero(as_tuple=False).view(-1):
                active = mask[row_idx].nonzero(as_tuple=True)[0]
                active_scores = scores[row_idx, active]
                _, order = torch.sort(active_scores)
                to_remove = int(counts[row_idx].item() - self.K)
                if to_remove > 0:
                    mask[row_idx, active[order[:to_remove]]] = 0

        counts = mask.sum(dim=1)
        zero = counts == 0
        if bool(zero.any()):
            for row_idx in zero.nonzero(as_tuple=False).view(-1):
                mask[row_idx, torch.argmax(scores[row_idx])] = 1
        return mask

    def active_sparse_mask(self) -> torch.Tensor:
        """Return active candidate mask with shape ``[out_features, candidate_size]``."""
        if self.selection == "stochastic":
            return self._stochastic_candidate_mask()
        scores = self._scores_for_selection()
        if self.noise_level > 0:
            scores = scores + torch.randn_like(scores) * self.noise_level
        return self._standard_candidate_mask(scores)

    def _normalized_sparse_weight(self) -> torch.Tensor:
        active_mask = self.active_sparse_mask()
        self._last_forward_active_sparse_mask = (
            active_mask.detach() if self.cache_mask else None
        )
        active_weight = self.sparse_weight() * active_mask
        if self.weight_norm_order is None:
            return active_weight
        norm = torch.norm(active_weight, p=self.weight_norm_order, dim=-1, keepdim=True)
        safe_norm = norm.clamp_min(torch.finfo(active_weight.dtype).eps)
        normalized = (active_weight / safe_norm) * self.gamma
        return torch.where(norm > 0, normalized, active_weight)

    def candidate_mask(self) -> torch.Tensor:
        """Return dense mask of all stored candidate synapses."""
        mask = torch.zeros(
            self.out_features,
            self.in_features,
            device=self.pre_w.device,
            dtype=self.pre_w.dtype,
        )
        values = torch.ones_like(self.pre_w, dtype=mask.dtype)
        mask.scatter_add_(1, self.connection_indices.to(mask.device), values)
        return mask.clamp_max(1)

    def weight(self) -> torch.Tensor:
        """Return dense candidate weights; non-candidates are zero."""
        dense = torch.zeros(
            self.out_features,
            self.in_features,
            device=self.pre_w.device,
            dtype=self.pre_w.dtype,
        )
        dense.scatter_add_(
            1, self.connection_indices.to(dense.device), self.sparse_weight()
        )
        return dense

    def log_weight(self) -> torch.Tensor:
        """Return log candidate weights in dense compatibility form."""
        dense = self.weight()
        if self.weight_transform == "identity":
            return (dense.abs() + 1e-8).log()
        return (dense + 1e-8).log()

    def weight_mask(self) -> torch.Tensor:
        """Return dense active TopK mask within the fixed candidate pool."""
        sparse_mask = self.active_sparse_mask()
        dense = torch.zeros(
            self.out_features,
            self.in_features,
            device=self.pre_w.device,
            dtype=self.pre_w.dtype,
        )
        dense.scatter_add_(1, self.connection_indices.to(dense.device), sparse_mask)
        return dense.clamp_max(1)

    def pruned_weight(self) -> torch.Tensor:
        """Return dense active TopK weights within the fixed candidate pool."""
        dense = torch.zeros(
            self.out_features,
            self.in_features,
            device=self.pre_w.device,
            dtype=self.pre_w.dtype,
        )
        active_weight = self.sparse_weight() * self.active_sparse_mask()
        dense.scatter_add_(1, self.connection_indices.to(dense.device), active_weight)
        if self.weight_norm_order is not None:
            dense_norm = torch.norm(
                dense, p=self.weight_norm_order, dim=-1, keepdim=True
            )
            safe_norm = dense_norm.clamp_min(torch.finfo(dense.dtype).eps)
            dense = torch.where(dense_norm > 0, (dense / safe_norm) * self.gamma, dense)
        return dense

    def log_pruned_weight(self) -> torch.Tensor:
        """Return log active weights with inactive entries shifted down."""
        return ((self.weight_mask() - 1) * 10) + self.log_weight()

    def decay_weights(self, weight_decay: float = 0.1, weight_boosting: bool = False):
        """Decay active candidates and optionally boost inactive candidates."""
        with torch.no_grad():
            active = self.active_sparse_mask()
            inactive = 1 - active
            if self.weight_transform == "exp":
                self.pre_w.data += log(1 - weight_decay) * active
                if weight_boosting:
                    self.pre_w.data += log(1 + weight_decay) * inactive
            elif self.weight_transform == "softplus":
                weight = nn.functional.softplus(self.pre_w.data)
                decayed = torch.where(active > 0, weight * (1 - weight_decay), weight)
                if weight_boosting:
                    decayed = torch.where(
                        inactive > 0, decayed * (1 + weight_decay), decayed
                    )
                update_mask = active > 0
                if weight_boosting:
                    update_mask = update_mask | (inactive > 0)
                updated_pre_w = self.pre_w.data.clone()
                updated_pre_w[update_mask] = inverse_softplus(decayed[update_mask])
                self.pre_w.data = updated_pre_w
            elif self.weight_transform in {"identity", "relu"}:
                scale = torch.where(active > 0, 1 - weight_decay, 1.0)
                if weight_boosting:
                    scale = torch.where(inactive > 0, scale * (1 + weight_decay), scale)
                self.pre_w.data *= scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dynamic TopK over the fixed candidate pool."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input to be a tensor, got {type(x)}")
        squeezed = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeezed = True
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Input feature dimension mismatch: got {x.shape[-1]}, "
                f"expected {self.in_features}"
            )

        original_shape = x.shape
        flat_x = x.reshape(-1, self.in_features)
        out = flat_x.new_empty(flat_x.shape[0], self.out_features)
        active_weight = self._normalized_sparse_weight()
        object.__setattr__(self, "_last_forward_param_tensor", active_weight)
        indices = self.connection_indices.to(flat_x.device)

        for start in range(0, self.out_features, self.output_chunk_size):
            end = min(start + self.output_chunk_size, self.out_features)
            idx_chunk = indices[start:end]
            selected = flat_x[:, idx_chunk.reshape(-1)].reshape(
                flat_x.shape[0], end - start, self.candidate_size
            )
            out[:, start:end] = (selected * active_weight[start:end].unsqueeze(0)).sum(
                dim=-1
            )

        out = out.reshape(*original_shape[:-1], self.out_features)
        if self.cache_mask:
            sparse_mask = self._last_forward_active_sparse_mask
            assert sparse_mask is not None
            dense_mask = torch.zeros(
                self.out_features,
                self.in_features,
                device=sparse_mask.device,
                dtype=sparse_mask.dtype,
            )
            dense_mask.scatter_add_(
                1, self.connection_indices.to(dense_mask.device), sparse_mask
            )
            self._last_forward_weight_mask = dense_mask.clamp_max(1)
        if squeezed:
            return out.squeeze(0)
        return out

    def register_forward_param_gradient_scale(self, param_scale: torch.Tensor) -> None:
        """Scale active sparse-candidate gradients for the most recent forward call."""
        self._use_forward_dynamic_grad_scaling = True
        target = self._last_forward_param_tensor
        if target is None or not target.requires_grad:
            return
        scale = _scale_for_sparse_target(
            param_scale,
            target=target,
            connection_indices=self.connection_indices,
            in_features=self.in_features,
        )
        target.register_hook(lambda grad, scale=scale: grad * scale)


__all__ = ["IndexedDynamicTopKLinear"]
