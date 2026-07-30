"""Fixed-index sparse linear synapses."""

from __future__ import annotations

from math import log

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_common import (
    _connection_generator,
    _indexed_sparse_projection,
    _resolve_index_dtype,
    _sample_indices_from_mask,
    _scale_for_sparse_target,
    _validate_forbidden_indices,
    _workspace_limited_chunk_size,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.structured_mask import (
    validate_connection_mask,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TOPK_INIT_METHODS,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.fused_gather import (
    fused_sparse_gather,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.triton_gather import (
    triton_gather_available,
    triton_sparse_gather,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.cuda_gather import (
    cuda_gather_available,
    cuda_sparse_gather,
)
from dendritic_modeling.networks.utils.weight_transforms import (
    WeightTransformType,
    apply_weight_transform,
    inverse_softplus,
)


class IndexedSparseLinear(nn.Module):
    """
    Sparse linear layer with fixed presynaptic indices and trainable weights.

    The forward pass computes

    .. math::

       y_j = \\sum_{k=1}^{K} w_{j,k} x_{i_{j,k}},

    where ``i[j, k]`` is a fixed integer index.  Unlike dynamic TopK, there is
    no dense trainable matrix.  The morphology/topology is fixed at
    construction time; only the synaptic weights are learned.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        K: int,
        *,
        param_space: str = "log",
        init_method: str = "xavier_normal",
        init_gain: float = 1.0,
        weight_transform: WeightTransformType = "exp",
        weight_norm_order: int | None = None,
        gamma: float = 1.0,
        connection_indices: torch.Tensor | None = None,
        connection_mask: torch.Tensor | None = None,
        forbidden_input_index_per_output: torch.Tensor | None = None,
        seed: int | None = None,
        output_chunk_size: int = 2048,
        use_fused_gather: bool = False,
        use_triton_gather: bool = False,
        use_cuda_gather: bool = False,
        cache_recurrent_weight: bool = False,
        index_dtype: str | torch.dtype = "auto",
        workspace_mb: float | None = None,
        init_mode: str = "per_rank",
        persistent_indices: bool = True,
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
        if output_chunk_size < 1:
            raise ValueError("output_chunk_size must be >= 1")
        if workspace_mb is not None and float(workspace_mb) <= 0:
            raise ValueError("workspace_mb must be > 0 when provided")
        init_mode = str(init_mode).strip().lower()
        if init_mode not in {"per_rank", "rank0_broadcast"}:
            raise ValueError("init_mode must be 'per_rank' or 'rank0_broadcast'")
        if not persistent_indices and connection_indices is None and seed is None:
            raise ValueError(
                "persistent_indices=False requires an explicit seed so the "
                "topology can be regenerated when loading a checkpoint"
            )

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.K = K
        # Assigned before the topology block below, which reads self.index_dtype
        # and self.init_mode. Defaults reproduce the pre-7d24e966 behaviour:
        # "auto" -> int32 whenever in_features fits, workspace_mb None -> no chunk
        # cap, per-rank topology sampling, indices kept in the state dict.
        self.index_dtype = _resolve_index_dtype(
            index_dtype, in_features=int(in_features)
        )
        self.workspace_mb = None if workspace_mb is None else float(workspace_mb)
        self.init_mode = init_mode
        self.persistent_indices = bool(persistent_indices)
        self.param_space = param_space
        self.init_method = init_method
        self.init_gain = init_gain
        self.weight_transform = weight_transform
        self.weight_norm_order = weight_norm_order
        self.gamma = float(gamma)
        self.output_chunk_size = int(output_chunk_size)
        self.use_fused_gather = bool(use_fused_gather)
        self.use_triton_gather = bool(use_triton_gather)
        self.use_cuda_gather = bool(use_cuda_gather)
        self.input_scale_vec = None
        self.param_scale_vec = None
        self._use_forward_dynamic_grad_scaling = False
        self.cache_mask = False
        self._last_forward_weight_mask: torch.Tensor | None = None
        self._last_forward_param_tensor: torch.Tensor | None = None
        # Opt-in hoist of the weight transform out of a recurrent unroll; see
        # ``set_recurrent_weight_cache``. Off by default so behaviour is unchanged
        # unless a config asks for it.
        self.cache_recurrent_weight = bool(cache_recurrent_weight)
        self._recurrent_weight_cache_enabled = False
        self._cached_recurrent_weight: torch.Tensor | None = None

        allowed_connection_mask = validate_connection_mask(
            connection_mask,
            self.out_features,
            self.in_features,
        )
        forbidden = _validate_forbidden_indices(
            forbidden_input_index_per_output,
            out_features=self.out_features,
            in_features=self.in_features,
        )

        topology_initialized = True
        if connection_indices is None:
            rank0_initializes = (
                self.init_mode == "rank0_broadcast"
                and torch.distributed.is_available()
                and torch.distributed.is_initialized()
            )
            topology_initialized = (
                not rank0_initializes or torch.distributed.get_rank() == 0
            )
            if topology_initialized:
                generator = _connection_generator(seed)
                connection_indices = _sample_indices_from_mask(
                    out_features=self.out_features,
                    in_features=self.in_features,
                    K=self.K,
                    connection_mask=allowed_connection_mask,
                    forbidden_input_index_per_output=forbidden,
                    generator=generator,
                    row_chunk_size=self.output_chunk_size,
                    index_dtype=self.index_dtype,
                )
            else:
                # FSDP ``sync_module_states=True`` broadcasts this buffer and
                # the initialization marker from rank 0 before first use.
                connection_indices = torch.zeros(
                    self.out_features,
                    self.K,
                    dtype=self.index_dtype,
                )
        else:
            connection_indices = torch.as_tensor(
                connection_indices, dtype=self.index_dtype
            )
            if tuple(connection_indices.shape) != (self.out_features, self.K):
                raise ValueError(
                    "connection_indices must have shape "
                    f"({self.out_features}, {self.K}), got "
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
                if not bool(
                    allowed_connection_mask[
                        row, connection_indices.to(torch.long)
                    ].all()
                ):
                    raise ValueError(
                        "connection_indices contains entries disallowed by "
                        "connection_mask"
                    )
            if forbidden is not None:
                violates_forbidden = (forbidden[:, None] >= 0) & (
                    connection_indices.to(torch.long) == forbidden[:, None]
                )
                if bool(violates_forbidden.any()):
                    raise ValueError(
                        "connection_indices contains entries disallowed by "
                        "forbidden_input_index_per_output"
                    )

        if allowed_connection_mask is None:
            self.register_buffer(
                "_allowed_connection_mask",
                torch.empty(0, dtype=torch.bool),
                persistent=False,
            )
        else:
            self.register_buffer(
                "_allowed_connection_mask",
                allowed_connection_mask.contiguous(),
                persistent=False,
            )
        # Store connection indices in the smallest safe integer dtype. The index
        # buffer is an FSDP-replicated (non-sharded) buffer, so it is the binding
        # per-GPU memory term at scale (issue #383): int32 halves it vs int64. Safe
        # whenever in_features fits int32 -- index values live in [0, in_features) --
        # and torch indexing / index_add_ / scatter_add_ all accept int32, so the
        # result is bit-identical (same integer values, narrower storage). Rewiring
        # writes in place, inheriting this dtype.
        # Honour self.index_dtype rather than forcing int32: it already resolves
        # to int32 under "auto" whenever in_features fits, so an unconditional
        # downcast here would silently override an explicit index_dtype="int64".
        connection_indices = connection_indices.to(self.index_dtype)
        # Mirrors the `_allowed_connection_mask` sentinel above: this buffer is
        # optional, so store an empty tensor rather than dereferencing None.
        self.register_buffer(
            "_forbidden_input_index_per_output",
            (
                torch.empty(0, dtype=torch.long)
                if forbidden is None
                else forbidden.contiguous()
            ),
            persistent=False,
        )
        self.register_buffer(
            "connection_indices",
            connection_indices.contiguous(),
            persistent=self.persistent_indices,
        )
        self.register_buffer(
            "_topology_initialized",
            torch.tensor(int(topology_initialized), dtype=torch.uint8),
            persistent=False,
        )
        self._topology_ready = topology_initialized
        self.pre_w = nn.Parameter(torch.empty(self.out_features, self.K))
        self.initialize()

    def initialize(self) -> None:
        """Initialize the sparse synaptic weights."""
        if self.init_method not in TOPK_INIT_METHODS:
            raise ValueError(
                f"Invalid initialization method: {self.init_method}. "
                f"Choose from {list(TOPK_INIT_METHODS.keys())}"
            )
        TOPK_INIT_METHODS[self.init_method](self.pre_w)
        self.pre_w.data.mul_(self.init_gain)

    def sparse_weight(self) -> torch.Tensor:
        """Return the trainable sparse weights with shape ``[out_features, K]``."""
        return apply_weight_transform(self.pre_w, self.weight_transform)

    def _normalized_sparse_weight(self) -> torch.Tensor:
        weight = self.sparse_weight()
        if self.weight_norm_order is None:
            return weight
        norm = torch.norm(weight, p=self.weight_norm_order, dim=-1, keepdim=True)
        safe_norm = norm.clamp_min(torch.finfo(weight.dtype).eps)
        normalized = (weight / safe_norm) * self.gamma
        return torch.where(norm > 0, normalized, weight)

    def set_recurrent_weight_cache(self, enabled: bool) -> None:
        """Cache the transformed fixed weights once per recurrent unroll."""
        self._recurrent_weight_cache_enabled = (
            bool(enabled)
            and self.cache_transformed_weights
            and not self._use_forward_dynamic_grad_scaling
        )
        self._recurrent_cached_weight = None

    def _forward_sparse_weight(self) -> torch.Tensor:
        if not self._recurrent_weight_cache_enabled:
            return self._normalized_sparse_weight()
        if self._recurrent_cached_weight is None:
            self._recurrent_cached_weight = self._normalized_sparse_weight()
        return self._recurrent_cached_weight

    def _forward_output_chunk_size(self, flat_x: torch.Tensor) -> int:
        return _workspace_limited_chunk_size(
            output_chunk_size=self.output_chunk_size,
            workspace_mb=self.workspace_mb,
            out_features=self.out_features,
            flattened_batch_size=flat_x.shape[0],
            synapses_per_output=self.K,
            element_size=flat_x.element_size(),
        )

    def _can_use_frozen_chunked_transform(self, flat_x: torch.Tensor) -> bool:
        """Return whether forward can transform each projection chunk in place."""
        return (
            (
                not self.pre_w.requires_grad
                or bool(getattr(self, "_fsdp_frozen_core", False))
            )
            and not flat_x.requires_grad
            and self.weight_norm_order is None
            and not self._recurrent_weight_cache_enabled
        )

    def _forward_frozen_chunked_transform(
        self,
        flat_x: torch.Tensor,
        connection_indices: torch.Tensor,
        output_chunk_size: int,
    ) -> torch.Tensor:
        """Project frozen weights without materializing one full transformed copy."""
        out = flat_x.new_empty(flat_x.shape[0], self.out_features)
        for start in range(0, self.out_features, output_chunk_size):
            end = min(start + output_chunk_size, self.out_features)
            idx_chunk = connection_indices[start:end].to(torch.long)
            selected = flat_x[:, idx_chunk.reshape(-1)].reshape(
                flat_x.shape[0],
                end - start,
                self.K,
            )
            weight_chunk = apply_weight_transform(
                self.pre_w[start:end],
                self.weight_transform,
            )
            out[:, start:end] = (selected * weight_chunk.unsqueeze(0)).sum(dim=-1)
        return out

    def weight(self) -> torch.Tensor:
        """Return a dense compatibility view of the effective weights."""
        return self.dense_weight()

    def dense_weight(self) -> torch.Tensor:
        """Materialize a dense ``[out_features, in_features]`` weight matrix."""
        dense = torch.zeros(
            self.out_features,
            self.in_features,
            device=self.pre_w.device,
            dtype=self.pre_w.dtype,
        )
        dense.scatter_add_(
            1, self.connection_indices.to(dense.device), self.sparse_weight()
        )
        if self.weight_norm_order is not None:
            dense_norm = torch.norm(
                dense, p=self.weight_norm_order, dim=-1, keepdim=True
            )
            safe_norm = dense_norm.clamp_min(torch.finfo(dense.dtype).eps)
            dense = torch.where(dense_norm > 0, (dense / safe_norm) * self.gamma, dense)
        return dense

    def log_weight(self) -> torch.Tensor:
        """Return a dense log-weight matrix for compatibility with analyzers."""
        dense = self.dense_weight()
        if self.weight_transform == "identity":
            return (dense.abs() + 1e-8).log()
        return (dense + 1e-8).log()

    def weight_mask(self) -> torch.Tensor:
        """Return a dense binary mask of active fixed synapses."""
        mask = torch.zeros(
            self.out_features,
            self.in_features,
            device=self.pre_w.device,
            dtype=self.pre_w.dtype,
        )
        values = torch.ones_like(self.pre_w, dtype=mask.dtype)
        mask.scatter_add_(1, self.connection_indices.to(mask.device), values)
        return mask.clamp_max(1)

    def pruned_weight(self) -> torch.Tensor:
        """Return the dense effective sparse matrix."""
        return self.dense_weight()

    def log_pruned_weight(self) -> torch.Tensor:
        """Return log effective weights with inactive entries shifted down."""
        return ((self.weight_mask() - 1) * 10) + self.log_weight()

    def decay_weights(self, weight_decay: float = 0.1, weight_boosting: bool = False):
        """Decay active sparse weights in-place."""
        with torch.no_grad():
            if self.weight_transform == "exp":
                self.pre_w.data += log(1 - weight_decay)
                if weight_boosting:
                    self.pre_w.data += log(1 + weight_decay)
            elif self.weight_transform == "softplus":
                weight = nn.functional.softplus(self.pre_w.data)
                scale = 1 - weight_decay
                if weight_boosting:
                    scale *= 1 + weight_decay
                updated = weight * scale
                self.pre_w.data = inverse_softplus(updated)
            elif self.weight_transform in {"identity", "relu"}:
                scale = 1 - weight_decay
                if weight_boosting:
                    scale *= 1 + weight_decay
                self.pre_w.data *= scale

    def set_recurrent_weight_cache(self, enabled: bool) -> None:
        """Hoist the weight transform out of a recurrent unroll's timestep loop.

        ``pre_w`` is constant for the duration of one forward pass, so
        ``exp(pre_w)`` (or softplus/…) is loop-invariant across timesteps. Unlike
        :class:`TopKLinear` -- whose mask is derived from the weights, forcing it
        to cache only the mask under autograd -- this layer's connectivity is a
        fixed index buffer, so the transformed weight has no per-timestep data
        dependence and is safe to compute once and reuse.

        This matters at scale because autograd saves each timestep's transformed
        weight for backward: the recomputing path costs ``T * out_features * K``
        elements of activation, which at T=58 is the single largest memory term in
        the model and does *not* shard under FSDP (every rank runs the full
        forward). Caching cuts it by a factor of T.

        The gradient is mathematically identical -- one shared node accumulating T
        incoming gradients instead of T nodes contributing one each -- but the
        floating-point accumulation order differs, so results are not bit-identical.

        Bypassed when dynamic forward gradient scaling is active, since that
        registers a per-forward hook on the returned tensor and a shared tensor
        would collect one hook per timestep. Always call with ``enabled=False``
        after the unroll to release the cached tensor.
        """
        self._recurrent_weight_cache_enabled = (
            bool(enabled)
            and self.cache_recurrent_weight
            and not getattr(self, "_use_forward_dynamic_grad_scaling", False)
        )
        # Drop any cached tensor on a state change so a stale weight can never be
        # reused across unrolls (and so the memory is actually released).
        self._cached_recurrent_weight = None

    def _transformed_weight(self) -> torch.Tensor:
        """Return the effective weight, reusing the unroll cache when enabled."""
        # Re-check the dynamic-grad-scaling guard here as well as in the setter:
        # ``register_forward_param_gradient_scale`` sets that attribute during a
        # forward, i.e. possibly after the cache was enabled for this unroll.
        if not self._recurrent_weight_cache_enabled or getattr(
            self, "_use_forward_dynamic_grad_scaling", False
        ):
            return self._normalized_sparse_weight()
        cached = self._cached_recurrent_weight
        if cached is None:
            cached = self._normalized_sparse_weight()
            self._cached_recurrent_weight = cached
        return cached

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the fixed sparse projection along the last input dimension."""
        if not self._topology_ready:
            if not bool(self._topology_initialized.item()):
                raise RuntimeError(
                    "Indexed topology is awaiting rank-0 broadcast. Construct "
                    "this module under FSDP with sync_module_states=True before "
                    "forward."
                )
            self._topology_ready = True
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
        weight = self._transformed_weight()
        # ``weight`` can be the Parameter itself for the identity transform.
        # Bypass ``nn.Module.__setattr__`` so this diagnostic cache is not
        # registered as a duplicate parameter/state-dict key.
        object.__setattr__(self, "_last_forward_param_tensor", weight)
        indices = self.connection_indices.to(flat_x.device)
        output_chunk_size = self._forward_output_chunk_size(flat_x)

        # Fastest-first precedence: cuda -> triton -> fused -> original loop. Each perf
        # path is guarded (CUDA tensor + toolchain/lib available); if a selected path is
        # unavailable we fall through to the next-best, so a flag can never break a run.
        # cuda_gather_available() lazily compiles the CUDA extension on first use only --
        # never reached when use_cuda_gather is off, so the default path compiles nothing.
        use_cuda = (
            self.use_cuda_gather
            and flat_x.is_cuda
            and cuda_gather_available()
        )
        use_triton = (
            self.use_triton_gather
            and flat_x.is_cuda
            and triton_gather_available()
        )
        if use_cuda:
            out = cuda_sparse_gather(flat_x, indices, weight)
        elif use_triton:
            out = triton_sparse_gather(flat_x, indices, weight)
        elif self.use_fused_gather:
            # custom-autograd gather+mul+sum: ~7.6x faster backward, ~28x less memory
            out = fused_sparse_gather(
                flat_x, indices, weight, chunk_size=self.output_chunk_size
            )
        else:
            out = flat_x.new_empty(flat_x.shape[0], self.out_features)
            for start in range(0, self.out_features, self.output_chunk_size):
                end = min(start + self.output_chunk_size, self.out_features)
                idx_chunk = indices[start:end]
                selected = flat_x[:, idx_chunk.reshape(-1)].reshape(
                    flat_x.shape[0], end - start, self.K
                )
                out[:, start:end] = (selected * weight[start:end].unsqueeze(0)).sum(dim=-1)

        out = out.reshape(*original_shape[:-1], self.out_features)
        if self.cache_mask:
            self._last_forward_weight_mask = self.weight_mask().detach()
        if squeezed:
            return out.squeeze(0)
        return out

    def register_forward_param_gradient_scale(self, param_scale: torch.Tensor) -> None:
        """Scale sparse parameter gradients for the most recent forward call."""
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


__all__ = ["IndexedSparseLinear"]
