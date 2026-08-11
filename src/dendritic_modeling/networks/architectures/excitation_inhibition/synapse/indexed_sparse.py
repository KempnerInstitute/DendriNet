"""Fixed-index sparse linear synapses."""

from __future__ import annotations

import logging
from math import log

import time

import torch
import torch.nn as nn

_logger = logging.getLogger(__name__)
_GATHER_PATHS_LOGGED: set[str] = set()

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
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.comparison_backends import (
    dispatch_iss373_backend,
    normalize_indexed_comparison_backend,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.hybrid_gather import (
    hybrid_gather_available,
    hybrid_sparse_gather,
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
        use_auto_gather: bool = False,
        comparison_backend: str | None = None,
        cache_recurrent_weight: bool = False,
        cache_transformed_weights: bool = False,
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
        self.use_auto_gather = bool(use_auto_gather)
        self.comparison_backend = normalize_indexed_comparison_backend(
            comparison_backend
        )
        self._last_resolved_projection_backend: str | None = None
        if self.comparison_backend is not None:
            # The comparison selector is explicit and therefore takes priority
            # over Ryanconfig's legacy booleans.  With no selector, those
            # booleans retain exactly their existing behavior.
            self.use_fused_gather = False
            self.use_triton_gather = False
            self.use_cuda_gather = False
            self.use_auto_gather = False
            if self.comparison_backend == "ryan_fused":
                self.use_fused_gather = True
            elif self.comparison_backend == "ryan_triton":
                self.use_triton_gather = True
            elif self.comparison_backend == "ryan_cuda":
                self.use_cuda_gather = True
            elif self.comparison_backend == "ryan_auto":
                # Fused is the portable pre-probe fallback; on CUDA the
                # existing measured race retains control of the final choice.
                self.use_fused_gather = True
                self.use_auto_gather = True
        self._auto_gather_choice: str | None = None  # per-module probe verdict
        self.input_scale_vec = None
        self.param_scale_vec = None
        self._use_forward_dynamic_grad_scaling = False
        self.cache_mask = False
        self._last_forward_weight_mask: torch.Tensor | None = None
        self._last_forward_param_tensor: torch.Tensor | None = None
        # Opt-in hoist of the weight transform out of a recurrent unroll; see
        # ``set_recurrent_weight_cache``. Off by default so behaviour is unchanged
        # unless a config asks for it.
        # Two config spellings reached this feature independently:
        # `indexed_cache_recurrent_weight` and `indexed_cache_transformed_weights`
        # (the latter is also used by indexed_dynamic.py). Accept either so no
        # existing config silently stops caching; a single flag drives the one
        # live implementation.
        self.cache_transformed_weights = bool(cache_transformed_weights)
        self.cache_recurrent_weight = bool(
            cache_recurrent_weight or cache_transformed_weights
        )
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

    # NOTE: a second `set_recurrent_weight_cache` / `_forward_sparse_weight` pair
    # used to live here. It was dead code -- the definition further down shadowed
    # it (later definition wins), and it read `self.cache_transformed_weights`,
    # which this class never assigned, so it would have raised AttributeError had
    # it ever been reached. The single live implementation is below; it honours
    # both config spellings. Do not reintroduce a second copy.

    # Measured on H200 at the 1M model's module geometry (bench_rnn_b4_crossover,
    # 2026-08-10): the cuda path is 6.8-25x faster than fused on >=1M-output
    # modules (adaptive-tile regime) but RAISES on some shapes (e.g. K=240), and
    # fused wins the small-module regime where the whole-step 5k verdict was
    # settled. Auto mode therefore probes capability per module ONCE (forward +
    # both grads, catching launch-time raises at probe time instead of epoch
    # time) and applies the measured size rule below.
    # Modules smaller than this many contacts aren't worth a startup race.
    _AUTO_GATHER_MIN_CONTACTS = 1_000_000
    # Shape-keyed, PROCESS-wide verdicts. Keying by instance broke under
    # activation checkpointing: a probe firing inside a backward-pass
    # recompute adds ops the original forward didn't run, and checkpoint
    # aborts on the divergence. Shape-keyed, every module touched in a
    # recompute inherits the verdict its shape earned during the original
    # pass (which always runs first), so probes can never fire in recompute.
    _AUTO_GATHER_VERDICTS: dict[tuple[int, int], str] = {}

    def _probe_auto_gather(self, flat_x, indices, weight) -> str:
        if (
            self.out_features * int(indices.shape[1])
            < self._AUTO_GATHER_MIN_CONTACTS
            or not cuda_gather_available()
        ):
            return "fused"
        key = (self.out_features, int(indices.shape[1]))
        cached = self._AUTO_GATHER_VERDICTS.get(key)
        if cached is not None:
            return cached

        # RACE the two paths on the real tensors: warmup + 3 timed fwd+bwd
        # each, keep the faster. The step profile (2026-08-10) showed the
        # fused gather/scatter on MEDIUM per-timestep modules is 58% of the
        # 1M step -- a size rule picked the wrong modules; a race cannot.
        def _time(fn) -> float:
            with torch.enable_grad():
                px = flat_x.detach().clone().requires_grad_(True)
                pw = weight.detach().requires_grad_(True)
                for _ in range(1):
                    out = fn(px, indices, pw)
                    torch.autograd.grad(out.sum(), (px, pw))
                torch.cuda.synchronize(flat_x.device)
                t0 = time.perf_counter()
                for _ in range(3):
                    out = fn(px, indices, pw)
                    torch.autograd.grad(out.sum(), (px, pw))
                torch.cuda.synchronize(flat_x.device)
            return time.perf_counter() - t0

        def _fused(px, idx, pw):
            return fused_sparse_gather(
                px, idx, pw, chunk_size=self.output_chunk_size
            )

        times = {"fused": _time(_fused)}
        try:
            times["cuda"] = _time(cuda_sparse_gather)
        except RuntimeError as exc:
            _logger.info("auto-gather race: cuda out (%s)", str(exc)[:80])
        if hybrid_gather_available():
            try:
                times["hybrid"] = _time(
                    lambda px, idx, pw: hybrid_sparse_gather(
                        px, idx, pw, chunk_size=self.output_chunk_size
                    )
                )
            except RuntimeError as exc:
                _logger.info("auto-gather race: hybrid out (%s)", str(exc)[:80])
        choice = min(times, key=times.get)
        _logger.info(
            "auto-gather race: out=%d K=%d %s -> %s",
            self.out_features, indices.shape[1],
            " ".join(f"{k}={v * 1e3 / 3:.1f}ms" for k, v in times.items()),
            choice,
        )
        self._AUTO_GATHER_VERDICTS[key] = choice
        return choice

    def _forward_output_chunk_size(self, flat_x: torch.Tensor) -> int:
        return _workspace_limited_chunk_size(
            output_chunk_size=self.output_chunk_size,
            workspace_mb=self.workspace_mb,
            out_features=self.out_features,
            flattened_batch_size=flat_x.shape[0],
            synapses_per_output=self.K,
            element_size=flat_x.element_size(),
        )

    def _record_comparison_backend(self, resolved: str) -> None:
        """Record the concrete implementation used by an A/B condition."""
        if resolved != self._last_resolved_projection_backend:
            _logger.info(
                "indexed comparison backend resolved: requested=%s resolved=%s",
                self.comparison_backend,
                resolved,
            )
            self._last_resolved_projection_backend = resolved

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

        # Keep upstream implementations physically separate from Ryanconfig's
        # kernels.  This is deliberately an early dispatch: neither path may
        # silently fall through to the other during a benchmark.
        if self.comparison_backend in {
            "iss_recompute",
            "iss_triton",
            "iss_cuda",
            "iss_auto",
        }:
            out, resolved = dispatch_iss373_backend(
                self.comparison_backend,
                flat_x,
                indices,
                weight,
                output_chunk_size=output_chunk_size,
            )
            self._record_comparison_backend(resolved)
            out = out.reshape(*original_shape[:-1], self.out_features)
            if self.cache_mask:
                self._last_forward_weight_mask = self.weight_mask().detach()
            return out.squeeze(0) if squeezed else out

        if self.comparison_backend == "ryan_hybrid":
            if not flat_x.is_cuda or not hybrid_gather_available():
                raise RuntimeError(
                    "indexed_comparison_backend='ryan_hybrid' is unavailable; "
                    "a CUDA device and the Ryan hybrid gather extension are required"
                )
            out = hybrid_sparse_gather(
                flat_x, indices, weight, chunk_size=output_chunk_size
            )
            self._record_comparison_backend("ryan_hybrid")
            out = out.reshape(*original_shape[:-1], self.out_features)
            if self.cache_mask:
                self._last_forward_weight_mask = self.weight_mask().detach()
            return out.squeeze(0) if squeezed else out

        if self.comparison_backend == "ryan_cuda" and (
            not flat_x.is_cuda or not cuda_gather_available()
        ):
            raise RuntimeError(
                "indexed_comparison_backend='ryan_cuda' is unavailable; "
                "a CUDA device and the Ryan CUDA gather extension are required"
            )
        if self.comparison_backend == "ryan_triton" and (
            not flat_x.is_cuda or not triton_gather_available()
        ):
            raise RuntimeError(
                "indexed_comparison_backend='ryan_triton' is unavailable; "
                "a CUDA device and the Ryan Triton gather implementation are required"
            )

        # Fastest-first precedence: cuda -> triton -> fused -> original loop. Each perf
        # path is guarded (CUDA tensor + toolchain/lib available); if a selected path is
        # unavailable we fall through to the next-best, so a flag can never break a run.
        # cuda_gather_available() lazily compiles the CUDA extension on first use only --
        # never reached when use_cuda_gather is off, so the default path compiles nothing.
        if (
            self.use_auto_gather
            and flat_x.is_cuda
            and self._auto_gather_choice is None
        ):
            if key_cached := self._AUTO_GATHER_VERDICTS.get(
                (self.out_features, int(indices.shape[1]))
            ):
                self._auto_gather_choice = key_cached
            elif not torch.is_grad_enabled():
                # Probe ONLY under no_grad: activation checkpointing is inert
                # there, so the probe's throwaway graph can't pollute a
                # checkpoint region's saved-tensor sequence (it aborts the
                # step with a metadata mismatch if it does). The trainer's
                # warmup forward triggers all probes before step 1; absent a
                # warmup, grad-mode forwards use fused until the first
                # no_grad pass (validation) resolves the verdict.
                self._auto_gather_choice = self._probe_auto_gather(
                    flat_x, indices, weight
                )
        use_cuda = (
            (
                self._auto_gather_choice == "cuda"
                if self.use_auto_gather and flat_x.is_cuda
                else self.use_cuda_gather
            )
            and flat_x.is_cuda
            and cuda_gather_available()
        )
        use_triton = (
            self.use_triton_gather
            and flat_x.is_cuda
            and triton_gather_available()
        )
        use_fused = (
            self.use_fused_gather
            or (self.use_auto_gather and self._auto_gather_choice == "fused")
        )
        resolved = ("cuda" if use_cuda else "triton" if use_triton
                    else "fused" if use_fused else "loop")
        if resolved not in _GATHER_PATHS_LOGGED:
            # Once per process per path. The fall-through above is deliberately
            # silent so a flag can never break a run -- but silence also means a
            # broken toolchain quietly demotes cuda to loop and every downstream
            # claim about "which kernel the run used" becomes inference. This
            # makes it a logged fact instead.
            _GATHER_PATHS_LOGGED.add(resolved)
            _logger.info(
                "indexed gather path resolved: %s (flags: cuda=%s triton=%s "
                "fused=%s; available: cuda=%s triton=%s)",
                resolved, self.use_cuda_gather, self.use_triton_gather,
                self.use_fused_gather,
                cuda_gather_available() if flat_x.is_cuda else "n/a",
                triton_gather_available() if flat_x.is_cuda else "n/a",
            )
        if self.use_auto_gather and self._auto_gather_choice == "hybrid":
            out = hybrid_sparse_gather(
                flat_x, indices, weight, chunk_size=self.output_chunk_size
            )
        elif use_cuda:
            out = cuda_sparse_gather(flat_x, indices, weight)
        elif use_triton:
            out = triton_sparse_gather(flat_x, indices, weight)
        elif use_fused:
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

        if self.comparison_backend is not None:
            comparison_resolved = {
                "loop": "control_eager",
                "fused": "ryan_fused",
                "triton": "ryan_triton",
                "cuda": "ryan_cuda",
            }.get(resolved, resolved)
            if self.comparison_backend == "ryan_auto":
                comparison_resolved = f"ryan_auto:{resolved}"
            self._record_comparison_backend(comparison_resolved)

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
