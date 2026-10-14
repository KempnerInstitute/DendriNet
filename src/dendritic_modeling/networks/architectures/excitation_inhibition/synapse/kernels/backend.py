"""Backend selection for fixed indexed sparse projections.

The portable eager and recompute implementations live in ``indexed_common``.
This module owns selection only; the optional Triton dependency stays lazy so
importing or constructing the default layer never compiles a kernel.

``contiguous_blocks`` is an explicit, portable (pure torch) backend for the
index-free layout in which every output row reads one aligned block of ``K``
consecutive inputs; it computes the projection as a batched GEMM and raises if
the indices do not have that layout. ``auto`` never selects it.

``scatter_dense`` is an explicit, portable backend for dense-ish fan-in: it
scatters every row's contacts into a dense ``[out, in]`` matrix and applies one
GEMM, so the forward, the weight gradient and the input gradient all run on
matrix units. It pays ``out * in`` workspace per call and wins when contacts per
row are a sizeable fraction of the input width. ``auto`` never selects it.
"""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Iterator

import torch

from dendritic_modeling.config.sparse_backends import (
    INDEXED_PROJECTION_BACKENDS,
    IndexedProjectionBackend,
    normalize_indexed_projection_backend,
    normalize_indexed_projection_options,
)
from dendritic_modeling.utils.optional_imports import is_missing_module

#: ``auto`` prefers the fused chunked kernel at and above this input width on a
#: Triton-capable CUDA device (measured 3.4x on H100 and 8.5x on Blackwell over
#: chunked recompute at giant-input, batch-4 recurrent shapes).
_AUTO_FUSED_MIN_IN_FEATURES = 100_000

#: Element offsets that some Triton kernels compute in int32 must stay below
#: this bound; past it they wrap, giving illegal memory accesses or silently
#: wrong outputs and gradients.
_INT32_OFFSET_LIMIT = 2**31

#: Backends that accumulate gradients with atomic adds, so the gradients vary
#: in the last bits from run to run: the input gradient of all three, and the
#: weight gradient of ``triton_fused`` when the batch exceeds 16 rows. Under
#: ``torch.use_deterministic_algorithms(True)`` ``auto`` never selects them and
#: an explicit request raises once gradients are computed (see
#: :func:`nondeterministic_gradient_reason`).
NONDETERMINISTIC_INDEXED_BACKENDS = frozenset(
    {"triton_transposed_atomic", "triton_fused", "triton_ell"}
)

_TRANSPOSED_KERNELS = "dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels.triton_indexed_gather_transposed"
_FUSED_KERNELS = "dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels.triton_fused_chunked"


def nondeterministic_gradient_reason(
    backend: object,
    *,
    batch_size: int,
    in_features: int,
    out_features: int,
    synapses_per_output: int,
    input_gradient: bool,
    weight_gradient: bool,
) -> str | None:
    """Return why ``backend`` computes these gradients nondeterministically.

    ``None`` means the requested gradients (``input_gradient``,
    ``weight_gradient``) are bitwise reproducible at this shape. Besides the
    :data:`NONDETERMINISTIC_INDEXED_BACKENDS`, the deterministic transposed
    kernels fall back to an atomic input gradient past their workspace limits.
    Forward outputs are deterministic on every backend.
    """
    normalized = normalize_indexed_projection_backend(backend)
    if normalized in NONDETERMINISTIC_INDEXED_BACKENDS and input_gradient:
        return "it accumulates the input gradient with atomic adds"
    if normalized == "triton_fused" and weight_gradient:
        is_atomic = importlib.import_module(
            _FUSED_KERNELS
        ).fused_weight_gradient_is_atomic
        if is_atomic(batch_size):
            return (
                "it accumulates the weight gradient with atomic adds at batches "
                "over 16 rows"
            )
    if (
        normalized in {"triton_transposed", "triton_transposed_bf16_compat"}
        and input_gradient
    ):
        limit = importlib.import_module(_TRANSPOSED_KERNELS).deterministic_gradx_limit(
            batch_size=batch_size,
            in_features=in_features,
            out_features=out_features,
            synapses_per_output=synapses_per_output,
        )
        if limit is not None:
            return (
                f"its deterministic input gradient cannot run this shape ({limit}) "
                "and falls back to atomic adds"
            )
    return None


def indexed_kernel_shape_violations(
    backend: object,
    *,
    batch_size: int,
    in_features: int,
    out_features: int,
    synapses_per_output: int,
) -> tuple[str, ...]:
    """Return the int32 offset products of ``backend`` that a shape overflows.

    The transposed kernels (``triton_transposed`` and its atomic and
    BF16-compatible variants) index ``x^T`` at ``input * batch``, the output and output gradient
    at ``batch * output`` and the indices/weights at ``output * K`` in int32.
    ``triton_fused`` uses int64 for the latter two but int32 gather addresses
    (``in_features * batch``). ELL also uses int32 row/weight/output products;
    conservatively require all three contiguous shape products to fit. An
    empty tuple means the shape is supported by these bounds.
    """
    normalized = normalize_indexed_projection_backend(backend)
    if normalized in {
        "triton_transposed",
        "triton_transposed_atomic",
        "triton_transposed_bf16_compat",
        "triton_ell",
    }:
        products = {
            "in_features*batch": in_features * batch_size,
            "batch*out_features": batch_size * out_features,
            "out_features*K": out_features * synapses_per_output,
        }
    elif normalized == "triton_fused":
        products = {"in_features*batch": in_features * batch_size}
    else:
        return ()
    return tuple(
        f"{name}={value}"
        for name, value in products.items()
        if value >= _INT32_OFFSET_LIMIT
    )


def indexed_projection_backend_available(
    backend: object,
    *,
    device: torch.device | str | None = None,
) -> bool:
    """Return whether a backend is usable for the requested device."""
    normalized = normalize_indexed_projection_backend(backend)
    if normalized in {
        "auto",
        "eager",
        "recompute",
        "contiguous_blocks",
        "scatter_dense",
    }:
        # ``auto`` always has the portable recompute implementation available,
        # even when neither optional accelerator backend can run.
        return True

    resolved_device = torch.device(device) if device is not None else None
    if resolved_device is not None and resolved_device.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False

    if normalized in {"triton_transposed", "triton_transposed_atomic"}:
        return _kernel_probe(
            "dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels.triton_indexed_gather_transposed",
            "triton_gather_available",
        )
    if normalized == "triton_transposed_bf16_compat":
        return _kernel_probe(
            "dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels.triton_indexed_gather_transposed_bf16_compat",
            "triton_bf16_compat_available",
        )
    if normalized == "triton_fused":
        return _kernel_probe(
            "dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels.triton_fused_chunked",
            "triton_fused_available",
        )
    if normalized == "triton_ell":
        return _kernel_probe(
            "dendritic_modeling.kernels.triton_ell", "triton_available"
        )
    raise AssertionError(f"unhandled indexed projection backend: {normalized}")


def _kernel_probe(module_name: str, probe: str) -> bool:
    """Call ``module_name.probe()``, or return ``False`` when the module is absent.

    General Triton kernels are optional core backends. Without
    them every Triton backend reports itself unavailable, ``auto`` resolves to
    the portable recompute path, and an explicit Triton request fails in
    :func:`resolve_indexed_projection_backend`. Any other import error
    propagates.
    """

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if not is_missing_module(exc, module_name):
            raise
        return False
    return getattr(module, probe)()


def resolve_indexed_projection_backend(
    backend: object,
    *,
    device: torch.device | str,
    in_features: int | None = None,
    batch_size: int | None = None,
    out_features: int | None = None,
    synapses_per_output: int | None = None,
) -> str:
    """Resolve an indexed backend without silently demoting explicit choices.

    ``in_features`` lets ``auto`` pick a shape-dependent kernel per layer;
    ``None`` preserves the shape-agnostic policy (the transposed kernel on a
    Triton-capable CUDA device, otherwise portable recompute). When the full
    projection shape (``batch_size``, ``in_features``, ``out_features``,
    ``synapses_per_output``) is given, ``auto`` skips kernels whose int32
    offsets it would overflow and an explicit request for such a kernel
    raises ``ValueError`` (see :func:`indexed_kernel_shape_violations`).

    Under ``torch.use_deterministic_algorithms(True)`` ``auto`` resolves only
    to backends with deterministic gradients: never to a
    :data:`NONDETERMINISTIC_INDEXED_BACKENDS` entry, and, when the full shape
    is given, not to the transposed kernel past its deterministic workspace
    limits. Explicit requests are checked when the projection runs.
    """
    normalized = normalize_indexed_projection_backend(backend)
    resolved_device = torch.device(device)
    if normalized in {"eager", "recompute", "contiguous_blocks", "scatter_dense"}:
        return normalized

    shape_known = None not in (
        batch_size,
        in_features,
        out_features,
        synapses_per_output,
    )
    deterministic = torch.are_deterministic_algorithms_enabled()

    def violations(candidate: str) -> tuple[str, ...]:
        if not shape_known:
            return ()
        return indexed_kernel_shape_violations(
            candidate,
            batch_size=int(batch_size),
            in_features=int(in_features),
            out_features=int(out_features),
            synapses_per_output=int(synapses_per_output),
        )

    def nondeterministic(candidate: str) -> bool:
        if not deterministic:
            return False
        if candidate in NONDETERMINISTIC_INDEXED_BACKENDS:
            return True
        return shape_known and (
            nondeterministic_gradient_reason(
                candidate,
                batch_size=int(batch_size),
                in_features=int(in_features),
                out_features=int(out_features),
                synapses_per_output=int(synapses_per_output),
                input_gradient=True,
                weight_gradient=True,
            )
            is not None
        )

    if normalized == "auto":
        # The transposed-coalesced Triton kernel was fastest at every
        # measured feedforward and transformer-FFN shape on H100 and
        # Blackwell GPUs (2.4-21x over the other sparse backends), so auto
        # prefers it wherever Triton runs. At giant input widths
        # (in_features >= 100k, the small-batch recurrent regime) the
        # fused chunked kernel measured 3.4x (H100) / 8.5x (Blackwell)
        # over chunked recompute, so auto resolves to it there when the
        # layer's width is known.
        # Under deterministic algorithms auto skips kernels whose gradients
        # use atomic adds.
        if resolved_device.type == "cuda":
            if (
                in_features is not None
                and in_features >= _AUTO_FUSED_MIN_IN_FEATURES
                and not violations("triton_fused")
                and indexed_projection_backend_available(
                    "triton_fused",
                    device=resolved_device,
                )
                and not nondeterministic("triton_fused")
            ):
                return "triton_fused"
            if (
                not violations("triton_transposed")
                and indexed_projection_backend_available(
                    "triton_transposed",
                    device=resolved_device,
                )
                and not nondeterministic("triton_transposed")
            ):
                return "triton_transposed"
        return "recompute"

    if indexed_projection_backend_available(normalized, device=resolved_device):
        overflow = violations(normalized)
        if overflow:
            raise ValueError(
                f"indexed_projection_backend={normalized!r} computes int32 element "
                f"offsets and cannot run this projection ({', '.join(overflow)} "
                f">= 2^31). Use 'auto', 'recompute' or 'eager' for this shape."
            )
        return normalized

    reason = "Triton, a CUDA device and the optional Triton kernel modules are required"
    raise RuntimeError(
        f"indexed_projection_backend={normalized!r} is unavailable on "
        f"device {resolved_device}: {reason}. Use 'auto', 'recompute', or 'eager' "
        "for a portable fallback."
    )


@contextlib.contextmanager
def indexed_projection_backend_override(
    module: torch.nn.Module, backend: object
) -> Iterator[None]:
    """Run every indexed projection inside ``module`` on ``backend`` temporarily.

    Each layer's configured backend is restored on exit, and the layers resolve
    their backend again at their next forward. Analyses that need second-order
    or forward-mode derivatives use the portable ``eager`` kernel this way:
    the Triton kernels compute the same operator but their backward is not
    differentiable.
    """
    normalized = normalize_indexed_projection_backend(backend)
    layers = [
        layer
        for layer in module.modules()
        if hasattr(layer, "projection_backend")
        and hasattr(layer, "_last_resolved_projection_backend")
    ]
    configured = [layer.projection_backend for layer in layers]
    try:
        for layer in layers:
            layer.projection_backend = normalized
        yield
    finally:
        for layer, original in zip(layers, configured):
            layer.projection_backend = original


__all__ = [
    "INDEXED_PROJECTION_BACKENDS",
    "NONDETERMINISTIC_INDEXED_BACKENDS",
    "IndexedProjectionBackend",
    "indexed_kernel_shape_violations",
    "indexed_projection_backend_available",
    "indexed_projection_backend_override",
    "nondeterministic_gradient_reason",
    "normalize_indexed_projection_backend",
    "normalize_indexed_projection_options",
    "resolve_indexed_projection_backend",
]
