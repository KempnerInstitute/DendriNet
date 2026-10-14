"""Names of the indexed sparse-projection backends.

Selection and normalization only: the backends themselves live in the synapse
kernels package, which imports this module, so configuration loading never
imports torch or Triton.
"""

from __future__ import annotations

import warnings
from typing import Literal

IndexedProjectionBackend = Literal[
    "eager",
    "recompute",
    "contiguous_blocks",
    "scatter_dense",
    "triton_transposed",
    "triton_transposed_atomic",
    "triton_transposed_bf16_compat",
    "triton_fused",
    "triton_ell",
    "auto",
]

INDEXED_PROJECTION_BACKENDS = frozenset(
    {
        "auto",
        "contiguous_blocks",
        "eager",
        "recompute",
        "scatter_dense",
        "triton_ell",
        "triton_fused",
        "triton_transposed",
        "triton_transposed_atomic",
        "triton_transposed_bf16_compat",
    }
)

#: Fused accelerator kernels: none materializes the ``[batch, out, K]`` gather.
#: ``triton_transposed`` computes the input gradient deterministically;
#: ``triton_transposed_atomic`` uses the atomic scatter (run-to-run
#: nondeterministic in the last bits) and ``triton_fused`` accumulates with
#: atomics as well.
ACCELERATOR_INDEXED_BACKENDS = frozenset(
    {
        "triton_ell",
        "triton_fused",
        "triton_transposed",
        "triton_transposed_atomic",
        "triton_transposed_bf16_compat",
    }
)


def normalize_indexed_projection_backend(value: object) -> str:
    """Return a validated indexed-projection backend name."""
    normalized = str(value).strip().lower()
    if normalized == "triton":
        # Retired: the batch-tiled Triton gather was slower than the
        # transposed-coalesced kernel at every measured shape (2.4-21x on
        # H100 and Blackwell GPUs). Old configs keep working and get the
        # faster kernel.
        warnings.warn(
            "indexed_projection_backend='triton' is retired; resolving to "
            "'triton_transposed'",
            DeprecationWarning,
            stacklevel=2,
        )
        normalized = "triton_transposed"
    if normalized == "cuda":
        # Retired: slower than the transposed Triton kernel at every
        # measured feedforward shape, and its JIT extension no longer
        # builds with current toolchains on H100 and Blackwell GPUs. The
        # sources remain in git history should an architecture where it
        # wins return. 'auto' picks the best available backend instead.
        warnings.warn(
            "indexed_projection_backend='cuda' is retired; resolving to 'auto'",
            DeprecationWarning,
            stacklevel=2,
        )
        normalized = "auto"
    if normalized not in INDEXED_PROJECTION_BACKENDS:
        choices = ", ".join(sorted(INDEXED_PROJECTION_BACKENDS))
        raise ValueError(
            f"indexed_projection_backend must be one of {{{choices}}}, got {value!r}"
        )
    return normalized


def normalize_indexed_projection_options(
    backend: object,
    *,
    recompute_backward: bool,
) -> str:
    """Resolve the legacy recompute flag into the single backend selector."""
    normalized = normalize_indexed_projection_backend(backend)
    if recompute_backward:
        if normalized not in {"eager", "recompute"}:
            raise ValueError(
                "indexed_recompute_backward cannot be combined with "
                f"indexed_projection_backend={normalized!r}; use "
                "projection_backend='recompute' instead"
            )
        return "recompute"
    return normalized


__all__ = [
    "ACCELERATOR_INDEXED_BACKENDS",
    "INDEXED_PROJECTION_BACKENDS",
    "IndexedProjectionBackend",
    "normalize_indexed_projection_backend",
    "normalize_indexed_projection_options",
]
