"""Backend selection for fixed indexed sparse projections.

The portable eager and recompute implementations live in ``indexed_common``.
This module owns selection only; optional CUDA and Triton dependencies remain
lazy so importing or constructing the default layer never compiles a kernel.
"""

from __future__ import annotations

from typing import Literal

import torch

IndexedProjectionBackend = Literal[
    "eager",
    "recompute",
    "triton",
    "cuda",
    "auto",
]

INDEXED_PROJECTION_BACKENDS = frozenset(
    {"auto", "cuda", "eager", "recompute", "triton"}
)


def normalize_indexed_projection_backend(value: object) -> str:
    """Return a validated indexed-projection backend name."""
    normalized = str(value).strip().lower()
    if normalized not in INDEXED_PROJECTION_BACKENDS:
        choices = ", ".join(sorted(INDEXED_PROJECTION_BACKENDS))
        raise ValueError(
            f"indexed_projection_backend must be one of {{{choices}}}, "
            f"got {value!r}"
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


def indexed_projection_backend_available(
    backend: object,
    *,
    device: torch.device | str | None = None,
) -> bool:
    """Return whether a backend is usable for the requested device.

    Checking the CUDA extension intentionally triggers its lazy one-time build.
    Callers should therefore use this helper only when ``cuda`` or ``auto`` was
    explicitly requested.
    """
    normalized = normalize_indexed_projection_backend(backend)
    if normalized in {"auto", "eager", "recompute"}:
        # ``auto`` always has the portable recompute implementation available,
        # even when neither optional accelerator backend can run.
        return True

    resolved_device = torch.device(device) if device is not None else None
    if resolved_device is not None and resolved_device.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False

    if normalized == "triton":
        from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels.triton_indexed_gather import (
            triton_gather_available,
        )

        return triton_gather_available()
    if normalized == "cuda":
        from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels.cuda_indexed_gather import (
            cuda_gather_available,
        )

        return cuda_gather_available()
    raise AssertionError(f"unhandled indexed projection backend: {normalized}")


def resolve_indexed_projection_backend(
    backend: object,
    *,
    device: torch.device | str,
) -> str:
    """Resolve an indexed backend without silently demoting explicit choices."""
    normalized = normalize_indexed_projection_backend(backend)
    resolved_device = torch.device(device)
    if normalized in {"eager", "recompute"}:
        return normalized

    if normalized == "auto":
        if resolved_device.type == "cuda":
            for candidate in ("cuda", "triton"):
                if indexed_projection_backend_available(
                    candidate,
                    device=resolved_device,
                ):
                    return candidate
        return "recompute"

    if indexed_projection_backend_available(normalized, device=resolved_device):
        return normalized

    if normalized == "cuda":
        from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels.cuda_indexed_gather import (
            cuda_gather_unavailability_reason,
        )

        reason = cuda_gather_unavailability_reason()
    else:
        reason = "Triton and a CUDA device are required"
    raise RuntimeError(
        f"indexed_projection_backend={normalized!r} is unavailable on "
        f"device {resolved_device}: {reason}. Use 'auto', 'recompute', or 'eager' "
        "for a portable fallback."
    )


__all__ = [
    "INDEXED_PROJECTION_BACKENDS",
    "IndexedProjectionBackend",
    "indexed_projection_backend_available",
    "normalize_indexed_projection_backend",
    "normalize_indexed_projection_options",
    "resolve_indexed_projection_backend",
]
