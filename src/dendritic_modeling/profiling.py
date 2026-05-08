"""Lightweight profiling helpers for DendriNet execution.

The helpers in this module are intentionally inert by default.  Profiling ranges
are enabled either by setting ``DENDRITIC_NVTX=1``/``DENDRINET_NVTX=1`` before
import, or temporarily through :func:`dendritic_nvtx`.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch

_TRUE_VALUES = {"1", "true", "yes", "on"}
_ENV_ENABLED = any(
    os.environ.get(name, "").strip().lower() in _TRUE_VALUES
    for name in ("DENDRITIC_NVTX", "DENDRINET_NVTX")
)
_FORCED_DEPTH = 0


def is_nvtx_enabled() -> bool:
    """Return whether DendriNet profiling annotations should be emitted."""
    return _ENV_ENABLED or _FORCED_DEPTH > 0


@contextmanager
def dendritic_nvtx(enabled: bool = True):
    """Temporarily enable DendriNet profiling annotations."""
    global _FORCED_DEPTH
    if not enabled:
        yield
        return

    _FORCED_DEPTH += 1
    try:
        yield
    finally:
        _FORCED_DEPTH -= 1


@contextmanager
def nvtx_range(message: str):
    """Emit a PyTorch profiler range and, on CUDA, an NVTX range."""
    if not is_nvtx_enabled():
        yield
        return

    name = str(message)
    with torch.autograd.profiler.record_function(name):
        pushed_cuda_range = False
        if torch.cuda.is_available():
            try:
                torch.cuda.nvtx.range_push(name)
                pushed_cuda_range = True
            except Exception:
                pushed_cuda_range = False

        try:
            yield
        finally:
            if pushed_cuda_range:
                try:
                    torch.cuda.nvtx.range_pop()
                except Exception:
                    pass


class _BackwardRange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, message: str) -> torch.Tensor:
        ctx.message = message
        return tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        with nvtx_range(ctx.message):
            return grad_output, None


def mark_backward(tensor: torch.Tensor, message: str) -> torch.Tensor:
    """Insert a named identity node so backward flow has a visible range."""
    if (
        not is_nvtx_enabled()
        or not torch.is_tensor(tensor)
        or not tensor.requires_grad
    ):
        return tensor
    return _BackwardRange.apply(tensor, str(message))

