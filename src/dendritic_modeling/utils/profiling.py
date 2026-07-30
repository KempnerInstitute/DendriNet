"""Lightweight profiling helpers used by training and network code."""

from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def nvtx_range(message: str) -> Iterator[None]:
    """Emit an NVTX range when CUDA/NVTX is available.

    The context is intentionally a no-op on CPU-only runs.  Profiling
    annotations must never make an otherwise valid training run fail, so a
    PyTorch build without NVTX support also falls back to the no-op path.
    """
    if not torch.cuda.is_available():
        yield
        return

    try:
        torch.cuda.nvtx.range_push(message)
    except (AttributeError, RuntimeError):
        yield
        return

    try:
        yield
    finally:
        try:
            torch.cuda.nvtx.range_pop()
        except (AttributeError, RuntimeError):
            # NVTX annotations are diagnostic only and must not break training.
            pass


__all__ = ["nvtx_range"]
