"""Explicit input-domain contracts without changing values or derivatives."""

import threading
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn

_INVALID_INPUT = "nonnegative input requires finite values >= 0"
# Checks pending in this thread's innermost deferring scope, or None.
_pending = threading.local()


def require_nonnegative_input(x: torch.Tensor) -> torch.Tensor:
    """Check each input batch and return it unchanged, including at zero.

    This is for measured or generated nonnegative rates. Unlike rectification,
    it rejects invalid data and preserves the identity derivative at zero.
    The explicit check synchronizes a CUDA input once per call, or once per
    scope inside :func:`deferred_nonnegative_checks`; use a rectifying adapter
    when the input domain is not guaranteed by the data source.
    """
    invalid = ~torch.isfinite(x).logical_and_(x >= 0).all()
    flags = getattr(_pending, "flags", None)
    if flags is None:
        _raise_if_invalid([invalid])
    else:
        flags.append(invalid)
    return x


@contextmanager
def deferred_nonnegative_checks() -> Iterator[None]:
    """Evaluate the nonnegative-input checks of a block together at its end.

    A recurrent unroll checks its input at every step: the checked input, a
    ``nonnegative`` transfer activation and each positive projection. Inside
    this scope those checks compute their flags on the device as before, and
    the host reads them once when the block ends, instead of synchronizing at
    every check and step. The checks see the same tensors at the same points,
    values and derivatives are unchanged, and invalid input still raises before
    the block's result is returned. A nested scope joins the enclosing one; an
    exception raised inside the block discards the pending checks. The scope
    also works as a decorator.
    """
    if getattr(_pending, "flags", None) is not None:
        yield
        return
    flags: list[torch.Tensor] = []
    _pending.flags = flags
    try:
        yield
    finally:
        _pending.flags = None
    if flags:
        _raise_if_invalid(flags)


def _raise_if_invalid(flags: list[torch.Tensor]) -> None:
    """Read pending check flags on the host, one synchronization per device."""
    by_device: dict[torch.device, list[torch.Tensor]] = {}
    for flag in flags:
        by_device.setdefault(flag.device, []).append(flag)
    for group in by_device.values():
        invalid = group[0] if len(group) == 1 else torch.stack(group).any()
        if bool(invalid):
            raise ValueError(_INVALID_INPUT)


class NonnegativeInputCheck(nn.Module):
    """An identity map with an enforced finite, nonnegative input domain."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return require_nonnegative_input(x)
