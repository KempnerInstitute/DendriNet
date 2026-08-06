"""
Contiguous prefix split of input features into E-stream and I-stream.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def resolve_e_stream_prefix_k(
    e_stream_dims: Any,
    *,
    input_dim: int,
) -> int:
    """Resolve ``e_stream_dims`` to a strict prefix length ``k``.

    Accepts:
    - ``int`` ``k``
    - ``slice`` with start in ``{None, 0}`` and integer stop
    - string forms ``":k"`` / ``"0:k"`` / ``"0:k:"`` (prefix only)

    Raises ``ValueError`` for non-prefix slices or invalid ``k``.
    """
    if e_stream_dims is None:
        raise ValueError("e_stream_dims must not be None when resolving a split")

    if isinstance(e_stream_dims, bool):
        raise ValueError(
            f"e_stream_dims must be an int or prefix slice, got {e_stream_dims!r}"
        )

    if isinstance(e_stream_dims, int):
        k = int(e_stream_dims)
    elif isinstance(e_stream_dims, slice):
        k = _prefix_k_from_slice(e_stream_dims, input_dim=input_dim)
    elif isinstance(e_stream_dims, str):
        k = _prefix_k_from_slice_string(e_stream_dims, input_dim=input_dim)
    else:
        raise ValueError(
            "e_stream_dims must be an int, slice, or prefix slice string "
            f"(':k' / '0:k'), got {type(e_stream_dims).__name__}: {e_stream_dims!r}"
        )

    if k <= 0 or k >= input_dim:
        raise ValueError(
            f"e_stream_dims prefix length must satisfy 0 < k < input_dim "
            f"({input_dim}), got k={k}"
        )
    return k


def _prefix_k_from_slice(sl: slice, *, input_dim: int) -> int:
    if sl.step not in (None, 1):
        raise ValueError(
            f"e_stream_dims slice step must be 1 or None, got {sl.step!r}"
        )
    if sl.start not in (None, 0):
        raise ValueError(
            "e_stream_dims only supports a prefix slice "
            f"(start None or 0), got start={sl.start!r}"
        )
    if sl.stop is None:
        raise ValueError(
            "e_stream_dims prefix slice requires an integer stop "
            "(e.g. ':784' or '0:784')"
        )
    try:
        return int(sl.stop)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"e_stream_dims slice stop must be an integer, got {sl.stop!r}"
        ) from exc


def _prefix_k_from_slice_string(value: str, *, input_dim: int) -> int:
    text = value.strip()
    if ":" not in text:
        try:
            return int(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "e_stream_dims string must be an int-like value or prefix slice "
                f"(':k' / '0:k'), got {value!r}"
            ) from exc
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(
            f"e_stream_dims slice string must look like ':k' or '0:k', got {value!r}"
        )
    start_s, stop_s = parts[0], parts[1]
    step_s = parts[2] if len(parts) == 3 else ""
    if step_s not in ("", "1"):
        raise ValueError(
            f"e_stream_dims slice step must be empty or 1, got {value!r}"
        )
    start = None if start_s == "" else int(start_s)
    if stop_s == "":
        raise ValueError(
            "e_stream_dims prefix slice requires an integer stop "
            f"(e.g. ':784' or '0:784'), got {value!r}"
        )
    stop = int(stop_s)
    return _prefix_k_from_slice(slice(start, stop, None), input_dim=input_dim)


class PrefixSplitInputTransform(nn.Module):
    """Split ``x`` contiguously into ``(x[..., :k], x[..., k:])``."""

    def __init__(self, input_dim: int, e_stream_dims: Any):
        super().__init__()
        if input_dim is None or int(input_dim) <= 0:
            raise ValueError(f"input_dim must be a positive integer, got {input_dim}")
        input_dim = int(input_dim)
        self.input_dim = input_dim
        self.k = resolve_e_stream_prefix_k(e_stream_dims, input_dim=input_dim)
        self.excitatory_dim = self.k
        self.inhibitory_dim = input_dim - self.k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x[None, :]
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                "PrefixSplitInputTransform expected last input dimension "
                f"{self.input_dim}, got {x.shape[-1]}"
            )
        return x[..., : self.k], x[..., self.k :]


__all__ = [
    "PrefixSplitInputTransform",
    "resolve_e_stream_prefix_k",
]
