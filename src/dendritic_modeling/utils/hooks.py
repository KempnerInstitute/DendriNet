"""Shared utilities for hook lifecycles."""

from __future__ import annotations

from collections.abc import Iterable

import torch

HookHandle = torch.utils.hooks.RemovableHandle


def remove_hook_handles(handles: Iterable[HookHandle]) -> None:
    """Remove all registered hook handles."""
    for handle in handles:
        handle.remove()


__all__ = ["HookHandle", "remove_hook_handles"]
