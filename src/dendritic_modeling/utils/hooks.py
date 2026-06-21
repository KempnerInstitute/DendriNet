"""Shared utilities for hook lifecycles."""

from __future__ import annotations

from collections.abc import Iterable

import torch

HookHandle = torch.utils.hooks.RemovableHandle


def remove_hook_handles(handles: Iterable[HookHandle]) -> None:
    """Remove all registered hook handles."""
    for handle in handles:
        handle.remove()


class ForwardHookRemovalMixin:
    """Mixin for analyzers whose hook teardown only removes handles."""

    def remove_forward_hooks(self, handles: Iterable[HookHandle]) -> None:
        """Remove registered forward hooks."""
        remove_hook_handles(handles)


__all__ = ["ForwardHookRemovalMixin", "HookHandle", "remove_hook_handles"]
