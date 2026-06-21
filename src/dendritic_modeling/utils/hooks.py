"""Shared utilities for hook lifecycles."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

import torch

HookHandle = torch.utils.hooks.RemovableHandle
ModuleT = TypeVar("ModuleT", bound=torch.nn.Module)


def remove_hook_handles(handles: Iterable[HookHandle]) -> None:
    """Remove all registered hook handles."""
    for handle in handles:
        handle.remove()


def iter_named_modules_of_type(
    model: torch.nn.Module,
    module_type: type[ModuleT],
) -> Iterator[tuple[str, ModuleT]]:
    """Yield named submodules matching ``module_type`` in model traversal order."""
    for name, module in model.named_modules():
        if isinstance(module, module_type):
            yield name, module


class ForwardHookRemovalMixin:
    """Mixin for analyzers whose hook teardown only removes handles."""

    def remove_forward_hooks(self, handles: Iterable[HookHandle]) -> None:
        """Remove registered forward hooks."""
        remove_hook_handles(handles)


__all__ = [
    "ForwardHookRemovalMixin",
    "HookHandle",
    "iter_named_modules_of_type",
    "remove_hook_handles",
]
