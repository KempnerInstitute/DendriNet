"""Shared utilities for hook lifecycles."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

import torch

HookHandle = torch.utils.hooks.RemovableHandle
ModuleT = TypeVar("ModuleT", bound=torch.nn.Module)
ForwardHook = Callable[..., object]


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


def register_named_forward_hooks(
    model: torch.nn.Module,
    module_type: type[ModuleT],
    hook: ForwardHook,
    *,
    predicate: Callable[[str, ModuleT], bool] | None = None,
    prepare: Callable[[str, ModuleT], None] | None = None,
) -> list[HookHandle]:
    """Register one forward hook per matching module with partial-failure cleanup."""
    handles: list[HookHandle] = []
    try:
        for name, module in iter_named_modules_of_type(model, module_type):
            if predicate is not None and not predicate(name, module):
                continue
            if prepare is not None:
                prepare(name, module)
            handles.append(module.register_forward_hook(hook))
        return handles
    except Exception:
        remove_hook_handles(handles)
        raise


class ForwardHookRemovalMixin:
    """Mixin for analyzers whose hook teardown only removes handles."""

    def remove_forward_hooks(self, handles: Iterable[HookHandle]) -> None:
        """Remove registered forward hooks."""
        remove_hook_handles(handles)


__all__ = [
    "ForwardHookRemovalMixin",
    "HookHandle",
    "iter_named_modules_of_type",
    "register_named_forward_hooks",
    "remove_hook_handles",
]
