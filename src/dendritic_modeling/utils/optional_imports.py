"""Helpers for importing optional dependencies.

Code that imports an optional dependency lazily (for example the Triton
kernels) treats that dependency's absence as "feature unavailable" while every
other import error, such as a missing package inside the dependency, still
propagates unchanged.
"""

from __future__ import annotations

__all__ = ["is_missing_module"]


def is_missing_module(exc: ModuleNotFoundError, *modules: str) -> bool:
    """Return whether ``exc`` reports one of ``modules``, or a parent package, absent."""

    missing = exc.name or ""
    if not missing:
        return False
    return any(
        module == missing or module.startswith(missing + ".") for module in modules
    )
