"""
Training utilities.

This module provides utilities for safe gradient handling and hook management.
"""

from .hooks import (
    safe_backward_hook,
    safe_register_hooks,
    register_safe_hook,
    safe_hook_fn,
    SafeHookModule
)

__all__ = [
    "safe_backward_hook",
    "safe_register_hooks",
    "register_safe_hook",
    "safe_hook_fn",
    "SafeHookModule"
] 