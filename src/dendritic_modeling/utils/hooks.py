"""
Hook utilities for safe backward hook management in PyTorch.

This module provides utilities for safely registering backward hooks to prevent
the "Backward hook for Modules where no input requires gradient" error.
"""

import functools
from typing import Callable, Optional, Tuple

import torch.nn as nn


def safe_backward_hook(hook_fn: Callable) -> Callable:
    """
    Decorator for backward hooks to ensure they return None when no inputs require gradients.
    This prevents the "Backward hook for Modules where no input requires gradient" error.

    Args:
        hook_fn: The original hook function to decorate

    Returns:
        A decorated hook function that properly handles the case when no inputs require gradients
    """

    @functools.wraps(hook_fn)
    def safe_hook_fn(module, grad_input, grad_output):
        # Check if any grad_input requires gradient
        requires_grad = False
        if grad_input is not None:
            for g in grad_input:
                if g is not None:
                    requires_grad = True
                    break

        # Check parameters too (some hooks need this)
        if not requires_grad:
            for p in module.parameters(recurse=True):
                if p.requires_grad:
                    requires_grad = True
                    break

        # If nothing requires gradients, return None
        if not requires_grad:
            return None

        # Otherwise, call the original hook function
        try:
            result = hook_fn(module, grad_input, grad_output)
            return result
        except Exception as e:
            # If anything goes wrong, return the original grad_input as a fallback
            print(f"Warning: Hook error: {e}")
            return grad_input

    return safe_hook_fn


def safe_register_hooks(module: nn.Module) -> None:
    """
    Recursively register safe hooks on a module and all its submodules.
    This safely handles the replacement of existing hooks to ensure PyTorch
    doesn't try to use both regular and full backward hooks on the same module.

    Args:
        module: The module to register safe hooks on
    """
    # First, check if module already has backward hooks
    has_backward_hooks = bool(module._backward_hooks)
    has_full_backward_hooks = bool(module._backward_pre_hooks)

    # Preserve gradient scaling hooks by checking if they exist
    gradient_hooks_exist = False
    if has_full_backward_hooks:
        for hook_fn in module._backward_pre_hooks.values():
            # Check if any hook function's name contains gradient-related terms
            hook_name = str(hook_fn).lower()
            if (
                "gradient" in hook_name
                or "scale" in hook_name
                or "grad_hook" in hook_name
                or "scale_activation" in hook_name
                or "scale_layer" in hook_name
            ):
                gradient_hooks_exist = True
                break

    # Only replace hooks if no gradient scaling hooks exist
    if not gradient_hooks_exist:
        # Remove all existing hooks to prevent conflicts
        if has_backward_hooks:
            module._backward_hooks.clear()
        if has_full_backward_hooks:
            module._backward_pre_hooks.clear()

        # Register a new safe full backward hook (preferred)
        module.register_full_backward_hook(safe_hook_fn)

    # Process child modules
    for child in module.children():
        safe_register_hooks(child)


def register_safe_hook(module: nn.Module, hook_fn: Callable) -> None:
    """
    Register a safe hook on a module

    Args:
        module: The module to register the hook on
        hook_fn: The hook function to register
    """
    # First, check if module already has hooks
    has_backward_hooks = bool(module._backward_hooks)
    has_full_backward_hooks = bool(module._backward_pre_hooks)

    # Remove all existing hooks to prevent conflicts
    if has_backward_hooks:
        module._backward_hooks.clear()
    if has_full_backward_hooks:
        module._backward_pre_hooks.clear()

    # Register the new safe hook
    safe_hook = safe_backward_hook(hook_fn)
    module.register_full_backward_hook(safe_hook)


def safe_hook_fn(module: nn.Module, grad_input: Tuple, grad_output: Tuple) -> Optional[Tuple]:
    """
    Simple safe hook function that just returns grad_input unchanged.
    
    Args:
        module: The module the hook is attached to
        grad_input: Gradients with respect to inputs
        grad_output: Gradients with respect to outputs
        
    Returns:
        Original grad_input unchanged
    """
    return grad_input


class SafeHookModule(nn.Module):
    """
    A wrapper module that ensures all backward hooks are safely registered.

    Args:
        module: The module to wrap
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        # Apply safe hooks to the wrapped module
        safe_register_hooks(module)

    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped module."""
        return self.module(*args, **kwargs) 