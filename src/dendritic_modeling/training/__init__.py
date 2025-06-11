"""
Training infrastructure for dendritic models.

This module contains all training-related functionality including strategies,
optimizers, and utilities.
"""

from . import optimizers
from . import strategies

__all__ = ["optimizers", "strategies"] 