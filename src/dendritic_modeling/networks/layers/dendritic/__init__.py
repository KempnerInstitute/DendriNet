"""
Dendritic-specific layer implementations.

This module provides layers for dendritic computation including
the DendriticBranchLayer and ExcitationInhibitionLayer.
"""

from .branch_layer import DendriticBranchLayer
from .excitation_inhibition import ExcitationInhibitionLayer

__all__ = [
    "DendriticBranchLayer",
    "ExcitationInhibitionLayer",
] 