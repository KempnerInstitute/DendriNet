"""
Dendritic-specific layer implementations.

This module provides layers for dendritic computation including
the DendriticBranchLayer and ExcitationInhibitionLayer.
"""

from dendritic_modeling.networks.layers.dendritic.branch_layer import DendriticBranchLayer
from dendritic_modeling.networks.layers.dendritic.excitation_inhibition import ExcitationInhibitionLayer

__all__ = [
    "DendriticBranchLayer",
    "ExcitationInhibitionLayer",
] 