"""
Excitation-Inhibition Network architectures.

This module contains network architectures based on excitation-inhibition principles.
"""

from .ei_network import ExcitationInhibitionNetwork
from .mlp_ei_network import MLPExcInhNetwork

__all__ = [
    "ExcitationInhibitionNetwork",
    "MLPExcInhNetwork",
] 