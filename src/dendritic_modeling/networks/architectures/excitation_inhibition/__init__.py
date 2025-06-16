"""
Excitation-Inhibition Network architectures.

This module contains network architectures based on excitation-inhibition principles.
"""

from dendritic_modeling.networks.architectures.excitation_inhibition.ei_network import ExcitationInhibitionNetwork
from dendritic_modeling.networks.architectures.excitation_inhibition.mlp_ei_network import MLPExcInhNetwork

__all__ = [
    "ExcitationInhibitionNetwork",
    "MLPExcInhNetwork",
] 