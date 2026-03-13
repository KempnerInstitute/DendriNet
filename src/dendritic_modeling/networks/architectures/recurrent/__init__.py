"""
Recurrent architecture module for dendritic modeling.

Unified E-I recurrent architecture with hierarchical dendrites, per-level
temporal traces, and proper 4-compartment synaptic structure (FF E, FF I,
REC E, REC I) per neuron.
"""

from dendritic_modeling.networks.architectures.recurrent.baseline_rnn import (
    BaselineRNN,
    BaselineRNNConfig,
)
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    EILayerConfig,
    EINetworkConfig,
    PopulationConfig,
)
from dendritic_modeling.networks.architectures.recurrent.ei_layer import EILayer
from dendritic_modeling.networks.architectures.recurrent.ei_network import EINetwork
from dendritic_modeling.networks.architectures.recurrent.ei_state import (
    DendriNetState,
    EIState,
)
from dendritic_modeling.networks.architectures.recurrent.stateful_dendrinet import (
    StatefulDendriNet,
)

__all__ = [
    "BaselineRNN",
    "BaselineRNNConfig",
    "DendriNetState",
    "EILayer",
    "EILayerConfig",
    "EINetwork",
    "EINetworkConfig",
    "EIState",
    "PopulationConfig",
    "StatefulDendriNet",
]
