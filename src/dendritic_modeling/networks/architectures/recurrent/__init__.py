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
from dendritic_modeling.networks.architectures.recurrent.multi_population import (
    MultiPopulationLayer,
    MultiPopulationLayerConfig,
    MultiPopulationLayerState,
    MultiPopulationNetwork,
    MultiPopulationNetworkConfig,
    PopulationProjectionConfig,
)
from dendritic_modeling.networks.architectures.recurrent.population_graph import (
    CellTypeConfig,
    PopulationConnectionConfig,
    PopulationGraphConfig,
    PopulationGraphNetwork,
    PopulationGraphState,
)
from dendritic_modeling.networks.architectures.recurrent.protocol import (
    RecurrentCore,
    SteppableRecurrentCore,
)
from dendritic_modeling.networks.architectures.recurrent.spiking_soma import (
    LIFSoma,
    SurrogateSpike,
)
from dendritic_modeling.networks.architectures.recurrent.stateful_dendrinet import (
    StatefulDendriNet,
)
from dendritic_modeling.networks.architectures.recurrent.synapse_types import (
    SynapseTypeConfig,
    SynapseTypeGroup,
    SynapseTypeSet,
    build_synapse_type_set,
)

__all__ = [
    "BaselineRNN",
    "BaselineRNNConfig",
    "CellTypeConfig",
    "DendriNetState",
    "EILayer",
    "EILayerConfig",
    "EINetwork",
    "EINetworkConfig",
    "EIState",
    "LIFSoma",
    "MultiPopulationLayer",
    "MultiPopulationLayerConfig",
    "MultiPopulationLayerState",
    "MultiPopulationNetwork",
    "MultiPopulationNetworkConfig",
    "PopulationConfig",
    "PopulationConnectionConfig",
    "PopulationGraphConfig",
    "PopulationGraphNetwork",
    "PopulationGraphState",
    "PopulationProjectionConfig",
    "RecurrentCore",
    "StatefulDendriNet",
    "SteppableRecurrentCore",
    "SurrogateSpike",
    "SynapseTypeConfig",
    "SynapseTypeGroup",
    "SynapseTypeSet",
    "build_synapse_type_set",
]
