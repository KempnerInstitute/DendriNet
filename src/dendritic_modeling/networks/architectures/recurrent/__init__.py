"""
Recurrent architecture module for dendritic modeling.

Exports:
    DendriticRNNCell: Single-timestep recurrent cell with per-branch dynamics.
    DendriticRNNNetwork: Sequence-level wrapper implementing BaseNetwork.
    RecurrentState: Typed state container for the cell.
"""

from dendritic_modeling.networks.architectures.recurrent.rnn_cell import (
    DendriticRNNCell,
)
from dendritic_modeling.networks.architectures.recurrent.rnn_network import (
    DendriticRNNNetwork,
)
from dendritic_modeling.networks.architectures.recurrent.state import RecurrentState

__all__ = ["DendriticRNNCell", "DendriticRNNNetwork", "RecurrentState"]
