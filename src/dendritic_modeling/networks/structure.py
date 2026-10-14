"""Queries about the dendritic-tree structure of a built model."""

from __future__ import annotations

import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
    DendriticBranchLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet import (
    DendriNet,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.ei_network import (
    ExcitationInhibitionNetwork,
)
from dendritic_modeling.networks.architectures.recurrent.population_network import (
    PopulationNetwork,
)


def inhibitory_branch_layer_ids(model: nn.Module) -> set[int]:
    """Return the ids of the branch layers of inhibitory cell populations.

    Inhibitory populations are the ``inhibitory_cells`` of feedforward EINet
    layers, the ``i_population`` of recurrent EI layers and the
    inhibitory-polarity populations of population-network layers.
    """
    populations: list[nn.Module] = []
    for module in model.modules():
        for attribute in ("inhibitory_cells", "i_population"):
            cells = getattr(module, attribute, None)
            if isinstance(cells, nn.Module):
                populations.append(cells)
        members = getattr(module, "populations", None)
        if isinstance(members, nn.ModuleDict):
            populations.extend(
                members[name]
                for name in getattr(module, "inhibitory_names", ())
                if name in members
            )
    return {
        id(layer)
        for population in populations
        for layer in population.modules()
        if isinstance(layer, DendriticBranchLayer)
    }


def dendritic_tree_of(model: nn.Module) -> dict[int, int]:
    """Map each branch layer's id to the id of the dendritic tree that owns it.

    A tree is a module holding its levels in a ``branch_layers`` list (a
    DendriNet or a population), ordered from the distal level to the soma.
    """
    return {
        id(layer): id(owner)
        for owner in model.modules()
        if isinstance(getattr(owner, "branch_layers", None), nn.ModuleList)
        for layer in owner.branch_layers
        if isinstance(layer, DendriticBranchLayer)
    }


def readout_tree(model: nn.Module) -> nn.Module | None:
    """Return the dendritic tree whose somas feed the model output, if known."""
    core = getattr(model, "core_network", model)
    if isinstance(core, ExcitationInhibitionNetwork) and len(core.layers):
        return core.layers[-1].excitatory_cells
    if isinstance(core, PopulationNetwork) and len(core.layers):
        last = core.layers[-1]
        return last.populations[last.readout_population]
    if isinstance(core, DendriNet):
        return core
    return None


__all__ = ["dendritic_tree_of", "inhibitory_branch_layer_ids", "readout_tree"]
