"""Tree-structure queries name the readout tree and each branch layer's tree."""

from __future__ import annotations

import pytest
import torch
from test_biological_constraints import _einet, _population
from test_trainer_configuration import IN_DIM, _wrap

from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.networks.structure import dendritic_tree_of, readout_tree


@pytest.mark.parametrize(
    "core_type, core, readout_name",
    [
        ("einet", _einet(), "core_network.layers.0.excitatory_cells"),
        ("population_network", _population(), "core_network.layers.0.populations.e"),
    ],
)
def test_readout_tree_is_the_population_feeding_the_output(
    core_type, core, readout_name
):
    torch.manual_seed(0)
    model = _wrap(get_architecture(core_type, core, input_dim=IN_DIM))
    names = {id(module): name for name, module in model.named_modules()}
    assert names[id(readout_tree(model))] == readout_name

    tree_of = dendritic_tree_of(model)
    branch_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, DendriticBranchLayer)
    ]
    assert all(id(module) in tree_of for _, module in branch_layers)
    assert {
        name
        for name, module in branch_layers
        if tree_of[id(module)] == id(readout_tree(model))
    } == {name for name, _ in branch_layers if name.startswith(f"{readout_name}.")}
