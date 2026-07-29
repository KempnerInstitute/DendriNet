from types import SimpleNamespace

import pytest

from dendritic_modeling.analysis.utils.dendritic_depth import (
    soma_relative_dendritic_depth,
)
from dendritic_modeling.networks.architectures.recurrent import (
    PopulationConfig,
    StatefulDendriNet,
)


@pytest.mark.parametrize("depth", [0, 1, 4])
def test_soma_relative_dendritic_depth_uses_layer_idx_directly(depth):
    assert soma_relative_dendritic_depth(SimpleNamespace(layer_idx=depth)) == depth


@pytest.mark.parametrize("depth", [-1, True, 1.5, None])
def test_soma_relative_dendritic_depth_rejects_ambiguous_indices(depth):
    with pytest.raises(ValueError, match="soma-relative"):
        soma_relative_dendritic_depth(SimpleNamespace(layer_idx=depth))


def test_stateful_dendrinet_assigns_soma_relative_depths():
    net = StatefulDendriNet(
        pop_config=PopulationConfig(
            n_neurons=3,
            branch_factors=[2, 3],
            ff_excitatory_synapses=2,
            rec_excitatory_synapses=0,
        ),
        excitatory_input_dim=4,
    )

    assert [layer.layer_idx for layer in net.branch_layers] == [2, 1, 0]
