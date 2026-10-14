"""Construction accepts exactly the supported additive integration modes."""

import pytest

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_dynamics import (
    ADDITIVE_MODES,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet import (
    DendriNet,
)


def test_the_supported_additive_modes_are_exactly_these():
    # Adding an integration rule is a reviewed change: extend this set together
    # with the rule's equation, gradient and default-identity tests.
    assert ADDITIVE_MODES == {"raw", "tangent_matched", "conductance_normalized"}


@pytest.mark.parametrize("mode", ["divisive", "gain", "unknown_rule"])
@pytest.mark.parametrize("use_shunting", [False, True])
def test_unknown_additive_modes_are_rejected_at_construction(mode, use_shunting):
    with pytest.raises(ValueError, match="additive_mode"):
        DendriNet(
            n_soma=2,
            branch_factors=[2, 2],
            excitatory_input_dim=8,
            excitatory_synapses_per_branch=4,
            inhibitory_input_dim=8,
            inhibitory_synapses_per_branch=2,
            use_shunting=use_shunting,
            additive_mode=mode,
        )
