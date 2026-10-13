"""The information analysis applies each layer's own voltage rule to projected signals.

Fed the currents the layer recorded in its forward pass, the projection must
reproduce the layer's output, for every integration rule, a non-unit leak and
non-leaf layers whose child coupling conductance enters the denominator.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch
from test_operator_config_contract import _config

from dendritic_modeling.analysis.core.information_parts.information_signal_projection import (
    compute_projected_branch_output,
)
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.factory import get_architecture


def _network(rule, gate, leak):
    config = deepcopy(_config("einet", rule, gate))
    config["morphology"]["leak_conductance"] = leak
    torch.manual_seed(31)
    return get_architecture("einet", config, input_dim=6).eval()


def _recorded(record, name):
    """A recorded current, or zeros for a pathway the layer does not have."""
    value = record[name]
    if value is None:
        return np.zeros(record["post_gate_output"].shape)
    return value.double().numpy()


@pytest.mark.parametrize("leak", [1.0, 0.3])
@pytest.mark.parametrize(
    "rule",
    ["shunting", "raw", "legacy_zscore", "conductance_normalized", "tangent_matched"],
)
@pytest.mark.parametrize("gate", ["identity", "param_tanh"])
def test_projection_of_recorded_currents_reproduces_each_layer(rule, gate, leak):
    if leak != 1.0 and rule != "shunting":
        pytest.skip("a non-unit leak requires shunting integration")
    network = _network(rule, gate, leak)
    branches = [m for m in network.modules() if isinstance(m, DendriticBranchLayer)]
    for branch in branches:
        branch._store_analysis_currents = True
    with torch.no_grad():
        network(torch.rand(5, 6, generator=torch.Generator().manual_seed(2)))

    non_leaf = 0
    for branch in branches:
        record = branch._last_analysis_currents
        projected = compute_projected_branch_output(
            branch,
            *(
                _recorded(record, name)
                for name in (
                    "pre_gate_excitation_current",
                    "pre_gate_inhibition_current",
                    "pre_gate_upstream_current",
                )
            ),
        )
        np.testing.assert_allclose(
            projected, record["post_gate_output"].double().numpy(), rtol=1e-5, atol=1e-6
        )
        non_leaf += bool(branch.input_branches)
    assert non_leaf >= 2  # the soma and the middle level have child branches


@pytest.mark.parametrize(
    "rule, leak",
    [
        ("shunting", 1.0),
        ("shunting", 0.3),
        ("raw", 1.0),
        ("legacy_zscore", 1.0),
        ("conductance_normalized", 1.0),
        ("tangent_matched", 1.0),
    ],
)
def test_receptive_field_recomposition_reproduces_the_soma(rule, leak):
    """Without inhibitory input, conditioning on the excitation is the forward."""
    from dendritic_modeling.analysis.tools.receptive_fields import (
        ReceptiveFieldAnalysisParams,
        ReceptiveFieldAnalyzer,
    )

    network = _network(rule, "param_tanh", leak)
    dendrinet = network.layers[0].excitatory_cells
    x = torch.rand(5, 6, generator=torch.Generator().manual_seed(2))
    analyzer = ReceptiveFieldAnalyzer(ReceptiveFieldAnalysisParams())

    records = analyzer._collect_component_branch_records(network, dendrinet, x)
    recomposed = analyzer._conditioned_soma_from_records(dendrinet, records, vary="exc")

    torch.testing.assert_close(recomposed, records[-1]["out"], rtol=1e-5, atol=1e-6)
