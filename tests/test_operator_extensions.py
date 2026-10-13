"""Tests for operator-spec extensions: param_relu, soma override, MLP corner.

Every feature here is additive: with the new fields at their defaults the
constructed networks must be identical to the historical ones (same state-dict
keys and values, same forward), which is asserted explicitly.
"""

from __future__ import annotations

import copy

import pytest
import torch

from dendritic_modeling.networks.activations.factory import ActivationFactory
from dendritic_modeling.networks.activations.parametric import (
    ParametricReLU,
    ParametricTanh,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet import (
    DendriNet,
)
from dendritic_modeling.networks.architectures.factory import get_architecture


def _spatial_config() -> dict:
    return {
        "spatial": {"kernel_size": 3, "stride": 1, "padding": 1},
        "architecture": {
            "excitatory_layer_sizes": [8],
            "inhibitory_layer_sizes": [2],
            "excitatory_branch_factors": [2],
            "inhibitory_branch_factors": [2],
            "inhibitory_network_type": "dendritic",
            "mlp_inhibitory_network_params": {"hidden_dims": [], "activation": "relu"},
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [4],
            "ei_synapses_per_branch_per_layer": [2],
            "ie_synapses_per_branch_per_layer": [2],
            "ii_synapses_per_branch_per_layer": [0],
        },
        "transfer": {
            "independent_pathways": False,
            "excitatory_dim": None,
            "inhibitory_dim": None,
            "output_activation": None,
        },
        "morphology": {
            "somatic_synapses": False,
            "use_shunting": True,
            "weight_transform": "softplus",
            "dbl_init_method": "analytical_expectation",
        },
        "sparsity": {
            "type": "indexed",
            "init_method": "xavier_normal",
            "indexed": {"seed": 3},
        },
        "reactivation": {
            "enabled": True,
            "type": "param_tanh",
            "init_m": 1.5,
            "init_b": 0.5,
        },
        "blocklinear": {"efficient": True},
        "implementation": {"adaptive_initialization": False, "print_hooks": False},
    }


# ---------------------------------------------------------------------------
# param_relu activation
# ---------------------------------------------------------------------------
def test_param_relu_is_threshold_linear() -> None:
    act = ActivationFactory.create(
        act_type="param_relu", output_dim=4, init_m=2.0, init_b=0.5
    )
    assert isinstance(act, ParametricReLU)

    V = torch.tensor([[-1.0, 0.0, 0.5, 2.0]] * 1).repeat(1, 1)
    out = act(torch.tensor([[-1.0, 0.4, 0.5, 2.0]]))

    torch.testing.assert_close(out, torch.tensor([[0.0, 0.0, 0.0, 3.0]]))
    assert (out == 0).sum() == 3  # exact zeros below/at threshold
    del V

    act.initialize(5.0, 1.0)
    out = act(torch.tensor([[2.0, 0.0, 1.0, 3.0]]))
    torch.testing.assert_close(out, torch.tensor([[5.0, 0.0, 0.0, 10.0]]))


def test_param_relu_rejects_nonpositive_slope() -> None:
    with pytest.raises(ValueError, match="init_m"):
        ParametricReLU(4, init_m=0.0)


# ---------------------------------------------------------------------------
# soma-level reactivation override
# ---------------------------------------------------------------------------
def test_soma_override_default_is_identical_construction() -> None:
    base = _spatial_config()
    explicit = copy.deepcopy(base)
    explicit["reactivation"].update(
        {"soma_type": None, "soma_init_m": 1.0, "soma_init_b": 0.0}
    )

    torch.manual_seed(0)
    core_base = get_architecture("hierarchical_dendritic_conv", base, input_dim=6)
    torch.manual_seed(0)
    core_explicit = get_architecture(
        "hierarchical_dendritic_conv", explicit, input_dim=6
    )

    state_base = core_base.state_dict()
    state_explicit = core_explicit.state_dict()
    assert list(state_base) == list(state_explicit)
    for key in state_base:
        torch.testing.assert_close(state_base[key], state_explicit[key])

    x = torch.rand(2, 6, 8, 8)
    with torch.no_grad():
        torch.testing.assert_close(core_base(x), core_explicit(x))


def test_soma_override_targets_only_excitatory_soma() -> None:
    config = _spatial_config()
    config["reactivation"].update(
        {"soma_type": "param_relu", "soma_init_m": 4.0, "soma_init_b": 0.25}
    )
    torch.manual_seed(0)
    core = get_architecture("hierarchical_dendritic_conv", config, input_dim=6)

    excitatory = core.einet.layers[0].excitatory_cells.branch_layers
    inhibitory = core.einet.layers[0].inhibitory_cells.branch_layers

    assert isinstance(excitatory[-1].reactivation, ParametricReLU)
    assert all(
        isinstance(layer.reactivation, ParametricTanh) for layer in excitatory[:-1]
    )
    assert all(isinstance(layer.reactivation, ParametricTanh) for layer in inhibitory)

    with torch.no_grad():
        out = core(torch.rand(2, 6, 8, 8))
    assert out.shape == (2, 8, 8, 8)
    assert (out == 0).float().mean() > 0  # threshold-linear soma emits exact zeros


# ---------------------------------------------------------------------------
# operator config diagnostics
# ---------------------------------------------------------------------------
def test_operator_warnings_flag_subtle_combinations() -> None:
    from dendritic_modeling.config.operator_validation import (
        collect_operator_config_warnings,
    )

    clean = _spatial_config()
    assert collect_operator_config_warnings(clean) == []

    subtle = _spatial_config()
    subtle["morphology"]["weight_transform"] = "identity"
    subtle["reactivation"]["init_policy"] = "occupancy_quantile"
    subtle["reactivation"]["type"] = "param_relu"
    messages = collect_operator_config_warnings(subtle)
    assert any("signed weight transform" in m for m in messages)
    assert any("occupancy_quantile" in m for m in messages)

    disabled = _spatial_config()
    disabled["reactivation"]["enabled"] = False
    disabled["reactivation"]["soma_type"] = "param_relu"
    assert any("soma_type" in m for m in collect_operator_config_warnings(disabled))


# ---------------------------------------------------------------------------
# MLP corner: exact equivalence to a linear layer
# ---------------------------------------------------------------------------
def test_signed_flat_dendrinet_equals_linear_layer() -> None:
    in_features, hidden = 12, 6
    torch.manual_seed(0)
    net = DendriNet(
        n_soma=hidden,
        branch_factors=[1],
        excitatory_input_dim=in_features,
        excitatory_synapses_per_branch=in_features,
        inhibitory_input_dim=None,
        inhibitory_synapses_per_branch=None,
        reactivate=False,
        somatic_synapses=False,
        use_shunting=False,
        weight_transform="identity",
        adaptive_initialization=False,
    )

    weight = torch.randn(hidden, in_features)
    with torch.no_grad():
        net.branch_layers[0].branch_excitation.pre_w.copy_(weight)
        # Branch-to-soma coupling of exactly one: with the identity transform
        # the stored blocklinear weight is the applied weight.
        net.branch_layers[-1].branches_to_output.log_weight.fill_(1.0)

    x = torch.randn(5, in_features)
    with torch.no_grad():
        out = net(x)

    torch.testing.assert_close(out, x @ weight.t())
