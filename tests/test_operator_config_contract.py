"""Exercise integration-rule choices through the supported configuration builders."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
import yaml

from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    PopulationConfig,
)
from dendritic_modeling.networks.architectures.recurrent.stateful_dendrinet import (
    StatefulDendriNet,
)
from dendritic_modeling.training.strategies.local_learning import LocalCreditAssignment


def _config(family, rule, gate):
    mode = rule if rule in {"tangent_matched", "conductance_normalized"} else "raw"
    integration = {
        "use_shunting": rule == "shunting",
        "additive_mode": mode,
        "use_additive_normalization": rule == "legacy_zscore",
    }
    if mode == "tangent_matched":
        integration.update(additive_tangent_n0=0.4, additive_tangent_t0=1.2)
    if family == "population_network":
        return {
            "biological_neuron": False,
            "population_network": {
                "input_transform": "identity",
                "layers": [
                    {
                        "name": "layer0",
                        "readout_population": "e",
                        "population_defaults": dict(
                            integration,
                            ff_excitatory_synapses=2,
                            ff_inhibitory_synapses=0,
                            rec_excitatory_synapses=0,
                            rec_inhibitory_synapses=0,
                            reactivate=gate != "identity",
                            reactivation_type=(
                                "param_tanh" if gate == "identity" else gate
                            ),
                            reactivation_init_policy="fixed",
                            somatic_synapses=False,
                            adaptive_initialization=False,
                            dbl_init_method="default",
                        ),
                        "populations": [
                            {
                                "name": "e",
                                "polarity": "excitatory",
                                "n_neurons": 3,
                                "branch_factors": [2, 2],
                            }
                        ],
                        "connections": [{"source": "input", "target": "e"}],
                    }
                ],
            },
        }
    return {
        "type": "einet",
        "biological_neuron": False,
        "architecture": {
            "excitatory_layer_sizes": [3],
            "inhibitory_layer_sizes": [],
            "inhibitory_branch_factors": [2, 2],
            "excitatory_branch_factors": [2, 2],
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [2],
            "ie_synapses_per_branch_per_layer": [0],
        },
        "morphology": dict(
            integration,
            somatic_synapses=False,
            weight_transform="softplus",
            dbl_init_method="default",
        ),
        "reactivation": {
            "enabled": gate != "identity",
            "init_policy": "fixed",
            "type": "param_tanh" if gate == "identity" else gate,
        },
        "implementation": {"adaptive_initialization": False},
    }


# Spatial cores apply an EINet core configuration to every patch.
_SPATIAL_FAMILIES = ("hierarchical_dendritic_conv", "hierarchical_dendritic_tensor_map")


def _builder_config(family, rule, gate):
    if family not in _SPATIAL_FAMILIES:
        return _config(family, rule, gate)
    config = _config("einet", rule, gate)
    config["spatial"] = {"kernel_size": 3, "stride": 1, "padding": 1}
    return config


@pytest.mark.parametrize("family", ["einet", "population_network", *_SPATIAL_FAMILIES])
@pytest.mark.parametrize(
    "rule",
    ["shunting", "raw", "legacy_zscore", "conductance_normalized", "tangent_matched"],
)
@pytest.mark.parametrize("gate", ["identity", "param_tanh", "param_relu"])
def test_yaml_rule_reaches_every_native_branch_and_preserves_gradients(
    family, rule, gate
):
    config = yaml.safe_load(yaml.safe_dump(_builder_config(family, rule, gate)))
    torch.manual_seed(918)
    network = get_architecture(family, deepcopy(config), input_dim=6)
    branches = [m for m in network.modules() if isinstance(m, DendriticBranchLayer)]
    assert len(branches) == 3
    for branch in branches:
        assert branch.use_shunting == (rule == "shunting")
        assert branch.use_additive_normalization == (rule == "legacy_zscore")
        assert branch.additive_mode == (
            rule if rule in {"tangent_matched", "conductance_normalized"} else "raw"
        )
        if rule == "tangent_matched":
            assert branch.additive_operating_point == pytest.approx((0.4, 1.2))
        # Check the voltage equation AND its current derivatives, independently
        # of the composed network's readout and activation.
        E, inhibitory, C, G = [
            torch.tensor([[0.2, 0.4, 0.9]], requires_grad=True) * scale
            for scale in (1.0, 0.7, 0.4, 0.6)
        ]
        observed, _ = branch.voltage_from_currents(E, 0, inhibitory, 0, C, G)
        n, t = E + C, E + inhibitory + G
        if rule == "shunting":
            expected = n / (1 + t + branch.epsilon)
        elif rule == "tangent_matched":
            d0 = 1 + 1.2 + branch.epsilon
            v0 = 0.4 / d0
            expected = v0 + ((n - 0.4) - v0 * (t - 1.2)) / d0
        else:
            expected = E - inhibitory + C
            if rule == "conductance_normalized":
                expected = expected / (1 + t + branch.epsilon)
            elif rule == "legacy_zscore":
                expected = (expected - expected.mean(-1, keepdim=True)) / expected.std(
                    -1, keepdim=True, unbiased=False
                ).clamp_min(branch.epsilon)
        torch.testing.assert_close(observed, expected)
        weights = torch.tensor([[0.3, 0.9, -0.4]])
        actual = torch.autograd.grad(
            (observed * weights).sum(),
            (E, inhibitory, C, G),
            retain_graph=True,
            allow_unused=True,
        )
        reference = torch.autograd.grad(
            (expected * weights).sum(), (E, inhibitory, C, G), allow_unused=True
        )
        for a, b in zip(actual, reference, strict=True):
            if a is None or b is None:
                assert a is None and b is None
            else:
                torch.testing.assert_close(a, b)
    if family == "hierarchical_dendritic_conv":
        x, expected_shape = torch.rand(2, 6, 4, 4, requires_grad=True), (2, 3, 4, 4)
    else:
        x, expected_shape = torch.rand(2, 6, requires_grad=True), (2, 3)
    y = network(x)
    assert y.shape == expected_shape
    assert torch.isfinite(y).all()
    assert torch.isfinite(torch.autograd.grad(y.sum(), x)[0]).all()


@pytest.mark.parametrize("denominator_anchors", [[1.2], [1.2, 1.4, 1.6, 1.8]])
def test_population_rejects_wrong_denominator_anchor_count(denominator_anchors):
    with pytest.raises(ValueError, match="exactly 3"):
        StatefulDendriNet(
            PopulationConfig(
                n_neurons=2,
                branch_factors=[2, 2],
                ff_excitatory_synapses=1,
                use_shunting=False,
                additive_mode="tangent_matched",
                additive_tangent_n0_by_depth=[0.2, 0.4, 0.6],
                additive_tangent_t0_by_depth=denominator_anchors,
            ),
            excitatory_input_dim=4,
        )


@pytest.mark.parametrize("rule", ["tangent_matched", "conductance_normalized"])
def test_local_analytic_updates_reject_unimplemented_integration_derivatives(rule):
    anchors = (
        {"additive_tangent_n0": 0.4, "additive_tangent_t0": 1.2}
        if rule == "tangent_matched"
        else {}
    )
    model = DendriticBranchLayer(
        output_dim=2,
        excitatory_input_dim=4,
        excitatory_synapses_per_branch=2,
        use_shunting=False,
        additive_mode=rule,
        reactivate=False,
        **anchors,
    )
    trainer = LocalCreditAssignment.__new__(LocalCreditAssignment)
    trainer._decoder_cache = {}
    with pytest.raises(ValueError, match=r"analytic local.*additive_mode"):
        trainer._attach_local_recorders(model)
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


@pytest.mark.parametrize("family", ["einet", "population_network"])
def test_factory_applies_soma_first_tangent_anchors(family):
    config = _config(family, "tangent_matched", "identity")
    options = (
        config["morphology"]
        if family == "einet"
        else config["population_network"]["layers"][0]["population_defaults"]
    )
    options.pop("additive_tangent_n0")
    options.pop("additive_tangent_t0")
    options.update(
        additive_tangent_n0_by_depth=[0.2, 0.4, 0.6],
        additive_tangent_t0_by_depth=[1.2, 1.4, 1.6],
    )
    network = get_architecture(family, config, input_dim=6)
    branches = [m for m in network.modules() if isinstance(m, DendriticBranchLayer)]
    for branch, anchor in zip(
        branches, [(0.6, 1.6), (0.4, 1.4), (0.2, 1.2)], strict=True
    ):
        assert branch.additive_operating_point == pytest.approx(anchor)
    assert torch.isfinite(network(torch.rand(2, 6))).all()


@pytest.mark.parametrize("family", ["einet", "population_network"])
def test_empty_anchor_defaults_leave_random_state_outputs_and_gradients_identical(
    family,
):
    config = _config(family, "shunting", "param_tanh")
    explicit = deepcopy(config)
    options = (
        explicit["morphology"]
        if family == "einet"
        else explicit["population_network"]["layers"][0]["population_defaults"]
    )
    options.update(additive_tangent_n0_by_depth=[], additive_tangent_t0_by_depth=[])
    torch.manual_seed(19)
    first = get_architecture(family, deepcopy(config), input_dim=6)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(19)
    second = get_architecture(family, deepcopy(explicit), input_dim=6)
    assert torch.equal(rng, torch.get_rng_state())
    assert first.state_dict().keys() == second.state_dict().keys()
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key]), key
    x = torch.rand(3, 6, requires_grad=True)
    a, b = first(x), second(x)
    assert torch.equal(a, b)
    assert torch.equal(
        torch.autograd.grad(a.sum(), x)[0], torch.autograd.grad(b.sum(), x)[0]
    )


def test_typed_feedforward_config_keeps_depth_anchors():
    from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet import (
        DendriNet,
    )
    from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet_config import (
        DendriNetConfig,
    )

    config = DendriNetConfig.from_kwargs(
        n_soma=2,
        branch_factors=[2],
        excitatory_input_dim=4,
        excitatory_synapses_per_branch=1,
        use_shunting=False,
        additive_mode="tangent_matched",
        reactivate=False,
        additive_tangent_n0_by_depth=[0.2, 0.4],
        additive_tangent_t0_by_depth=[1.2, 1.4],
    )
    assert DendriNetConfig.from_kwargs(config.to_kwargs()) == config
    network = DendriNet(config=config)
    assert network.branch_layers[0].additive_operating_point == pytest.approx(
        (0.4, 1.4)
    )
    assert network.branch_layers[1].additive_operating_point == pytest.approx(
        (0.2, 1.2)
    )


@pytest.mark.parametrize(
    "options, match",
    [
        (
            {"use_shunting": True, "additive_mode": "conductance_normalized"},
            "additive_mode",
        ),
        ({"use_shunting": True, "use_additive_normalization": True}, "z-score"),
        (
            {
                "use_shunting": False,
                "additive_mode": "conductance_normalized",
                "additive_tangent_n0": 0.4,
                "additive_tangent_t0": 1.2,
            },
            "tangent_matched",
        ),
        (
            {
                "use_shunting": True,
                "additive_tangent_n0": 0.4,
                "additive_tangent_t0": 1.2,
            },
            "tangent_matched",
        ),
        (
            {
                "use_shunting": False,
                "blocklinear_strategy": "conductance_dynamic",
                "weight_transform": "relu",
            },
            "relu weight transform",
        ),
    ],
)
def test_contradictory_integration_options_raise(options, match):
    with pytest.raises(ValueError, match=match):
        DendriticBranchLayer(
            output_dim=2,
            excitatory_input_dim=4,
            excitatory_synapses_per_branch=2,
            input_branch_factor=1,
            reactivate=False,
            **options,
        )


@pytest.mark.parametrize(
    "option",
    [
        {"leak_conductance": 2.0},
        {"learnable_leak": True},
        {"reactivation_type": "endpoint_tanh", "reactivation_init_policy": "fixed"},
        {"topk_type": "indexed_release"},
        {"signal_rounding": "bfloat16", "topk_type": "indexed_release"},
    ],
    ids=lambda option: next(iter(option)),
)
@pytest.mark.parametrize(
    "context",
    [
        {"dynamics_mode": "spike"},
        {"dendritic_spikes_enabled": True},
        {"soma_feedback_enabled": True},
        {"cross_level_mode": "parallel_readout"},
        "recurrent",
    ],
    ids=lambda context: context if isinstance(context, str) else next(iter(context)),
)
def test_stateful_population_keeps_extended_compartments_to_feedforward_rate_cells(
    option, context
):
    """Leak, endpoint gates, release synapses and rounding are qualified only
    for feedforward hierarchical rate populations; anything else is rejected."""
    population = {
        "n_neurons": 2,
        "branch_factors": [2],
        "ff_excitatory_synapses": 1,
        **option,
    }
    recurrent = {}
    if context == "recurrent":
        population["rec_excitatory_synapses"] = 1
        recurrent = {"recurrent_excitatory_input_dim": 2}
    else:
        population.update(context)
    with pytest.raises(ValueError, match="feedforward hierarchical rate populations"):
        StatefulDendriNet(
            PopulationConfig(**population), excitatory_input_dim=4, **recurrent
        )
    # The same option is accepted on a feedforward hierarchical rate population.
    feedforward = {
        key: value for key, value in population.items() if key not in context
    }
    feedforward.pop("rec_excitatory_synapses", None)
    StatefulDendriNet(PopulationConfig(**feedforward), excitatory_input_dim=4)
