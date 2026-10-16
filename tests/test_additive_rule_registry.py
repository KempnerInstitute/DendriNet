"""Additive integration rules that plugins register.

A registered rule is selected with ``additive_mode: <name>`` and configured with
``additive_options``; it must reach every native branch builder, keep its
equation and gradients, survive checkpoints, and leave the built-in rules and
their defaults unchanged.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

import pytest
import torch
import yaml
from torch import nn

from dendritic_modeling.config import config_from_mapping
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_dynamics import (
    ADDITIVE_MODES,
    AdditiveRule,
    available_additive_modes,
    get_additive_rule,
    normalize_additive_mode,
    register_additive_rule,
    unregister_additive_rule,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.training.strategies.local_learning import LocalCreditAssignment

RULE = "test_gain_scaled"


def _setup(layer, options):
    if options["start_log_gain"] > 0:
        raise ValueError("start_log_gain must be at most 0")
    layer.test_log_gain = nn.Parameter(
        torch.full((layer.n_branches,), float(options["start_log_gain"]))
    )
    layer.register_buffer("test_reference", torch.ones(layer.n_branches))
    layer.test_rule_inputs = []


def _voltage(
    layer,
    *,
    excitation,
    inhibition,
    child_current,
    child_conductance,
    raw_voltage,
    inputs,
):
    layer.test_rule_inputs.append(None if inputs is None else sorted(inputs))
    gain = layer.test_log_gain.exp() / layer.test_reference
    scale = layer.additive_options["scale"]
    return scale * raw_voltage / (1 + gain * torch.as_tensor(excitation).abs())


def _expected(branch, E, inhibitory, C):
    gain = branch.test_log_gain.exp() / branch.test_reference
    return (
        branch.additive_options["scale"] * (E - inhibitory + C) / (1 + gain * E.abs())
    )


@pytest.fixture
def rule():
    register_additive_rule(
        RULE,
        AdditiveRule(
            voltage=_voltage,
            options={"scale": 1.0, "start_log_gain": -1.0},
            setup=_setup,
        ),
    )
    yield RULE
    unregister_additive_rule(RULE)


def _family_config(family, options):
    integration = {
        "use_shunting": False,
        "additive_mode": RULE,
        "additive_options": options,
    }
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
                            reactivate=True,
                            reactivation_type="param_tanh",
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
    config = {
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
        "reactivation": {"enabled": True, "init_policy": "fixed", "type": "param_tanh"},
        "implementation": {"adaptive_initialization": False},
    }
    if family != "einet":
        config["spatial"] = {"kernel_size": 3, "stride": 1, "padding": 1}
    return config


_FAMILIES = [
    "einet",
    "population_network",
    "hierarchical_dendritic_conv",
    "hierarchical_dendritic_tensor_map",
]


def _branches(network):
    return [m for m in network.modules() if isinstance(m, DendriticBranchLayer)]


def _input(family):
    if family == "hierarchical_dendritic_conv":
        return torch.rand(2, 6, 4, 4, requires_grad=True)
    return torch.rand(2, 6, requires_grad=True)


@pytest.mark.usefixtures("rule")
@pytest.mark.parametrize("family", _FAMILIES)
def test_a_registered_rule_reaches_every_native_branch_with_its_gradients(family):
    config = yaml.safe_load(yaml.safe_dump(_family_config(family, {"scale": 0.5})))
    torch.manual_seed(918)
    network = get_architecture(family, deepcopy(config), input_dim=6)
    branches = _branches(network)
    assert len(branches) == 3
    for branch in branches:
        assert branch.additive_mode == RULE
        assert branch.additive_options == {"scale": 0.5, "start_log_gain": -1.0}
        # Parameters a rule creates belong to the branch, ahead of its synapses.
        names = list(branch.state_dict())
        assert names[:2] == ["test_log_gain", "test_reference"]
        base = torch.linspace(-0.4, 0.9, branch.n_branches).unsqueeze(0)
        E, inhibitory, C, G = [
            (base * scale).requires_grad_() for scale in (1.0, 0.7, 0.4, 0.6)
        ]
        observed, _ = branch.voltage_from_currents(E, 0, inhibitory, 0, C, G)
        # Voltages from stored currents carry no layer inputs.
        assert branch.test_rule_inputs[-1] is None
        expected = _expected(branch, E, inhibitory, C)
        torch.testing.assert_close(observed, expected)
        weights = torch.linspace(0.3, -0.4, branch.n_branches).unsqueeze(0)
        sources = (E, inhibitory, C, branch.test_log_gain)
        actual = torch.autograd.grad(
            (observed * weights).sum(), sources, retain_graph=True
        )
        reference = torch.autograd.grad((expected * weights).sum(), sources)
        for a, b in zip(actual, reference, strict=True):
            torch.testing.assert_close(a, b)

    for branch in branches:
        branch.test_rule_inputs.clear()
    x = _input(family)
    y = network(x)
    assert torch.isfinite(y).all()
    y.sum().backward()
    for branch in branches:
        # The forward pass hands the rule the layer's inputs.
        assert branch.test_rule_inputs
        assert branch.test_rule_inputs[-1] == ["branch_input", "inhibitory_input", "x"]
        assert branch.test_log_gain.grad is not None
        assert torch.isfinite(branch.test_log_gain.grad).all()
    assert torch.isfinite(x.grad).all()

    # A checkpoint restores the rule's parameters and buffers.
    with torch.no_grad():
        for branch in branches:
            branch.test_log_gain.fill_(-0.25)
            branch.test_reference.fill_(2.0)
    restored = get_architecture(family, deepcopy(config), input_dim=6)
    restored.load_state_dict(network.state_dict())
    probe = _input(family).detach()
    assert torch.equal(network(probe), restored(probe))


@pytest.mark.usefixtures("rule")
def test_the_configuration_carries_a_registered_rule_to_the_model():
    config = config_from_mapping(
        {
            "model": {
                "core": {
                    "biological_neuron": False,
                    "morphology": {
                        "use_shunting": False,
                        "additive_mode": RULE,
                        "additive_options": {"scale": 2.0},
                    },
                }
            }
        }
    )
    assert config.model.core.morphology.additive_options == {"scale": 2.0}
    assert asdict(config_from_mapping(asdict(config))) == asdict(config)
    model, _ = initialize_model(config.model)
    branches = _branches(model)
    assert branches
    for branch in branches:
        assert branch.additive_mode == RULE
        assert branch.additive_options == {"scale": 2.0, "start_log_gain": -1.0}


@pytest.mark.usefixtures("rule")
@pytest.mark.parametrize(
    "options, match",
    [
        ({"additive_mode": RULE, "additive_options": {"scael": 2.0}}, "no options"),
        (
            {"additive_mode": RULE, "additive_options": {"start_log_gain": 1.0}},
            "at most 0",
        ),
        ({"additive_mode": "raw", "additive_options": {"scale": 2.0}}, "takes none"),
        (
            {
                "additive_mode": "conductance_normalized",
                "additive_options": {"scale": 2.0},
            },
            "takes none",
        ),
        (
            {"additive_mode": RULE, "use_additive_normalization": True},
            "use_additive_normalization",
        ),
    ],
)
def test_options_are_checked_against_the_selected_rule(options, match):
    with pytest.raises(ValueError, match=match):
        DendriticBranchLayer(
            output_dim=2,
            excitatory_input_dim=4,
            excitatory_synapses_per_branch=2,
            use_shunting=False,
            reactivate=False,
            **options,
        )


@pytest.mark.usefixtures("rule")
def test_a_registered_rule_is_additive_only():
    with pytest.raises(ValueError, match="applies only to additive"):
        DendriticBranchLayer(
            output_dim=2,
            excitatory_input_dim=4,
            excitatory_synapses_per_branch=2,
            use_shunting=True,
            additive_mode=RULE,
        )


@pytest.mark.usefixtures("rule")
def test_analytic_local_updates_reject_a_registered_rule():
    model = DendriticBranchLayer(
        output_dim=2,
        excitatory_input_dim=4,
        excitatory_synapses_per_branch=2,
        use_shunting=False,
        additive_mode=RULE,
        reactivate=False,
    )
    trainer = LocalCreditAssignment.__new__(LocalCreditAssignment)
    trainer._decoder_cache = {}
    with pytest.raises(ValueError, match=r"analytic local.*additive_mode"):
        trainer._attach_local_recorders(model)


def test_the_registry_keeps_built_in_names_and_names_rules_in_errors(rule):
    assert get_additive_rule(rule) is not None
    assert get_additive_rule("raw") is None
    assert available_additive_modes() == ADDITIVE_MODES | {rule}
    assert normalize_additive_mode(" Test_Gain_Scaled ") == rule
    with pytest.raises(ValueError, match="built in"):
        register_additive_rule("raw", AdditiveRule(voltage=_voltage))
    with pytest.raises(ValueError, match="already registered"):
        register_additive_rule(rule, AdditiveRule(voltage=_voltage))
    with pytest.raises(TypeError, match="AdditiveRule"):
        register_additive_rule("other_rule", _voltage)
    with pytest.raises(ValueError, match=rule):
        normalize_additive_mode("not_a_rule")


def test_a_rule_from_a_plugin_is_found_when_first_requested(monkeypatch):
    import dendritic_modeling.plugins as plugins

    class EntryPoint:
        name = "rules"
        value = "tests.rules:register"
        dist = None

        @staticmethod
        def load():
            return lambda: register_additive_rule(
                "plugin_rule", AdditiveRule(voltage=_voltage)
            )

    monkeypatch.setattr(plugins, "_LOADED", {})
    monkeypatch.setitem(plugins._STATE, "discovered", False)
    monkeypatch.setattr(plugins, "entry_points", lambda group: [EntryPoint])
    monkeypatch.delenv(plugins.PLUGINS_ENVIRONMENT_VARIABLE, raising=False)
    try:
        assert normalize_additive_mode("plugin_rule") == "plugin_rule"
    finally:
        unregister_additive_rule("plugin_rule")


@pytest.mark.filterwarnings("ignore:Additive integration:UserWarning")
@pytest.mark.parametrize("family", ["einet", "population_network"])
def test_empty_additive_options_leave_random_state_outputs_and_gradients_identical(
    family,
):
    config = _family_config(family, {})
    options = (
        config["morphology"]
        if family == "einet"
        else config["population_network"]["layers"][0]["population_defaults"]
    )
    options["additive_mode"] = "conductance_normalized"
    implicit = deepcopy(config)
    implicit_options = (
        implicit["morphology"]
        if family == "einet"
        else implicit["population_network"]["layers"][0]["population_defaults"]
    )
    implicit_options.pop("additive_options")
    torch.manual_seed(19)
    first = get_architecture(family, deepcopy(implicit), input_dim=6)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(19)
    second = get_architecture(family, deepcopy(config), input_dim=6)
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


@pytest.mark.parametrize("weight_transform", ["identity", "softplus"])
@pytest.mark.parametrize("weight_norm_order", [None, 2])
def test_projection_with_other_weights_uses_the_layer_contacts(
    weight_transform, weight_norm_order
):
    torch.manual_seed(5)
    layer = IndexedSparseLinear(
        in_features=7,
        out_features=4,
        K=3,
        weight_transform=weight_transform,
        weight_norm_order=weight_norm_order,
    )
    x = torch.randn(2, 5, 7, requires_grad=True)
    weight = layer.effective_sparse_weight()
    torch.testing.assert_close(layer.project_with_weights(x, weight), layer(x))
    torch.testing.assert_close(
        x @ layer.dense_weight().T, layer.project_with_weights(x, weight)
    )

    magnitude = layer.project_with_weights(x.abs(), weight.abs())
    dense = torch.zeros(4, 7).scatter_add_(
        1, layer.connection_indices, weight.detach().abs()
    )
    torch.testing.assert_close(magnitude, x.detach().abs() @ dense.T)
    magnitude.sum().backward()
    assert x.grad is not None and layer.pre_w.grad is not None

    with pytest.raises(ValueError, match="weight must have shape"):
        layer.project_with_weights(x, weight[:, :2])
    with pytest.raises(ValueError, match="dimension mismatch"):
        layer.project_with_weights(x[..., :6], weight)
