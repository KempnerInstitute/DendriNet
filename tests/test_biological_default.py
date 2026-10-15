"""Biological defaults, historical opt-out and explicit input contracts."""

import copy

import pytest
import torch
from test_biological_constraints import _einet, _population, _recurrent
from torch import nn

from dendritic_modeling.config import CONFIG_VERSION, config_from_mapping
from dendritic_modeling.config.model import CoreConfig
from dendritic_modeling.config.operator_card import build_operator_card
from dendritic_modeling.networks.architectures.biological_constraints import (
    biological_violations,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform.transfer import (
    TransferLayer,
)
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.networks.architectures.replacement.adapters import (
    NonNegativeInputAdapter,
    SpatialNonNegativeInputAdapter,
)
from dendritic_modeling.networks.utils.nonnegative_linear import NonnegativeLinear
from dendritic_modeling.scripts.tools.upgrade_config import upgrade_mapping


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("einet", True),
        ("dendritic_additive", True),
        ("population_network", True),
        ("dendritic_signed", False),
        ("MLP", False),
        ("gru", False),
    ],
)
def test_typed_and_yaml_defaults(kind, expected):
    assert CoreConfig(type=kind).biological_neuron is expected
    payload = {"model": {"core": {"type": kind}}}
    if kind == "population_network":
        payload["model"]["core"].update(_population())
    assert config_from_mapping(payload).model.core.biological_neuron is expected


@pytest.mark.parametrize(
    "kind, core",
    [
        ("einet", _einet(**{"transfer.output_activation": None})),
        ("population_network", _population(input_transform="identity")),
    ],
)
def test_factory_default_enforces_and_explicit_false_preserves_exploration(kind, core):
    with pytest.raises(ValueError, match="biological constraint set"):
        get_architecture(kind, core, input_dim=8)
    assert get_architecture(kind, {**core, "biological_neuron": False}, input_dim=8)


@pytest.mark.parametrize("version", [1, 2, 3])
@pytest.mark.parametrize("written", [None, False, True])
def test_upgrade_freezes_historical_declaration_and_is_idempotent(version, written):
    data = {
        "config_version": version,
        "model": {"core": {"type": "population_network", **_population()}},
    }
    if written is not None:
        data["model"]["core"]["biological_neuron"] = written
    upgraded, _, unresolved = upgrade_mapping(data)
    assert not unresolved
    assert upgraded["config_version"] == CONFIG_VERSION
    assert upgraded["model"]["core"]["biological_neuron"] is (written is True)
    assert upgrade_mapping(upgraded) == (upgraded, [], [])


@pytest.mark.parametrize("spatial", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_checked_input_preserves_values_and_derivatives_including_zero(spatial, device):
    x = (
        torch.tensor([0.0, 0.2, 1.0], dtype=torch.float64).reshape(1, 3, 1, 1)
        if spatial
        else torch.tensor([[0.0, 0.2, 1.0]], dtype=torch.float64)
    )
    x = x.to(device).requires_grad_()
    adapter = (
        SpatialNonNegativeInputAdapter(3, "nonnegative")
        if spatial
        else NonNegativeInputAdapter(3, "nonnegative")
    )
    y = adapter(x)
    assert y is x
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    for bad in (-0.001, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite values >= 0"):
            adapter(torch.full_like(x, bad))


@pytest.mark.parametrize(
    "kind, base",
    [
        ("einet", _einet()),
        ("einet", _recurrent(_einet())),
        ("population_network", _population()),
    ],
)
def test_explicit_biological_flag_does_not_change_state_outputs_or_gradients(
    kind, base
):
    models = []
    for flag in (False, True):
        torch.manual_seed(21)
        models.append(
            get_architecture(
                kind, {**copy.deepcopy(base), "biological_neuron": flag}, input_dim=8
            ).double()
        )
    recurrent = getattr(models[0], "is_recurrent", False)
    inputs = (
        torch.rand(2, 3, 8, dtype=torch.float64)
        if recurrent
        else torch.rand(2, 8, dtype=torch.float64)
    )
    results = []
    for model in models:
        x = inputs.clone().requires_grad_()
        out = model(x)
        out.sum().backward()
        results.append(
            (
                out,
                x.grad,
                model.state_dict(),
                {k: p.grad for k, p in model.named_parameters()},
            )
        )
    for a, b in zip(results[0][:2], results[1][:2]):
        assert torch.equal(a, b)
    for a, b in zip(results[0][2:], results[1][2:]):
        assert a.keys() == b.keys()
        assert all(
            (a[k] is None and b[k] is None) or torch.equal(a[k], b[k]) for k in a
        )


def test_paired_afferents_are_explicit_distinct_sources_with_identical_rates():
    core = _einet(
        **{
            "architecture.inhibitory_layer_sizes": [],
            "connectivity.ei_synapses_per_branch_per_layer": [0],
            "transfer.output_activation": "nonnegative",
        }
    )
    with pytest.raises(ValueError, match="both excites and inhibits"):
        get_architecture("einet", core, input_dim=8)
    core["transfer"]["input_source_identity"] = "paired_afferents"
    model = get_architecture("einet", core, input_dim=8)
    assert biological_violations(model) == []
    x = torch.rand(2, 8, requires_grad=True)
    e, i = model.transfer_fn(x)
    assert torch.equal(e, x) and torch.equal(i, x)
    (e + i).sum().backward()
    assert torch.equal(x.grad, torch.full_like(x, 2))
    with pytest.raises(ValueError, match="requires independent_pathways"):
        TransferLayer(
            8,
            {"independent_pathways": True, "input_source_identity": "paired_afferents"},
        )
    with pytest.raises(ValueError, match="input_source_identity"):
        TransferLayer(8, {"input_source_identity": "paired_afferent"})


def test_operator_card_scopes_declaration_to_core_not_signed_decoder():
    core = _population()
    model = nn.Module()
    model.core_network = get_architecture("population_network", core, input_dim=8)
    model.decoder = nn.Linear(5, 2)
    card = build_operator_card({"type": "population_network", **core}, model)
    assert card["biological_neuron"] == {
        "declared": True,
        "scope": "neuronal_core",
        "violations": [],
    }


def test_positive_projection_equation_gradients_and_roundtrip():
    torch.manual_seed(10)
    module = NonnegativeLinear(3, 4).double()
    x = torch.rand(2, 3, dtype=torch.float64, requires_grad=True)
    expected = x @ torch.nn.functional.softplus(module.raw_weight).T + module.bias
    torch.testing.assert_close(module(x), expected, rtol=0, atol=0)
    assert torch.autograd.gradcheck(module, (x,))
    grads = torch.autograd.grad(module(x).sum(), (x, module.raw_weight, module.bias))
    refs = torch.autograd.grad(expected.sum(), (x, module.raw_weight, module.bias))
    for grad, ref in zip(grads, refs):
        torch.testing.assert_close(grad, ref)
    other = NonnegativeLinear(3, 4).double()
    other.load_state_dict(module.state_dict())
    assert torch.equal(module(x), other(x))
    with pytest.raises(ValueError, match="nonnegative input"):
        module(-x)


def test_population_positive_projection_is_checked_through_typed_builder():
    core = {
        "type": "population_network",
        **_population(
            input_transform="nonnegative",
            input_projection_dims=[7],
            input_projection_weight_transform="softplus",
        ),
    }
    config = config_from_mapping({"model": {"core": core}})
    model = get_architecture(config.model.core.type, config.model.core, input_dim=8)
    assert biological_violations(model) == []
    x = torch.rand(2, 8)
    out = model(x)
    assert (out >= 0).all()
    out.sum().backward()
    assert model.input_projection[0].raw_weight.grad is not None


def test_signed_projection_keeps_historical_initialization_and_rejects_biological_claim():
    core = _population(input_projection_dims=[7])
    with pytest.raises(ValueError, match="input projection"):
        get_architecture("population_network", core, input_dim=8)
    states = []
    for explicit in (False, True):
        payload = copy.deepcopy(core)
        if explicit:
            payload["population_network"][
                "input_projection_weight_transform"
            ] = "identity"
        torch.manual_seed(42)
        model = get_architecture(
            "population_network", {**payload, "biological_neuron": False}, input_dim=8
        )
        assert isinstance(model.input_projection[0], nn.Linear)
        states.append(model.state_dict())
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])


@pytest.mark.parametrize("family", ["feedforward", "recurrent", "population"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_paired_afferent_adoption_preserves_historical_computation(family, device):
    if family == "population":
        core_type = "population_network"
        base = _population(
            input_transform="identity",
            use_transfer=True,
            transfer_params={"output_activation": None},
            connections=[
                {"source": "input_e", "target": "e"},
                {"source": "input_i", "target": "e"},
                {"source": "input_e", "target": "i"},
            ],
        )
        new = copy.deepcopy(base)
        transfer = new["population_network"]["transfer_params"]
    else:
        core_type = "einet"
        base = _einet(
            **{
                "architecture.inhibitory_layer_sizes": [],
                "connectivity.ei_synapses_per_branch_per_layer": [0],
                "transfer.output_activation": None,
            }
        )
        if family == "recurrent":
            base = _recurrent(base)
            base["connectivity"].update(
                rec_ei_synapses_per_branch=[0], rec_ie_synapses_per_branch=[0]
            )
        new = copy.deepcopy(base)
        transfer = new["transfer"]
    transfer.update(
        input_source_identity="paired_afferents", output_activation="nonnegative"
    )
    reference = {**base, "biological_neuron": False}
    adopted = {**new, "biological_neuron": True}
    results = []
    x = torch.rand(2, 3, 8) if family == "recurrent" else torch.rand(2, 8)
    x[..., 0] = 0
    for payload in (reference, adopted):
        torch.manual_seed(42)
        net = get_architecture(core_type, payload, input_dim=8).to(device)
        inp = x.to(device).clone().requires_grad_()
        out = net(inp)
        out.sum().backward()
        results.append((net, out, inp.grad))
    left, right = results
    assert biological_violations(right[0]) == []
    assert torch.equal(left[1], right[1]) and torch.equal(left[2], right[2])
    a, b = left[0].state_dict(), right[0].state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    for a, b in zip(left[0].parameters(), right[0].parameters(), strict=True):
        assert (a.grad is None and b.grad is None) or torch.equal(a.grad, b.grad)
    right[0].load_state_dict(left[0].state_dict(), strict=True)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_positive_projection_trains_with_the_normal_optimizer(device):
    from dendritic_modeling.config.training import OptimizerConfig
    from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
    from dendritic_modeling.training.optimizers.factory import create_optimizer

    config = config_from_mapping(
        {
            "model": {
                "encoder": {"type": "identity", "params": {"input_dim": 8}},
                "core": {
                    "type": "population_network",
                    **_population(
                        input_transform="nonnegative",
                        input_projection_dims=[7],
                        input_projection_weight_transform="softplus",
                    ),
                },
                "decoder": {
                    "type": "MLP",
                    "params": {"hidden_dims": [], "output_dim": 2},
                },
            }
        }
    )
    torch.manual_seed(32)
    model, _ = initialize_model(config.model)
    model = model.to(device)
    optimizer = create_optimizer(
        model.parameters(), OptimizerConfig(name="adam", lr=0.01)
    )
    projection = model.core_network.input_projection[0]
    before = projection.raw_weight.detach().clone()
    x, y = torch.rand(8, 8, device=device), torch.arange(8, device=device) % 2
    for _ in range(3):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    assert not torch.equal(before, projection.raw_weight)
    assert (projection.weight >= 0).all()
    assert biological_violations(model) == []


@pytest.mark.parametrize("option", ["bad", [], None])
def test_typed_afferent_identity_rejects_invalid_values(option):
    with pytest.raises(ValueError, match="input_source_identity"):
        config_from_mapping(
            {"model": {"core": {"transfer": {"input_source_identity": option}}}}
        )


def test_default_model_requires_an_explicit_input_contract():
    from dendritic_modeling.config.model import ModelConfig
    from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

    config = ModelConfig()
    config.encoder.params.input_dim = 8
    with pytest.raises(ValueError, match="biological constraint set"):
        initialize_model(config)


@pytest.mark.parametrize(
    "family",
    [
        "total_mlp",
        "active_mlp",
        "structured_mlp",
        "tensor_map",
        "spatial_conv",
        "alexnet",
    ],
)
def test_direct_composite_constructors_keep_their_nonbiological_default(family):
    from test_dendritic_alexnet import _tiny_config

    from dendritic_modeling.networks.architectures.classical.mlp.matched_param import (
        MatchedActiveParamMLP,
        MatchedTotalParamMLP,
        SparseStructuredMLP,
    )
    from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.spatial_dendritic import (
        HierarchicalDendriticConv,
        HierarchicalDendriticTensorMap,
    )
    from dendritic_modeling.networks.architectures.vision.dendritic_alexnet import (
        DendriticAlexNet,
    )

    core = (
        _tiny_config()
        if family == "alexnet"
        else _einet(**{"transfer.output_activation": None})
    )
    core.pop("biological_neuron", None)
    unchanged = copy.deepcopy(core)
    x = torch.rand(2, 3, 32, 32) if family == "alexnet" else torch.rand(2, 8)
    if family == "spatial_conv":
        x = x[:, :, None, None]
    models = []
    for explicit in (False, True):
        payload = copy.deepcopy(core)
        if explicit:
            payload["biological_neuron"] = False
        torch.manual_seed(19)
        if family == "alexnet":
            net = DendriticAlexNet(payload)
        elif family == "spatial_conv":
            net = HierarchicalDendriticConv(payload, in_channels=8, kernel_size=1)
        elif family == "tensor_map":
            net = HierarchicalDendriticTensorMap(payload, input_dim=8)
        else:
            cls = {
                "total_mlp": MatchedTotalParamMLP,
                "active_mlp": MatchedActiveParamMLP,
                "structured_mlp": SparseStructuredMLP,
            }[family]
            net = cls(payload, input_dim=8)
        models.append(net.eval())
    assert core == unchanged
    a, b = (m.state_dict() for m in models)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    with torch.no_grad():
        assert torch.equal(models[0](x), models[1](x))
