"""``blocklinear_strategy: conductance_dynamic`` on additive branch layers.

The modified backward rule divides the gradient passed to each child (the
input of ``branches_to_output``) by its coupling weight ``w``, or by
``|w| + epsilon`` for identity weights. The forward pass and the coupling
gradients are unchanged. Shunting layers keep their own rule.
"""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from test_operator_config_contract import _config

from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.factory import get_architecture

POSITIVE = ("softplus", "exp")


def _layer(transform, strategy, *, efficient=False, forward_hooks=False):
    torch.manual_seed(3)
    layer = DendriticBranchLayer(
        output_dim=3,
        excitatory_input_dim=5,
        excitatory_synapses_per_branch=2,
        input_branch_factor=2,
        use_shunting=False,
        reactivate=False,
        weight_transform=transform,
        blocklinear_strategy=strategy,
        efficient_blocklinear=efficient,
    )
    if forward_hooks:
        layer.set_forward_dynamic_grad_scaling(True)
    return layer


def _coupling(layer):
    """The coupling weights the rule divides by, as the rule computes them."""
    weight = layer.branches_to_output.weight().detach()
    if layer.weight_transform == "identity":
        weight = weight.abs() + layer.epsilon
    return weight


def _random_couplings_(layer, transform):
    generator = torch.Generator().manual_seed(17)
    raw = torch.randn(layer.branches_to_output.log_weight.shape, generator=generator)
    if transform == "identity":
        raw[0, 0] = 0.0  # a zero coupling of a signed layer
    with torch.no_grad():
        layer.branches_to_output.log_weight.copy_(raw)


@pytest.mark.parametrize("forward_hooks", [False, True])
@pytest.mark.parametrize("efficient", [False, True])
@pytest.mark.parametrize("transform", ["softplus", "exp", "identity"])
def test_child_gradients_are_divided_by_their_coupling_weight(
    transform, efficient, forward_hooks
):
    plain = _layer(transform, "none", efficient=efficient)
    scaled = _layer(
        transform,
        "conductance_dynamic",
        efficient=efficient,
        forward_hooks=forward_hooks,
    )
    _random_couplings_(plain, transform)
    scaled.load_state_dict(plain.state_dict())
    x = torch.rand(4, 5, generator=torch.Generator().manual_seed(1))
    child = torch.randn(4, 6, generator=torch.Generator().manual_seed(2))
    upstream = torch.randn(4, 3, generator=torch.Generator().manual_seed(4))

    results = []
    for layer in (plain, scaled):
        layer.train()
        branch_input = child.clone().requires_grad_(True)
        out = layer(x, branch_input=branch_input)
        (out * upstream).sum().backward()
        results.append((out.detach(), branch_input.grad, layer))

    (out_plain, grad_plain, _), (out_scaled, grad_scaled, _) = results
    assert torch.equal(out_scaled, out_plain)
    weight = _coupling(plain).flatten()
    assert torch.equal(grad_scaled, grad_plain * torch.exp(-1 * weight.log()))
    torch.testing.assert_close(grad_scaled, grad_plain / weight)
    assert torch.isfinite(grad_scaled).all()
    for (name, a), (_, b) in zip(
        plain.named_parameters(), scaled.named_parameters(), strict=True
    ):
        assert torch.equal(a.grad, b.grad), name

    # Evaluation-mode gradients are exact.
    scaled.eval()
    branch_input = child.clone().requires_grad_(True)
    (scaled(x, branch_input=branch_input) * upstream).sum().backward()
    assert torch.equal(branch_input.grad, grad_plain)


@pytest.mark.parametrize(
    "transform, raw",
    [("identity", 0.0), ("identity", -1e-12), ("softplus", -20.0), ("exp", -30.0)],
)
def test_small_couplings_give_finite_gradients(transform, raw):
    layer = _layer(transform, "conductance_dynamic")
    with torch.no_grad():
        layer.branches_to_output.log_weight.fill_(raw)
    layer.train()
    child = torch.randn(4, 6, requires_grad=True)
    layer(torch.rand(4, 5), branch_input=child).sum().backward()
    scale = layer.branches_to_output.input_scale_vec
    assert torch.isfinite(scale).all() and bool((scale > 0).all())
    torch.testing.assert_close(scale, 1 / _coupling(layer).flatten())
    assert torch.isfinite(child.grad).all()
    assert all(
        torch.isfinite(p.grad).all() for p in layer.parameters() if p.grad is not None
    )


def test_relu_couplings_are_rejected():
    with pytest.raises(ValueError, match="relu"):
        _layer("relu", "conductance_dynamic")
    _layer("relu", "none")  # unchanged without the rule


def test_shunting_rule_is_unchanged():
    torch.manual_seed(5)
    layer = DendriticBranchLayer(
        output_dim=3,
        excitatory_input_dim=5,
        excitatory_synapses_per_branch=2,
        input_branch_factor=2,
        use_shunting=True,
        reactivate=False,
        weight_transform="softplus",
        blocklinear_strategy="conductance_dynamic",
    )
    g_total = torch.tensor([[1.5, 2.0, 3.0], [2.5, 1.0, 4.0]])
    layer.compute_grad_scales(g_total=g_total)
    mean = g_total.mean(dim=0)
    weight = layer.branches_to_output.weight().detach().clamp_min(layer.epsilon)
    assert torch.equal(
        layer.branches_to_output.input_scale_vec, (mean[:, None] / weight).flatten()
    )
    assert torch.equal(layer.branches_to_output.param_scale_vec, mean[:, None])


# --- Networks: feedforward (backward hooks) and stepped (per-step hooks) ---


def _divide_child_gradients_(model):
    """Reference rule: register 1 / w on every BlockLinear input by hand."""
    for layer in model.modules():
        if not isinstance(layer, DendriticBranchLayer) or not layer.input_branches:
            continue
        aggregator = layer.branches_to_output
        forward = aggregator.forward

        def divided_forward(x, forward=forward, layer=layer):
            if torch.is_grad_enabled() and x.requires_grad:
                scale = torch.exp(-1 * _coupling(layer).log()).flatten()
                x.register_hook(lambda grad, scale=scale: grad * scale)
            return forward(x)

        aggregator.forward = divided_forward


def _randomize_couplings_(model):
    generator = torch.Generator().manual_seed(23)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("branches_to_output.log_weight"):
                parameter.copy_(0.5 * torch.randn(parameter.shape, generator=generator))


def _run(model, shape):
    model.train()
    model.zero_grad(set_to_none=True)
    x = torch.rand(shape, generator=torch.Generator().manual_seed(3))
    torch.manual_seed(7)
    out = model(x)
    out = out[0] if isinstance(out, tuple) else out
    upstream = torch.randn(out.shape, generator=torch.Generator().manual_seed(11))
    (out * upstream).sum().backward()
    grads = {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return out.detach(), grads


def _population(*, recurrent=False, **defaults):
    config = deepcopy(_config("population_network", "additive", "param_tanh"))
    layer = config["population_network"]["layers"][0]
    layer["population_defaults"].update(weight_transform="softplus", **defaults)
    if recurrent:
        layer["recurrent"] = True
        layer["population_defaults"]["rec_excitatory_synapses"] = 1
        layer["connections"].append({"source": "e", "target": "e", "timing": "delayed"})
    return "population_network", config, 6


def _einet(transform, family="einet"):
    config = deepcopy(_config("einet", "additive", "param_tanh"))
    config["morphology"]["weight_transform"] = transform
    if family != "einet":  # spatial cores apply the EINet core to every patch
        config["spatial"] = {"kernel_size": 3, "stride": 1, "padding": 1}
    return family, config, 6


def _recurrent_ei():
    return (
        "dendritic_additive",
        {
            "biological_neuron": False,
            "architecture": {
                "excitatory_layer_sizes": [4],
                "inhibitory_layer_sizes": [],
                "excitatory_branch_factors": [2],
            },
            "connectivity": {
                "ee_synapses_per_branch_per_layer": [2],
                "ie_synapses_per_branch_per_layer": [0],
                "rec_ee_synapses_per_branch": [1],
            },
            "morphology": {"somatic_synapses": True, "weight_transform": "softplus"},
            "reactivation": {"enabled": False},
        },
        5,
    )


_TYPED = {
    "enabled": True,
    "excitatory": [{"name": "AMPA", "fraction": 1.0, "tau": 2.0}],
    "inhibitory": [{"name": "GABA_A", "fraction": 1.0, "tau": 4.0}],
}

NETWORKS = {
    # Feedforward: backward hooks on BlockLinear.
    "einet_softplus": (lambda: _einet("softplus"), (5, 6)),
    "einet_identity": (lambda: _einet("identity"), (5, 6)),
    "spatial_conv": (
        lambda: _einet("softplus", "hierarchical_dendritic_conv"),
        (2, 6, 4, 4),
    ),
    "spatial_tensor_map": (
        lambda: _einet("softplus", "hierarchical_dendritic_tensor_map"),
        (2, 6),
    ),
    "population_feedforward": (lambda: _population(), (5, 6)),
    # Stepped through time: scales registered on each step's tensors.
    "population_recurrent": (lambda: _population(recurrent=True), (3, 4, 6)),
    "population_typed_synapses": (lambda: _population(synapse_types=_TYPED), (3, 6)),
    "recurrent_ei_layer": (_recurrent_ei, (2, 3, 5)),
}


def _with_strategy(family, config, strategy):
    config = deepcopy(config)
    if family == "population_network":
        defaults = config["population_network"]["layers"][0]["population_defaults"]
        defaults["blocklinear_gradient_scaling"] = strategy
    else:
        config["blocklinear"] = {"gradient_scaling": strategy}
    return config


@pytest.mark.parametrize("name", sorted(NETWORKS))
def test_networks_apply_the_additive_rule_to_every_coupling(name):
    make, shape = NETWORKS[name]
    family, config, input_dim = make()
    torch.manual_seed(1234)
    scaled = get_architecture(
        family, _with_strategy(family, config, "conductance_dynamic"), input_dim
    )
    layers = [m for m in scaled.modules() if isinstance(m, DendriticBranchLayer)]
    assert layers and not any(layer.use_shunting for layer in layers)
    assert any(layer.input_branches for layer in layers)
    _randomize_couplings_(scaled)
    torch.manual_seed(1234)
    plain = get_architecture(family, _with_strategy(family, config, "none"), input_dim)
    plain.load_state_dict(scaled.state_dict())

    out_scaled, grads_scaled = _run(scaled, shape)
    out_plain, grads_plain = _run(plain, shape)
    assert torch.equal(out_scaled, out_plain)
    assert grads_scaled.keys() == grads_plain.keys()
    assert any(not torch.equal(grads_scaled[k], grads_plain[k]) for k in grads_plain)

    _divide_child_gradients_(plain)
    out_reference, grads_reference = _run(plain, shape)
    assert torch.equal(out_reference, out_plain)
    for key, gradient in grads_reference.items():
        assert torch.equal(grads_scaled[key], gradient), key
