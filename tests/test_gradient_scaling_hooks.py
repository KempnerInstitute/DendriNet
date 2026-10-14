"""Regression tests for gradient-scaling hook failure handling."""

import pytest
import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear import (
    BlockLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
    DendriticBranchLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.gradient_scaling import (
    GradientScaler,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)


def test_topk_forward_param_scale_is_per_call_and_bypasses_stale_param_hook():
    layer = TopKLinear(
        in_features=1,
        out_features=1,
        K=1,
        weight_transform="identity",
    )
    with torch.no_grad():
        layer.pre_w.fill_(1.0)
    GradientScaler(topk_strategy="conductance_dynamic").register_topk_dynamic(layer)

    x1 = torch.tensor([[5.0]])
    x2 = torch.tensor([[7.0]])
    y1 = layer(x1)
    layer.register_forward_param_gradient_scale(torch.tensor([[2.0]]))
    assert layer.param_scale_vec is None
    y2 = layer(x2)
    layer.param_scale_vec = torch.tensor([[99.0]])  # stale module attribute
    layer.register_forward_param_gradient_scale(torch.tensor([[3.0]]))
    torch.testing.assert_close(layer.param_scale_vec, torch.tensor([[99.0]]))

    (y1 + y2).sum().backward()

    torch.testing.assert_close(layer.pre_w.grad, torch.tensor([[31.0]]))


def test_topk_drops_dense_forward_tensor_without_dynamic_param_scaling():
    layer = TopKLinear(
        in_features=2,
        out_features=2,
        K=1,
        weight_transform="identity",
    )

    y = layer(torch.ones(1, 2))

    assert layer._last_forward_param_tensor is None
    y.sum().backward()


def test_blocklinear_forward_scales_param_and_input_per_call():
    layer = BlockLinear(in_features=1, out_features=1, weight_transform="identity")
    with torch.no_grad():
        layer.log_weight.fill_(1.0)
    GradientScaler(
        blocklinear_strategy="conductance_dynamic"
    ).register_block_linear_dynamic(layer)

    x1 = torch.tensor([[5.0]], requires_grad=True)
    x2 = torch.tensor([[7.0]], requires_grad=True)
    layer(x1)  # outside forward-hook mode nothing is kept
    assert layer._last_forward_input_tensor is None
    with pytest.raises(RuntimeError, match="set_forward_dynamic_grad_scaling"):
        layer.register_forward_gradient_scales(param_scale=torch.tensor([[2.0]]))

    layer._use_forward_dynamic_grad_scaling = True  # as the owning branch sets it
    y1 = layer(x1)
    layer.register_forward_gradient_scales(
        param_scale=torch.tensor([[2.0]]),
        input_scale=torch.tensor([4.0]),
    )
    assert layer.param_scale_vec is None
    assert layer.input_scale_vec is None
    y2 = layer(x2)
    layer.param_scale_vec = torch.tensor([[99.0]])  # stale module attribute
    layer.input_scale_vec = torch.tensor([99.0])
    layer.register_forward_gradient_scales(
        param_scale=torch.tensor([[3.0]]),
        input_scale=torch.tensor([5.0]),
    )
    torch.testing.assert_close(layer.param_scale_vec, torch.tensor([[99.0]]))
    torch.testing.assert_close(layer.input_scale_vec, torch.tensor([99.0]))

    (y1 + y2).sum().backward()

    torch.testing.assert_close(layer.log_weight.grad, torch.tensor([[31.0]]))
    torch.testing.assert_close(x1.grad, torch.tensor([[4.0]]))
    torch.testing.assert_close(x2.grad, torch.tensor([[5.0]]))


def test_branch_layer_legacy_topk_scale_keeps_param_scale_only():
    layer = DendriticBranchLayer(
        output_dim=2,
        excitatory_input_dim=4,
        excitatory_synapses_per_branch=2,
        input_branch_factor=None,
        reactivate=False,
        topk_strategy="conductance_dynamic",
        use_shunting=True,
        dbl_init_method="default",
    )

    layer.compute_grad_scales(
        g_total=torch.tensor([1.5, 2.5]),
        use_forward_hooks=False,
    )

    assert layer.branch_excitation.param_scale_vec is not None
    assert layer.branch_excitation.input_scale_vec is None
    assert not layer.branch_excitation._use_forward_dynamic_grad_scaling


def test_a_failing_scaling_hook_raises_instead_of_dropping_the_scale():
    import pytest

    layer = TopKLinear(in_features=3, out_features=2, K=2, weight_transform="identity")
    GradientScaler(topk_strategy="conductance_dynamic").register_topk_dynamic(layer)
    layer._use_forward_dynamic_grad_scaling = False
    layer.param_scale_vec = torch.ones(5, 7)  # cannot scale the pre_w gradient
    layer.train()
    with pytest.raises(RuntimeError):
        layer(torch.ones(4, 3)).sum().backward()


def test_registering_scaling_hooks_keeps_other_backward_hooks():
    layer = TopKLinear(in_features=3, out_features=2, K=2, weight_transform="identity")
    calls = []
    layer.register_full_backward_hook(lambda *args: calls.append("analysis"))
    GradientScaler(topk_strategy="conductance_dynamic").register_topk_dynamic(layer)
    layer.train()
    layer(torch.ones(4, 3, requires_grad=True)).sum().backward()
    assert calls == ["analysis"]


# Populations stepped through time scale the BlockLinear gradients of each
# step on that step's own tensors (blocklinear_gradient_scaling alone).


def _population(rule, *, recurrent, **defaults):
    from copy import deepcopy

    from test_operator_config_contract import _config

    config = deepcopy(_config("population_network", rule, "param_tanh"))
    layer = config["population_network"]["layers"][0]
    layer["population_defaults"].update(defaults)
    if recurrent:
        layer["recurrent"] = True
        layer["population_defaults"]["rec_excitatory_synapses"] = 1
        layer["connections"].append({"source": "e", "target": "e", "timing": "delayed"})
    return config


def _build_population(config):
    from copy import deepcopy

    from dendritic_modeling.networks.architectures.factory import get_architecture

    torch.manual_seed(1234)
    return get_architecture("population_network", deepcopy(config), input_dim=6)


def _branch_layers(model):
    return [m for m in model.modules() if isinstance(m, DendriticBranchLayer)]


_SYNAPSE_BANKS = (
    "branch_excitation",
    "branch_inhibition",
    "branch_recurrent",
    "branch_rec_inhibition",
)


def _gradients(model, x):
    model.train()
    model.zero_grad(set_to_none=True)
    out = model(x)
    weights = torch.randn(out.shape, generator=torch.Generator().manual_seed(11))
    (out * weights).sum().backward()
    return {
        name: param.grad.clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


def test_stepped_population_scales_blocklinear_gradients_of_each_step():
    config = _population(
        "shunting", recurrent=True, blocklinear_gradient_scaling="conductance_dynamic"
    )
    model = _build_population(config)
    aggregators = [
        layer
        for layer in _branch_layers(model)
        if getattr(layer, "branches_to_output", None) is not None
    ]
    assert aggregators
    for layer in _branch_layers(model):
        # Only the aggregator records per-forward tensors; the unscaled
        # synapses keep none.
        assert all(
            not getattr(layer, bank)._use_forward_dynamic_grad_scaling
            for bank in _SYNAPSE_BANKS
            if getattr(layer, bank, None) is not None
        )
    assert all(
        layer._forward_dynamic_grad_scaling
        and layer.branches_to_output._use_forward_dynamic_grad_scaling
        for layer in aggregators
    )

    # Record the scales each step registers on its own tensors.
    recorded = []
    for layer in aggregators:
        register = layer.branches_to_output.register_forward_gradient_scales

        def spy(*, param_scale=None, input_scale=None, register=register, layer=layer):
            recorded.append((layer, param_scale.clone(), input_scale.clone()))
            register(param_scale=param_scale, input_scale=input_scale)

        layer.branches_to_output.register_forward_gradient_scales = spy
    x = torch.rand(3, 4, 6, generator=torch.Generator().manual_seed(3))
    torch.manual_seed(7)
    scaled = _gradients(model, x)
    assert len(recorded) == 4 * len(aggregators)  # one per layer and step
    assert not torch.equal(recorded[0][2], recorded[-len(aggregators)][2])

    # Reference: the unscaled model, with the recorded scales registered on
    # each call's tensors.
    unscaled_config = _population(
        "shunting", recurrent=True, blocklinear_gradient_scaling="none"
    )
    reference = _build_population(unscaled_config)
    reference.load_state_dict(model.state_dict())
    torch.manual_seed(7)
    unscaled = _gradients(reference, x)
    assert any(not torch.equal(unscaled[name], scaled[name]) for name in scaled)
    calls = iter(recorded)
    for layer in _branch_layers(reference):
        aggregator = getattr(layer, "branches_to_output", None)
        if aggregator is None:
            continue
        aggregator._use_forward_dynamic_grad_scaling = True
        forward = aggregator.forward

        def scaled_forward(x, forward=forward, aggregator=aggregator):
            out = forward(x)
            _, param_scale, input_scale = next(calls)
            aggregator.register_forward_gradient_scales(
                param_scale=param_scale, input_scale=input_scale
            )
            return out

        aggregator.forward = scaled_forward
    torch.manual_seed(7)
    expected = _gradients(reference, x)
    assert scaled.keys() == expected.keys()
    for name, gradient in expected.items():
        assert torch.equal(scaled[name], gradient), name


def test_feedforward_population_keeps_backward_hook_blocklinear_scaling():
    model = _build_population(
        _population(
            "shunting",
            recurrent=False,
            blocklinear_gradient_scaling="conductance_dynamic",
        )
    )
    for layer in _branch_layers(model):
        assert not layer._forward_dynamic_grad_scaling
        aggregator = getattr(layer, "branches_to_output", None)
        if aggregator is not None:
            assert not aggregator._use_forward_dynamic_grad_scaling
    _gradients(model, torch.rand(5, 6))
    assert any(
        layer.branches_to_output.input_scale_vec is not None
        for layer in _branch_layers(model)
        if getattr(layer, "branches_to_output", None) is not None
    )


def test_recurrent_ei_layer_scales_blocklinear_gradients_per_step():
    from dendritic_modeling.networks.architectures.factory import get_architecture

    params = {
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
        "morphology": {"use_shunting": True, "somatic_synapses": True},
        "blocklinear": {"gradient_scaling": "conductance_dynamic"},
        "reactivation": {"enabled": False},
    }
    torch.manual_seed(5)
    net = get_architecture("dendritic_shunting", params, input_dim=5)
    soma = net.layers[0].e_population.branch_layers[-1]
    assert soma.branches_to_output._use_forward_dynamic_grad_scaling
    assert not soma.branch_excitation._use_forward_dynamic_grad_scaling
    gradients = _gradients(net, torch.rand(2, 3, 5))
    assert all(torch.isfinite(gradient).all() for gradient in gradients.values())


def test_aggregator_only_forward_mode_rejects_scaled_synapses():
    layer = DendriticBranchLayer(
        output_dim=2,
        excitatory_input_dim=4,
        excitatory_synapses_per_branch=2,
        input_branch_factor=2,
        reactivate=False,
        topk_strategy="conductance_dynamic",
        blocklinear_strategy="conductance_dynamic",
        use_shunting=True,
        dbl_init_method="default",
    )
    with pytest.raises(ValueError, match="synapses=True"):
        layer.set_forward_dynamic_grad_scaling(True, synapses=False)
    layer.set_forward_dynamic_grad_scaling(True)
    layer.set_forward_dynamic_grad_scaling(False, synapses=False)
    assert not layer.branch_excitation._use_forward_dynamic_grad_scaling
    assert not layer.branches_to_output._use_forward_dynamic_grad_scaling
