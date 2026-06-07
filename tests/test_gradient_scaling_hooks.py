"""Regression tests for gradient-scaling hook failure handling."""

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic import (
    gradient_scaling,
)
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


def test_safe_hook_warns_when_falling_back_to_unscaled_gradient(monkeypatch):
    def broken_hook(module, grad_input, grad_output):
        raise RuntimeError("synthetic hook failure")

    warnings = []
    monkeypatch.setattr(
        gradient_scaling.logger,
        "warning",
        lambda *args, **kwargs: warnings.append((args, kwargs)),
    )
    gradient_scaling._WARNED_HOOK_FAILURES.clear()
    module = nn.Linear(2, 2)
    grad_input = (torch.ones(1, 2),)
    grad_output = (torch.ones(1, 2),)
    wrapped = gradient_scaling.safe_hook(broken_hook)

    result = wrapped(module, grad_input, grad_output)

    assert result[0] is grad_input[0]
    assert warnings
    assert "returning unscaled gradients" in warnings[0][0][0]


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


def test_blocklinear_forward_scales_param_and_input_per_call():
    layer = BlockLinear(in_features=1, out_features=1, weight_transform="identity")
    with torch.no_grad():
        layer.log_weight.fill_(1.0)
    GradientScaler(
        blocklinear_strategy="conductance_dynamic"
    ).register_block_linear_dynamic(layer)

    x1 = torch.tensor([[5.0]], requires_grad=True)
    x2 = torch.tensor([[7.0]], requires_grad=True)
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
