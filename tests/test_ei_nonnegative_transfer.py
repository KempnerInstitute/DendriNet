import pytest
import torch

from dendritic_modeling.config import TransferFunctionConfig
from dendritic_modeling.networks import ExcitationInhibitionNetwork


def _make_transfer_params(output_activation=None):
    params = TransferFunctionConfig().asdict()
    params["output_activation"] = output_activation
    return params


def _make_network(weight_transform="softplus", output_activation=None):
    return ExcitationInhibitionNetwork(
        input_dim=8,
        transfer_params=_make_transfer_params(output_activation),
        excitatory_layer_sizes=[4],
        inhibitory_layer_sizes=[0],
        excitatory_branch_factors=[2],
        inhibitory_branch_factors=[2],
        ee_synapses_per_branch_per_layer=[1],
        ei_synapses_per_branch_per_layer=[0],
        ie_synapses_per_branch_per_layer=[1],
        ii_synapses_per_branch_per_layer=[0],
        weight_transform=weight_transform,
    )


def test_positive_conductance_core_keeps_identity_transfer_when_unspecified():
    net = _make_network(weight_transform="softplus", output_activation=None)
    assert net.transfer_params["output_activation"] is None

    x = torch.tensor([[-1.0, -0.5, 0.0, 0.25, 1.0, -2.0, 0.5, 3.0]])
    x_e, x_i = net.transfer_fn(x)
    assert torch.any(x_e < 0)
    assert torch.any(x_i < 0)


def test_positive_conductance_core_supports_explicit_nonnegative_transfer():
    net = _make_network(weight_transform="softplus", output_activation="relu")

    assert net.transfer_params["output_activation"] == "relu"

    x = torch.tensor([[-1.0, -0.5, 0.0, 0.25, 1.0, -2.0, 0.5, 3.0]])
    x_e, x_i = net.transfer_fn(x)
    assert torch.all(x_e >= 0)
    assert torch.all(x_i >= 0)


def test_signed_weight_core_keeps_identity_transfer():
    net = _make_network(weight_transform="identity", output_activation=None)
    assert net.transfer_params["output_activation"] is None

    x = torch.tensor([[-1.0, 0.5, -0.25, 1.0, 0.0, -2.0, 0.5, 3.0]])
    x_e, x_i = net.transfer_fn(x)
    assert torch.any(x_e < 0)
    assert torch.any(x_i < 0)


def test_positive_conductance_core_rejects_signed_transfer_activation():
    with pytest.raises(ValueError, match="requires nonnegative transfer output"):
        _make_network(weight_transform="softplus", output_activation="tanh")
