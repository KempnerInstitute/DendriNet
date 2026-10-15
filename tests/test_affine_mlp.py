"""Numerical and configuration contracts for ordinary affine MLP controls."""

from dataclasses import asdict

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from dendritic_modeling.config import AffineMLPConfig, CoreConfig
from dendritic_modeling.networks.architectures.classical.mlp import (
    MLP,
    SparseAffine,
    build_affine_mlp,
)
from dendritic_modeling.networks.architectures.factory import get_architecture


@pytest.mark.parametrize("dropout", [None, 0.0, 0.2])
@pytest.mark.parametrize("activation", ["relu", "leaky_relu"])
def test_dense_checkpoint_rng_and_training_match_explicit_sequential(
    dropout, activation
):
    # Explicit legacy layout is the regression oracle. Construct both affine
    # layers before initializing either, including no-op dropout modules.
    def act():
        return nn.ReLU() if activation == "relu" else nn.LeakyReLU(0.1)

    torch.manual_seed(71)
    if dropout is None:
        expected = nn.Sequential(nn.Linear(13, 7), act(), nn.Linear(7, 5), act())
        last = 2
    else:
        expected = nn.Sequential(
            nn.Linear(13, 7),
            act(),
            nn.Dropout(dropout),
            nn.Linear(7, 5),
            act(),
            nn.Dropout(dropout),
        )
        last = 3
    for layer in [expected[0], expected[last]]:
        nn.init.kaiming_normal_(layer.weight)
        nn.init.zeros_(layer.bias)
    with torch.no_grad():
        expected[last].weight.mul_(0.1)
    expected_rng = torch.get_rng_state()
    config = AffineMLPConfig(
        dimensions=[13, 7, 5],
        activation=activation,
        dropout=dropout,
        output_weight_scale=0.1,
    )
    torch.manual_seed(71)
    actual = build_affine_mlp(config)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert list(actual.state_dict()) == list(expected.state_dict())
    for key, value in expected.state_dict().items():
        assert torch.equal(actual.state_dict()[key], value)
    actual.load_state_dict(expected.state_dict(), strict=True)
    x = torch.linspace(-1, 1, 39).reshape(3, 13)
    trajectories = []
    for model in [expected, actual]:
        torch.manual_seed(82)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        history = []
        for _ in range(3):
            optimizer.zero_grad()
            inputs = x.clone().requires_grad_(True)
            y = model(inputs)
            y.square().sum().backward()
            history.append((y.detach().clone(), inputs.grad.clone()))
            optimizer.step()
        trajectories.append(history)
    for old, new in zip(*trajectories):
        assert all(torch.equal(a, b) for a, b in zip(old, new))
    for key, value in expected.state_dict().items():
        assert torch.equal(actual.state_dict()[key], value)


@pytest.mark.parametrize("learned", [False, True])
def test_sparse_affine_matches_explicit_masked_dense_forward_and_input_gradient(
    learned,
):
    torch.manual_seed(81)
    layer = SparseAffine(13, 7, 4, learned=learned, seed=4)
    weights = torch.zeros(7, 13)
    if learned:
        indices = layer.linear.pre_w.abs().topk(4, dim=1).indices
        values = layer.linear.pre_w.gather(1, indices)
        torch.testing.assert_close(values.square().sum(1), torch.full((7,), 2.0))
    else:
        indices = layer.linear.connection_indices.long()
        values = layer.linear.pre_w
    weights.scatter_(1, indices, values.detach())
    x = torch.linspace(-1, 1, 39).reshape(3, 13).requires_grad_(True)
    actual = layer(x)
    expected = nn.functional.linear(x, weights, layer.bias)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    actual_grad = torch.autograd.grad(actual.square().sum(), x)[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), x)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("mode", ["dense", "fixed", "topk"])
def test_typed_yaml_and_factory_have_identical_state_and_rng(mode):
    config = AffineMLPConfig(
        dimensions=[13, 7, 5],
        mode=mode,
        fanin=[] if mode == "dense" else [4, 3],
        connectivity_seeds=[6, 10006] if mode == "fixed" else [],
    )
    structured = OmegaConf.structured(CoreConfig(type="affine_mlp", affine_mlp=config))
    # Exercise schema serialization, not only direct dataclass construction.
    loaded = OmegaConf.create(OmegaConf.to_yaml(structured))
    torch.manual_seed(82)
    expected = build_affine_mlp(config)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(82)
    actual = get_architecture("affine_mlp", loaded, input_dim=13, suffix_input_dim=5)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert actual.affine_mlp_config == asdict(config)
    actual.load_state_dict(expected.state_dict(), strict=True)
    for key, value in expected.state_dict().items():
        assert torch.equal(actual.state_dict()[key], value)
    with pytest.raises(ValueError, match="encoder"):
        get_architecture("affine_mlp", config, input_dim=14)
    with pytest.raises(ValueError, match="decoder"):
        get_architecture("affine_mlp", config, suffix_input_dim=6)


@pytest.mark.parametrize(
    "changes",
    [
        {"dimensions": [0, 3]},
        {"dimensions": [True, 3]},
        {"dimensions": [3]},
        {"mode": "unknown"},
        {"fanin": [1, 1]},
        {"mode": "fixed", "fanin": [4, 3]},
        {"mode": "fixed", "fanin": [14, 3], "connectivity_seeds": [1, 2]},
        {"mode": "topk", "fanin": [4]},
        {"mode": "topk", "fanin": [4, 0]},
        {"mode": "topk", "fanin": [4, 3], "connectivity_seeds": [1, 2]},
        {"activation": "unknown"},
        {"dropout": -0.1},
        {"dropout": float("nan")},
        {"output_weight_scale": float("inf")},
        {"workspace_mb": 0},
        {"index_dtype": "int16"},
        {"negative_slope": -0.1},
    ],
)
def test_invalid_config_fails_before_rng_consumption(changes):
    config = AffineMLPConfig(**dict({"dimensions": [13, 7, 5]}, **changes))
    before = torch.get_rng_state()
    with pytest.raises(ValueError):
        build_affine_mlp(config)
    assert torch.equal(before, torch.get_rng_state())


def test_existing_mlp_factory_defaults_remain_identical():
    params = {"input_dim": 13, "hidden_dims": [7], "output_dim": 5}
    torch.manual_seed(71)
    expected = MLP(**params)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(71)
    actual = get_architecture("mlp", params)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for key, value in expected.state_dict().items():
        assert torch.equal(actual.state_dict()[key], value)


def test_arbitrary_depth_without_output_activation():
    config = AffineMLPConfig(dimensions=[13, 7, 5, 2], activate_output=False)
    model = build_affine_mlp(config)
    assert len(model) == 5
    assert isinstance(model[-1], nn.Linear)
    assert model(torch.ones(3, 13)).shape == (3, 2)


def test_structured_core_config_with_mlp_type_raises():
    params = {
        "biological_neuron": False,
        "architecture": {"excitatory_layer_sizes": [4]},
        "connectivity": {"ee_synapses_per_branch_per_layer": [2]},
    }
    with pytest.raises(ValueError, match="point_mlp"):
        get_architecture("mlp", params, input_dim=6)
