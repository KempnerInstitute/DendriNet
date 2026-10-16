"""Equation, gradient, configuration and stored-resource contracts."""

import copy
import io
import json
from dataclasses import asdict, replace

import pytest
import torch
from omegaconf import OmegaConf
from torch.nn import functional as F

from dendritic_modeling.config import CoreConfig, load_config
from dendritic_modeling.config.shared_projection import SharedProjectionShuntingConfig
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet import (
    DendriNet,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.shared_projection import (
    SharedProjectionShunting,
    build_shared_projection_shunting,
)
from dendritic_modeling.networks.architectures.factory import get_architecture


def spec(**changes):
    return replace(
        SharedProjectionShuntingConfig(
            input_dim=7,
            supports=[[0, 2], [3, 5]],
            nodes=[1e-4, 0.03, 2.0],
            dtype="float64",
        ),
        **changes,
    )


def reference(model, x):
    positive = F.softplus(model.projection.pre_w)
    theta = positive / positive.sum(-1, keepdim=True)
    v = (x[:, model.projection.connection_indices] * theta).sum(-1)
    raw = torch.stack(
        [b.branch_layers[0].branch_excitation.pre_w.flatten() for b in model.banks]
    )
    z = (1 + model.config.epsilon) * torch.exp(-raw)
    phi = v[:, :, None] / (v[:, :, None] + z)
    h = (phi * model.pool.log_weight).sum(-1)
    return h @ model.readout.weight.T + model.readout.bias


@pytest.mark.parametrize("epsilon", [0.0, 1e-8, 0.1])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_native_equation_and_all_gradients(epsilon, dtype):
    m = SharedProjectionShunting(spec(epsilon=epsilon, dtype=dtype))
    with torch.no_grad():
        m.projection.pre_w.copy_(torch.tensor([[0.2, -0.7], [-0.3, 0.8]]))
        m.pool.log_weight.copy_(torch.tensor([[0.3, -0.4, 0.2], [-0.2, 0.7, 0.1]]))
    x = (
        torch.linspace(0, 1, 35, dtype=getattr(torch, dtype))
        .reshape(5, 7)
        .requires_grad_()
    )
    actual, expected = m(x), reference(m, x)
    tolerance = 3e-6 if dtype == "float32" else 3e-13
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    args = [x, *m.parameters()]
    a = torch.autograd.grad(actual.square().sum(), args)
    b = torch.autograd.grad(expected.square().sum(), args)
    for aa, bb in zip(a, b):
        torch.testing.assert_close(aa, bb, rtol=tolerance, atol=tolerance)
    # Distractors do not contribute to the function or its input gradient.
    assert torch.count_nonzero(a[0][:, [1, 4, 6]]) == 0
    zero = torch.zeros(1, 7, dtype=x.dtype)
    torch.testing.assert_close(m(zero), m.readout.bias[None])


def test_finite_difference_all_trainable_parameters_and_inputs():
    m = SharedProjectionShunting(spec())
    named = dict(m.named_parameters())
    names = list(named)
    x = torch.linspace(0.1, 0.9, 14, dtype=torch.float64).reshape(2, 7).requires_grad_()

    def forward(x, *parameters):
        return torch.func.functional_call(m, dict(zip(names, parameters)), (x,))

    assert torch.autograd.gradcheck(
        forward, (x, *named.values()), eps=1e-6, atol=1e-6, rtol=1e-4
    )


def test_native_execution_counts_freezing_and_roundtrip():
    config = spec(train_projection=False, train_conductance=False, train_output=False)
    m = build_shared_projection_shunting(config)
    seen = []
    handles = [b.register_forward_hook(lambda *args: seen.append(1)) for b in m.banks]
    x = torch.linspace(0.01, 1, 35, dtype=torch.float64).reshape(5, 7)
    y = m(x)
    for h in handles:
        h.remove()
    assert len(seen) == 2
    assert all(isinstance(b, DendriNet) for b in m.banks)
    report = m.resource_report()
    json.dumps(report, allow_nan=False)
    assert report["total_parameters"] == report["expected_stored_parameters"] == 19
    assert report["trainable_parameters"] == 7  # qW pool weights plus intercept
    assert sum(report["parameter_roles"].values()) == 19
    assert report["shunting_divisions_per_example"] == 6
    assert report["dendrinet_inventory"]["population_count"] == 2
    assert report["dendrinet_inventory"]["total_nonsomatic_compartments"] == 0
    torch.testing.assert_close(
        m.effective_nodes(),
        torch.tensor([config.nodes] * 2, dtype=torch.float64),
        rtol=2e-15,
        atol=0,
    )
    stream = io.BytesIO()
    torch.save(m.state_dict(), stream)
    stream.seek(0)
    rebuilt = SharedProjectionShunting(SharedProjectionShuntingConfig(**asdict(config)))
    rebuilt.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    assert torch.equal(rebuilt(x), y)
    assert [p.requires_grad for p in rebuilt.parameters()] == [
        p.requires_grad for p in m.parameters()
    ]


def test_factory_yaml_and_loader_routes(tmp_path):
    config = spec()
    core = CoreConfig(
        type="shared_projection_shunting", shared_projection_shunting=config
    )
    yaml = OmegaConf.to_yaml(OmegaConf.structured(core))
    loaded = OmegaConf.create(yaml)
    rng = torch.random.get_rng_state().clone()
    direct = build_shared_projection_shunting(config)
    assert torch.equal(rng, torch.random.get_rng_state())
    via = get_architecture(
        "shared_projection_shunting", loaded, input_dim=7, suffix_input_dim=1
    )
    for name, value in direct.state_dict().items():
        assert torch.equal(value, via.state_dict()[name])
    path = tmp_path / "config.yaml"
    path.write_text(
        OmegaConf.to_yaml(
            OmegaConf.create(
                {
                    "config_version": 4,
                    "model": {
                        "core": {
                            "type": "shared_projection_shunting",
                            "shared_projection_shunting": asdict(config),
                        }
                    },
                }
            )
        )
    )
    parsed = load_config(str(path))
    assert asdict(parsed.model.core.shared_projection_shunting) == asdict(config)
    from_loaded = get_architecture(
        parsed.model.core.type, parsed.model.core, input_dim=7, suffix_input_dim=1
    )
    x = torch.ones(2, 7, dtype=torch.float64)
    assert torch.equal(direct(x), from_loaded(x))
    with pytest.raises(ValueError, match="encoder"):
        get_architecture("shared_projection_shunting", asdict(config), input_dim=8)
    with pytest.raises(ValueError, match="decoder"):
        get_architecture(
            "shared_projection_shunting", asdict(config), suffix_input_dim=2
        )


@pytest.mark.parametrize(
    "change",
    [
        {"nodes": [0.0]},
        {"nodes": [float("nan")]},
        {"nodes": [True]},
        {"supports": [[0, 0]]},
        {"supports": [[7, 0]]},
        {"supports": [[0], [1, 2]]},
        {"dtype": "float16"},
        {"epsilon": -1.0},
        {"train_projection": 1},
        {"input_dim": True},
    ],
)
def test_reject_invalid_config(change):
    with pytest.raises((ValueError, TypeError)):
        build_shared_projection_shunting(spec(**change))


def test_reject_unused_options_and_training_strategies():
    with pytest.raises(ValueError, match="shared_projection_shunting"):
        load_config(
            overrides={
                "config_version": 4,
                "model.core.type": "affine_mlp",
                "model.core.shared_projection_shunting": asdict(spec()),
            }
        )
    for overrides in [
        {"model.core.morphology.use_shunting": False},
        {"training.main.strategy": "local_learning"},
        {"model.core.shared_projection_shunting.nodes": [0.0]},
        {"model.core.shared_projection_shunting.misspelled": 1},
    ]:
        with pytest.raises(ValueError):
            load_config(
                overrides={
                    "config_version": 4,
                    "model.core.type": "shared_projection_shunting",
                    "model.core.shared_projection_shunting": asdict(spec()),
                    **overrides,
                }
            )
    with pytest.raises(ValueError):
        get_architecture(
            "shared_projection_shunting",
            {
                "type": "shared_projection_shunting",
                "shared_projection_shunting": asdict(spec()),
                "morphology": {"use_shunting": False},
            },
        )
    m = SharedProjectionShunting(spec())
    for x in [
        torch.ones(2, 8, dtype=torch.float64),
        -torch.ones(2, 7, dtype=torch.float64),
        torch.full((2, 7), float("nan"), dtype=torch.float64),
    ]:
        with pytest.raises(ValueError):
            m(x)


def test_joint_optimizer_matches_equivalent_circuit():
    native, ordinary = SharedProjectionShunting(spec()), SharedProjectionShunting(
        spec()
    )
    x = torch.linspace(0.01, 1, 63, dtype=torch.float64).reshape(9, 7)
    target = (x[:, 0:1] + 0.2 * x[:, 3:4]).sqrt()
    a = torch.optim.Adam(native.parameters(), lr=0.01)
    b = torch.optim.Adam(ordinary.parameters(), lr=0.01)
    initial = {n: p.detach().clone() for n, p in native.named_parameters()}
    for _ in range(5):
        for model, opt, fn in [
            (native, a, lambda m, x: m(x)),
            (ordinary, b, reference),
        ]:
            opt.zero_grad()
            (fn(model, x) - target).square().mean().backward()
            opt.step()
    for (name, p), other in zip(native.named_parameters(), ordinary.parameters()):
        torch.testing.assert_close(p, other, rtol=1e-10, atol=1e-12)
        assert torch.max(torch.abs(p - initial[name])) > 0


def test_section_wrapped_mapping_builds_without_a_type():
    config = spec()
    direct = build_shared_projection_shunting(config)
    via = get_architecture(
        "shared_projection_shunting",
        {"shared_projection_shunting": asdict(config)},
        input_dim=7,
        suffix_input_dim=1,
    )
    for name, value in direct.state_dict().items():
        assert torch.equal(value, via.state_dict()[name])


@pytest.mark.parametrize("level", [-20.0, -40.0, -90.0])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_small_finite_projection_weights_keep_unit_sum(dtype, level):
    m = SharedProjectionShunting(spec(dtype=dtype))
    with torch.no_grad():
        # The projection norm lies below machine epsilon; at -90 the float32
        # weights are subnormal.
        m.projection.pre_w.copy_(level + torch.tensor([[0.0, 0.0], [-2.0, 1.0]]))
    weights = m.projection._normalized_sparse_weight()
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, dtype=weights.dtype))
    x = (
        torch.linspace(0, 1, 35, dtype=getattr(torch, dtype))
        .reshape(5, 7)
        .requires_grad_()
    )
    # The reference equation, evaluated independently in float64.
    m64 = copy.deepcopy(m).double()
    x64 = x.detach().double().requires_grad_()
    actual, expected = m(x), reference(m64, x64)
    tolerance = 3e-6 if dtype == "float32" else 3e-13
    torch.testing.assert_close(
        actual.double(), expected, rtol=tolerance, atol=tolerance
    )
    a = torch.autograd.grad(actual.square().sum(), [x, *m.parameters()])
    b = torch.autograd.grad(expected.square().sum(), [x64, *m64.parameters()])
    for aa, bb in zip(a, b):
        torch.testing.assert_close(aa.double(), bb, rtol=10 * tolerance, atol=tolerance)
