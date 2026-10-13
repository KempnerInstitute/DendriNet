"""Opt-in impact, regrowth, bounded-conductance and coupling-law options."""

import math
from copy import deepcopy

import pytest
import torch
from test_operator_config_contract import _config

from dendritic_modeling.config.model import SparsityConfig
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.synapse_config import (
    DendriticSynapseConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.impact_rewiring import (
    CellwiseImpactRewiring,
    ImpactRewiringConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_release import (
    IndexedReleaseLinear,
)
from dendritic_modeling.networks.architectures.factory import get_architecture

NEW_DEFAULTS = {
    "release_impact_metric": "drive",
    "release_max_conductance": None,
    "release_track_source_activity": False,
}


def bank(seed=8, **kwargs):
    options = {"weight_transform": "exp"}
    options.update(kwargs)
    return IndexedReleaseLinear(12, 4, 4, seed=seed, **options)


def branch(**overrides):
    kwargs = {
        "output_dim": 4,
        "excitatory_input_dim": 10,
        "inhibitory_input_dim": 5,
        "excitatory_synapses_per_branch": 4,
        "inhibitory_synapses_per_branch": 2,
        "input_branch_factor": 2,
        "topk_type": "indexed_release",
        "indexed_seed": 41,
        "weight_transform": "exp",
        "dbl_init_method": "fan_in_lognormal",
        "reactivation_type": "endpoint_tanh",
        "reactivation_init_policy": "fixed",
        "reactivation_init_m": 3.0,
        "reactivation_init_b": 1.5,
        "adaptive_initialization": False,
        "initial_child_conductance": 0.3,
        "leak_conductance": 0.03,
        "learnable_leak": True,
    }
    kwargs.update(overrides)
    torch.manual_seed(52)
    return DendriticBranchLayer(**kwargs)


def test_explicit_defaults_preserve_state_output_impact_and_rng():
    torch.manual_seed(3)
    a = bank()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(3)
    b = bank(**NEW_DEFAULTS)
    assert torch.equal(torch.get_rng_state(), rng)
    assert a.state_dict().keys() == b.state_dict().keys()
    assert "source_activity" not in a.state_dict()
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key])
    x = torch.rand(7, 12)
    torch.manual_seed(5)
    ya = a(x)
    torch.manual_seed(5)
    yb = b(x)
    assert torch.equal(ya, yb)
    assert torch.equal(a.impact, b.impact)
    a.eval()
    b.eval()
    assert torch.equal(a(x), b(x))
    assert torch.equal(
        branch().state_dict()["branch_excitation.pre_w"],
        branch(initial_child_conductance_log_sd=0.0).state_dict()[
            "branch_excitation.pre_w"
        ],
    )


def test_active_drive_averages_over_examples_with_active_sources():
    layer = bank(release_p_min=1.0, release_impact_metric="active_drive").double()
    x = torch.rand(6, 12, dtype=torch.double) + 0.1
    x[:, layer.connection_indices[0, 0]] = 0
    x[:4, layer.connection_indices[1, 1]] = 0
    layer(x)
    selected = x[:, layer.connection_indices.long()]
    drive = selected * layer.pre_w.exp()
    active = selected.ne(0).sum(0)
    expected = 0.01 * drive.sum(0) / active.clamp_min(1)
    torch.testing.assert_close(layer.impact, expected.detach())
    assert layer.impact[0, 0] == 0 and layer.contact_age[0, 0] == 0


@pytest.mark.parametrize("transform", ["exp", "softplus"])
def test_weight_and_saliency_metrics_accumulate_after_steps(transform):
    torch.manual_seed(9)
    layer = bank(weight_transform=transform, release_impact_metric="saliency")
    layer.eval()
    x = torch.rand(5, 12)
    w = layer.sparse_weight().detach().requires_grad_()
    p = layer.release_probability(w)
    reference = (x[:, layer.connection_indices.long()] * w * p).sum(-1)
    (reference**2).sum().backward()
    (layer(x) ** 2).sum().backward()
    layer.accumulate_step_impact()
    torch.testing.assert_close(layer.impact, 0.01 * (w * w.grad).abs())
    weight_metric = bank(weight_transform=transform, release_impact_metric="weight")
    weight_metric.accumulate_step_impact()
    torch.testing.assert_close(
        weight_metric.impact, 0.01 * weight_metric.sparse_weight().detach()
    )
    drive = bank(weight_transform=transform)
    drive.accumulate_step_impact()
    assert drive.impact.eq(0).all()


def test_softplus_release_matches_exp_initialization_function_and_gradients():
    a = branch(weight_transform="exp").double().eval()
    b = branch(weight_transform="softplus").double().eval()
    for pathway in ("branch_excitation", "branch_inhibition"):
        left, right = getattr(a, pathway), getattr(b, pathway)
        torch.testing.assert_close(left.sparse_weight(), right.sparse_weight())
        assert right.initial_log_mean == pytest.approx(left.initial_log_mean)
    torch.testing.assert_close(
        a.branches_to_output.weight(), b.branches_to_output.weight()
    )
    x, i, c = (
        torch.rand(3, 10, dtype=torch.double),
        torch.rand(3, 5, dtype=torch.double),
        torch.rand(3, 8, dtype=torch.double),
    )
    torch.testing.assert_close(
        a(x, inhibitory_input=i, branch_input=c),
        b(x, inhibitory_input=i, branch_input=c),
    )
    b(x, inhibitory_input=i, branch_input=c).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in b.parameters())
    replay = branch(weight_transform="softplus").double().eval()
    replay.load_state_dict(b.state_dict(), strict=True)
    torch.testing.assert_close(
        replay(x, inhibitory_input=i, branch_input=c),
        b(x, inhibitory_input=i, branch_input=c),
        rtol=0,
        atol=0,
    )


def test_conductance_cap_clamps_release_and_blocks_gradient_above_cap():
    layer = bank(release_max_conductance=0.5).double().eval()
    with torch.no_grad():
        layer.pre_w[0, 0] = math.log(3.0)
        layer.pre_w[0, 1] = math.log(0.1)
    w = layer.pre_w.exp().clamp(max=0.5)
    x = torch.rand(4, 12, dtype=torch.double)
    p = 0.5 + 0.5 * w / (w + 0.25 * layer.release_weight_scale)
    reference = (x[:, layer.connection_indices.long()] * w * p).sum(-1)
    actual = layer(x)
    torch.testing.assert_close(actual, reference)
    actual.sum().backward()
    assert layer.pre_w.grad[0, 0] == 0
    assert layer.pre_w.grad[0, 1] != 0


def test_active_regrowth_never_samples_silent_sources():
    layer = bank(release_track_source_activity=True)
    assert "source_activity" in layer.state_dict()
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
    x = torch.rand(8, 12) + 0.1
    silent = torch.tensor([1, 2, 3, 5, 7])
    x[:, silent] = 0
    layer(x).sum().backward()
    optimizer.step()
    assert layer.source_activity[silent].eq(0).all()
    assert layer.source_activity[0] > 0
    layer.impact.fill_(1.0)
    layer.impact[:, 0] = 0.0
    layer.contact_age.fill_(5)
    old = layer.connection_indices.clone()
    rewire = CellwiseImpactRewiring(
        [(2, [layer])],
        ImpactRewiringConfig(
            frequency=1,
            maturity_steps=1,
            eligible_fraction=1.0,
            segment_fraction=0.25,
            regrowth_sources="active",
            min_source_activity=0.0,
        ),
    )
    assert rewire.step(optimizer) == 4
    changed = old.ne(layer.connection_indices)
    assert changed.sum() == 4
    new_sources = layer.connection_indices[changed].long()
    assert not torch.isin(new_sources, silent).any()
    with pytest.raises(ValueError, match="release_track_source_activity"):
        CellwiseImpactRewiring(
            [(2, [bank()])], ImpactRewiringConfig(regrowth_sources="active")
        )


def test_initial_regrowth_draws_from_recorded_initialization_law():
    layer = branch()
    synapses = layer.branch_excitation
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
    assert synapses.initial_log_mean == pytest.approx(-math.log(6) - 0.5 * 0.5**2)
    synapses.impact.fill_(1.0)
    synapses.impact[:, :2] = 0.0
    synapses.contact_age.fill_(5)
    before = synapses.pre_w.detach().clone()
    rewire = CellwiseImpactRewiring(
        [(4, [synapses])],
        ImpactRewiringConfig(
            frequency=1,
            maturity_steps=1,
            eligible_fraction=1.0,
            segment_fraction=0.5,
            regrowth_weight="initial",
            seed=4,
        ),
    )
    assert rewire.step(optimizer) == 8
    changed = before.ne(synapses.pre_w.detach())
    assert changed.sum() == 8
    assert synapses.sparse_weight()[changed].gt(0).all()
    assert synapses.contact_age[changed].eq(0).all()
    with pytest.raises(ValueError, match="recorded log-normal law"):
        CellwiseImpactRewiring(
            [(2, [bank()])], ImpactRewiringConfig(regrowth_weight="initial")
        )


def test_initial_regrowth_law_matches_recorded_moments():
    torch.manual_seed(1)
    big = IndexedReleaseLinear(400, 400, 100, seed=3, weight_transform="exp")
    big.initial_log_mean, big.initial_log_sd = -2.0, 0.5
    big.impact.fill_(1.0)
    big.impact[:, :25] = 0.0
    big.contact_age.fill_(5)
    optimizer = torch.optim.Adam(big.parameters(), lr=0.01)
    before = big.pre_w.detach().clone()
    rewire = CellwiseImpactRewiring(
        [(400, [big])],
        ImpactRewiringConfig(
            frequency=1,
            maturity_steps=1,
            eligible_fraction=1.0,
            segment_fraction=0.25,
            regrowth_weight="initial",
        ),
    )
    assert rewire.step(optimizer) == 400 * 25
    log_new = big.pre_w.detach()[before.ne(big.pre_w.detach())]
    assert log_new.mean().item() == pytest.approx(-2.0, abs=0.03)
    assert log_new.std().item() == pytest.approx(0.5, abs=0.03)


def test_child_coupling_lognormal_law_and_validation():
    constant = branch()
    rng = torch.get_rng_state().clone()
    same = branch(initial_child_conductance_log_sd=0.0)
    assert torch.equal(torch.get_rng_state(), rng)
    torch.testing.assert_close(
        constant.branches_to_output.weight(), same.branches_to_output.weight()
    )
    wide = branch(
        output_dim=500,
        initial_child_conductance_log_sd=0.4,
        initialization_seed=11,
    )
    coupling = wide.branches_to_output.weight().detach()
    assert coupling.std() > 0
    assert coupling.mean().item() == pytest.approx(0.3, rel=0.05)
    assert coupling.log().std().item() == pytest.approx(0.4, rel=0.1)
    replay = branch(
        output_dim=500,
        initial_child_conductance_log_sd=0.4,
        initialization_seed=11,
    )
    assert torch.equal(
        replay.branches_to_output.weight(), wide.branches_to_output.weight()
    )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        branch(initial_child_conductance_log_sd=-0.1)
    with pytest.raises(ValueError, match="requires fan_in_lognormal"):
        branch(initial_child_conductance_log_sd=0.2, dbl_init_method="default")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"release_impact_metric": "drives"}, "release_impact_metric"),
        ({"release_max_conductance": 0.0}, "finite and positive"),
        ({"release_max_conductance": float("nan")}, "finite and positive"),
        (
            {"release_impact_metric": "saliency", "release_max_conductance": 2.0},
            "saliency",
        ),
        ({"weight_transform": "identity"}, "exp or softplus"),
    ],
)
def test_invalid_release_options_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        bank(**kwargs)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"regrowth_sources": "dead"}, "regrowth_sources"),
        ({"regrowth_weight": "mean"}, "regrowth_weight"),
        ({"min_source_activity": 0.1}, "requires regrowth_sources='active'"),
        (
            {"regrowth_sources": "active", "min_source_activity": -1.0},
            "finite and nonnegative",
        ),
    ],
)
def test_invalid_rewiring_options_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ImpactRewiringConfig(**kwargs)


def test_typed_and_structured_config_roundtrip():
    options = {
        "release_impact_metric": "weight",
        "release_max_conductance": 25.0,
        "release_track_source_activity": True,
        "initial_child_conductance_log_sd": 0.25,
    }
    cfg = DendriticSynapseConfig.from_kwargs(options)
    assert cfg.indexed.release_impact_metric == "weight"
    assert cfg.initialization.initial_child_conductance_log_sd == 0.25
    for key, value in options.items():
        assert cfg.to_kwargs()[key] == value
    SparsityConfig(
        indexed={
            "release_impact_metric": "active_drive",
            "release_max_conductance": 10.0,
            "release_track_source_activity": True,
        }
    )


@pytest.mark.parametrize("family", ["population_network", "einet"])
@pytest.mark.parametrize("transform", ["exp", "softplus"])
def test_options_route_through_each_supported_builder(family, transform):
    config = _config(family, "shunting", "endpoint_tanh")
    options = (
        config["population_network"]["layers"][0]["population_defaults"]
        if family == "population_network"
        else config["morphology"]
    )
    options.update(
        weight_transform=transform,
        dbl_init_method="fan_in_lognormal",
        leak_conductance=0.03,
        learnable_leak=True,
    )
    release = {
        "release_impact_metric": "active_drive",
        "release_max_conductance": 50.0,
        "release_track_source_activity": True,
    }
    if family == "population_network":
        options.update(
            topk_type="indexed_release",
            initial_child_conductance_log_sd=0.3,
            **release,
        )
    else:
        config["sparsity"] = {"type": "indexed_release", "indexed": dict(release)}
        config["implementation"]["initial_child_conductance_log_sd"] = 0.3
    torch.manual_seed(61)
    model = get_architecture(family, deepcopy(config), input_dim=6)
    banks = [m for m in model.modules() if isinstance(m, IndexedReleaseLinear)]
    assert banks
    for synapses in banks:
        assert synapses.release_impact_metric == "active_drive"
        assert synapses.release_max_conductance == 50.0
        assert synapses.source_activity is not None
        assert synapses.weight_transform == transform
        assert synapses.initial_log_sd == pytest.approx(0.5)
    branches = [m for m in model.modules() if isinstance(m, DendriticBranchLayer)]
    assert all(b.initial_child_conductance_log_sd == 0.3 for b in branches)
    assert any(
        b.input_branches and b.branches_to_output.weight().std() > 0 for b in branches
    )
    x = torch.rand(4, 6)
    model.train()
    model(x).sum().backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    model.eval()
    replay = get_architecture(family, deepcopy(config), input_dim=6).eval()
    replay.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(replay(x), model(x), rtol=0, atol=0)
