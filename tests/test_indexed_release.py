import pytest
import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.impact_rewiring import (
    CellwiseImpactRewiring,
    ImpactRewiringConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_release import (
    IndexedReleaseLinear,
)


def test_mean_release_and_parameter_gradient():
    layer = (
        IndexedReleaseLinear(
            5, 2, 4, seed=8, weight_transform="exp", release_weight_scale=0.2
        )
        .double()
        .eval()
    )
    x = torch.rand(3, 5, dtype=torch.double, requires_grad=True)
    w = layer.pre_w.exp()
    p = 0.5 + 0.5 * w / (w + 0.05)
    reference = (x[:, layer.connection_indices.long()] * w * p).sum(-1)
    actual = layer(x)
    torch.testing.assert_close(actual, reference)
    for a, b in zip(
        torch.autograd.grad(actual.sum(), (x, layer.pre_w)),
        torch.autograd.grad(reference.sum(), (x, layer.pre_w)),
    ):
        torch.testing.assert_close(a, b)


def test_stochastic_mean_and_checkpoint():
    layer = IndexedReleaseLinear(5, 2, 4, seed=8, weight_transform="exp")
    x = torch.ones(30000, 5)
    torch.manual_seed(14)
    actual = layer(x).mean(0)
    # Training forwards update impact; contact ages count optimizer steps.
    assert layer.contact_age.eq(0).all() and layer.impact.gt(0).all()
    state = layer.state_dict()
    replay = IndexedReleaseLinear(5, 2, 4, seed=9, weight_transform="exp")
    replay.load_state_dict(state)
    layer.eval()
    replay.eval()
    expected = layer(x[:1])[0]
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0)
    torch.testing.assert_close(replay(x[:1]), layer(x[:1]), rtol=0, atol=0)


def test_cellwise_rewiring_caps_legality_and_optimizer_reset():
    layer = IndexedReleaseLinear(12, 4, 4, seed=8, weight_transform="exp")
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
    layer(torch.ones(3, 12)).sum().backward()
    optimizer.step()
    layer.impact.fill_(0)
    layer.impact[:, 0] = 10
    layer.contact_age.fill_(5)
    old = layer.connection_indices.clone()
    rewire = CellwiseImpactRewiring(
        [(2, [layer])],
        ImpactRewiringConfig(
            frequency=1,
            maturity_steps=1,
            eligible_fraction=1.0,
            segment_fraction=0.25,
            until_step=3,
        ),
    )
    assert rewire.step(optimizer) == 4
    changed = old.ne(layer.connection_indices)
    assert changed.sum(1).eq(1).all()
    assert layer.contact_age[changed].eq(0).all()
    assert optimizer.state[layer.pre_w]["exp_avg"][changed].eq(0).all()
    assert all(len(row.unique()) == 4 for row in layer.connection_indices)
    # The controller step ages every surviving contact by one optimizer step.
    assert layer.contact_age[~changed].eq(6).all()
    rewire.step(optimizer)
    frozen = layer.connection_indices.clone()
    assert rewire.step(optimizer) == 0
    assert torch.equal(frozen, layer.connection_indices)


def test_release_forward_waits_for_a_broadcast_topology():
    layer = IndexedReleaseLinear(5, 2, 4, seed=8)
    layer._topology_initialized.zero_()
    layer._topology_ready = False
    with pytest.raises(RuntimeError, match="awaiting rank-0 broadcast"):
        layer(torch.rand(3, 5))
    layer._topology_initialized.fill_(1)
    assert layer(torch.rand(3, 5)).shape == (3, 2)
