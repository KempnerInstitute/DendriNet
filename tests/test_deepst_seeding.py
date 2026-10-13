"""Seeded DeepST topology is reproducible and independent of the global RNG."""

from __future__ import annotations

import pytest
import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.credit_deepst import (
    CreditGatedDeepstLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.deepst import (
    DeepstLinear,
)


def _layer(seed, **kwargs):
    return DeepstLinear(20, 6, target_density=0.3, seed=seed, **kwargs)


def test_seeded_initial_contacts_ignore_and_preserve_the_global_rng():
    torch.manual_seed(0)
    first = _layer(5)
    state = torch.random.get_rng_state()
    torch.manual_seed(123)
    second = _layer(5)
    assert torch.equal(first.mask, second.mask)
    torch.manual_seed(0)
    _layer(5)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert not torch.equal(first.mask, _layer(6).mask)


def test_seeded_rewiring_events_replay_after_a_resume():
    def run(layer, steps):
        for _ in range(steps):
            with torch.no_grad():
                layer.pre_w.fill_(-10.0)  # every contact falls below threshold
            layer.apply_rewiring()
        return layer.mask.clone()

    torch.manual_seed(0)
    reference = _layer(5)
    initial_mask = reference.mask.clone()
    reference_mask = run(reference, 4)
    assert not torch.equal(reference_mask, initial_mask)

    resumed = _layer(5)
    run(resumed, 2)
    restored = _layer(5)
    restored.load_state_dict(resumed.state_dict())
    torch.manual_seed(999)  # a different global stream must not matter
    assert torch.equal(run(restored, 2), reference_mask)


def test_rank_dependent_rewiring_is_rejected_for_data_parallel_runs():
    from dendritic_modeling.scripts.training.train_experiments import (
        _reject_rank_dependent_rewiring,
    )

    _reject_rank_dependent_rewiring(torch.nn.Sequential(_layer(5)))
    with pytest.raises(ValueError, match="unseeded DeepST"):
        _reject_rank_dependent_rewiring(torch.nn.Sequential(_layer(None)))
    credit = CreditGatedDeepstLinear(
        20, 6, rewiring_mode="constant_branch", synapses_per_branch=3, seed=5
    )
    with pytest.raises(ValueError, match="credit_deepst"):
        _reject_rank_dependent_rewiring(torch.nn.Sequential(credit))


def test_deepst_init_method_has_one_default_on_every_construction_path():
    from dendritic_modeling.config.keys import field_default
    from dendritic_modeling.config.model import SparsityConfig
    from dendritic_modeling.config.synapse_payloads import DeepSTConfig

    default = field_default(DeepSTConfig, "init_method")
    assert DeepstLinear(4, 2).init_method == default
    assert SparsityConfig().deepst["init_method"] == default


@pytest.mark.parametrize("init_method", ["xavier_normal", None])
def test_relu_contacts_are_born_and_regrown_alive(init_method):
    layer = _layer(5, weight_transform="relu", init_method=init_method, use_noise=False)
    assert bool((layer.weight()[layer.mask] > 0).all())
    with torch.no_grad():
        layer.pre_w.fill_(-1.0)  # relu weight 0: every contact is pruned
    layer.apply_rewiring()
    # The rewired mask also marks pruned slots; regrown ones are active.
    regrown = layer._last_rewired_mask.bool() & layer.mask
    assert bool(regrown.any())
    assert bool((layer.weight()[regrown] > 0).all())
    layer.weight()[regrown].sum().backward()
    assert bool((layer.pre_w.grad[regrown] != 0).all())


def test_exp_regrowth_is_unchanged():
    layer = _layer(
        5, weight_transform="exp", init_method="xavier_normal", use_noise=False
    )
    with torch.no_grad():
        layer.pre_w.fill_(-20.0)
    layer.apply_rewiring()
    regrown = layer._last_rewired_mask.bool() & layer.mask
    torch.testing.assert_close(
        layer.pre_w[regrown],
        torch.full_like(layer.pre_w[regrown], float(torch.log(torch.tensor(1e-6)))),
    )
