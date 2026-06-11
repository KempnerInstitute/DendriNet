"""Tests for the per-forward recurrent TopK cache.

The cache reuses deterministic TopK selection across the timesteps of a single
recurrent unroll. It caches only the mask under autograd (preserving the original
per-timestep weight-transform graph exactly) and caches the full pruned weight
under no_grad inference. It must never engage for non-deterministic variants,
noisy masks, or dynamic forward grad scaling.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.dense_to_sparse import (
    DenseToSparseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.stochastic import (
    StochasticTopKLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
    apply_recurrent_weight_cache,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.variance import (
    VarianceTopKLinear,
)
from dendritic_modeling.networks.architectures.factory import get_architecture


def _topk(seed: int = 0) -> TopKLinear:
    torch.manual_seed(seed)
    return TopKLinear(in_features=8, out_features=5, K=3, weight_transform="softplus")


def test_cache_preserves_forward_and_gradients():
    """3 repeated forwards (recurrent-like) must match cached vs uncached, in
    both the forward outputs and the accumulated gradients."""
    x = torch.randn(4, 8)

    m_ref = _topk()
    y_ref = m_ref(x) + m_ref(x) + m_ref(x)
    y_ref.sum().backward()

    m_cache = _topk()  # identical init (same seed)
    m_cache.set_recurrent_weight_cache(True)
    y_cache = m_cache(x) + m_cache(x) + m_cache(x)
    m_cache.set_recurrent_weight_cache(False)
    y_cache.sum().backward()

    torch.testing.assert_close(y_ref, y_cache)
    torch.testing.assert_close(m_ref.pre_w.grad, m_cache.pre_w.grad)


def test_cache_reuses_mask_then_clears_under_autograd():
    m = _topk()
    x = torch.randn(2, 8)
    m.set_recurrent_weight_cache(True)
    _ = m(x)
    cached = m._recurrent_cached_weight_mask
    assert cached is not None
    assert m._recurrent_cached_pruned_weight is None
    _ = m(x)
    assert m._recurrent_cached_weight_mask is cached  # reused, not recomputed
    m.set_recurrent_weight_cache(False)
    assert m._recurrent_cached_weight_mask is None  # released
    assert m._recurrent_cached_pruned_weight is None


def test_cache_reuses_pruned_weight_under_no_grad():
    m = _topk()
    x = torch.randn(2, 8)
    m.set_recurrent_weight_cache(True)
    with torch.no_grad():
        _ = m(x)
        cached = m._recurrent_cached_pruned_weight
        assert cached is not None
        _ = m(x)
        assert m._recurrent_cached_pruned_weight is cached
    m.set_recurrent_weight_cache(False)
    assert m._recurrent_cached_weight_mask is None
    assert m._recurrent_cached_pruned_weight is None


def test_noise_level_bypasses_cache():
    torch.manual_seed(0)
    m = TopKLinear(
        in_features=8,
        out_features=5,
        K=3,
        weight_transform="softplus",
        noise_level=0.1,
    )
    m.set_recurrent_weight_cache(True)
    assert m._recurrent_weight_cache_enabled is True  # class supports it...
    assert m._recurrent_weight_cache_active() is False  # ...but noise bypasses
    _ = m(torch.randn(2, 8))
    assert m._recurrent_cached_weight_mask is None
    assert m._recurrent_cached_pruned_weight is None


def test_dynamic_grad_scaling_bypasses_cache():
    m = _topk()
    m.set_recurrent_weight_cache(True)
    m._use_forward_dynamic_grad_scaling = True
    assert m._recurrent_weight_cache_active() is False
    _ = m(torch.randn(2, 8))
    assert m._recurrent_cached_weight_mask is None
    assert m._recurrent_cached_pruned_weight is None


@pytest.mark.parametrize(
    "cls", [StochasticTopKLinear, VarianceTopKLinear, DenseToSparseLinear]
)
def test_nondeterministic_subclasses_opt_out(cls):
    assert cls._supports_recurrent_weight_cache is False
    m = cls(in_features=8, out_features=5, K=3, weight_transform="softplus")
    m.set_recurrent_weight_cache(True)
    assert m._recurrent_weight_cache_enabled is False  # request ignored
    _ = m(torch.rand(3, 8))
    assert m._recurrent_cached_weight_mask is None
    assert m._recurrent_cached_pruned_weight is None


def test_apply_recurrent_weight_cache_toggles_tree():
    net = nn.Sequential(_topk(0), nn.ReLU(), _topk(1))
    apply_recurrent_weight_cache(net, True)
    assert all(
        m._recurrent_weight_cache_enabled for m in net if isinstance(m, TopKLinear)
    )
    apply_recurrent_weight_cache(net, False)
    assert all(
        not m._recurrent_weight_cache_enabled for m in net if isinstance(m, TopKLinear)
    )


def _recurrent_population_config() -> dict:
    base = {
        "ff_excitatory_synapses": 2,
        "ff_inhibitory_synapses": 1,
        "rec_excitatory_synapses": 1,
        "rec_inhibitory_synapses": 1,
        "somatic_synapses": True,
        "reactivate": True,
        "dbl_init_method": "default",
        "weight_transform": "exp",
    }
    return {
        "population_network": {
            "output_mode": "last",
            "layers": [
                {
                    "name": "layer0",
                    "recurrent": True,
                    "readout_population": "e",
                    "population_defaults": base,
                    "populations": [
                        {
                            "name": "e",
                            "polarity": "excitatory",
                            "n_neurons": 5,
                            "branch_factors": [1],
                        },
                        {
                            "name": "i",
                            "polarity": "inhibitory",
                            "n_neurons": 3,
                            "branch_factors": [1],
                        },
                    ],
                    "connections": [
                        {"source": "input", "target": "e"},
                        {"source": "input", "target": "i"},
                        {"source": "e", "target": "i", "timing": "delayed"},
                        {"source": "i", "target": "e", "timing": "delayed"},
                    ],
                }
            ],
        }
    }


def test_population_network_recurrent_unroll_parity_with_and_without_cache():
    """The wired cache must not change a recurrent unroll's output."""
    torch.manual_seed(0)
    net = get_architecture(
        "population_network", _recurrent_population_config(), input_dim=4
    )
    assert net.is_recurrent is True

    x = torch.randn(2, 6, 4)
    net.eval()
    with torch.no_grad():
        out_cached = net(x)
        out_cached = out_cached[0] if isinstance(out_cached, tuple) else out_cached

        # Force the cache off everywhere (request becomes a no-op) for reference.
        for module in net.modules():
            if isinstance(module, TopKLinear):
                module._supports_recurrent_weight_cache = False
        out_ref = net(x)
        out_ref = out_ref[0] if isinstance(out_ref, tuple) else out_ref

    assert torch.isfinite(out_cached).all()
    torch.testing.assert_close(out_cached, out_ref)


def test_population_network_recurrent_grad_parity_with_and_without_cache():
    """Training gradients must match exactly with cache enabled vs disabled."""
    torch.manual_seed(0)
    cfg = _recurrent_population_config()
    cfg["population_network"]["output_mode"] = "all"
    net_cached = get_architecture("population_network", cfg, input_dim=4)
    net_ref = copy.deepcopy(net_cached)

    for module in net_ref.modules():
        if isinstance(module, TopKLinear):
            module._supports_recurrent_weight_cache = False

    x = torch.randn(3, 7, 4)
    y = torch.randn(3, 7, 5)

    out_cached = net_cached(x)
    loss_cached = torch.nn.functional.mse_loss(out_cached, y)
    loss_cached.backward()

    out_ref = net_ref(x)
    loss_ref = torch.nn.functional.mse_loss(out_ref, y)
    loss_ref.backward()

    torch.testing.assert_close(out_cached, out_ref, rtol=0, atol=0)
    torch.testing.assert_close(loss_cached, loss_ref, rtol=0, atol=0)
    for (name_cached, param_cached), (name_ref, param_ref) in zip(
        net_cached.named_parameters(),
        net_ref.named_parameters(),
    ):
        assert name_cached == name_ref
        if param_cached.grad is None and param_ref.grad is None:
            continue
        torch.testing.assert_close(
            param_cached.grad,
            param_ref.grad,
            rtol=0,
            atol=0,
            msg=f"gradient mismatch for {name_cached}",
        )


def test_population_network_cache_released_on_forward_error():
    """A failing recurrent forward must not leave TopK cache mode enabled."""
    torch.manual_seed(0)
    net = get_architecture(
        "population_network", _recurrent_population_config(), input_dim=4
    )

    with pytest.raises(ValueError):
        net(torch.randn(2, 6, 5))

    for module in net.modules():
        if isinstance(module, TopKLinear):
            assert module._recurrent_weight_cache_enabled is False
            assert module._recurrent_cached_weight_mask is None
            assert module._recurrent_cached_pruned_weight is None
