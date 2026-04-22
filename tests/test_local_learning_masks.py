"""Tests for TopK forward-mask caching and local-learning mask usage."""

from unittest.mock import patch

import torch
import torch.nn as nn

from dendritic_modeling.networks import DendriticBranchLayer, TopKLinear
from dendritic_modeling.training.strategies.local_learning import LocalCreditAssignment


class _SingleBranchModel(nn.Module):
    """Minimal model exposing one dendritic branch layer."""

    def __init__(self):
        super().__init__()
        self.branch = DendriticBranchLayer(
            output_dim=4,
            excitatory_input_dim=8,
            excitatory_synapses_per_branch=2,
            topk_noise_level=0.5,
            reactivate=False,
        )

    def forward(self, x):
        return self.branch(x)


def test_topk_forward_does_not_cache_by_default():
    """Forward pass should NOT cache the mask when cache_mask is False."""
    torch.manual_seed(7)
    layer = TopKLinear(in_features=12, out_features=5, K=3, noise_level=0.5)
    layer.initialize()

    x = torch.rand(4, 12)
    _ = layer(x)

    assert layer._last_forward_weight_mask is None


def test_topk_forward_caches_exact_mask_when_enabled():
    """Forward pass should persist the exact sampled TopK mask when cache_mask=True."""
    torch.manual_seed(7)
    layer = TopKLinear(in_features=12, out_features=5, K=3, noise_level=0.5)
    layer.initialize()
    layer.cache_mask = True

    x = torch.rand(4, 12)
    out = layer(x)

    cached_mask = layer._last_forward_weight_mask
    assert isinstance(cached_mask, torch.Tensor)
    assert cached_mask.shape == (5, 12)
    assert torch.all(cached_mask.sum(dim=1) == 3)

    expected = x @ (cached_mask * layer.weight()).t()
    torch.testing.assert_close(out, expected)


def test_local_recorders_use_cached_topk_mask_without_extra_sampling():
    """Local-learning hooks should not trigger a second noisy TopK sampling."""
    torch.manual_seed(11)
    model = _SingleBranchModel()
    topk = model.branch.branch_excitation
    assert isinstance(topk, TopKLinear)

    trainer = LocalCreditAssignment.__new__(LocalCreditAssignment)
    trainer._decoder_cache = {}

    records, handles = trainer._attach_local_recorders(model)
    topk.cache_mask = True
    x = torch.rand(6, 8)

    try:
        with patch.object(topk, "weight_mask", wraps=topk.weight_mask) as wrapped_mask:
            _ = model(x)
            assert wrapped_mask.call_count == 1
    finally:
        for handle in handles:
            handle.remove()

    assert len(records) == 1
    rec = records[0]
    assert isinstance(rec.get("exc_mask"), torch.Tensor)
    torch.testing.assert_close(rec["exc_mask"], topk._last_forward_weight_mask)
