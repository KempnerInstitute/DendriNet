"""Weight decay reuses the training forward's Top-K selection when it is current.

A deterministic selection is a function of ``pre_w`` alone and decay runs
before the optimizer update, so the selection the step's forward made is the
one a recomputation would give. Any change to ``pre_w`` in between, and every
stochastic or state-dependent selector, falls back to selecting again.
"""

from __future__ import annotations

import copy
import pickle

import pytest
import torch
from torch import nn

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    StochasticTopKLinear,
    TopKLinear,
    VarianceTopKLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.dense_to_sparse import (
    DenseToSparseLinear,
)
from dendritic_modeling.training.optimizers.custom import CustomWeightDecayOptimizer


def _layer(cls=TopKLinear, seed=0, **options):
    torch.manual_seed(seed)
    layer = cls(8, 4, 3, **options)
    layer.initialize()
    return layer


class _Net(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x)


def _train(layer, steps=4, *, boosting=False):
    net = _Net(layer)
    optimizer = CustomWeightDecayOptimizer(
        net,
        torch.optim.SGD(net.parameters(), lr=0.1),
        weight_decay=0.05,
        weight_boosting=boosting,
    )
    data = torch.rand(steps, 5, 8, generator=torch.Generator().manual_seed(1))
    for x in data:
        optimizer.zero_grad()
        net(x).square().mean().backward()
        optimizer.step()
    return layer


def _count_topk(monkeypatch):
    calls = []
    topk = torch.topk

    def counting(*args, **kwargs):
        calls.append(1)
        return topk(*args, **kwargs)

    monkeypatch.setattr(torch, "topk", counting)
    return calls


@pytest.mark.parametrize("boosting", [False, True])
@pytest.mark.parametrize("transform", ["exp", "softplus", "identity"])
def test_decay_with_the_forward_selection_matches_a_new_selection(
    monkeypatch, transform, boosting
):
    options = {
        "weight_transform": transform,
        "forbidden_input_index_per_output": [1, -1, 4, 0],
    }
    reference = _layer(**options)
    monkeypatch.setattr(reference, "_remember_forward_selection", lambda mask: None)
    _train(reference, boosting=boosting)

    calls = _count_topk(monkeypatch)
    reused = _train(_layer(**options), boosting=boosting)

    assert torch.equal(reused.pre_w, reference.pre_w)
    assert len(calls) == 4 + 1  # one per forward, plus the first decay


def test_decay_selects_again_after_pre_w_changes():
    layer = _train(_layer(), steps=1)  # a decay has run: forwards now record
    fresh = copy.deepcopy(layer)
    x = torch.rand(5, 8)
    layer(x)
    fresh(x)
    with torch.no_grad():  # an update between forward and decay
        flipped = -layer.pre_w
        layer.pre_w.copy_(flipped)
        fresh.pre_w.copy_(flipped)
    fresh._forward_selection = None

    layer.decay_weights(0.05)
    fresh.decay_weights(0.05)

    assert torch.equal(layer.pre_w, fresh.pre_w)


def test_the_forward_selection_is_used_once_and_only_from_training():
    layer = _train(_layer(), steps=1)
    layer.eval()
    layer(torch.rand(5, 8))
    assert layer._forward_selection is None

    layer.train()
    layer(torch.rand(5, 8))
    assert layer._forward_selection is not None
    assert copy.deepcopy(layer)._forward_selection is None
    assert pickle.loads(pickle.dumps(layer))._forward_selection is None
    layer.decay_weights(0.05)
    assert layer._forward_selection is None


def test_runs_without_decay_keep_no_selection():
    layer = _layer()
    layer(torch.rand(5, 8))
    assert layer._forward_selection is None


@pytest.mark.parametrize(
    "build",
    [
        lambda: _layer(noise_level=0.1),
        lambda: _layer(StochasticTopKLinear),
        lambda: _layer(VarianceTopKLinear),
        lambda: _layer(DenseToSparseLinear),
    ],
    ids=["noisy", "stochastic", "variance", "dense_to_sparse"],
)
def test_non_deterministic_selections_are_never_reused(build):
    layer = _train(build(), steps=2)
    layer(torch.rand(5, 8))
    assert layer._forward_selection is None
