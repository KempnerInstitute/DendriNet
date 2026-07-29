"""Regression tests for realized sparse-operator analysis."""

from __future__ import annotations

import pytest
import torch

from dendritic_modeling.analysis.utils.effective_synapses import (
    effective_synapse_snapshot,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.stochastic import (
    StochasticTopKLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)


def test_snapshot_matches_the_deterministic_topk_forward_operator() -> None:
    layer = TopKLinear(
        in_features=4,
        out_features=2,
        K=2,
        weight_transform="identity",
    )
    with torch.no_grad():
        layer.pre_w.copy_(torch.tensor([[1.0, 4.0, 2.0, 3.0], [8.0, 5.0, 7.0, 6.0]]))

    snapshot = effective_synapse_snapshot(layer)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    torch.testing.assert_close(layer(x), x @ snapshot.effective_weight.T)
    torch.testing.assert_close(snapshot.realized_k, torch.tensor([2, 2]))
    assert snapshot.mask_source == "deterministic_current"
    assert torch.count_nonzero(snapshot.candidate_weight) == 8
    assert torch.count_nonzero(snapshot.effective_weight) == 4


def test_large_inactive_candidate_cannot_enter_effective_snapshot() -> None:
    allowed = torch.tensor([[True, True, False]])
    layer = TopKLinear(
        in_features=3,
        out_features=1,
        K=1,
        weight_transform="identity",
        connection_mask=allowed,
    )
    with torch.no_grad():
        layer.pre_w.copy_(torch.tensor([[1.0, 2.0, 1_000.0]]))

    snapshot = effective_synapse_snapshot(layer)

    assert snapshot.candidate_weight[0, 2] == 1_000.0
    assert snapshot.candidate_mask[0, 2] == 0
    assert snapshot.active_mask[0, 2] == 0
    assert snapshot.effective_weight[0, 2] == 0
    assert snapshot.effective_weight[0, 1] == 2.0


def test_stochastic_snapshot_requires_and_reuses_realized_forward_mask() -> None:
    layer = StochasticTopKLinear(
        in_features=5,
        out_features=2,
        K=2,
        weight_transform="identity",
        temperature=0.5,
    )
    with pytest.raises(RuntimeError, match="realized forward mask"):
        effective_synapse_snapshot(layer)

    layer.cache_mask = True
    x = torch.ones(1, 5)
    output = layer(x)
    snapshot = effective_synapse_snapshot(layer)

    assert snapshot.mask_source == "cached_forward"
    torch.testing.assert_close(output, x @ snapshot.effective_weight.T)
