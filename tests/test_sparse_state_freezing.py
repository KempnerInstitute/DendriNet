"""Auxiliary forwards and recomputation must not advance sparse-synapse state."""

from __future__ import annotations

import pytest
import torch

from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.utils.sparse_maintenance import (
    forward_stateful_modules,
    frozen_sparse_state,
)
from dendritic_modeling.training.fsdp_utils import _apply_gradient_checkpointing


def _layer(topk_type):
    torch.manual_seed(0)
    layer = DendriticBranchLayer(
        output_dim=3,
        excitatory_input_dim=8,
        excitatory_synapses_per_branch=2,
        reactivate=False,
        topk_type=topk_type,
    )
    layer.train()
    return layer


def _state(layer):
    return {k: v.clone() for k, v in layer.state_dict().items()}


@pytest.mark.parametrize("topk_type", ["dense_to_sparse", "variance"])
def test_frozen_forwards_leave_state_and_outputs_unchanged(topk_type):
    layer = _layer(topk_type)
    x = torch.rand(4, 8)
    before = _state(layer)
    with frozen_sparse_state():
        frozen = layer(x)
    after = _state(layer)
    assert all(torch.equal(before[k], after[k]) for k in before)
    bank = layer.branch_excitation
    if topk_type == "variance":
        # A frozen forward selects from the statistics without this batch.
        bank.stats_tracking = False
        torch.testing.assert_close(frozen, layer(x))
        bank.stats_tracking = True
    else:
        torch.testing.assert_close(frozen, layer(x))
    # Outside the context the training forward changes the state again.
    layer(x)
    assert any(not torch.equal(after[k], v) for k, v in _state(layer).items())


@pytest.mark.parametrize("topk_type", ["dense_to_sparse", "variance"])
def test_gradient_checkpointing_rejects_stateful_forwards(topk_type):
    layer = _layer(topk_type)
    assert forward_stateful_modules(layer)
    with pytest.raises(ValueError, match="replaying training-forward state"):
        _apply_gradient_checkpointing(torch.nn.Sequential(layer))


def test_gradient_checkpointing_accepts_stateless_synapses():
    layer = _layer("standard")
    assert forward_stateful_modules(layer) == []
    _apply_gradient_checkpointing(torch.nn.Sequential(layer))
