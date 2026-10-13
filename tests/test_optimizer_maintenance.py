"""Optimizer groups and shared branches preserve their declared update rules."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_rewire import (
    IndexedRewireLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.networks.utils.sparse_maintenance import apply_maintenance
from dendritic_modeling.training.optimizers.custom import CustomWeightDecayOptimizer
from dendritic_modeling.training.optimizers.muonh import MuonH


@pytest.mark.parametrize("kind", [TopKLinear, IndexedSparseLinear])
@pytest.mark.parametrize("transform", ["exp", "softplus", "identity"])
@pytest.mark.parametrize("reverse", [False, True])
def test_sparse_decay_uses_each_groups_learning_rate(kind, transform, reverse):
    banks = [kind(6, 2, 2, weight_transform=transform).double() for _ in range(2)]
    with torch.no_grad():
        for bank in banks:
            bank.pre_w.copy_(
                torch.linspace(0.2, 1.5, bank.pre_w.numel()).reshape_as(bank.pre_w)
            )
    refs = [deepcopy(bank) for bank in banks]
    groups = [
        {"params": bank.parameters(), "lr": lr} for bank, lr in zip(banks, [0.0, 0.2])
    ]
    if reverse:
        groups.reverse()
    wrapped = CustomWeightDecayOptimizer(
        nn.ModuleList(banks), torch.optim.SGD(groups), weight_decay=0.1
    )
    wrapped.step()
    for bank, ref, lr in zip(banks, refs, [0.0, 0.2]):
        if lr:
            ref.decay_weights(lr * 0.1, False)
        torch.testing.assert_close(bank.pre_w, ref.pre_w, rtol=0, atol=0)


def test_native_shared_branch_bank_is_decayed_once():
    first = DendriticBranchLayer(
        output_dim=2, excitatory_input_dim=6, excitatory_synapses_per_branch=2
    )
    second = deepcopy(first)
    second.branch_excitation = first.branch_excitation
    ref = deepcopy(first.branch_excitation)
    apply_maintenance(nn.ModuleList([first, second]), "decay_weights", 0.1, False)
    ref.decay_weights(0.1, False)
    torch.testing.assert_close(first.branch_excitation.pre_w, ref.pre_w, rtol=0, atol=0)


def test_native_shared_branch_rewiring_advances_once():
    first = DendriticBranchLayer(
        output_dim=2, excitatory_input_dim=6, excitatory_synapses_per_branch=2
    )
    first.branch_excitation = IndexedRewireLinear(6, 2, 2, seed=17, rewire_frequency=1)
    second = deepcopy(first)
    second.branch_excitation = first.branch_excitation
    ref = deepcopy(first.branch_excitation)
    apply_maintenance(nn.ModuleList([first, second]), "apply_rewiring")
    ref.apply_rewiring()
    assert int(first.branch_excitation.rewire_step) == 1
    assert torch.equal(
        first.branch_excitation.connection_indices, ref.connection_indices
    )


def test_fresh_rewiring_uses_topology_rng_without_changing_training_rng():
    bank = IndexedRewireLinear(
        6, 3, 2, seed=17, rewire_frequency=1, rewire_init_policy="fresh"
    )
    reference = deepcopy(bank)
    state = torch.get_rng_state().clone()
    bank.apply_rewiring()
    assert torch.equal(torch.get_rng_state(), state)
    torch.randn(37)
    reference.apply_rewiring()
    assert torch.equal(bank.connection_indices, reference.connection_indices)
    assert torch.equal(bank.pre_w, reference.pre_w)


def test_muon_added_matrix_group_matches_constructor_group():
    torch.manual_seed(52)
    vector = nn.Parameter(torch.randn(3, dtype=torch.float64))
    matrix = nn.Parameter(torch.randn(3, 4, dtype=torch.float64))
    ref_vector, ref_matrix = nn.Parameter(vector.detach().clone()), nn.Parameter(
        matrix.detach().clone()
    )
    dynamic = MuonH([vector], lr=0.01)
    dynamic.add_param_group({"params": [matrix], "lr": 0.03, "name": "later"})
    reference = MuonH(
        [
            {"params": [ref_vector], "lr": 0.01},
            {"params": [ref_matrix], "lr": 0.03, "name": "later"},
        ]
    )
    for _ in range(3):
        for p in (vector, matrix, ref_vector, ref_matrix):
            p.grad = torch.ones_like(p)
        dynamic.step()
        reference.step()
    torch.testing.assert_close(matrix, ref_matrix, rtol=0, atol=0)


def test_wrapper_accepts_zero_grad_set_to_none():
    model = nn.Linear(3, 2)
    optimizer = CustomWeightDecayOptimizer(
        model, torch.optim.SGD(model.parameters(), lr=0.1), weight_decay=0
    )
    model(torch.ones(2, 3)).sum().backward()
    optimizer.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in model.parameters())
