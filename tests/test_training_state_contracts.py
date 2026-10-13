"""Sparse ownership, activation trainability and stream loss contracts."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dendritic_modeling.config.regularization import RegularizationConfig
from dendritic_modeling.networks.activations.endpoint_tanh import EndpointNormalizedTanh
from dendritic_modeling.networks.activations.parametric import ParametricTanhOnlyM
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.dense_to_sparse import (
    DenseToSparseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_rewire import (
    IndexedRewireLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.networks.utils.sparse_maintenance import (
    apply_maintenance,
    forward_maintenance,
)
from dendritic_modeling.training.optimizers.custom import (
    apply_sparse_topology_updates_after_step,
)
from dendritic_modeling.training.strategies.presentation_stream import (
    PresentationStreamTrainer,
)
from dendritic_modeling.training.strategies.specialized import (
    _freeze_parametric_activations,
    _restore_parametric_activations,
)


def test_plain_optimizer_does_not_rewire_unowned_layer():
    trained = IndexedRewireLinear(
        8, 3, 2, rewire_frequency=1, rewire_quantile=0.5, seed=7
    )
    unowned = IndexedRewireLinear(
        8, 3, 2, rewire_frequency=1, rewire_quantile=0.5, seed=8
    )
    model = nn.ModuleList([trained, unowned])
    before = {k: v.clone() for k, v in unowned.state_dict().items()}
    optimizer = torch.optim.SGD(trained.parameters(), lr=0.01)
    apply_sparse_topology_updates_after_step(model, optimizer)
    assert int(trained.rewire_step) == 1
    for key, value in before.items():
        assert torch.equal(unowned.state_dict()[key], value), key


def test_plain_optimizer_does_not_advance_unowned_schedule():
    a, b = [DenseToSparseLinear(8, 3, 2, advance_on_forward=False) for _ in range(2)]
    optimizer = torch.optim.SGD(a.parameters(), lr=0.01)
    apply_sparse_topology_updates_after_step(nn.ModuleList([a, b]), optimizer)
    assert int(a.sparsity_step) == 1
    assert int(b.sparsity_step) == 0


def test_shared_leaf_is_maintained_once_across_forwarding_containers():
    class Container(nn.Module):
        def __init__(self, bank):
            super().__init__()
            self.bank = bank

        def decay_weights(self, *args):
            forward_maintenance(self, "decay_weights", *args)

    bank = TopKLinear(8, 3, 2)
    bank.initialize()
    expected = (
        bank.pre_w.detach().clone() + torch.tensor(0.9).log() * bank.weight_mask()
    )
    apply_maintenance(
        nn.ModuleList([Container(bank), Container(bank)]), "decay_weights", 0.1, False
    )
    torch.testing.assert_close(bank.pre_w, expected)


@pytest.mark.parametrize("unroll", [False, True])
def test_frozen_forward_schedule_does_not_advance(unroll):
    bank = DenseToSparseLinear(8, 3, 2, advance_on_forward=True)
    bank.initialize()
    bank.pre_w.requires_grad_(False)
    if unroll:
        bank.set_recurrent_weight_cache(True)
    bank(torch.rand(2, 8))
    if unroll:
        bank.set_recurrent_weight_cache(False)
    assert int(bank.sparsity_step) == 0


def test_warmup_freezes_endpoint_gate_and_restores_original_flags():
    gate = EndpointNormalizedTanh(3)
    gate.b.requires_grad_(False)
    states = _freeze_parametric_activations(gate)
    assert not any(p.requires_grad for p in gate.parameters())
    _restore_parametric_activations(SimpleNamespace(), states)
    assert gate.log_m.requires_grad and not gate.b.requires_grad


@pytest.mark.parametrize("dtype", [torch.float64, torch.float16])
def test_fixed_midpoint_checkpoint_preserves_python_value(dtype):
    gate = ParametricTanhOnlyM(3, fixed_b=0.123456789123).to(dtype=dtype)
    restored = ParametricTanhOnlyM(3).to(dtype=dtype)
    restored.load_state_dict(gate.state_dict())
    assert restored.fixed_b == gate.fixed_b


def test_stream_rejects_unsupported_periodic_calibration():
    model = nn.Linear(4, 3)
    with pytest.raises(ValueError, match="recalibrate_reactivation_every"):
        PresentationStreamTrainer(
            torch.optim.SGD(model.parameters(), lr=0.1),
            loss_function="ce",
            recalibrate_reactivation_every=1,
        )


def test_stream_reports_regularization_instead_of_zero():
    class Stream:
        def to(self, device):
            return self

        def sample(self, count, **kwargs):
            return torch.ones(count, 4), torch.zeros(count, dtype=torch.long)

    model = nn.Linear(4, 3)
    trainer = PresentationStreamTrainer(
        torch.optim.SGD(model.parameters(), lr=0.0),
        loss_function="ce",
        epochs=1,
        batch_size=4,
        regularization_config=RegularizationConfig(l2_weight=0.1),
        presentation_stream_config={"presentations_per_epoch": 4},
    )
    trainer.device = "cpu"
    valid = torch.utils.data.TensorDataset(
        torch.ones(4, 4), torch.zeros(4, dtype=torch.long)
    )
    result = trainer.train(model, SimpleNamespace(stream=Stream()), valid)
    assert result["train_losses_reg"][0] > 0
    assert result["train_losses"][0] == pytest.approx(
        result["train_losses_base"][0] + result["train_losses_reg"][0]
    )
