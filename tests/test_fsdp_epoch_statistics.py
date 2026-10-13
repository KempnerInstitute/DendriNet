"""FSDP epoch losses and accuracies are float64 sample-weighted means.

The sums stay on the device during the epoch and are reduced across ranks in
float64 (float32 had rounded the loss sum and, beyond 2**24 samples, the
counts).
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dendritic_modeling.training.strategies.fsdp_standard import (
    FSDPTrainer,
    _correct_predictions,
    _mean_over_ranks,
)


@pytest.fixture
def process_group(tmp_path):
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'group'}", rank=0, world_size=1
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.mark.usefixtures("process_group")
def test_rank_means_keep_float64_sums_and_exact_counts():
    loss_sum = torch.tensor(1 + 2**-40, dtype=torch.float64)  # not a float32
    correct = torch.tensor(2**24 + 1)  # not a float32 either

    loss, accuracy = _mean_over_ranks(loss_sum, correct, 2**25)

    assert loss == (1 + 2**-40) / 2**25
    assert accuracy == (2**24 + 1) / 2**25


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 4)

    def forward(self, x, seq_lengths=None, precomputed_core=False):
        return self.linear(x)


@pytest.mark.requires_gpu
@pytest.mark.usefixtures("process_group")
def test_fsdp_epochs_return_sample_weighted_float64_means(tmp_path):
    trainer = FSDPTrainer(
        rank=0,
        world_size=1,
        fsdp_config={"sharding_strategy": "FULL_SHARD"},
        checkpoint_dir=str(tmp_path),
        optimizer_config={"param_groups": {"lr": 0.0}},
        epochs=1,
        loss_function="cross_entropy",
    )
    torch.manual_seed(0)
    model = _Model().to(trainer.device)
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    generator = torch.Generator().manual_seed(1)
    data = TensorDataset(
        torch.randn(10, 3, generator=generator),
        torch.randint(0, 4, (10,), generator=generator),
    )
    loader = DataLoader(data, batch_size=4)  # batches of 4, 4 and 2

    expected_loss, expected_correct = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            outputs = model(x.to(trainer.device))
            y = y.to(trainer.device)
            expected_loss += trainer.loss_fn(outputs, y).item() * y.numel()
            expected_correct += int((outputs.argmax(dim=1) == y).sum())

    for epoch in (trainer._epoch_train_fsdp, trainer._epoch_valid_fsdp):
        loss, accuracy = epoch(model, loader)
        assert loss == expected_loss / 10
        assert accuracy == expected_correct / 10


def test_accuracy_counts_class_predictions_only():
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0], [3.0, 1.0]])
    assert _correct_predictions(logits, torch.tensor([0, 1, 1])) == 2
    regression = _correct_predictions(torch.rand(3, 1), torch.rand(3, 1))
    assert torch.isnan(regression)


def _trainer(tmp_path, **options):
    return FSDPTrainer(
        rank=0,
        world_size=1,
        fsdp_config={"sharding_strategy": "FULL_SHARD"},
        checkpoint_dir=str(tmp_path),
        optimizer_config={"param_groups": {"lr": 0.0}},
        epochs=1,
        **options,
    )


@pytest.mark.requires_gpu
@pytest.mark.usefixtures("process_group")
def test_fsdp_trains_the_registry_loss(tmp_path):
    trainer = _trainer(tmp_path, loss_function="mse", task="regression")
    torch.manual_seed(0)
    model = _Model().to(trainer.device)
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    generator = torch.Generator().manual_seed(2)
    data = TensorDataset(
        torch.randn(6, 3, generator=generator), torch.randn(6, 4, generator=generator)
    )
    loader = DataLoader(data, batch_size=3)
    with torch.no_grad():
        expected = sum(
            nn.functional.mse_loss(model(x.to(trainer.device)), y.to(trainer.device))
            .double()
            .item()
            * len(y)
            for x, y in loader
        )

    loss, accuracy = trainer._epoch_valid_fsdp(model, loader)

    assert loss == pytest.approx(expected / 6, rel=1e-12)
    assert accuracy != accuracy  # NaN: regression has no accuracy


@pytest.mark.requires_gpu
@pytest.mark.usefixtures("process_group")
def test_fsdp_refuses_sequence_losses_and_sequence_class_outputs(tmp_path):
    with pytest.raises(ValueError, match="sequence loss of the recurrent trainer"):
        _trainer(tmp_path, loss_function="temporal_event_ce")
    trainer = _trainer(tmp_path, loss_function="ce", task="classification")
    with pytest.raises(ValueError, match=r"\[batch, classes\] logits"):
        trainer._batch_loss(torch.rand(2, 5, 3), torch.zeros(2, 5, dtype=torch.long))
