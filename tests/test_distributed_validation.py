import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dendritic_modeling.training.distributed_samplers import (
    PaddedDistributedEvalSampler,
)
from dendritic_modeling.training.strategies.recurrent_trainer import RecurrentTrainer
from dendritic_modeling.training.strategies.standard import Trainer


class FixedLogitModel(torch.nn.Module):
    def forward(self, x):
        return x


class FixedSequenceLogitModel(torch.nn.Module):
    def forward(self, x, seq_lengths=None):
        return x


def test_padded_distributed_eval_sampler_pads_with_sentinels():
    dataset = TensorDataset(torch.randn(5, 2), torch.randint(0, 2, (5,)))
    sampler = PaddedDistributedEvalSampler(dataset, num_replicas=2, rank=1)

    assert sampler.real_indices == [1, 3]
    assert sampler.num_padding == 1
    assert list(iter(sampler)) == [1, 3, -1]


def test_epoch_valid_distributed_excludes_padding_examples(monkeypatch):
    x = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [0.5, 0.5],
        ]
    )
    y = torch.tensor([0, 1, 0, 1, 0])
    dataset = TensorDataset(x, y)
    sampler = PaddedDistributedEvalSampler(dataset, num_replicas=2, rank=1)
    loader = DataLoader(dataset, batch_size=2, sampler=sampler, shuffle=False)

    trainer = Trainer(
        optimizer=torch.optim.Adam([torch.tensor(1.0, requires_grad=True)]),
        loss_function="cross_entropy",
        batch_size=2,
        suppress_prints=True,
    )
    trainer.device = "cpu"

    model = FixedLogitModel()
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op=None: None)
    valid_loss = trainer._epoch_valid_distributed(model, loader)

    loss_fn = torch.nn.CrossEntropyLoss()
    manual_loss = loss_fn(model(x[[1, 3]]), y[[1, 3]]).item()

    assert valid_loss == pytest.approx(manual_loss)


def test_recurrent_epoch_valid_distributed_uses_sequence_loss(monkeypatch):
    x = torch.randn(5, 3, 2)
    y = torch.randint(0, 2, (5, 3))
    seq_lengths = torch.full((5,), 3, dtype=torch.long)
    dataset = TensorDataset(x, y, seq_lengths)
    sampler = PaddedDistributedEvalSampler(dataset, num_replicas=2, rank=1)
    loader = DataLoader(dataset, batch_size=2, sampler=sampler, shuffle=False)

    trainer = RecurrentTrainer(
        optimizer=torch.optim.Adam([torch.tensor(1.0, requires_grad=True)]),
        task="classification",
        batch_size=2,
        suppress_prints=True,
    )
    trainer.device = "cpu"

    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op=None: None)
    valid_loss = trainer._epoch_valid_distributed(FixedSequenceLogitModel(), loader)

    loss_fn = torch.nn.CrossEntropyLoss()
    manual_loss = loss_fn(x[[1, 3]].reshape(-1, 2), y[[1, 3]].reshape(-1)).item()

    assert valid_loss == pytest.approx(manual_loss)


def test_dataloaders_use_padded_eval_sampler_under_distributed(monkeypatch):
    train_data = TensorDataset(torch.randn(16, 4), torch.randint(0, 2, (16,)))
    valid_data = TensorDataset(torch.randn(10, 4), torch.randint(0, 2, (10,)))
    trainer = Trainer(
        optimizer=torch.optim.Adam([torch.tensor(1.0, requires_grad=True)]),
        loss_function="cross_entropy",
        batch_size=4,
        suppress_prints=True,
        num_workers=0,
    )

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

    train_loader, valid_loader = trainer._dataloaders(train_data, valid_data)

    assert isinstance(train_loader.sampler, torch.utils.data.DistributedSampler)
    assert isinstance(valid_loader.sampler, PaddedDistributedEvalSampler)
