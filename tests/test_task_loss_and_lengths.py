"""Trainers check the loss against the model task and keep sequence lengths."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dendritic_modeling.training.loss.functions import (
    get_loss_function,
    validate_task_loss,
)
from dendritic_modeling.training.strategies.recurrent_trainer import RecurrentTrainer
from dendritic_modeling.training.strategies.standard import Trainer


def _optimizer():
    return torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)


@pytest.mark.parametrize(
    "task, loss",
    [("regression", "ce"), ("regression", "cat_nll"), ("regression", "bce")],
)
def test_a_loss_for_another_task_raises(task, loss):
    with pytest.raises(ValueError, match="defined for"):
        Trainer(optimizer=_optimizer(), loss_function=loss, task=task)


@pytest.mark.parametrize(
    "task, loss",
    [
        ("classification", "ce"),
        ("classification", "cat_nll"),
        ("regression", "mse"),
        ("regression", "log_mse"),
        ("classification", "mse"),
        ("regression", "temporal_event_ce"),
        (None, "mse"),
    ],
)
def test_matching_or_self_checking_losses_build(task, loss):
    Trainer(optimizer=_optimizer(), loss_function=loss, task=task)


def test_the_recurrent_trainer_substitutes_its_task_loss():
    # ce is the typed default; the recurrent loop trains regression with MSE.
    RecurrentTrainer(optimizer=_optimizer(), loss_function="ce", task="regression")


def test_validate_task_loss_names_the_owner():
    with pytest.raises(ValueError, match="FSDPTrainer"):
        validate_task_loss("regression", get_loss_function("ce"), owner="FSDPTrainer")


def test_a_recurrent_model_does_not_train_without_its_sequence_lengths():
    trainer = Trainer(optimizer=_optimizer(), loss_function="ce", device="cpu")
    batch = (torch.rand(2, 3, 4), torch.tensor([0, 1]), torch.tensor([3, 2]))
    loader = DataLoader(TensorDataset(*batch), batch_size=2)
    trainer._recurrent_model = False
    x, _y = trainer._move_training_batch_to_device(batch, loader)
    assert x.shape == (2, 3, 4)
    trainer._recurrent_model = True
    with pytest.raises(ValueError, match="strategy: recurrent"):
        trainer._move_training_batch_to_device(batch, loader)
