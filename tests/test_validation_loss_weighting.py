"""Regression tests for sample-weighted checkpoint validation losses."""

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dendritic_modeling.training.loss.functions import CrossEntropyLoss
from dendritic_modeling.training.strategies.recurrent_trainer import RecurrentTrainer
from dendritic_modeling.training.strategies.standard import Trainer


class _FixedLogitModel(torch.nn.Module):
    """Expose dataset values as logits while retaining an optimizer parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.0 * self.offset


class _FixedSequenceLogitModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        x: torch.Tensor,
        seq_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del seq_lengths
        return x + 0.0 * self.offset


def _uneven_classification_data() -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.tensor(
        [
            [8.0, 0.0],
            [0.0, 8.0],
            [7.0, 0.0],
            [0.0, 7.0],
            [8.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 1, 0, 1, 1])
    return logits, labels


def test_standard_checkpoint_validation_is_exact_sample_mean() -> None:
    """The smaller final batch must not receive a full batch's weight."""
    logits, labels = _uneven_classification_data()
    dataset = TensorDataset(logits, labels)
    model = _FixedLogitModel()
    trainer = Trainer(
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        loss_function="cross_entropy",
        epochs=1,
        batch_size=2,
        suppress_prints=True,
        num_workers=0,
    )

    results = trainer.train(model, dataset, dataset)
    device = next(model.parameters()).device
    direct_loss = F.cross_entropy(model(logits.to(device)), labels.to(device)).item()

    assert results["valid_losses"] == pytest.approx([direct_loss])
    assert results["best_loss"] == pytest.approx(direct_loss)


def test_standard_epoch_valid_keeps_legacy_tuple_semantics() -> None:
    """The tuple remains batch-loss sum plus batch count for existing callers."""
    logits, labels = _uneven_classification_data()
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2, shuffle=False)
    model = _FixedLogitModel()
    trainer = Trainer(
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        loss_function="cross_entropy",
        batch_size=2,
        suppress_prints=True,
    )
    trainer.device = "cpu"

    batch_losses = [
        F.cross_entropy(logits[start : start + 2], labels[start : start + 2]).item()
        for start in range(0, len(labels), 2)
    ]
    loss_sum, batch_count = trainer._epoch_valid(model, loader)

    assert loss_sum == pytest.approx(sum(batch_losses))
    assert batch_count == len(batch_losses)
    assert trainer._non_distributed_validation_loss(model, loader) == pytest.approx(
        F.cross_entropy(logits, labels).item()
    )


def test_non_mean_standard_loss_retains_batch_aggregation() -> None:
    """Sample weighting must not reinterpret an explicitly non-mean loss."""
    logits, labels = _uneven_classification_data()
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2, shuffle=False)
    model = _FixedLogitModel()
    trainer = Trainer(
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        loss_function="cross_entropy",
        batch_size=2,
        suppress_prints=True,
    )
    trainer.loss_function = CrossEntropyLoss(reduction="sum")
    trainer.device = "cpu"

    batch_sums = [
        F.cross_entropy(
            logits[start : start + 2],
            labels[start : start + 2],
            reduction="sum",
        ).item()
        for start in range(0, len(labels), 2)
    ]

    assert trainer._non_distributed_validation_loss(model, loader) == pytest.approx(
        sum(batch_sums) / len(batch_sums)
    )


def test_recurrent_validation_matches_full_sequence_sample_mean() -> None:
    """Sequence CE uses sample weighting and preserves its legacy tuple API."""
    flat_logits, flat_labels = _uneven_classification_data()
    logits = torch.stack((flat_logits, flat_logits.flip(dims=(1,))), dim=1)
    labels = torch.stack((flat_labels, 1 - flat_labels), dim=1)
    seq_lengths = torch.full((len(labels),), 2, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(logits, labels, seq_lengths),
        batch_size=2,
        shuffle=False,
    )
    model = _FixedSequenceLogitModel()
    trainer = RecurrentTrainer(
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        task="classification",
        batch_size=2,
        suppress_prints=True,
    )
    trainer.device = "cpu"

    batch_losses = [
        F.cross_entropy(
            logits[start : start + 2].reshape(-1, 2),
            labels[start : start + 2].reshape(-1),
        ).item()
        for start in range(0, len(labels), 2)
    ]
    loss_sum, batch_count = trainer._epoch_valid(model, loader)
    direct_loss = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1)).item()

    assert loss_sum == pytest.approx(sum(batch_losses))
    assert batch_count == len(batch_losses)
    assert trainer._non_distributed_validation_loss(model, loader) == pytest.approx(
        direct_loss
    )
