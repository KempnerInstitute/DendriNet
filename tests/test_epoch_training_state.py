"""Epoch continuation must preserve optimization and every random stream."""

import random
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.training.state import (
    capture_epoch_state,
    capture_random_state,
    restore_epoch_state,
    restore_random_state,
)
from dendritic_modeling.training.strategies.standard import Trainer


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _equal(a, b)
    else:
        assert left == right


def _setup(device="cpu", **options):
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.Tanh(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(5, 2),
    ).to(device)
    trainer = Trainer(
        torch.optim.Adam(model.parameters(), lr=0.01),
        loss_function="cross_entropy",
        epochs=4,
        batch_size=7,
        num_workers=0,
        seed=73,
        suppress_prints=True,
        **options,
    )
    trainer.device = device
    data = TensorDataset(
        torch.arange(69, dtype=torch.float32).reshape(23, 3) / 69, torch.arange(23) % 2
    )
    train, valid = trainer._dataloaders(data, data)
    trainer._initialize_attributes(model, 4)
    trainer._build_scheduler(4)
    return trainer, model, {"train_loader": train, "valid_loader": valid}


def _advance(trainer, model, loaders, histories, count):
    return trainer._run_training_loop(
        n_epochs=count,
        model=model,
        **loaders,
        **dict(
            zip(
                (
                    "train_losses",
                    "valid_losses",
                    "train_losses_base",
                    "train_losses_reg",
                ),
                histories,
            )
        ),
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_resume_matches_uninterrupted_training_and_legacy_layout(device):
    torch.manual_seed(51)
    trainer, model, loaders = _setup(device)
    histories = _advance(trainer, model, loaders, ([], [], [], []), 2)
    snapshot = capture_epoch_state(
        trainer, model, loaders=loaders, loss_lists=histories, source_commit="origin"
    )
    # Independent reference serializer: preserve the artifact schema.
    legacy = dict(  # noqa: C408 - mirror the historical serializer
        epoch=trainer.epoch_counter,
        current_model=deepcopy(model.state_dict()),
        best_model=deepcopy(trainer.best_state_dict),
        optimizer=deepcopy(trainer.optimizer.state_dict()),
        optimizer_wrappers=[],
        rng=capture_random_state(),
        loader_generators={k: v.generator.get_state() for k, v in loaders.items()},
        loss_lists=deepcopy(histories),
        best_epoch=trainer.best_epoch,
        best_loss=trainer.best_loss,
        patience_counter=trainer.patience_counter,
        source_commit="origin",
    )
    _equal(snapshot, legacy)
    histories = _advance(trainer, model, loaders, histories, 2)
    expected = capture_epoch_state(
        trainer, model, loaders=loaders, loss_lists=histories, source_commit="origin"
    )
    random.random()
    np.random.random()
    torch.rand(7)
    other, rebuilt, new_loaders = _setup(device)
    histories = restore_epoch_state(
        other, rebuilt, snapshot, loaders=new_loaders, expected_source_commit="origin"
    )
    histories = _advance(other, rebuilt, new_loaders, histories, 2)
    actual = capture_epoch_state(
        other,
        rebuilt,
        loaders=new_loaders,
        loss_lists=histories,
        source_commit="origin",
    )
    _equal(actual, expected)
    _equal(
        snapshot, legacy
    )  # A retained snapshot must not alias live optimizer/model state.


def test_random_state_roundtrip_without_consuming_draws():
    before = capture_random_state()
    expected = (random.random(), np.random.random(), torch.rand(4))
    restore_random_state(before)
    _equal(expected, (random.random(), np.random.random(), torch.rand(4)))


def test_wrong_source_rejected_before_model_mutation():
    trainer, model, loaders = _setup()
    state = capture_epoch_state(
        trainer,
        model,
        loaders=loaders,
        loss_lists=([], [], [], []),
        source_commit="origin",
    )
    before = deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="source commit"):
        restore_epoch_state(
            trainer, model, state, loaders=loaders, expected_source_commit="other"
        )
    _equal(before, model.state_dict())


@pytest.mark.parametrize("bad", ["amp", "workers", "generator", "pruning", "histories"])
def test_incomplete_state_scope_is_rejected(bad):
    trainer, model, loaders = _setup()
    histories = ([], [], [], [])
    if bad == "amp":
        trainer.enable_amp = True
    if bad == "workers":
        loaders["train_loader"].num_workers = 1
    if bad == "generator":
        loaders["valid_loader"].generator = None
    if bad == "pruning":
        trainer.pruning_config.enabled = True
    if bad == "histories":
        histories = ([],)
    with pytest.raises(ValueError):
        capture_epoch_state(
            trainer,
            model,
            loaders=loaders,
            loss_lists=histories,
            source_commit="origin",
        )


def test_resume_continues_the_trainer_lr_schedule():
    torch.manual_seed(51)
    options = {"lr_schedule": "cosine", "lr_warmup_epochs": 1}
    trainer, model, loaders = _setup(**options)
    histories = _advance(trainer, model, loaders, ([], [], [], []), 2)
    snapshot = capture_epoch_state(
        trainer, model, loaders=loaders, loss_lists=histories, source_commit="origin"
    )
    assert snapshot["scheduler"]["last_epoch"] == 2
    histories = _advance(trainer, model, loaders, histories, 2)
    expected = capture_epoch_state(
        trainer, model, loaders=loaders, loss_lists=histories, source_commit="origin"
    )

    other, rebuilt, new_loaders = _setup(**options)
    histories = restore_epoch_state(
        other, rebuilt, snapshot, loaders=new_loaders, expected_source_commit="origin"
    )
    histories = _advance(other, rebuilt, new_loaders, histories, 2)
    actual = capture_epoch_state(
        other,
        rebuilt,
        loaders=new_loaders,
        loss_lists=histories,
        source_commit="origin",
    )
    _equal(actual, expected)

    unscheduled, model3, loaders3 = _setup()
    with pytest.raises(ValueError, match="LR schedule"):
        restore_epoch_state(
            unscheduled,
            model3,
            snapshot,
            loaders=loaders3,
            expected_source_commit="origin",
        )
