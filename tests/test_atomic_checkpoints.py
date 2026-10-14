"""Checkpoint writes are atomic, durable and shared-readable."""

from __future__ import annotations

import os
import stat

import numpy as np
import pytest
import torch
from torch import nn

from dendritic_modeling.config.multi_stage_training import TrainingStageConfig
from dendritic_modeling.networks.checkpoints import (
    atomic_torch_save,
    load_model_state_checkpoint,
)
from dendritic_modeling.training.strategies.multi_stage import (
    _load_stage_checkpoint,
    _save_stage_checkpoint,
)


@pytest.fixture
def umask_022():
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


def test_saved_files_follow_the_umask(tmp_path, umask_022):
    path = tmp_path / "model.pt"
    atomic_torch_save({"w": torch.ones(2)}, path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o644


class _Unpicklable:
    def __reduce__(self):
        raise RuntimeError("interrupted")


def test_a_failed_save_keeps_the_previous_file(tmp_path):
    path = tmp_path / "best_model.pt"
    atomic_torch_save({"w": torch.ones(2)}, path)
    with pytest.raises(RuntimeError, match="interrupted"):
        atomic_torch_save({"w": _Unpicklable()}, path)
    assert torch.equal(torch.load(path, weights_only=True)["w"], torch.ones(2))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["best_model.pt"]


def test_stage_checkpoints_load_with_the_weights_only_loader(tmp_path):
    model = nn.Linear(3, 2)
    path = tmp_path / "stage.pt"
    _save_stage_checkpoint(
        model,
        TrainingStageConfig(stage_name="warmup", epochs=2),
        {"loss": [np.float32(0.5), 0.25], "accuracy": np.float64(0.75)},
        stage_trainer=object(),
        stage_optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        checkpoint_path=str(path),
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["stage_config"]["stage_name"] == "warmup"
    assert payload["stage_results"]["loss"] == [0.5, 0.25]
    target = nn.Linear(3, 2)
    _load_stage_checkpoint(target, str(path))
    assert torch.equal(target.weight, model.weight)
    assert set(load_model_state_checkpoint(path)) == {"weight", "bias"}
