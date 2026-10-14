"""FSDP checkpoints carry every rank's random generators.

A resumed run then draws the same dropout masks and stochastic synapse
selections as the uninterrupted one (two-rank continuation is covered in
``test_distributed_training_contracts.py``).
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch import nn

from dendritic_modeling.training.fsdp_utils import (
    _rank_random_state,
    _restore_rank_random_state,
    get_fsdp_config,
    load_fsdp_checkpoint,
    save_fsdp_checkpoint,
    wrap_model_with_fsdp,
)


def _draws():
    values = [random.random(), float(np.random.rand()), float(torch.rand(()))]
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        values.append(float(torch.rand((), device="cuda")))
    return values


def test_rank_random_state_round_trips_through_a_weights_only_load(tmp_path):
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    state = _rank_random_state()
    expected = _draws()

    torch.save({"random_states": [state]}, tmp_path / "state.pt")
    loaded = torch.load(tmp_path / "state.pt", weights_only=True)
    _draws()  # move every generator on
    _restore_rank_random_state(loaded["random_states"][0])

    assert _draws() == expected


def test_a_checkpoint_without_generators_does_not_resume_silently(tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"epoch": 1, "model_state_dict": {}}, path)
    with pytest.raises(ValueError, match="no random-generator state"):
        load_fsdp_checkpoint(None, None, str(path))


def test_generators_of_another_rank_count_are_refused(tmp_path):
    path = tmp_path / "two_ranks.pt"
    torch.save({"random_states": [_rank_random_state()] * 2}, path)
    with pytest.raises(ValueError, match="2 ranks, but this run has 1"):
        load_fsdp_checkpoint(None, None, str(path))


@pytest.fixture
def process_group(tmp_path):
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'group'}", rank=0, world_size=1
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.mark.requires_gpu
@pytest.mark.usefixtures("process_group")
def test_resumed_fsdp_training_redraws_the_same_dropout_masks(tmp_path):
    def build(seed):
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(6, 8), nn.Dropout(0.5), nn.Linear(8, 2))
        model = wrap_model_with_fsdp(
            model.cuda(),
            get_fsdp_config(fsdp_config_dict={"mixed_precision": False}),
        )
        return model, torch.optim.Adam(model.parameters(), lr=0.01)

    x = torch.rand(5, 6, generator=torch.Generator().manual_seed(4)).cuda()

    def steps(model, optimizer):
        for _ in range(3):
            optimizer.zero_grad()
            model(x).square().mean().backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            output = model(x)
        model.train()
        return output

    checkpoint = str(tmp_path / "state.pt")
    model, optimizer = build(0)
    steps(model, optimizer)
    save_fsdp_checkpoint(model, optimizer, 3, checkpoint)
    expected = steps(model, optimizer)

    model, optimizer = build(1)
    assert load_fsdp_checkpoint(model, optimizer, checkpoint) == 3
    assert torch.equal(steps(model, optimizer), expected)
