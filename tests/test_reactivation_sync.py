"""Calibrated reactivation state is the same on every distributed rank."""

from __future__ import annotations

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from dendritic_modeling.networks.activations.parametric import ParametricTanhOnlyM
from dendritic_modeling.training.reactivation_sync import (
    broadcast_reactivation_state,
    refresh_fixed_b_from_buffers,
)

pytestmark = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="CPU gloo distributed backend is unavailable",
)


def _model(rank: int) -> nn.Module:
    torch.manual_seed(rank)  # ranks start from different gate parameters
    model = nn.Sequential(nn.Linear(4, 3), ParametricTanhOnlyM(3))
    model[1].fixed_b = 0.123 if rank == 0 else 0.5  # calibrated on rank 0 only
    return model


def _worker(rank: int, rendezvous: str) -> None:
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        # DDP-style: calibration ran on every rank after wrapping.
        model = _model(rank)
        broadcast_reactivation_state(model)
        gathered = [None, None]
        dist.all_gather_object(
            gathered, (model[1].fixed_b, model[1].log_m.detach().tolist())
        )
        assert gathered[0] == gathered[1]
        assert model[1].fixed_b == pytest.approx(0.123)

        # FSDP: sync_module_states broadcasts the buffer, not the float. On CPU
        # (gloo) FSDP cannot sync states itself, so the broadcast is emulated.
        from torch.distributed.fsdp import FullyShardedDataParallel

        wrapped = FullyShardedDataParallel(
            _model(rank), device_id=torch.device("cpu"), use_orig_params=True
        )
        gate = next(m for m in wrapped.modules() if isinstance(m, ParametricTanhOnlyM))
        dist.broadcast(gate.fixed_b_buffer, src=0)
        assert gate.fixed_b == (0.123 if rank == 0 else 0.5)
        refresh_fixed_b_from_buffers(wrapped)
        assert gate.fixed_b == pytest.approx(0.123)
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.multiprocess
def test_calibrated_gates_match_across_ranks(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "init"),), nprocs=2)


def test_broadcast_is_a_no_op_in_one_process():
    model = _model(1)
    broadcast_reactivation_state(model)
    assert model[1].fixed_b == 0.5
