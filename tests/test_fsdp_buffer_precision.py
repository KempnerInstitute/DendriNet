"""FSDP mixed precision: buffer dtypes, their record, and decays it freezes."""

from __future__ import annotations

import math

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel

from dendritic_modeling.training.fsdp_utils import (
    frozen_decay_buffers,
    get_fsdp_config,
    mixed_precision_record,
)


class _Leaky(nn.Module):
    def __init__(self, taus):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.register_buffer(
            "decays", torch.exp(-1.0 / torch.tensor(taus, dtype=torch.float32))
        )
        self.register_buffer("record", torch.tensor(0.123456789, dtype=torch.float64))

    def forward(self, x):
        return self.linear(x) * self.decays.to(x.dtype)


def test_buffers_follow_the_parameter_dtype_unless_kept(monkeypatch):
    # The configuration does not need a GPU; only its device_id field reads one.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    default = mixed_precision_record(get_fsdp_config())
    assert default == {
        "param_dtype": "torch.bfloat16",
        "reduce_dtype": "torch.bfloat16",
        "buffer_dtype": "torch.bfloat16",
    }
    kept = get_fsdp_config(fsdp_config_dict={"buffer_dtype": "keep"})
    assert kept["mixed_precision"].buffer_dtype is None
    assert mixed_precision_record(kept)["buffer_dtype"] == "keep"
    assert mixed_precision_record(get_fsdp_config(mixed_precision=False)) is None


def test_decays_that_bf16_rounds_to_one_are_reported():
    model = _Leaky([10.0, 625.0, 1000.0])
    assert frozen_decay_buffers(model, torch.bfloat16) == [
        "decays: 2 of 3 decay factors"
    ]
    assert frozen_decay_buffers(model, torch.float32) == []
    assert math.isclose(float(torch.tensor(math.exp(-1 / 625)).bfloat16()), 1.0)


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
@pytest.mark.filterwarnings("ignore:FSDP is switching to use `NO_SHARD`")
@pytest.mark.parametrize(
    "buffer_dtype, expected",
    [(None, torch.bfloat16), ("keep", torch.float64)],
)
def test_fsdp_keeps_buffer_precision_only_when_asked(
    buffer_dtype, expected, monkeypatch
):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    settings = {"buffer_dtype": buffer_dtype, "use_orig_params": True}
    policy = get_fsdp_config(fsdp_config_dict=settings)["mixed_precision"]
    wrapped = FullyShardedDataParallel(
        _Leaky([10.0]), device_id=torch.device("cpu"), mixed_precision=policy
    )
    wrapped(torch.rand(2, 4))
    assert wrapped.module.record.dtype == expected
