"""Batch statistics of sparse synapses cover the batches of all ranks.

Under data-parallel training each rank sees its own batch. Variance Top-K
moments and release impact/source activity are reduced over the ranks, so
every rank keeps the statistics of the global batch (and selects alike).
"""

from __future__ import annotations

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_release import (
    IndexedReleaseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.variance import (
    VarianceTopKLinear,
)
from dendritic_modeling.networks.utils.sparse_maintenance import (
    mark_rank_reduced_statistics,
    reduces_statistics_across_ranks,
)

# Unequal per-rank batches: the global statistics weight samples, not ranks.
_BATCHES = (3, 5)


def _inputs():
    generator = torch.Generator().manual_seed(4)
    return [torch.rand(size, 6, generator=generator) for size in _BATCHES]


def _variance():
    torch.manual_seed(1)
    return VarianceTopKLinear(6, 3, 2, weight_transform="exp")


def _release(metric):
    # Certain release keeps the drive deterministic across ranks.
    torch.manual_seed(2)
    return IndexedReleaseLinear(
        6,
        3,
        4,
        seed=8,
        weight_transform="exp",
        release_p_min=1.0,
        release_p_max=1.0,
        release_impact_metric=metric,
        release_track_source_activity=True,
    )


def _builders():
    return {
        "variance": _variance,
        "release_drive": lambda: _release("drive"),
        "release_active_drive": lambda: _release("active_drive"),
    }


def _statistics(module):
    return {name: value.clone() for name, value in module.named_buffers()}


def _worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        inputs = _inputs()
        for name, build in _builders().items():
            module = build().train()
            assert mark_rank_reduced_statistics(nn.Sequential(module)) == ["0"]
            assert reduces_statistics_across_ranks(module)
            module(inputs[rank])
            local = _statistics(module)

            # Both ranks hold the same statistics ...
            for key, value in local.items():
                copies = [torch.empty_like(value) for _ in range(2)]
                dist.all_gather(copies, value)
                assert torch.equal(copies[0], copies[1]), (name, key)

            # ... equal to one process seeing the combined batch.
            reference = build().train()
            reference(torch.cat(inputs))
            for key, value in _statistics(reference).items():
                torch.testing.assert_close(
                    local[key], value, rtol=1e-6, atol=1e-7, msg=f"{name}: {key}"
                )
            if name == "variance":
                assert torch.equal(module.weight_mask(), reference.weight_mask())
    finally:
        dist.destroy_process_group()


@pytest.mark.multiprocess
def test_statistics_cover_the_batches_of_all_ranks(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2)


def test_a_single_process_keeps_its_own_statistics():
    # Marked modules reduce only when more than one rank trains.
    module = _variance()
    mark_rank_reduced_statistics(module)
    assert not reduces_statistics_across_ranks(module)
    reference = _variance()
    x = _inputs()[0]
    module.train()(x)
    reference.train()(x)
    for key, value in _statistics(reference).items():
        assert torch.equal(_statistics(module)[key], value)


def test_only_batch_statistics_synapses_are_marked():
    model = nn.Sequential(_variance(), nn.Linear(3, 3), _release("drive"))
    assert mark_rank_reduced_statistics(model) == ["0", "2"]
    assert not hasattr(model[1], "reduce_statistics_across_ranks")
