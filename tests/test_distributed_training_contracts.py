"""Distributed topology and sharded checkpoint continuation use real processes."""

import gc
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from distributed_test_utils import spawn_workers
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_rewire import (
    IndexedRewireLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)
from dendritic_modeling.training.fsdp_utils import (
    get_fsdp_config,
    load_fsdp_checkpoint,
    save_fsdp_checkpoint,
    wrap_model_with_fsdp,
)
from dendritic_modeling.training.optimizers.custom import CustomWeightDecayOptimizer


def _worker(rank, backend, protocol, rendezvous, checkpoint, kind):
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group(
        backend,
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    model = optimizer = None
    try:
        torch.manual_seed(51 + rank)
        x = torch.randn(4, 6, device=device)
        if protocol == "ddp":
            bank = IndexedRewireLinear(
                6,
                4,
                2,
                seed=None if kind == "inherit_pruned" else 7,
                rewire_frequency=1,
                rewire_init_policy=kind,
                rewire_quantile=0.5,
                projection_backend="eager",
            ).to(device)
            model = DistributedDataParallel(
                bank, device_ids=[rank] if backend == "nccl" else None
            )
            optimizer = CustomWeightDecayOptimizer(
                model, torch.optim.SGD(model.parameters(), lr=0.01), weight_decay=0
            )
            for _ in range(3):
                optimizer.zero_grad()
                model(x).square().mean().backward()
                optimizer.step()
                for value in (
                    bank.pre_w,
                    bank.connection_indices,
                    bank.rewire_seed,
                    bank.rewire_step,
                ):
                    copies = [torch.empty_like(value) for _ in range(2)]
                    dist.all_gather(copies, value.detach())
                    torch.testing.assert_close(copies[0], copies[1], rtol=0, atol=0)
        else:
            first = (
                IndexedSparseLinear(6, 6, 3, seed=7, projection_backend="eager")
                if kind == "indexed"
                else nn.Linear(6, 6)
            )
            # Dropout draws from each rank's generator, which the checkpoint
            # restores.
            middle = nn.Dropout(0.5) if kind == "dropout" else nn.Tanh()
            model = wrap_model_with_fsdp(
                nn.Sequential(first, middle, nn.Linear(6, 2)),
                get_fsdp_config(
                    fsdp_config_dict={
                        "mixed_precision": False,
                        "auto_wrap_policy": "none",
                        "use_orig_params": True,
                    }
                ),
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            def step():
                optimizer.zero_grad()
                model(x).square().mean().backward()
                optimizer.step()

            step()
            step()
            save_fsdp_checkpoint(model, optimizer, 2, checkpoint, rank=rank)
            dist.barrier()
            step()
            step()
            expected = model(x).detach().clone()
            assert load_fsdp_checkpoint(model, optimizer, checkpoint) == 2
            step()
            step()
            actual = model(x).detach()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        # The optimizer retains the DDP wrapper, whose reducer owns process-
        # group resources. Release these objects while the group is live,
        # rather than from frame cleanup after destroy_process_group().
        optimizer = None
        model = None
        gc.collect()
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.multiprocess
@pytest.mark.parametrize("backend", ["gloo", "nccl"])
@pytest.mark.parametrize("policy", ["inherit_pruned", "fresh"])
def test_rewiring_matches_across_ddp_replicas(tmp_path, backend, policy):
    if backend == "nccl" and torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    spawn_workers(
        _worker,
        args=(
            backend,
            "ddp",
            str(tmp_path / "init"),
            str(tmp_path / "state.pt"),
            policy,
        ),
        nprocs=2,
    )


@pytest.mark.slow
@pytest.mark.multiprocess
@pytest.mark.requires_gpu
@pytest.mark.parametrize("kind", ["dense", "indexed", "dropout"])
def test_fsdp_checkpoint_restores_optimizer_continuation(tmp_path, kind):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    spawn_workers(
        _worker,
        args=("nccl", "fsdp", str(tmp_path / "init"), str(tmp_path / "state.pt"), kind),
        nprocs=2,
    )
