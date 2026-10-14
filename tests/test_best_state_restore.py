"""load_best_state_dict restores a trained epoch, never the untrained snapshot."""

from __future__ import annotations

import math

import pytest
import torch
from test_trainer_configuration import (
    _data,
    _einet_model,
    _make_trainer,
    _max_abs_diff,
    _state,
)


def _scripted_validation(trainer, losses):
    values = iter(losses)
    trainer._non_distributed_validation_loss = lambda *_args: next(values)


@pytest.mark.parametrize("strategy", ["standard", "local_ca"])
def test_never_finite_validation_fails_instead_of_restoring_untrained_weights(
    strategy,
):
    model = _einet_model()
    extra = {"local_rule_config": {}} if strategy == "local_ca" else {}
    trainer, _ = _make_trainer(strategy, model, load_best_state_dict=True, **extra)
    _scripted_validation(trainer, [math.nan, math.nan])
    with pytest.raises(RuntimeError, match="no epoch had a finite validation loss"):
        trainer.train(model, _data(), _data(seed=1))


def test_without_load_best_state_dict_the_final_weights_are_kept():
    model = _einet_model()
    initial = _state(model)
    trainer, _ = _make_trainer("standard", model, load_best_state_dict=False)
    _scripted_validation(trainer, [math.nan, math.nan])
    results = trainer.train(model, _data(), _data(seed=1))
    assert results["best_loss"] == math.inf
    assert _max_abs_diff(initial, model.state_dict()) > 0


def test_a_run_that_recovers_from_a_non_finite_epoch_restores_its_best_epoch():
    model = _einet_model()
    trainer, _ = _make_trainer("standard", model, load_best_state_dict=True, epochs=3)
    _scripted_validation(trainer, [math.nan, 0.5, 0.7])
    results = trainer.train(model, _data(), _data(seed=1))
    assert results["best_epoch"] == 2
    assert results["best_loss"] == 0.5
    assert _max_abs_diff(results["best_state_dict"], model.state_dict()) == 0


def _ddp_local_ca_worker(rank, world_size, init_file, queue):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        model = _einet_model()  # identical initialization on every rank
        ddp_model = torch.nn.parallel.DistributedDataParallel(model)
        trainer, _ = _make_trainer(
            "local_ca", model, local_rule_config={}, load_best_state_dict=True
        )
        results = trainer.train(ddp_model, _data(64), _data(32, seed=1))
        flat = torch.cat([param.detach().flatten() for param in model.parameters()])
        gathered = [torch.zeros_like(flat) for _ in range(world_size)]
        dist.all_gather(gathered, flat)
        if rank == 0:
            queue.put(
                {
                    "replica_gap": float((gathered[0] - gathered[1]).abs().max()),
                    "restored_gap": _max_abs_diff(
                        results["best_state_dict"], model.state_dict()
                    ),
                    "keys": sorted(results),
                }
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.multiprocess
def test_local_ca_restores_its_best_state_under_ddp(tmp_path):
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    mp.start_processes(
        _ddp_local_ca_worker,
        args=(2, str(tmp_path / "process_group"), queue),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
    outcome = queue.get()
    assert outcome["replica_gap"] == 0.0
    assert outcome["restored_gap"] == 0.0
    assert {"train_losses_base", "train_losses_reg"} <= set(outcome["keys"])


@pytest.mark.parametrize(
    "options, writes",
    [
        ({}, "every_improvement"),
        ({"save_best_checkpoint": "end"}, "end"),
        ({"save_best_checkpoint": "never"}, "never"),
    ],
)
def test_best_checkpoint_writes_follow_save_best_checkpoint(
    tmp_path, monkeypatch, options, writes
):
    from dendritic_modeling.training.strategies import standard

    saved = []
    monkeypatch.setattr(
        standard, "atomic_torch_save", lambda state, path: saved.append(path)
    )
    model = _einet_model()
    trainer, _ = _make_trainer(
        "standard", model, epochs=3, save_path=str(tmp_path), **options
    )
    _scripted_validation(trainer, [0.9, 0.5, 0.4])  # improves every epoch

    trainer.train(model, _data(), _data(seed=1))

    best = [path for path in saved if path.endswith("best_model.pt")]
    assert len(best) == {"every_improvement": 3, "end": 1, "never": 0}[writes]


def test_a_cpu_best_state_restores_the_same_weights():
    states = {}
    for device in ("model", "cpu"):
        model = _einet_model()
        trainer, _ = _make_trainer(
            "standard",
            model,
            epochs=2,
            load_best_state_dict=True,
            best_state_device=device,
        )
        _scripted_validation(trainer, [0.5, 0.7])
        results = trainer.train(model, _data(), _data(seed=1))
        assert (
            all(
                value.device.type == "cpu"
                for value in results["best_state_dict"].values()
            )
            or device == "model"
        )
        states[device] = _state(model)
    assert _max_abs_diff(states["model"], states["cpu"]) == 0


def test_best_state_options_are_checked():
    with pytest.raises(ValueError, match="best_state_device='disk' is not supported"):
        _make_trainer("standard", _einet_model(), best_state_device="disk")
