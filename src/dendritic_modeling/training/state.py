"""Explicit epoch-boundary state for single-process standard/local training.

These helpers do not change Trainer defaults, checkpoint scheduling or stopping
rules. The caller owns the resolved configuration and checkpoint provenance.
The dictionary layout preserves existing local-learning continuation artifacts.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from dendritic_modeling.networks.checkpoints import copy_state_metadata

if TYPE_CHECKING:
    from dendritic_modeling.training.strategies.standard import Trainer


def capture_random_state() -> dict[str, Any]:
    """Capture Python, NumPy and all visible Torch generators without draws."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_random_state(state: Mapping[str, Any]) -> None:
    """Restore a trusted state on the same device topology and environment."""
    cuda_states = state["cuda"]
    if len(cuda_states) != torch.cuda.device_count():
        raise ValueError("Checkpoint CUDA generator count differs from this process")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


def _optimizer(trainer: Trainer):
    optimizer = trainer.optimizer
    wrappers = []
    while not hasattr(optimizer, "state_dict"):
        # This wrapper has no evolving state beyond its contained optimizer.
        from dendritic_modeling.training.optimizers.custom import (
            CustomWeightDecayOptimizer,
        )

        if type(optimizer) is not CustomWeightDecayOptimizer:
            raise ValueError("Unsupported stateful optimizer wrapper")
        wrappers.append(type(optimizer).__name__)
        optimizer = optimizer.optimizer
    return optimizer, wrappers


def _check_scope(trainer: Trainer, loaders: Mapping[str, DataLoader]) -> None:
    if trainer._is_distributed or trainer.enable_amp:
        raise ValueError("Epoch state requires single-process training without AMP")
    if trainer.pruning_config.enabled:
        raise ValueError("Pruning-manager state is not covered by this checkpoint")
    if set(loaders) != {"train_loader", "valid_loader"}:
        raise ValueError("Both training and validation loaders must be supplied")
    for loader in loaders.values():
        if loader.generator is None or loader.num_workers != 0:
            raise ValueError(
                "Checkpointed loaders need private generators and zero workers"
            )


def capture_epoch_state(
    trainer: Trainer,
    model: torch.nn.Module,
    *,
    loaders: Mapping[str, DataLoader],
    loss_lists: Sequence[Sequence[float]],
    source_commit: str,
) -> dict[str, Any]:
    """Snapshot a completed epoch, including best/current weights and optimizer.

    Use with matching resolved configuration and standard/local-learning epoch
    loops. The trainer's own LR schedule (``lr_schedule``) is captured under
    ``scheduler``; external schedulers, custom sampler state, recurrent
    persistent state and subclass-specific state require separate handling and
    qualification.
    """
    _check_scope(trainer, loaders)
    if len(loss_lists) != 4 or any(len(x) != trainer.epoch_counter for x in loss_lists):
        raise ValueError("Expected four complete per-epoch loss histories")
    optimizer, wrappers = _optimizer(trainer)
    state = {
        "epoch": trainer.epoch_counter,
        "current_model": deepcopy(trainer._unwrap_model(model).state_dict()),
        "best_model": deepcopy(trainer.best_state_dict),
        "optimizer": deepcopy(optimizer.state_dict()),
        "optimizer_wrappers": wrappers,
        "rng": capture_random_state(),
        "loader_generators": {k: v.generator.get_state() for k, v in loaders.items()},
        "loss_lists": deepcopy(loss_lists),
        "best_epoch": trainer.best_epoch,
        "best_loss": trainer.best_loss,
        "patience_counter": trainer.patience_counter,
        "source_commit": source_commit,
    }
    scheduler = getattr(trainer, "scheduler", None)
    if scheduler is not None:
        state["scheduler"] = deepcopy(scheduler.state_dict())
    return state


def restore_epoch_state(
    trainer: Trainer,
    model: torch.nn.Module,
    state: Mapping[str, Any],
    *,
    loaders: Mapping[str, DataLoader],
    expected_source_commit: str,
) -> tuple[list[float], ...]:
    """Restore an initialized trainer at an epoch boundary; return loss histories.

    Restore after model/optimizer/loader construction and trainer initialization,
    immediately before continuing the epoch loop. The caller specifies remaining
    epochs and must honor a checkpoint that has already reached its stopping rule.
    An explicitly reviewed source migration may name the historical commit here;
    that does not certify numerical equivalence to the currently imported source.
    """
    _check_scope(trainer, loaders)
    if state["source_commit"] != expected_source_commit:
        raise ValueError("Checkpoint source commit does not match the declared origin")
    optimizer, wrappers = _optimizer(trainer)
    if wrappers != state["optimizer_wrappers"]:
        raise ValueError("Checkpoint optimizer wrapper layout differs")
    if set(state["loader_generators"]) != set(loaders):
        raise ValueError("Checkpoint loader generator layout differs")
    if len(state["rng"]["cuda"]) != torch.cuda.device_count():
        raise ValueError("Checkpoint CUDA generator count differs from this process")
    histories = state["loss_lists"]
    if len(histories) != 4 or any(len(x) != state["epoch"] for x in histories):
        raise ValueError("Checkpoint loss histories do not cover every completed epoch")
    scheduler = getattr(trainer, "scheduler", None)
    if (scheduler is None) != ("scheduler" not in state):
        raise ValueError(
            "The checkpoint and the trainer differ in their LR schedule: the "
            "resumed run would follow another learning-rate trajectory"
        )
    trainer._unwrap_model(model).load_state_dict(state["current_model"])
    device = next(model.parameters()).device
    trainer.best_state_dict = copy_state_metadata(
        state["best_model"],
        OrderedDict((k, v.to(device).clone()) for k, v in state["best_model"].items()),
    )
    trainer.best_epoch = state["best_epoch"]
    trainer.best_loss = state["best_loss"]
    trainer.epoch_counter = state["epoch"]
    trainer.patience_counter = state["patience_counter"]
    optimizer.load_state_dict(deepcopy(state["optimizer"]))
    if scheduler is not None:
        scheduler.load_state_dict(deepcopy(state["scheduler"]))
    for name, value in state["loader_generators"].items():
        loaders[name].generator.set_state(value.cpu())
    restore_random_state(state["rng"])
    return tuple(list(values) for values in histories)
