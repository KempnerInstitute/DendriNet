#!/usr/bin/env python
"""Profile bounded optimizer steps for the raw-ImageNet spatial DendriNet.

This benchmark uses the ordinary configuration, dataset construction, and
model factory. It intentionally stops after a small number of optimizer steps
and records CUDA memory, data-wait time, and compute throughput. It does not
save a model checkpoint or run final analyses.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader

from dendritic_modeling.config import load_config
from dendritic_modeling.scripts.script_utils.setup_utils import setup_environment
from dendritic_modeling.training.optimizers import create_optimizer

logger = logging.getLogger(__name__)


def _git_provenance() -> dict[str, Any]:
    """Return the checked-out commit and tracked-worktree state."""

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "commit": commit,
        "tracked_worktree_dirty": bool(status.strip()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile raw-ImageNet spatial DendriNet training steps."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--lr-scale",
        type=float,
        default=1.0,
        help="Multiply every configured optimizer-group learning rate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/profiles/spatial_imagenet"),
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("batch_size", "steps"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps cannot be negative")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.lr_scale <= 0:
        raise ValueError("--lr-scale must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")


def _cuda_gib(value: int) -> float:
    return float(value) / (1024**3)


def _synchronize() -> None:
    torch.cuda.synchronize()


def _model_resource_counts(model: torch.nn.Module) -> dict[str, int]:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    index_elements = sum(
        buffer.numel()
        for name, buffer in model.named_buffers()
        if name.endswith("connection_indices")
    )
    index_bytes = sum(
        buffer.numel() * buffer.element_size()
        for name, buffer in model.named_buffers()
        if name.endswith("connection_indices")
    )
    return {
        "parameters": int(parameters),
        "trainable_parameters": int(trainable_parameters),
        "connection_index_elements": int(index_elements),
        "connection_index_bytes": int(index_bytes),
    }


def _make_loader(
    train_dataset,
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    worker_kwargs: dict[str, Any] = {}
    if num_workers > 0:
        worker_kwargs.update(
            persistent_workers=True,
            prefetch_factor=4,
        )
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        **worker_kwargs,
    )


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _optimizer_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    images: torch.Tensor,
    labels: torch.Tensor,
    grad_clip_value: float | None,
    use_amp: bool,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        logits = model(images)
        loss = F.cross_entropy(logits, labels)

    scaler.scale(loss).backward()
    if grad_clip_value:
        scaler.unscale_(optimizer)
        clip_grad_value_(model.parameters(), float(grad_clip_value))
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach())


@torch.no_grad()
def _loss_only(
    *,
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    use_amp: bool,
) -> float:
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        return float(F.cross_entropy(model(images), labels))


def _profile(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    batch_size: int,
    warmup_steps: int,
    steps: int,
    optimizer_config,
    param_groups_config,
    learning_rate_scale: float,
    grad_clip_value: float | None,
    use_amp: bool,
) -> dict[str, Any]:
    device = torch.device("cuda")
    model.to(device)
    model.train()
    optimizer = create_optimizer(
        model.get_param_groups(param_groups_config),
        optimizer_config,
    )
    for group in optimizer.param_groups:
        group["lr"] = float(group["lr"]) * learning_rate_scale
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    iterator = iter(loader)

    initial_batch, iterator = _next_batch(iterator, loader)
    initial_images, initial_labels = initial_batch
    initial_loss = _loss_only(
        model=model,
        images=initial_images.to(device, non_blocking=True),
        labels=initial_labels.to(device, non_blocking=True),
        use_amp=use_amp,
    )

    for _ in range(warmup_steps):
        batch, iterator = _next_batch(iterator, loader)
        images, labels = batch
        _optimizer_step(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            images=images.to(device, non_blocking=True),
            labels=labels.to(device, non_blocking=True),
            grad_clip_value=grad_clip_value,
            use_amp=use_amp,
        )
    _synchronize()

    torch.cuda.reset_peak_memory_stats()
    data_wait_seconds = 0.0
    step_seconds = 0.0
    losses = []

    for _ in range(steps):
        wait_start = time.perf_counter()
        batch, iterator = _next_batch(iterator, loader)
        images, labels = batch
        data_wait_seconds += time.perf_counter() - wait_start

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        _synchronize()
        step_start = time.perf_counter()
        losses.append(
            _optimizer_step(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                images=images,
                labels=labels,
                grad_clip_value=grad_clip_value,
                use_amp=use_amp,
            )
        )
        _synchronize()
        step_seconds += time.perf_counter() - step_start

    samples = batch_size * steps
    total_seconds = data_wait_seconds + step_seconds
    optimizer_groups = [
        {
            "name": str(group.get("name", f"group_{idx}")),
            "lr": float(group["lr"]),
            "parameters": int(sum(parameter.numel() for parameter in group["params"])),
        }
        for idx, group in enumerate(optimizer.param_groups)
    ]
    return {
        "warmup_steps": warmup_steps,
        "measured_steps": steps,
        "measured_samples": samples,
        "initial_loss": initial_loss,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "all_losses_finite": all(torch.isfinite(torch.tensor(losses)).tolist()),
        "data_wait_seconds": data_wait_seconds,
        "optimizer_step_seconds": step_seconds,
        "total_measured_seconds": total_seconds,
        "compute_samples_per_second": samples / step_seconds,
        "end_to_end_samples_per_second": samples / total_seconds,
        "data_wait_fraction": data_wait_seconds / total_seconds,
        "peak_cuda_allocated_gib": _cuda_gib(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_gib": _cuda_gib(torch.cuda.max_memory_reserved()),
        "optimizer_groups": optimizer_groups,
    }


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(str(args.config))
    config.wandb.use_wandb = False
    config.outputs.results_dir = str(args.output_dir / "setup")
    config.outputs.run_name = f"setup_batch_{args.batch_size}"
    config.experiment.checkpointing.enabled = False

    (
        _run_save_path,
        train_dataset,
        _valid_dataset,
        _test_dataset,
        model,
        _encoder,
    ) = setup_environment(
        model_config=config.model,
        training_config=config.training,
        data_config=config.data,
        wandb_config=config.wandb,
        outputs_config=config.outputs,
        experiment_config=config.experiment,
        is_main=True,
    )

    loader = _make_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    common = config.training.main.common
    optimizer_config = config.training.main.optimizer
    param_groups = common.param_groups

    result: dict[str, Any] = {
        "schema_version": 1,
        "config": str(args.config.resolve()),
        "hostname": platform.node(),
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "learning_rate_scale": args.lr_scale,
        "train_dataset_samples": len(train_dataset),
        "use_amp": bool(common.use_amp),
        **_git_provenance(),
        **_model_resource_counts(model),
    }
    result.update(
        _profile(
            model=model,
            loader=loader,
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            steps=args.steps,
            optimizer_config=optimizer_config,
            param_groups_config=param_groups,
            learning_rate_scale=args.lr_scale,
            grad_clip_value=common.grad_clip_value,
            use_amp=bool(common.use_amp),
        )
    )

    output_path = args.output_dir / f"batch_{args.batch_size}.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Saved profile: {output_path}")


if __name__ == "__main__":
    main()
