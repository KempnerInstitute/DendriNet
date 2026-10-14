#!/usr/bin/env python
"""Profile a bounded number of optimizer steps of a configured model on its dataset.

This benchmark uses the ordinary configuration, dataset construction, and
model factory, so it includes data loading (``benchmark_configured_step``
reuses one resident batch instead). It stops after a small number of optimizer
steps and records CUDA memory, data-wait time, and compute throughput. It does
not save a model checkpoint or run final analyses.
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
from dendritic_modeling.utils.logging_config import setup_logging

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
        description="Profile bounded training steps of a configured model on its dataset."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=0,
        help="Number of held-out validation batches to evaluate after training.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--lr-scale",
        type=float,
        default=1.0,
        help="Multiply every configured optimizer-group learning rate.",
    )
    parser.add_argument(
        "--group-lr-scale",
        action="append",
        default=[],
        metavar="NAME=SCALE",
        help=(
            "Additional multiplier for one optimizer group. May be repeated; "
            "for example --group-lr-scale decoder=0.01."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/profiles/training_steps"),
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("batch_size", "steps"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps cannot be negative")
    if args.eval_steps < 0:
        raise ValueError("--eval-steps cannot be negative")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.lr_scale <= 0:
        raise ValueError("--lr-scale must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")


def _parse_group_lr_scales(values: list[str]) -> dict[str, float]:
    scales: dict[str, float] = {}
    for value in values:
        name, separator, raw_scale = value.partition("=")
        if not separator or not name.strip():
            raise ValueError(
                f"--group-lr-scale must use NAME=SCALE syntax, got {value!r}"
            )
        scale = float(raw_scale)
        if scale <= 0:
            raise ValueError(f"--group-lr-scale values must be positive, got {value!r}")
        scales[name.strip()] = scale
    return scales


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
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    worker_kwargs: dict[str, Any] = {}
    if num_workers > 0:
        worker_kwargs.update(
            persistent_workers=True,
            prefetch_factor=4,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
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
) -> tuple[float, int]:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
    correct = int((logits.detach().argmax(dim=-1) == labels).sum())

    scaler.scale(loss).backward()
    if grad_clip_value:
        scaler.unscale_(optimizer)
        clip_grad_value_(model.parameters(), float(grad_clip_value))
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach()), correct


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
    group_learning_rate_scales: dict[str, float],
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
    available_groups = {str(group.get("name", "")) for group in optimizer.param_groups}
    unknown_groups = set(group_learning_rate_scales) - available_groups
    if unknown_groups:
        raise ValueError(
            "Unknown optimizer groups in --group-lr-scale: "
            f"{sorted(unknown_groups)}; available groups are "
            f"{sorted(available_groups)}"
        )
    for group in optimizer.param_groups:
        group_name = str(group.get("name", ""))
        group["lr"] = (
            float(group["lr"])
            * learning_rate_scale
            * group_learning_rate_scales.get(group_name, 1.0)
        )
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
    correct_predictions = 0

    for _ in range(steps):
        wait_start = time.perf_counter()
        batch, iterator = _next_batch(iterator, loader)
        images, labels = batch
        data_wait_seconds += time.perf_counter() - wait_start

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        _synchronize()
        step_start = time.perf_counter()
        loss, correct = _optimizer_step(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            images=images,
            labels=labels,
            grad_clip_value=grad_clip_value,
            use_amp=use_amp,
        )
        losses.append(loss)
        correct_predictions += correct
        _synchronize()
        step_seconds += time.perf_counter() - step_start

    samples = batch_size * steps
    total_seconds = data_wait_seconds + step_seconds
    loss_window = max(1, steps // 4)
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
        "loss_mean": sum(losses) / len(losses),
        "loss_first_window_mean": sum(losses[:loss_window]) / loss_window,
        "loss_last_window_mean": sum(losses[-loss_window:]) / loss_window,
        "online_top1_accuracy": correct_predictions / samples,
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


@torch.no_grad()
def _evaluate(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    steps: int,
    use_amp: bool,
) -> dict[str, Any]:
    if steps <= 0:
        return {}
    device = torch.device("cuda")
    model.eval()
    losses = []
    correct = 0
    samples = 0
    start = time.perf_counter()
    for step_idx, (images, labels) in enumerate(loader):
        if step_idx >= steps:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            losses.append(float(F.cross_entropy(logits, labels)))
        correct += int((logits.argmax(dim=-1) == labels).sum())
        samples += int(labels.numel())
    _synchronize()
    elapsed = time.perf_counter() - start
    model.train()
    return {
        "eval_steps": len(losses),
        "eval_samples": samples,
        "eval_loss": sum(losses) / max(len(losses), 1),
        "eval_top1_accuracy": correct / max(samples, 1),
        "eval_samples_per_second": samples / max(elapsed, 1e-12),
    }


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    group_lr_scales = _parse_group_lr_scales(args.group_lr_scale)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(str(args.config))
    config.outputs.results_dir = str(args.output_dir / "setup")
    config.outputs.run_name = f"setup_batch_{args.batch_size}"
    config.experiment.checkpointing.enabled = False

    (
        _run_save_path,
        train_dataset,
        valid_dataset,
        _test_dataset,
        model,
        _encoder,
    ) = setup_environment(
        model_config=config.model,
        training_config=config.training,
        data_config=config.data,
        outputs_config=config.outputs,
        experiment_config=config.experiment,
        is_main=True,
    )

    loader = _make_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    eval_loader = _make_loader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
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
        "group_learning_rate_scales": group_lr_scales,
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
            group_learning_rate_scales=group_lr_scales,
            grad_clip_value=common.grad_clip_value,
            use_amp=bool(common.use_amp),
        )
    )
    result.update(
        _evaluate(
            model=model,
            loader=eval_loader,
            steps=args.eval_steps,
            use_amp=bool(common.use_amp),
        )
    )

    output_path = args.output_dir / f"batch_{args.batch_size}.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Saved profile: {output_path}")


if __name__ == "__main__":
    setup_logging()
    main()
