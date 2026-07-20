"""Benchmark dense and dendritic FFN operators without loading a full LM."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from dendritic_modeling.config import load_config
from dendritic_modeling.training._transformer_replacement.builders import (
    _make_replacement,
)
from dendritic_modeling.training._transformer_replacement.common import (
    SyntheticTransformerMLP,
    _resolve_device,
    _resolve_dtype,
    _to_plain_mapping,
)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timings(
    operation: Callable[[], None],
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> list[float]:
    for _ in range(max(0, int(warmup))):
        operation()
    _synchronize(device)
    samples = []
    for _ in range(max(1, int(iterations))):
        started = time.perf_counter()
        operation()
        _synchronize(device)
        samples.append(time.perf_counter() - started)
    return samples


def _timing_summary(samples: list[float], tokens: int) -> dict[str, float]:
    ordered = sorted(samples)

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
        return float(ordered[index])

    median = float(statistics.median(ordered))
    return {
        "median_seconds": median,
        "p10_seconds": percentile(0.10),
        "p90_seconds": percentile(0.90),
        "tokens_per_second": float(tokens) / max(median, 1e-12),
    }


def _benchmark_module(
    module: nn.Module,
    inputs: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, Any]:
    tokens = int(inputs.numel() // inputs.shape[-1])
    module.eval()

    def inference() -> None:
        with torch.no_grad():
            module(inputs)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        memory_start = int(torch.cuda.memory_allocated(device))
    else:
        memory_start = 0
    inference_samples = _timings(
        inference,
        warmup=warmup,
        iterations=iterations,
        device=device,
    )
    inference_peak = (
        int(torch.cuda.max_memory_allocated(device)) - memory_start
        if device.type == "cuda"
        else None
    )

    module.train()

    def training_step() -> None:
        module.zero_grad(set_to_none=True)
        output = module(inputs)
        output.float().square().mean().backward()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        memory_start = int(torch.cuda.memory_allocated(device))
    training_samples = _timings(
        training_step,
        warmup=warmup,
        iterations=iterations,
        device=device,
    )
    training_peak = (
        int(torch.cuda.max_memory_allocated(device)) - memory_start
        if device.type == "cuda"
        else None
    )
    return {
        "stored_parameters": sum(param.numel() for param in module.parameters()),
        "inference": _timing_summary(inference_samples, tokens),
        "training_forward_backward": _timing_summary(training_samples, tokens),
        "inference_peak_memory_bytes_delta": inference_peak,
        "training_peak_memory_bytes_delta": training_peak,
    }


def benchmark_operator_pair(
    config,
    *,
    hidden_size: int,
    intermediate_size: int,
    batch_size: int,
    sequence_length: int,
    warmup: int,
    iterations: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    train_cfg = config.training.transformer_replacement
    teacher_cfg = _to_plain_mapping(train_cfg.synthetic_teacher)
    dense = SyntheticTransformerMLP(
        int(hidden_size),
        int(intermediate_size),
        kind=str(teacher_cfg.get("kind", "gated_mlp")),
        activation=str(teacher_cfg.get("activation", "silu")),
        bias=bool(teacher_cfg.get("bias", False)),
    ).to(device=device, dtype=dtype)
    dendritic = _make_replacement(
        hidden_size=int(hidden_size),
        config=config,
        dtype=dtype,
        device=device,
    )
    seed = config.experiment.seed if train_cfg.seed is None else train_cfg.seed
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    inputs = torch.randn(
        int(batch_size),
        int(sequence_length),
        int(hidden_size),
        generator=generator,
    ).to(device=device, dtype=dtype)

    dense_report = _benchmark_module(
        dense,
        inputs,
        warmup=warmup,
        iterations=iterations,
        device=device,
    )
    dendritic_report = _benchmark_module(
        dendritic,
        inputs,
        warmup=warmup,
        iterations=iterations,
        device=device,
    )
    estimate = dendritic.parameter_estimate()
    dendritic_report["parameter_estimate"] = estimate
    return {
        "schema_version": 1,
        "device": str(device),
        "dtype": str(dtype),
        "shape": {
            "batch_size": int(batch_size),
            "sequence_length": int(sequence_length),
            "hidden_size": int(hidden_size),
            "intermediate_size": int(intermediate_size),
        },
        "warmup": int(warmup),
        "iterations": int(iterations),
        "dense_gated_mlp": dense_report,
        "dendritic_ffn": dendritic_report,
        "ratios": {
            "stored_parameter_reduction": dense_report["stored_parameters"]
            / max(int(estimate["stored_total"]), 1),
            "inference_speedup": dense_report["inference"]["median_seconds"]
            / max(dendritic_report["inference"]["median_seconds"], 1e-12),
            "training_speedup": dense_report["training_forward_backward"][
                "median_seconds"
            ]
            / max(
                dendritic_report["training_forward_backward"]["median_seconds"],
                1e-12,
            ),
        },
        "interpretation": (
            "Timing includes the current PyTorch gather/branch implementation. "
            "Active-weight count is not a FLOP estimate."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path")
    parser.add_argument("--hidden-size", type=int, default=0)
    parser.add_argument("--intermediate-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_config.cache_clear()
    config = load_config(args.config_path)
    train_cfg = config.training.transformer_replacement
    hidden_size = int(args.hidden_size or train_cfg.hidden_size or 128)
    intermediate_size = int(
        args.intermediate_size or train_cfg.intermediate_size or 4 * hidden_size
    )
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32
    report = benchmark_operator_pair(
        config,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
        dtype=dtype,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    print(rendered, end="", flush=True)


if __name__ == "__main__":
    main()
