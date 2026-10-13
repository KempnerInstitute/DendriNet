"""Benchmark fused ELL kernels against the eager indexed path and dense GEMM.

Shapes cover three regimes: the AlexNet fc6 bank (largest parameter mass), a
transformer feed-forward row, and the conv3 patch bank (spatial regime, many
rows = batch x positions). For each:
eager reference (gather-materialize-reduce), fused Triton kernel with
row-major [M, D] activations, fused kernel with [D, M]-contiguous
activations (coalesced gathers), the fused E/I shunting kernel, and the
dense fp32 GEMM of the same layer shape (the speed ceiling any dense
baseline enjoys).
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from dendritic_modeling.kernels import (
    ell_gather_gemm,
    ell_shunting_forward,
    reference_gather_gemm,
    reference_shunting_forward,
    triton_available,
)
from dendritic_modeling.utils.logging_config import setup_logging

SHAPES = {
    "fc6_alexnet": {"m": 64, "d": 9216, "o": 4096, "k": 1152},
    "ffn_row": {"m": 2048, "d": 2048, "o": 8192, "k": 256},
    "conv3_patches": {"m": 64 * 169, "d": 1728, "o": 384, "k": 216},
}


def _timed(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3 / iterations


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    if not triton_available():
        raise SystemExit("CUDA + triton required")
    device = "cuda"
    record: dict = {"device_name": torch.cuda.get_device_name(0), "shapes": {}}
    for name, shape in SHAPES.items():
        m, d, o, k = shape["m"], shape["d"], shape["o"], shape["k"]
        generator = torch.Generator().manual_seed(0)
        x = torch.rand(m, d, generator=generator).to(device)
        x_dm = x.t().contiguous().t()
        indices = torch.stack(
            [torch.randperm(d, generator=generator)[:k] for _ in range(o)]
        ).to(device=device, dtype=torch.int32)
        pre_w = torch.randn(o, k, generator=generator).to(device)
        idx_inh = indices.flip(0).contiguous()
        w_inh = pre_w.flip(0).contiguous()
        dense_w = torch.randn(o, d, generator=generator).to(device)

        # Correctness spot check before timing.
        torch.testing.assert_close(
            ell_gather_gemm(x, indices, pre_w),
            reference_gather_gemm(x, indices, pre_w),
            rtol=1e-3,
            atol=1e-3,
        )

        entry = {
            **shape,
            "eager_ms": _timed(
                lambda a=x, b=indices, c=pre_w: reference_gather_gemm(a, b, c),
                args.warmup,
                args.iterations,
            ),
            "fused_ms": _timed(
                lambda a=x, b=indices, c=pre_w: ell_gather_gemm(a, b, c),
                args.warmup,
                args.iterations,
            ),
            "fused_dm_layout_ms": _timed(
                lambda a=x_dm, b=indices, c=pre_w: ell_gather_gemm(a, b, c),
                args.warmup,
                args.iterations,
            ),
            "eager_shunting_ms": _timed(
                lambda a=x, b=indices, c=pre_w, d=idx_inh, e=w_inh: (
                    reference_shunting_forward(a, b, c, d, e)
                ),
                args.warmup,
                args.iterations,
            ),
            "fused_shunting_ms": _timed(
                lambda a=x, b=indices, c=pre_w, d=idx_inh, e=w_inh: (
                    ell_shunting_forward(a, b, c, d, e)
                ),
                args.warmup,
                args.iterations,
            ),
            "dense_gemm_ms": _timed(
                lambda a=x, b=dense_w: a @ b.t(), args.warmup, args.iterations
            ),
        }
        # Row-major and feature-major speedups reported separately: the
        # headline 7.4-8.8x figures are the feature-major ([D,M]) layout;
        # row-major fusion loses to eager and must not be conflated.
        entry["fused_rowmajor_speedup_vs_eager"] = round(
            entry["eager_ms"] / entry["fused_ms"], 2
        )
        entry["fused_dm_speedup_vs_eager"] = round(
            entry["eager_ms"] / entry["fused_dm_layout_ms"], 2
        )
        entry["fused_shunting_speedup"] = round(
            entry["eager_shunting_ms"] / entry["fused_shunting_ms"], 2
        )
        entry["fused_vs_dense_gemm"] = round(
            entry["dense_gemm_ms"] / entry["fused_ms"], 2
        )
        record["shapes"][name] = {
            key: (round(value, 3) if isinstance(value, float) else value)
            for key, value in entry.items()
        }
        print(name, record["shapes"][name], flush=True)

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(record, handle, indent=1)
        print("wrote", args.output)


if __name__ == "__main__":
    setup_logging()
    main()
