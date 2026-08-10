"""Baseline for the Triton optimization pass: where does shipped Triton stand?

Per-shape forward AND backward timings for the three shipped implementations:
    fused   -- custom autograd over torch ops (the Python-level path)
    triton  -- the shipped Triton kernels, default tiles (B_M=16, B_O=64)
    cuda    -- the CUDA extension through its measured picker

on the standard FF crossover grid. This table is the "before" that every
Triton change gets judged against; the gap column (cuda/triton) is the
theoretical room, with the caveat that cuda's staged-shared design is not
expressible in Triton -- the honest target is the no-stage arm's territory.

    python bench_triton_baseline.py [--batch 512] [--K 40]
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.fused_gather import (
    fused_sparse_gather,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    cuda_gather as cg,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.triton_gather import (
    triton_sparse_gather,
    triton_gather_available,
)

SHAPES = [(3072, 2000), (3072, 16000), (8000, 16000), (8000, 48000),
          (16000, 16000), (16000, 48000), (16000, 144000), (24000, 48000)]


def timeit(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def fwd_bwd_ms(callable_gather, x, idx, w):
    """(forward ms, backward ms) for one gather implementation."""
    def fwd():
        with torch.no_grad():
            return callable_gather(x, idx, w)

    def step():
        xg = x.detach().requires_grad_(True)
        wg = w.detach().requires_grad_(True)
        out = callable_gather(xg, idx, wg)
        out.sum().backward()

    f = timeit(fwd)
    s = timeit(step)
    return f, max(s - f, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--K", type=int, default=40)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("needs CUDA")
        return 1
    if not triton_gather_available():
        print("needs triton")
        return 1
    has_cuda_ext = cg.cuda_gather_available()
    dev = torch.device("cuda")
    M, K = a.batch, a.K
    print(f"GPU {torch.cuda.get_device_name(0)}  batch={M} K={K}")
    print(f"{'in':>7}{'out':>8} | {'fused f/b':>15} | {'triton f/b':>15} | "
          f"{'cuda f/b':>15} | {'tri/fused':>9} {'cuda/tri':>9}")
    for IN, O in SHAPES:
        idx = torch.randint(0, IN, (O, K), dtype=torch.int32, device=dev)
        idx64 = idx.to(torch.int64)
        w = (torch.randn(O, K, device=dev) * 0.05).contiguous()
        x = torch.randn(M, IN, device=dev).abs().contiguous()
        ff, fb = fwd_bwd_ms(lambda x_, i_, w_: fused_sparse_gather(x_, idx64, w_), x, idx, w)
        tf, tb = fwd_bwd_ms(lambda x_, i_, w_: triton_sparse_gather(x_, idx64, w_), x, idx, w)
        if has_cuda_ext:
            cf, cb = fwd_bwd_ms(lambda x_, i_, w_: cg.cuda_sparse_gather(x_, idx, w_), x, idx, w)
        else:
            cf = cb = float("nan")
        t_tot, f_tot, c_tot = tf + tb, ff + fb, cf + cb
        print(f"{IN:>7,}{O:>8,} | {ff:>6.2f} /{fb:>7.2f} | {tf:>6.2f} /{tb:>7.2f} | "
              f"{cf:>6.2f} /{cb:>7.2f} | {f_tot/t_tot:>8.2f}x {t_tot/c_tot:>8.2f}x")
        del idx, idx64, w, x
        torch.cuda.empty_cache()
    print("\n  tri/fused > 1 means shipped triton beats fused; cuda/tri is the gap"
          "\n  the optimization pass is trying to close (staged-shared excluded).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
