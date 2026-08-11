"""The high-K starting line: existing kernels at the 1M model's HOT shapes.

The step profile + race (2026-08-11) named the bottleneck: per-timestep
recurrent gathers at out=200k-4M with K=300-3968 are 58% of the 1M step, and
the auto-race only pitted cuda (48KB tile cap -> forfeits) against fused.
Triton v3 was measured only to K=240. This bench runs the EXACT hot-shape
census from the race log, bf16 inputs, batch 4, fwd and bwd separately, for:
    fused (aten index machinery -- the incumbent, wins today by forfeit)
    triton v3 (untested at high K; its fwd streams K and might already win)
    cuda (expected forfeit -- confirms the tile cap at every hot shape)
Correctness gate: bf16-input fp64 reference on the smallest hot shape.

    python bench_highk.py
"""

from __future__ import annotations

import sys
import time

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.fused_gather import (
    fused_sparse_gather,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.triton_gather import (
    triton_sparse_gather, triton_gather_available,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    cuda_gather as cg,
)

# The hot-shape census, verbatim from the 1M race log (out, K, in_features).
# in_features: recurrent modules read the neuron state (1M) or population
# slices; sizes below reproduce the gather-traffic geometry.
HOT = [
    (200_000, 1_000, 1_000_000),
    (200_000, 3_968, 1_000_000),
    (800_000, 300, 1_000_000),
    (800_000, 1_168, 1_000_000),
    (4_000_000, 240, 1_000_000),
]
M = 4  # the campaign batch


def timeit(fn, iters=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def fwd_bwd(fn, x, w):
    f = timeit(lambda: fn(x, w))
    xg = x.detach().requires_grad_(True)
    wg = w.detach().requires_grad_(True)
    out = fn(xg, wg)
    go = torch.randn_like(out)
    b = timeit(lambda: torch.autograd.grad(out, (xg, wg), go, retain_graph=True))
    return f, b


def main():
    if not (torch.cuda.is_available() and triton_gather_available()):
        print("needs CUDA + triton")
        return 1
    dev = torch.device("cuda")
    has_cuda = cg.cuda_gather_available()
    print(f"GPU {torch.cuda.get_device_name(0)}  batch={M}\n")

    print("[1] correctness at (200k, K=1000): bf16-input fp64 reference")
    O, K, IN = 200_000, 1_000, 400_000
    idx32 = torch.randint(0, IN, (O, K), dtype=torch.int32, device=dev)
    idx64 = idx32.long()
    x = torch.randn(M, IN, device=dev).abs().to(torch.bfloat16)
    w = (torch.randn(O, K, device=dev) * 0.02).to(torch.bfloat16)
    ref = torch.zeros(M, O, dtype=torch.float64, device=dev)
    for k0 in range(0, K, 100):
        sel = idx64[:, k0:k0 + 100]
        ref += (x.double()[:, sel.reshape(-1)]
                .reshape(M, O, -1) * w.double()[:, k0:k0 + 100]).sum(-1)
    e_f = (fused_sparse_gather(x, idx64, w).double() - ref).abs().max().item()
    e_t = (triton_sparse_gather(x, idx32, w).double() - ref).abs().max().item()
    print(f"  fused {e_f:.1e}   triton {e_t:.1e}   "
          f"{'OK' if max(e_f, e_t) < 1e-1 else 'FAIL'}")
    del idx32, idx64, x, w, ref
    torch.cuda.empty_cache()

    print(f"\n[2] hot shapes, bf16, batch {M} (ms)")
    print(f"{'out':>10}{'K':>6} | {'fused f/b':>17} | {'triton f/b':>17} | "
          f"{'cuda':>7} | {'tri/fused':>9}")
    for O, K, IN in HOT:
        idx32 = torch.randint(0, IN, (O, K), dtype=torch.int32, device=dev)
        idx64 = idx32.long()
        w = (torch.randn(O, K, device=dev) * 0.02).to(torch.bfloat16).contiguous()
        x = torch.randn(M, IN, device=dev).abs().to(torch.bfloat16).contiguous()
        ff, fb = fwd_bwd(lambda a, b: fused_sparse_gather(a, idx64, b), x, w)
        tf, tb = fwd_bwd(lambda a, b: triton_sparse_gather(a, idx32, b), x, w)
        if has_cuda:
            try:
                cf, _cb = fwd_bwd(lambda a, b: cg.cuda_sparse_gather(a, idx32, b), x, w)
                cnote = f"{cf:.0f}ms"
            except RuntimeError:
                cnote = "forfeit"
        else:
            cnote = "n/a"
        r = (ff + fb) / (tf + tb)
        print(f"{O:>10,}{K:>6} | {ff:>7.1f} /{fb:>8.1f} | {tf:>7.1f} /{tb:>8.1f} | "
              f"{cnote:>7} | {r:>8.2f}x")
        del idx32, idx64, w, x
        torch.cuda.empty_cache()
    print("\n  tri/fused > 1 anywhere at a hot shape => instant win available by"
          "\n  adding triton to the auto-race BEFORE any new kernel is written."
          "\n  The per-column gaps (f vs b) define what a v4 kernel must beat.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
