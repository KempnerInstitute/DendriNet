"""RNN gather crossover at batch 4, 1M-class shapes, bf16 inputs.

Every "fused wins for the RNN" verdict was measured at BATCH 1 with fp32-
upcasting kernel wrappers. The 1M campaign now runs batch 4 under FSDP bf16
mixed precision, which changes the byte economics of the dominant op:

    fused   : bf16 x/w (native)  but int64 indices (index_select requirement,
              the measured 11.6 GB/fwd traffic term)
    triton  : bf16 x/w (new)     and int32 indices, coalesced transposed layout
    cuda    : fp32-upcast x/w    and int32 indices (picker-dispatched)

[1] correctness: fp64 reference computed FROM the bf16-rounded inputs (both
    paths see identical inputs, so only accumulation differs; fp32-acc bounds).
[2] timing at batch 1 and 4, fwd and bwd separately, on shapes scaled from the
    5k module list to the 1M model's class.

    python bench_rnn_b4_crossover.py
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

# The 5k model's three heaviest gather modules, scaled x200 to the 1M class.
SHAPES = [(800_000, 20_000_000, 64), (200_000, 20_000_000, 16),
          (800_000, 4_000_000, 240)]
BATCHES = (1, 4)


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
    print(f"GPU {torch.cuda.get_device_name(0)}\n")

    print("[1] correctness, bf16 inputs (fp64 ref from the SAME rounded inputs)")
    IN, O, K = 200_000, 1_000_000, 16   # small cousin for the fp64 reference
    idx32 = torch.randint(0, IN, (O, K), dtype=torch.int32, device=dev)
    idx64 = idx32.long()
    x = torch.randn(4, IN, device=dev).abs().to(torch.bfloat16)
    w = (torch.randn(O, K, device=dev) * 0.05).to(torch.bfloat16)
    ref = torch.zeros(4, O, dtype=torch.float64, device=dev)
    for k in range(K):
        ref += x.double()[:, idx64[:, k]] * w.double()[:, k]
    e_f = (fused_sparse_gather(x, idx64, w).double() - ref).abs().max().item()
    e_t = (triton_sparse_gather(x, idx32, w).double() - ref).abs().max().item()
    print(f"  fused {e_f:.1e}   triton-bf16 {e_t:.1e}   "
          f"{'OK' if max(e_f, e_t) < 1e-2 else 'FAIL'}")
    del idx32, idx64, x, w, ref
    torch.cuda.empty_cache()

    print("\n[2] timing (ms), bf16 inputs")
    print(f"{'in':>8}{'out':>12}{'K':>5}{'batch':>6} | {'fused f/b':>16} | "
          f"{'triton f/b':>16} | {'cuda f/b':>15} | {'tri/fused':>9}")
    for IN, O, K in SHAPES:
        idx32 = torch.randint(0, IN, (O, K), dtype=torch.int32, device=dev)
        idx64 = idx32.long()
        w = (torch.randn(O, K, device=dev) * 0.05).to(torch.bfloat16).contiguous()
        for M in BATCHES:
            x = torch.randn(M, IN, device=dev).abs().to(torch.bfloat16).contiguous()
            ff, fb = fwd_bwd(lambda a, b: fused_sparse_gather(a, idx64, b), x, w)
            tf, tb = fwd_bwd(lambda a, b: triton_sparse_gather(a, idx32, b), x, w)
            if has_cuda:
                try:
                    cf, cb = fwd_bwd(lambda a, b: cg.cuda_sparse_gather(a, idx32, b), x, w)
                except RuntimeError:
                    cf = cb = float("nan")
            else:
                cf = cb = float("nan")
            r = (ff + fb) / (tf + tb)
            print(f"{IN:>8,}{O:>12,}{K:>5}{M:>6} | {ff:>7.1f} /{fb:>7.1f} | "
                  f"{tf:>7.1f} /{tb:>7.1f} | {cf:>6.1f} /{cb:>7.1f} | {r:>8.2f}x")
            del x
            torch.cuda.empty_cache()
        del idx32, idx64, w
        torch.cuda.empty_cache()
    print("\n  tri/fused > 1.03 at batch 4 -> the RNN gather path verdict flips;"
          "\n  <= 1.03 -> fused keeps the RNN, verdict recorded either way.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
