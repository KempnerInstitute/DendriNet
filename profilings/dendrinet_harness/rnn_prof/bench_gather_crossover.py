"""Where does un-staged beat staged? Map the boundary instead of guessing it.

The head-to-head showed no-stage winning 1.60x at (in=16,000, out=144,000) and
LOSING at (in=3,072, out=144,000) and (in=16,000, out=2,000). So the right
kernel is shape-dependent and the picker needs a real decision boundary.

The mechanism predicts the shape of that boundary: staged asks for in_features*4
bytes of shared, so blocks/SM = optin / (in*4). At in=16,000 that is 1 block --
occupancy-crippled. At in=4,000 it is 6 blocks -- fine. And the extra L2 traffic
no-stage pays only comes back if there is enough output work to fill the grid,
which is why out=2,000 goes the other way.

This sweeps the (in, out) grid, sweeps P within each variant, and prints the
winner per cell so the picker thresholds are measured.

    python bench_gather_crossover.py [--batch 512]
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    cuda_gather as cg,
)

INS = [3072, 4000, 8000, 16000, 24000]
OUTS = [2000, 8000, 16000, 48000, 144000]
K = 40
P_STAGED = [4096, 2048, 1024, 512]
P_NOSTAGE = [2048, 512, 128, 64, 32]


def timeit(fn, iters=6, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def best(fn, ladder):
    out = {}
    for P in ladder:
        try:
            out[P] = timeit(lambda P=P: fn(P))
        except RuntimeError:
            pass
    if not out:
        return None, float("inf")
    p = min(out, key=out.get)
    return p, out[p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()
    if not (torch.cuda.is_available() and cg.cuda_gather_available()):
        print("needs CUDA + built extension")
        return 1
    ext = cg._load_ext()
    dev = torch.device("cuda")
    optin = cg._optin_shared()
    M = a.batch
    print(f"GPU {torch.cuda.get_device_name(0)}  optin={optin:,} B  "
          f"batch={M}  K={K}\n")
    print("cell = speedup of NO-STAGE over STAGED (>1 means drop the shared staging)")
    header = "in \\ out"
    print("\n" + f"{header:>10}" + "".join(f"{o:>12,}" for o in OUTS)
          + f"{'blk/SM':>9}")

    winners = {}
    for in_f in INS:
        blocks = optin // (in_f * 4)
        row = f"{in_f:>10,}"
        for out_f in OUTS:
            idx = torch.randint(0, in_f, (out_f, K), device=dev,
                                dtype=torch.int32).contiguous()
            w = (torch.randn(out_f, K, device=dev) * 0.05).contiguous()
            x = torch.randn(M, in_f, device=dev).abs().contiguous()
            _, ts = best(lambda P: ext.gather_fwd_wpo_cuda(x, idx, w, P, 512), P_STAGED)
            pn, tn = best(lambda P: ext.gather_fwd_wpo_ns_cuda(x, idx, w, P, 512),
                          P_NOSTAGE)
            r = ts / tn if tn else float("nan")
            winners[(in_f, out_f)] = (r, pn)
            row += f"{r:>11.2f}x"
            del idx, w, x
            torch.cuda.empty_cache()
        print(row + f"{blocks:>9}")

    print("\n  blk/SM = optin / (in_features*4): how many staged blocks fit an SM.")
    print("  A cell > 1.00 means the un-staged kernel is faster at that shape.\n")
    print(f"  {'shape':>22}  {'best no-stage P':>16}")
    for (in_f, out_f), (r, pn) in sorted(winners.items(), key=lambda kv: -kv[1][0]):
        if r > 1.0:
            print(f"  in={in_f:>6,} out={out_f:>7,}  {str(pn):>16}   {r:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
