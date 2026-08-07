"""Time the SHIPPED picker dispatch against best-staged at every winner cell.

The crossover sweeps report per-cell BEST-P ratios; the picker ships banded
P values. This closes that gap: for each cell in the arch's expected winner
set (check_picker_dispatch.EXPECTED), time exactly the (variant, P, T, S)
that _pick_fwd_variant returns, against a P-swept staged baseline.

PASS = no winner cell where the shipped config is >3% slower than staged
(the sweep noise floor). A FAIL cell means the P band -- not the band shape
-- needs a refit at that cell.

    python bench_shipped_bands.py [--batch 512]
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from check_picker_dispatch import EXPECTED
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    cuda_gather as cg,
)


def timeit(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()
    if not (torch.cuda.is_available() and cg.cuda_gather_available()):
        print("needs CUDA + built extension")
        return 1
    cc = torch.cuda.get_device_capability()
    print(torch.cuda.get_device_name(0), cc, f"batch={a.batch}")
    if cc not in EXPECTED:
        print(f"no expected winner set for cc={cc}; nothing to verify")
        return 0
    ext = cg._load_ext()
    dev = torch.device("cuda")
    M = a.batch
    bad = []
    cells = sorted(EXPECTED[cc]().items())
    for (IN, O, K), want in cells:
        v, P, T = cg._pick_fwd_variant(IN, O, K)
        if v != want:
            print(f"DISPATCH MISMATCH at {(IN, O, K)}: {v} != {want}")
            return 1
        idx = torch.randint(0, IN, (O, K), dtype=torch.int32, device=dev)
        w = (torch.randn(O, K, device=dev) * 0.05).contiguous()
        x = torch.randn(M, IN, device=dev).abs().contiguous()
        if v == "nostage":
            fn = lambda: ext.gather_fwd_wpo_ns_cuda(x, idx, w, P, T)
        else:
            S = 8192 if K <= 20 else 12288
            fn = lambda: ext.gather_fwd_wpo_hy_cuda(x, idx, w, P, T, S)
        t_ship = timeit(fn)
        t_staged = min(timeit(lambda p=p: ext.gather_fwd_wpo_cuda(x, idx, w, p, 512))
                       for p in (4096, 2048, 1024))
        r = t_staged / t_ship
        flag = "" if r >= 0.97 else "   <-- SHIPPED CONFIG SLOWER THAN STAGED"
        print(f"IN={IN:>6,} O={O:>7,} K={K:>3} {v:>8} P={P:>5} "
              f"shipped {t_ship:8.3f} ms  staged {t_staged:8.3f}  {r:5.2f}x{flag}")
        if r < 0.97:
            bad.append((IN, O, K, v, P, round(r, 3)))
        del idx, w, x
        torch.cuda.empty_cache()
    print(f"\n{'PASS' if not bad else 'FAIL'}: {len(bad)} of {len(cells)} "
          f"winner cells slower than staged at the shipped P")
    for b in bad:
        print("  ", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
