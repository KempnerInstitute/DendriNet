"""Hybrid prefix-staged forward vs the two incumbents. Three-way, all swept.

Motivation is fully measured: the staged kernel is occupancy-capped at 1
block/SM on Blackwell by its IN*4-byte row (a HARD limit -- 2 blocks need
128,000 B vs a 102,400 B/SM budget), while the un-staged kernel gets occupancy
but pays ~5x the L2 requests and wins only 3/25 shapes at K=40. The hybrid
stages a PREFIX of S elements (S*4 bytes -> several blocks/SM) and reads the
tail through L2 (99.4% hit rate, measured).

  [1] EQUIVALENCE -- accumulation order and (o,k) ownership are identical to
      both incumbents, so outputs must be BIT-identical. Asserted, not toleranced.
  [2] SWEEP -- hybrid over S in {4096, 8192, 12288} x the P ladder, against the
      per-shape BEST staged and BEST un-staged (both P-swept), so the comparison
      cannot be rigged by a mistuned baseline. K=40 grid + the K=10 row where
      no-stage currently sweeps.

Verdict rule, fixed in advance: the hybrid earns a picker band only where it
beats BOTH incumbents by >3% (outside timing noise) at that shape.

    python bench_gather_hybrid.py [--batch 512]
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    cuda_gather as cg,
)

SHAPES_K40 = [(16000, 144000, 40), (16000, 48000, 40), (16000, 16000, 40),
              (16000, 2000, 40), (24000, 144000, 40), (24000, 48000, 40),
              (8000, 144000, 40)]
SHAPES_K10 = [(16000, 144000, 10), (16000, 2000, 10), (24000, 48000, 10)]
P_LADDER = [4096, 2048, 1024, 512, 128, 64, 32]
S_LADDER = [4096, 8192, 12288]


def timeit(fn, iters=6, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def best(fn, params):
    res = {}
    for prm in params:
        try:
            res[prm] = timeit(lambda prm=prm: fn(prm))
        except RuntimeError:
            pass
    if not res:
        return None, float("inf")
    k = min(res, key=res.get)
    return k, res[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()
    if not (torch.cuda.is_available() and cg.cuda_gather_available()):
        print("needs CUDA + built extension")
        return 1
    ext = cg._load_ext()
    if not hasattr(ext, "gather_fwd_wpo_hy_cuda"):
        print("no gather_fwd_wpo_hy_cuda in extension -- stale build?")
        return 1
    dev = torch.device("cuda")
    M = a.batch
    print(f"GPU {torch.cuda.get_device_name(0)}  batch={M}\n")

    print("=" * 78)
    print("[1] EQUIVALENCE -- must be bit-identical to the staged kernel")
    print("=" * 78)
    ok = True
    for in_f, out_f, K in SHAPES_K40[:3] + SHAPES_K10[:1]:
        idx = torch.randint(0, in_f, (out_f, K), dtype=torch.int32, device=dev)
        w = (torch.randn(out_f, K, device=dev) * 0.05).contiguous()
        x = torch.randn(128, in_f, device=dev).abs().contiguous()
        ref = ext.gather_fwd_wpo_cuda(x, idx, w, 2048, 512)
        for S in S_LADDER:
            hy = ext.gather_fwd_wpo_hy_cuda(x, idx, w, 2048, 512, S)
            same = torch.equal(ref, hy)
            ok &= same
            if not same:
                print(f"  in={in_f} out={out_f} K={K} S={S}: DIFFERS "
                      f"max|d|={(ref-hy).abs().max().item():.2e}")
        del idx, w, x, ref
        torch.cuda.empty_cache()
    print("  all shapes x all S: BIT-IDENTICAL" if ok else "  FAIL")

    print("\n" + "=" * 78)
    print("[2] SWEEP -- hybrid (best S,P) vs best staged vs best un-staged")
    print("=" * 78)
    print(f"{'in':>7}{'out':>8}{'K':>4}  {'staged':>10}{'no-stage':>10}"
          f"{'hybrid':>10}{'  hy best':>12}{'vs best inc':>12}")
    for in_f, out_f, K in SHAPES_K40 + SHAPES_K10:
        idx = torch.randint(0, in_f, (out_f, K), dtype=torch.int32, device=dev)
        w = (torch.randn(out_f, K, device=dev) * 0.05).contiguous()
        x = torch.randn(M, in_f, device=dev).abs().contiguous()
        _, ts = best(lambda P: ext.gather_fwd_wpo_cuda(x, idx, w, P, 512),
                     [4096, 2048, 1024])
        _, tn = best(lambda P: ext.gather_fwd_wpo_ns_cuda(x, idx, w, P, 512),
                     [2048, 512, 128, 64, 32])
        hp, th = best(lambda sp: ext.gather_fwd_wpo_hy_cuda(x, idx, w, sp[1], 512, sp[0]),
                      [(S, P) for S in S_LADDER for P in P_LADDER])
        inc = min(ts, tn)
        print(f"{in_f:>7,}{out_f:>8,}{K:>4}  {ts:>9.2f} {tn:>9.2f} {th:>9.2f} "
              f"{str(hp):>12}{inc/th:>11.2f}x")
        del idx, w, x
        torch.cuda.empty_cache()

    print("\n  >1.03x over the best incumbent earns a picker band; else the")
    print("  hybrid is recorded as a measured negative and not shipped.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
