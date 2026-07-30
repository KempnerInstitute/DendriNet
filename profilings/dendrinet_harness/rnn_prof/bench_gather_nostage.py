"""Staged vs un-staged forward gather: does dropping shared memory win?

ncu says the staged forward is occupancy-capped -- 33.1% achieved occupancy,
1 block/SM, DRAM 1.11% of peak, SM 35.3%, and a 99.44% L2 hit rate on the very
row the staging is protecting. The un-staged variant drops shared entirely and
lets L2 serve the gathers: ~5x more L2 requests in exchange for full occupancy.

Which wins is not decidable from the counters, so this measures it. Both
variants get a full P sweep, because the optimal tile is not the same once
shared memory stops constraining residency -- comparing a swept kernel against
a fixed-P one would rig the result.

  [1] equivalence -- the two kernels accumulate in identical order, so they
      should be BIT-identical, not merely close. Asserts the strong property.
  [2] P sweep     -- best-of-each per shape, weighted by the stress config mix.
  [3] ceiling     -- a shape the staged kernel cannot run at all (IN*4 > optin).

    python bench_gather_nostage.py [--batch 512]
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    cuda_gather as cg,
)

# (in_features, out_features, K, layers of this shape in the stress config)
SHAPES = [(16000, 144000, 40, 11), (16000, 48000, 40, 11),
          (4000, 144000, 10, 11), (16000, 2000, 40, 11), (3072, 144000, 40, 1)]
P_LADDER = [4096, 2048, 1024, 512, 384, 256, 128, 64, 32]


def timeit(fn, iters=8, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def make(in_f, out_f, K, M, dev):
    idx = torch.randint(0, in_f, (out_f, K), device=dev, dtype=torch.int32).contiguous()
    w = (torch.randn(out_f, K, device=dev) * 0.05).contiguous()
    x = torch.randn(M, in_f, device=dev).abs().contiguous()
    return x, idx, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()
    if not (torch.cuda.is_available() and cg.cuda_gather_available()):
        print("needs CUDA + built extension")
        return 1
    ext = cg._load_ext()
    if not hasattr(ext, "gather_fwd_wpo_ns_cuda"):
        print("extension has no gather_fwd_wpo_ns_cuda -- stale build cache.")
        return 1
    dev = torch.device("cuda")
    M = a.batch
    props = torch.cuda.get_device_properties(0)
    optin = cg._optin_shared()
    print(f"GPU {props.name}  SMs={props.multi_processor_count}  "
          f"optin shared={optin:,} B  batch={M}\n")

    # ---------------- [1] equivalence -------------------------------------- #
    print("=" * 78)
    print("[1] EQUIVALENCE -- same accumulation order, so expect bit-identical")
    print("=" * 78)
    ok = True
    for in_f, out_f, K, _ in SHAPES[:4]:
        x, idx, w = make(in_f, out_f, K, 128, dev)
        s = ext.gather_fwd_wpo_cuda(x, idx, w, 2048, 512)
        n = ext.gather_fwd_wpo_ns_cuda(x, idx, w, 2048, 512)
        same = torch.equal(s, n)
        ok &= same
        print(f"  in={in_f:>6,} out={out_f:>7,} K={K:>3}   max|d| "
              f"{(s - n).abs().max().item():.3e}   "
              f"{'BIT-IDENTICAL' if same else 'DIFFERS'}")
        del x, idx, w, s, n
        torch.cuda.empty_cache()

    # ---------------- [2] P sweep ------------------------------------------ #
    print("\n" + "=" * 78)
    print("[2] P SWEEP -- both variants swept, best of each")
    print("=" * 78)
    print(f"{'in':>7}{'out':>8}{'K':>4}   {'staged best':>20}   {'no-stage best':>20}"
          f"{'speedup':>9}")
    tot_s = tot_n = 0.0
    for in_f, out_f, K, count in SHAPES:
        x, idx, w = make(in_f, out_f, K, M, dev)
        res_s, res_n = {}, {}
        for P in P_LADDER:
            if in_f * 4 <= optin:
                try:
                    res_s[P] = timeit(lambda P=P: ext.gather_fwd_wpo_cuda(x, idx, w, P, 512))
                except RuntimeError:
                    pass
            try:
                res_n[P] = timeit(lambda P=P: ext.gather_fwd_wpo_ns_cuda(x, idx, w, P, 512))
            except RuntimeError:
                pass
        bs = min(res_s, key=res_s.get) if res_s else None
        bn = min(res_n, key=res_n.get) if res_n else None
        ts = res_s[bs] if bs else float("nan")
        tn = res_n[bn] if bn else float("nan")
        tot_s += (ts if bs else 0) * count
        tot_n += (tn if bn else 0) * count
        print(f"{in_f:>7,}{out_f:>8,}{K:>4}   "
              f"{('P=%-5d %7.2f ms' % (bs, ts)) if bs else 'cannot run':>20}   "
              f"{('P=%-5d %7.2f ms' % (bn, tn)) if bn else 'cannot run':>20}"
              f"{(ts / tn if bs and bn else float('nan')):>8.2f}x")
        del x, idx, w
        torch.cuda.empty_cache()

    if tot_n:
        print(f"\n  weighted by the stress config's layer mix:")
        print(f"    staged   : {tot_s:8.1f} ms per forward")
        print(f"    no-stage : {tot_n:8.1f} ms per forward   -> {tot_s/tot_n:.2f}x")

    # ---------------- [3] ceiling ------------------------------------------ #
    big = optin // 4 + 8000          # deliberately past the staged kernel's limit
    print("\n" + "=" * 78)
    print(f"[3] CEILING -- in_features={big:,} needs {big*4:,} B shared "
          f"> optin {optin:,}")
    print("=" * 78)
    x, idx, w = make(big, 8000, 40, 64, dev)
    try:
        ext.gather_fwd_wpo_cuda(x, idx, w, 2048, 512)
        torch.cuda.synchronize()
        print("  staged  : ran (unexpected)")
    except RuntimeError as e:
        print(f"  staged  : FAILS -- {str(e).splitlines()[0][:70]}")
    t = timeit(lambda: ext.gather_fwd_wpo_ns_cuda(x, idx, w, 512, 512), iters=4, warmup=2)
    print(f"  no-stage: runs, {t:.2f} ms  <- removes the in_features ceiling")

    print("\nPASS" if ok else "\nFAIL (equivalence)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
