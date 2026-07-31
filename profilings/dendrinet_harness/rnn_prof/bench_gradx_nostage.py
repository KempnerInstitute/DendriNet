"""grad_x: shared staging + flush vs plain global atomics. Which wins?

ncu splits the backward: gather_gradx_stage 21.01 ms, gather_gradw_transpose
3.16 ms -- grad_x is 87% of it and is capped at 1 block/SM by its IN-float shared
row (registers would allow 4).

Two hypotheses were already killed by measurement, so this one gets tested and
not assumed:
  * "the row flush is the cost" -- FALSE. Raising P cuts the flush 100x and makes
    it 17x SLOWER (bench_gradx_tile.py). Parallelism dominates; the shipped
    picker is already optimal at 1.00x.
  * "the forward fix transfers" -- untested, and NOT obvious: the staged kernel
    exists precisely because the v2 backward's global grad_x atomics were slow.

Un-staged trades shared atomics + flush for global atomics at full occupancy.
Both P ladders are swept, since the optimum moves when shared stops constraining
residency.

Note grad_x accumulation ORDER changes between the two, so gradients agree to
float tolerance, not bitwise -- unlike the forward, where the fix was exact.

    python bench_gradx_nostage.py [--batch 512]
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    cuda_gather as cg,
)

SHAPES = [(16000, 144000, 40, 11), (16000, 48000, 40, 11),
          (4000, 144000, 10, 11), (16000, 2000, 40, 11), (3072, 144000, 40, 1)]
P_LADDER = [2048, 1024, 512, 384, 256, 128, 96, 64, 32]


def timeit(fn, iters=6, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def best(fn):
    res = {}
    for P in P_LADDER:
        try:
            res[P] = timeit(lambda P=P: fn(P))
        except RuntimeError:
            pass
    if not res:
        return None, float("inf")
    p = min(res, key=res.get)
    return p, res[p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()
    if not (torch.cuda.is_available() and cg.cuda_gather_available()):
        print("needs CUDA + built extension")
        return 1
    ext = cg._load_ext()
    if not hasattr(ext, "gather_bwd_wpo_cuda2"):
        print("extension has no gather_bwd_wpo_cuda2 -- stale build cache.")
        return 1
    dev = torch.device("cuda")
    M = a.batch
    print(f"GPU {torch.cuda.get_device_name(0)}  batch={M}\n")

    print("=" * 80)
    print("[1] AGREEMENT -- accumulation order differs, so tolerance not bitwise")
    print("=" * 80)
    for in_f, out_f, K, _ in SHAPES[:3]:
        idx = torch.randint(0, in_f, (out_f, K), device=dev, dtype=torch.int32).contiguous()
        w = (torch.randn(out_f, K, device=dev) * 0.05).contiguous()
        x = torch.randn(128, in_f, device=dev).abs().contiguous()
        gof = torch.randn(128, out_f, device=dev).contiguous()
        goT, xT = gof.t().contiguous(), x.t().contiguous()
        gx0, gw0 = ext.gather_bwd_wpo_cuda2(gof, goT, xT, idx, w, 512, 512, 0)
        gx1, gw1 = ext.gather_bwd_wpo_cuda2(gof, goT, xT, idx, w, 512, 512, 1)
        rx = (gx0 - gx1).abs().max().item() / (gx0.abs().max().item() or 1)
        rw = (gw0 - gw1).abs().max().item() / (gw0.abs().max().item() or 1)
        print(f"  in={in_f:>6,} out={out_f:>7,}  grad_x rel {rx:.2e}  "
              f"grad_w rel {rw:.2e}  {'OK' if rx < 1e-5 and rw < 1e-6 else 'CHECK'}")
        del idx, w, x, gof, goT, xT, gx0, gw0, gx1, gw1
        torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("[2] SPEED -- both variants P-swept, best of each")
    print("=" * 80)
    print(f"{'in':>7}{'out':>8}{'K':>4}   {'staged best':>19}   {'no-stage best':>19}"
          f"{'speedup':>9}")
    tot_s = tot_n = 0.0
    for in_f, out_f, K, count in SHAPES:
        idx = torch.randint(0, in_f, (out_f, K), device=dev, dtype=torch.int32).contiguous()
        w = (torch.randn(out_f, K, device=dev) * 0.05).contiguous()
        x = torch.randn(M, in_f, device=dev).abs().contiguous()
        gof = torch.randn(M, out_f, device=dev).contiguous()
        goT, xT = gof.t().contiguous(), x.t().contiguous()
        ps, ts = best(lambda P: ext.gather_bwd_wpo_cuda2(gof, goT, xT, idx, w, P, 512, 0))
        pn, tn = best(lambda P: ext.gather_bwd_wpo_cuda2(gof, goT, xT, idx, w, P, 512, 1))
        tot_s += ts * count
        tot_n += tn * count
        print(f"{in_f:>7,}{out_f:>8,}{K:>4}   "
              f"{('P=%-4d %8.2f ms' % (ps, ts)):>19}   "
              f"{('P=%-4d %8.2f ms' % (pn, tn)):>19}{ts/tn:>8.2f}x")
        del idx, w, x, gof, goT, xT
        torch.cuda.empty_cache()

    print(f"\n  weighted by the stress config's layer mix:")
    print(f"    staged   : {tot_s:8.1f} ms per backward")
    print(f"    no-stage : {tot_n:8.1f} ms per backward   -> {tot_s/tot_n:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
