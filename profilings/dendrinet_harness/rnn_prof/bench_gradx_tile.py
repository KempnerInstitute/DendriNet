"""The grad_x backward's tile, swept UPWARD -- the flush is quadratic in block count.

ncu splits the backward cleanly: gather_gradx_stage 21.01 ms vs
gather_gradw_transpose 3.16 ms. grad_x is 87% of it, and it is occupancy-capped
at 1 block/SM by the IN-float shared row (registers would allow 4).

But occupancy is not the only thing P controls here. Each block stages a FULL
grad_x row, then flushes all IN entries back with GLOBAL atomics -- and there are
ceil(O/P) blocks per m, every one of them flushing the whole row:

    global atomics = ceil(O/P) * IN * M

At P=1024, O=144,000, IN=16,000, M=512 that is 1.16 BILLION atomic adds, just to
merge partial rows. Raising P cuts that linearly. The existing picker tops out at
P=2048 because the ladder it was fitted on topped out there -- the same
bracketing mistake that hid P=4096 in the forward.

So this sweeps P upward to O itself, and reports the flush volume alongside the
time so the mechanism is visible rather than inferred.

    python bench_gradx_tile.py [--batch 512]
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
          (4000, 144000, 10, 11), (16000, 2000, 40, 11)]
P_LADDER = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 147456]


def timeit(fn, iters=6, warmup=2):
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
    ext = cg._load_ext()
    dev = torch.device("cuda")
    M = a.batch
    SM = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"GPU {torch.cuda.get_device_name(0)}  SMs={SM}  batch={M}\n")

    tot_ship = tot_best = 0.0
    picks = {}
    for in_f, out_f, K, count in SHAPES:
        idx = torch.randint(0, in_f, (out_f, K), device=dev,
                            dtype=torch.int32).contiguous()
        w = (torch.randn(out_f, K, device=dev) * 0.05).contiguous()
        x = torch.randn(M, in_f, device=dev).abs().contiguous()
        gof = torch.randn(M, out_f, device=dev).contiguous()
        goT = gof.t().contiguous()
        xT = x.t().contiguous()

        shipP = cg._pick_bwd_wpo_params(out_f)[0]
        print(f"  in={in_f:,}  out={out_f:,}  K={K}   (shipped P={shipP})")
        print(f"{'P':>9}{'blocks/m':>10}{'flush atomics':>16}{'ms':>9}{'vs ship':>9}")
        res = {}
        for P in P_LADDER:
            if P > out_f * 2:
                continue
            try:
                res[P] = timeit(lambda P=P: ext.gather_bwd_wpo_cuda(
                    gof, goT, xT, idx, w, P, 512))
            except RuntimeError:
                continue
        ship = res.get(shipP)
        for P in sorted(res):
            blocks = -(-out_f // P)
            flush = blocks * in_f * M
            mark = "  <- shipped" if P == shipP else ""
            print(f"{P:>9}{blocks:>10}{flush/1e9:>14.2f}G{res[P]:>9.2f}"
                  f"{(ship / res[P]) if ship else float('nan'):>8.2f}x{mark}")
        bestP = min(res, key=res.get)
        picks[out_f] = bestP
        tot_ship += (ship or res[bestP]) * count
        tot_best += res[bestP] * count
        print(f"    best P={bestP} -> {res[bestP]:.2f} ms "
              f"({(ship / res[bestP]) if ship else float('nan'):.2f}x over shipped)\n")
        del idx, w, x, gof, goT, xT
        torch.cuda.empty_cache()

    print(f"weighted by the stress config's layer mix:")
    print(f"  shipped picker : {tot_ship:8.1f} ms per backward")
    print(f"  per-shape best : {tot_best:8.1f} ms per backward  "
          f"-> {tot_ship/tot_best:.2f}x")
    print(f"\n  measured optima: {picks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
