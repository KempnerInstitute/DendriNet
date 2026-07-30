"""Sweep the wpo tile (P) at the real feedforward layer shapes, fwd or bwd.

`_pick_bwd_wpo_params()` returns a hard-coded P = 2048 outputs per block, and the
kernel launches `grid = ceil(O/P)`. At the 12-layer stress config's shapes that
is 71 blocks for out=144,000 and 24 for out=48,000 -- on a 188-SM Blackwell,
38% and 13% of the machine. Nsight confirms it: 33.1% achieved occupancy against
100% theoretical, with DRAM at 4.2% and SM at 6.6%. The kernel is not bandwidth-
or compute-bound; most of the GPU is idle.

Lower P means more blocks (better fill) but more per-block staging overhead, so
there is a crossover. This measures it rather than assuming a formula: for each
real shape it times the kernel across a P ladder and reports the optimum against
the shipped value.

Run against the already-built extension -- no recompile.
"""
import argparse
import sys
import time

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    cuda_gather as cg,
)

# (in_features, out_features, K, how many such layers in the config)
SHAPES = [
    (16000, 144000, 40, 11), (3072, 144000, 40, 1),
    (16000, 48000, 40, 11), (3072, 48000, 40, 1),
    (4000, 144000, 10, 11), (4000, 48000, 10, 11),
    (16000, 2000, 40, 11),
]
P_LADDER = [2048, 1024, 512, 384, 256, 192, 128, 96, 64, 32]
BATCH = 512


def timeit(fn, iters=6, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    if not (torch.cuda.is_available() and cg.cuda_gather_available()):
        print("needs CUDA + built extension "
              "(source profilings/weight_transform_opt/setup_cuda_env.sh)")
        return 1
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["fwd", "bwd"], default="bwd")
    args = ap.parse_args()
    ext = cg._load_ext()
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    SM = props.multi_processor_count
    shipP, shipT = (cg._pick_wpo_params() if args.phase == "fwd"
                    else cg._pick_bwd_wpo_params())
    print(f"GPU {props.name}  SMs={SM}  phase={args.phase.upper()}  "
          f"shipped P={shipP} T={shipT}  batch={BATCH}\n")

    weighted_ship = weighted_best = 0.0
    per_shape = {}
    for in_f, out_f, K, count in SHAPES:
        idx = torch.stack(
            [torch.randperm(in_f, device=dev)[:K] for _ in range(out_f)]
        ).to(torch.int32).contiguous()
        gof = torch.randn(BATCH, out_f, device=dev).contiguous()
        goT = gof.t().contiguous()
        xT = torch.randn(in_f, BATCH, device=dev).abs().contiguous()
        wf = (torch.randn(out_f, K, device=dev) * 0.05).contiguous()

        xf = xT.t().contiguous()          # [M, IN] for the forward kernel
        res = {}
        for P in P_LADDER:
            try:
                if args.phase == "fwd":
                    res[P] = timeit(lambda P=P: ext.gather_fwd_wpo_cuda(xf, idx, wf, P, shipT))
                else:
                    res[P] = timeit(
                        lambda P=P: ext.gather_bwd_wpo_cuda(gof, goT, xT, idx, wf, P, shipT)
                    )
            except RuntimeError:
                continue
        if not res:
            continue
        bestP = min(res, key=res.get)
        ship = res.get(shipP, float("nan"))
        per_shape[(in_f, out_f, K)] = (bestP, ship, res[bestP], count)
        weighted_ship += ship * count
        weighted_best += res[bestP] * count
        blocks_ship, blocks_best = -(-out_f // shipP), -(-out_f // bestP)
        print(f"in={in_f:>6,} out={out_f:>7,} K={K:>4}  "
              f"P={shipP} {ship:7.2f} ms ({blocks_ship:>4} blk, {100*min(blocks_ship,SM)/SM:>3.0f}% SM)"
              f"   ->  best P={bestP:<5} {res[bestP]:7.2f} ms "
              f"({blocks_best:>4} blk, {100*min(blocks_best,SM)/SM:>3.0f}% SM)"
              f"   {ship/res[bestP]:5.2f}x")
        del idx, gof, goT, xT, wf, xf
        torch.cuda.empty_cache()

    if weighted_best:
        print(f"\nWeighted by how many layers of each shape the config actually has:")
        print(f"  shipped P={shipP} : {weighted_ship:8.1f} ms per {args.phase}")
        print(f"  per-shape best   : {weighted_best:8.1f} ms per {args.phase}"
              f"   -> {weighted_ship/weighted_best:.2f}x")
        print("\n  suggested adaptive rule, vs measured optimum:")
        for (i, o, k), (bestP, _, _, _) in per_shape.items():
            rule = max(32, min(2048, 1 << max(0, (o // (2 * SM)).bit_length() - 1)))
            print(f"    out={o:>7,}  measured {bestP:>5}   rule ceil(O/2*SM)->pow2 {rule:>5}"
                  f"   {'match' if rule == bestP else 'differs'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
