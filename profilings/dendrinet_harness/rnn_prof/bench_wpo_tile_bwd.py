"""Same tile question for the CUDA gather's BACKWARD kernels.

Backward is ~57% of the batch-1 recurrent step and, unlike the forward, it runs
at ~100% GPU-busy -- so a kernel win there translates straight into wall clock.
`_pick_bwd_wpo_params` returns the same hard-coded P=2048 as the forward, so if
the forward's under-occupancy story holds, backward should show it too.
"""
import sys
import time

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    cuda_gather as cg,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.fused_gather import (
    fused_sparse_gather,
)

SHAPES = [
    (4000, 100000, 64), (1000, 100000, 16), (4000, 20000, 240),
    (1000, 20000, 60), (4000, 4000, 1168), (1000, 4000, 300),
    (4000, 1000, 3968), (1000, 1000, 1000), (64, 4000, 32), (64, 1000, 32),
]
P_GRID = [2048, 512, 128, 64, 32, 8, 4, 2]
T_GRID = [512, 256, 128]


def timeit(fn, iters=6, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def fused_bwd(x, idx, w, go):
    x = x.detach().requires_grad_(True)
    w = w.detach().requires_grad_(True)
    out = fused_sparse_gather(x, idx, w, chunk_size=65536)
    out.backward(go)


def main():
    if not (torch.cuda.is_available() and cg.cuda_gather_available()):
        print("needs CUDA + built extension")
        return 1
    ext = cg._load_ext()
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  SMs={props.multi_processor_count}  "
          f"shipped bwd P={cg._pick_bwd_wpo_params()[0]}")
    print(f"\n{'in':>6} {'out':>7} {'K':>5} {'fused ms':>9} {'P=2048':>9} "
          f"{'best ms':>9} {'best P':>7} {'best T':>7} {'gain':>7}")

    tot_fused = tot_ship = tot_best = 0.0
    best_by_shape = {}
    for in_f, out_f, K in SHAPES:
        if not cg._bwd_wpo_ok(in_f):
            print(f"{in_f:>6} {out_f:>7} {K:>5}   (bwd wpo not applicable)")
            continue
        idx_l = torch.stack(
            [torch.randperm(in_f, device=dev)[:K] for _ in range(out_f)]
        ).contiguous()
        idx = idx_l.to(torch.int32).contiguous()
        xf = torch.randn(1, in_f, device=dev).abs().contiguous()
        wf = (torch.randn(out_f, K, device=dev) * 0.05).contiguous()
        gof = torch.randn(1, out_f, device=dev).contiguous()
        xT = xf.t().contiguous()
        goT = gof.t().contiguous()

        t_fused = timeit(lambda: fused_bwd(
            xf.to(torch.bfloat16), idx_l, wf.to(torch.bfloat16),
            gof.to(torch.bfloat16)))
        sP, sT = cg._pick_bwd_wpo_params()
        t_ship = timeit(
            lambda: ext.gather_bwd_wpo_cuda(gof, goT, xT, idx, wf, sP, sT))

        best = (float("inf"), None, None)
        for P in P_GRID:
            for T in T_GRID:
                try:
                    t = timeit(
                        lambda: ext.gather_bwd_wpo_cuda(gof, goT, xT, idx, wf, P, T),
                        iters=4, warmup=2)
                except RuntimeError:
                    continue
                if t < best[0]:
                    best = (t, P, T)
        tot_fused += t_fused
        tot_ship += t_ship
        tot_best += best[0]
        best_by_shape[(in_f, out_f, K)] = (best[1], best[2])
        print(f"{in_f:>6} {out_f:>7} {K:>5} {t_fused:>9.3f} {t_ship:>9.3f} "
              f"{best[0]:>9.3f} {best[1]:>7} {best[2]:>7} {t_ship / best[0]:>6.1f}x")
        del idx, idx_l, xf, wf, gof, xT, goT
        torch.cuda.empty_cache()

    print(f"\n{'TOTAL/timestep':>26} {tot_fused:>9.3f} {tot_ship:>9.3f} {tot_best:>9.3f}")
    print(f"  shipped P=2048 vs fused : {tot_ship / tot_fused:5.2f}x "
          f"({'SLOWER' if tot_ship > tot_fused else 'faster'})")
    print(f"  best tile      vs fused : {tot_best / tot_fused:5.2f}x "
          f"({'SLOWER' if tot_best > tot_fused else 'faster'})")
    print(f"  tile tuning alone recovers {tot_ship / tot_best:.1f}x")
    print("\n  best (P, T) per shape:")
    for k, v in best_by_shape.items():
        print(f"    in={k[0]:>5} out={k[1]:>7} K={k[2]:>5} -> P={v[0]:>5} T={v[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
