"""Is the CUDA gather's batch-1 recurrent slowdown just a mis-tuned tile?

`_pick_wpo_params` returns a hard-coded P = 2048 outputs per block, and the
forward launches `grid = (ceil(O/P), M)`. That was tuned on feedforward layers
where O is 10^4-10^5. The dendritic RNN's proximal/somatic layers have
O = 1000-4000 with K = 1168-3968, so at batch 1 the grid collapses to 1-2 blocks
on a 100+ SM GPU while each warp serially walks K.

This sweeps P per layer shape to find the best tile, and reports it against the
shipped P=2048 and against the eager loop path. If a smaller P closes the gap,
the CUDA path is recoverable for RNN configs by making the tile adaptive rather
than by writing a new kernel.
"""
import sys
import time

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    cuda_gather as cg,
)

SHAPES = [
    (4000, 100000, 64), (1000, 100000, 16), (4000, 20000, 240),
    (1000, 20000, 60), (4000, 4000, 1168), (1000, 4000, 300),
    (4000, 1000, 3968), (1000, 1000, 1000), (64, 4000, 32), (64, 1000, 32),
]
P_GRID = [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
T_GRID = [512, 256, 128]


def timeit(fn, iters=8, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def loop_gather(x, idx, w, chunk=65536):
    out = x.new_empty(x.shape[0], idx.shape[0])
    for s in range(0, idx.shape[0], chunk):
        e = min(s + chunk, idx.shape[0])
        sel = x[:, idx[s:e].reshape(-1)].reshape(x.shape[0], e - s, idx.shape[1])
        out[:, s:e] = (sel * w[s:e].unsqueeze(0)).sum(dim=-1)
    return out


def main():
    if not (torch.cuda.is_available() and cg.cuda_gather_available()):
        print("needs CUDA + built extension "
              "(source profilings/weight_transform_opt/setup_cuda_env.sh)")
        return 1
    ext = cg._load_ext()
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  SMs={props.multi_processor_count}  "
          f"shipped P={cg._pick_wpo_params()[0]}")
    print(f"\n{'in':>6} {'out':>7} {'K':>5} {'blocks@2048':>12} {'loop ms':>9} "
          f"{'P=2048':>9} {'best ms':>9} {'best P':>7} {'best T':>7} {'gain':>7}")

    tot_loop = tot_ship = tot_best = 0.0
    best_by_shape = {}
    for in_f, out_f, K in SHAPES:
        if not cg._wpo_ok(in_f, K):
            print(f"{in_f:>6} {out_f:>7} {K:>5}   (wpo not applicable, v2 fallback)")
            continue
        idx = torch.stack(
            [torch.randperm(in_f, device=dev)[:K] for _ in range(out_f)]
        ).to(torch.int32).contiguous()
        x = torch.randn(1, in_f, device=dev).abs().contiguous()
        w = (torch.randn(out_f, K, device=dev) * 0.05).contiguous()

        t_loop = timeit(lambda: loop_gather(
            x.to(torch.bfloat16), idx.long(), w.to(torch.bfloat16)))
        ship_P, ship_T = cg._pick_wpo_params()
        t_ship = timeit(lambda: ext.gather_fwd_wpo_cuda(x, idx, w, ship_P, ship_T))

        best = (float("inf"), None, None)
        for P in P_GRID:
            for T in T_GRID:
                try:
                    t = timeit(lambda: ext.gather_fwd_wpo_cuda(x, idx, w, P, T),
                               iters=5, warmup=2)
                except RuntimeError:
                    continue
                if t < best[0]:
                    best = (t, P, T)
        blocks = -(-out_f // ship_P)
        tot_loop += t_loop
        tot_ship += t_ship
        tot_best += best[0]
        best_by_shape[(in_f, out_f, K)] = (best[1], best[2])
        print(f"{in_f:>6} {out_f:>7} {K:>5} {blocks:>12} {t_loop:>9.3f} "
              f"{t_ship:>9.3f} {best[0]:>9.3f} {best[1]:>7} {best[2]:>7} "
              f"{t_ship / best[0]:>6.1f}x")
        del idx, x, w
        torch.cuda.empty_cache()

    print(f"\n{'TOTAL/timestep':>26} {tot_loop:>9.3f} {tot_ship:>9.3f} {tot_best:>9.3f}")
    print(f"  shipped P=2048 vs loop : {tot_ship / tot_loop:5.2f}x  "
          f"({'SLOWER' if tot_ship > tot_loop else 'faster'})")
    print(f"  best tile      vs loop : {tot_best / tot_loop:5.2f}x  "
          f"({'SLOWER' if tot_best > tot_loop else 'faster'})")
    print(f"  tile tuning alone recovers {tot_ship / tot_best:.1f}x")
    print("\n  best (P, T) per shape:")
    for k, v in best_by_shape.items():
        print(f"    in={k[0]:>5} out={k[1]:>7} K={k[2]:>5} -> P={v[0]:>5} T={v[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
