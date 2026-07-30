"""Per-call cost of each gather path at batch-1 recurrent shapes, and where it goes.

The recurrent unroll calls each IndexedSparseLinear once per timestep (58x), so
anything the wrapper redoes per call is paid 58x per forward. The cuda/triton
wrappers cast `weight` to fp32 and `connection_indices` to int32 on every call,
even though both are invariant across the whole unroll. This benchmark separates:

    cuda (wrapper)  -- what the model actually pays today
    cuda (kernel)   -- the same kernel with inputs pre-converted once

so the gap is exactly the recoverable per-call conversion overhead.
"""
import sys
import time

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.fused_gather import (
    fused_sparse_gather,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    cuda_gather as cg,
)

# The 10 IndexedSparseLinear layers of the 5k/5k dendritic-RNN config.
SHAPES = [
    (4000, 100000, 64), (1000, 100000, 16), (4000, 20000, 240),
    (1000, 20000, 60), (4000, 4000, 1168), (1000, 4000, 300),
    (4000, 1000, 3968), (1000, 1000, 1000), (64, 4000, 32), (64, 1000, 32),
]
BATCH = 1
T = 58


def timeit(fn, iters=5, warmup=2):
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
    if not torch.cuda.is_available():
        print("CUDA required")
        return 1
    have_cuda_ext = cg.cuda_gather_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
    print(f"cuda_gather_available: {have_cuda_ext}")
    if not have_cuda_ext:
        print("  -> extension not built; source profilings/weight_transform_opt/setup_cuda_env.sh")

    dev = torch.device("cuda")
    hdr = (f"{'in':>6} {'out':>7} {'K':>5} {'loop':>9} {'fused':>9} "
           f"{'cuda wrap':>10} {'cuda kern':>10} {'conv tax':>9}")
    print("\nforward, per call, ms (batch 1)")
    print(hdr)
    tot = dict(loop=0.0, fused=0.0, wrap=0.0, kern=0.0)

    for in_f, out_f, K in SHAPES:
        idx = torch.stack(
            [torch.randperm(in_f, device=dev)[:K] for _ in range(out_f)]
        )
        x = torch.randn(BATCH, in_f, device=dev, dtype=torch.bfloat16).abs()
        w = torch.randn(out_f, K, device=dev, dtype=torch.bfloat16) * 0.05

        r = {}
        r["loop"] = timeit(lambda: loop_gather(x, idx, w))
        r["fused"] = timeit(lambda: fused_sparse_gather(x, idx, w, chunk_size=65536))

        if have_cuda_ext:
            r["wrap"] = timeit(lambda: cg.cuda_sparse_gather(x, idx, w))
            # Same kernel, inputs converted ONCE (what a cached wrapper would pay).
            ext = cg._load_ext()
            xf = x.float().contiguous()
            wf = w.float().contiguous()
            i32 = idx.contiguous().to(torch.int32)
            if cg._wpo_ok(in_f, K):
                P, Tt = cg._pick_wpo_params()
                r["kern"] = timeit(lambda: ext.gather_fwd_wpo_cuda(xf, i32, wf, P, Tt))
            else:
                B_O, B_M = cg._pick_tile()
                i64 = idx.contiguous()
                r["kern"] = timeit(lambda: ext.gather_fwd_cuda(xf, i64, wf, B_O, B_M))
        else:
            r["wrap"] = r["kern"] = float("nan")

        for k in tot:
            tot[k] += r[k]
        tax = r["wrap"] - r["kern"]
        print(f"{in_f:>6} {out_f:>7} {K:>5} {r['loop']:>9.3f} {r['fused']:>9.3f} "
              f"{r['wrap']:>10.3f} {r['kern']:>10.3f} {tax:>9.3f}")
        del idx, x, w
        torch.cuda.empty_cache()

    print(f"{'TOTAL/timestep':>20} {tot['loop']:>9.3f} {tot['fused']:>9.3f} "
          f"{tot['wrap']:>10.3f} {tot['kern']:>10.3f} "
          f"{tot['wrap'] - tot['kern']:>9.3f}")
    print(f"{'x' + str(T) + ' timesteps':>20} {tot['loop'] * T:>9.1f} "
          f"{tot['fused'] * T:>9.1f} {tot['wrap'] * T:>10.1f} "
          f"{tot['kern'] * T:>10.1f} {(tot['wrap'] - tot['kern']) * T:>9.1f}")
    if have_cuda_ext and tot["kern"]:
        print(f"\n  cuda wrapper is {tot['wrap'] / tot['loop']:.1f}x the loop path, "
              f"but the KERNEL alone is {tot['kern'] / tot['loop']:.2f}x")
        print(f"  -> {100 * (tot['wrap'] - tot['kern']) / tot['wrap']:.0f}% of the cuda "
              f"path is per-call dtype conversion of invariant tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
