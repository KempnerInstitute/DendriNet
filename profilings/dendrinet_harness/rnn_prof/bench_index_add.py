"""Isolate the fused-gather backward's grad_x scatter: index_add_ under contention.

The fused path computes grad_x with
    grad_x.index_add_(1, idx_chunk, grad_sel)
which lowers to at::native::indexFuncLargeIndex -> one atomicAdd per synapse into
a tiny [M, in_features] tensor. At batch 1 that is ~6.4M atomics landing on 4000
addresses. This benchmark runs exactly that op, at the real layer shapes, in bf16
vs fp32, so the arch-dependence can be attributed to the atomic dtype rather than
to anything about the gather itself.
"""
import sys
import time

import torch

# (in_features, out_features, K) for the 10 IndexedSparseLinear layers of the
# 5k/5k dendritic-RNN config, biggest fan-in first.
SHAPES = [
    (4000, 100000, 64),
    (4000, 20000, 240),
    (4000, 4000, 1168),
    (4000, 1000, 3968),
    (1000, 100000, 16),
    (1000, 1000, 1000),
]


def bench(in_features, out_features, K, dtype, batch=1, iters=5):
    dev = torch.device("cuda")
    nnz = out_features * K
    idx = torch.randint(0, in_features, (nnz,), device=dev, dtype=torch.long)
    src = torch.randn(batch, nnz, device=dev, dtype=dtype)
    dst = torch.zeros(batch, in_features, device=dev, dtype=dtype)

    for _ in range(2):
        dst.index_add_(1, idx, src)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        dst.index_add_(1, idx, src)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    if not torch.cuda.is_available():
        print("CUDA required")
        return 1
    gpu = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {gpu}  sm_{cap[0]}{cap[1]}  torch {torch.__version__}")
    print(f"{'in':>6} {'out':>7} {'K':>5} {'nnz':>10} {'bf16 ms':>10} "
          f"{'fp32 ms':>10} {'bf16/fp32':>10}")
    tot = {"bf16": 0.0, "fp32": 0.0}
    for in_f, out_f, K in SHAPES:
        r = {}
        for name, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
            try:
                r[name] = bench(in_f, out_f, K, dt)
                tot[name] += r[name]
            except RuntimeError as e:
                r[name] = float("nan")
                print(f"  {name} failed: {str(e).splitlines()[0]}")
        ratio = r["bf16"] / r["fp32"] if r["fp32"] else float("nan")
        print(f"{in_f:>6} {out_f:>7} {K:>5} {out_f * K:>10} "
              f"{r['bf16']:>10.2f} {r['fp32']:>10.2f} {ratio:>10.1f}x")
    print(f"{'TOTAL':>21} {'':>10} {tot['bf16']:>10.2f} {tot['fp32']:>10.2f} "
          f"{tot['bf16'] / tot['fp32']:>10.1f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
