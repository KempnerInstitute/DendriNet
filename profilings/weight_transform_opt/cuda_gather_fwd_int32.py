"""Forward-kernel experiment: int32 connection indices (+ optional L1 carveout).

MOTIVATION (from the ncu_full.sh L2 profile of the production v2 forward):
  * L1 hit rate 5.4%, L2 hit rate 99.5%, DRAM idle -> we are L2-THROUGHPUT-bound.
  * ~28.8 sectors/request on the global load (max 32, coalesced floor 4) -> the x-gather
    is essentially fully uncoalesced; each 4-byte element drags its own 32-byte sector.
  * Data is L2-resident and reused ~360x per element, but L1 catches only 5% of that reuse.

This script tests the CHEAP, low-risk lever first: stage `idx` as int32 instead of int64.
  * Halves the idx shared-memory footprint (B_O*K*8 -> B_O*K*4 bytes).
  * Frees shared -> more room for L1 (on Blackwell L1/shared share one pool), and lets us
    push the L1/shared carveout toward L1 (--carveout).
  * Halves the idx staging traffic (a small slice of total L2 traffic).

HONEST PRIOR: idx staging is only ~1-2% of forward L2 traffic, and x (5 MB) can't fit in a
~128 KB L1 no matter how we carve it -- so I expect a SMALL forward win. The real value here
is DIAGNOSTIC: if L1 hit rate barely moves, that proves L1-capacity is not the lever and the
next step is structural (stage x itself in shared). If it moves a lot, even better.

Runs BOTH kernels in one process for a clean A/B, checks each against an eager reference, and
times them. Distinct kernel names (gather_fwd_i64_kernel / gather_fwd_i32_kernel) so you can
also profile them directly:

    ncu --metrics l1tex__t_sector_hit_rate.pct,lts__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
        -k "regex:gather_fwd_i(32|64)_kernel" --launch-skip 10 --launch-count 2 \
        python cuda_gather_fwd_int32.py

Usage:  python cuda_gather_fwd_int32.py [--carveout 0..100] [--M ..] [--IN ..] [--O ..] [--K ..]
"""
from __future__ import annotations

import argparse
import torch
from torch.utils.cpp_extension import load_inline

_CPP_SRC = (
    "torch::Tensor fwd_i64(torch::Tensor x, torch::Tensor idx, torch::Tensor w, int B_O, int B_M);\n"
    "torch::Tensor fwd_i32(torch::Tensor x, torch::Tensor idx, torch::Tensor w, int B_O, int B_M, int carveout);"
)

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ===== baseline: production v2 forward, int64 idx (copy of cuda_gather.py) =====
__global__ void gather_fwd_i64_kernel(
        const float* __restrict__ x, const long* __restrict__ idx,
        const float* __restrict__ w, float* __restrict__ out,
        int M, int O, int IN, int K, int B_O, int B_M) {
    extern __shared__ char smem[];
    long*  s_idx = reinterpret_cast<long*>(smem);
    float* s_w   = reinterpret_cast<float*>(smem + (size_t)B_O * K * sizeof(long));
    int o_base = blockIdx.x * B_O, m_base = blockIdx.y * B_M;
    int tx = threadIdx.x, ty = threadIdx.y, T = B_M * B_O, tid = ty * B_O + tx, tile = B_O * K;
    for (int i = tid; i < tile; i += T) {
        int oo = o_base + i / K, kk = i % K;
        if (oo < O) { s_idx[i] = idx[(long)oo * K + kk]; s_w[i] = w[(long)oo * K + kk]; }
    }
    __syncthreads();
    int o = o_base + tx, m = m_base + ty;
    if (o < O && m < M) {
        const float* x_m = x + (long)m * IN;
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += x_m[s_idx[tx * K + k]] * s_w[tx * K + k];
        out[(long)m * O + o] = acc;
    }
}

// ===== variant: int32 idx (half the idx shared footprint) =====================
__global__ void gather_fwd_i32_kernel(
        const float* __restrict__ x, const int* __restrict__ idx,
        const float* __restrict__ w, float* __restrict__ out,
        int M, int O, int IN, int K, int B_O, int B_M) {
    extern __shared__ char smem[];
    int*   s_idx = reinterpret_cast<int*>(smem);
    float* s_w   = reinterpret_cast<float*>(smem + (size_t)B_O * K * sizeof(int));
    int o_base = blockIdx.x * B_O, m_base = blockIdx.y * B_M;
    int tx = threadIdx.x, ty = threadIdx.y, T = B_M * B_O, tid = ty * B_O + tx, tile = B_O * K;
    for (int i = tid; i < tile; i += T) {
        int oo = o_base + i / K, kk = i % K;
        if (oo < O) { s_idx[i] = idx[(long)oo * K + kk]; s_w[i] = w[(long)oo * K + kk]; }
    }
    __syncthreads();
    int o = o_base + tx, m = m_base + ty;
    if (o < O && m < M) {
        const float* x_m = x + (long)m * IN;
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += x_m[s_idx[tx * K + k]] * s_w[tx * K + k];
        out[(long)m * O + o] = acc;
    }
}

torch::Tensor fwd_i64(torch::Tensor x, torch::Tensor idx, torch::Tensor w, int B_O, int B_M) {
    int M = x.size(0), IN = x.size(1), O = idx.size(0), K = idx.size(1);
    auto out = torch::empty({M, O}, x.options());
    dim3 block(B_O, B_M), grid((O + B_O - 1) / B_O, (M + B_M - 1) / B_M);
    size_t shmem = (size_t)B_O * K * (sizeof(long) + sizeof(float));
    TORCH_CHECK(shmem <= 48 * 1024, "i64 tile > 48KB; lower B_O");
    gather_fwd_i64_kernel<<<grid, block, shmem>>>(
        x.data_ptr<float>(), idx.data_ptr<long>(), w.data_ptr<float>(),
        out.data_ptr<float>(), M, O, IN, K, B_O, B_M);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fwd_i64 launch failed");
    return out;
}

torch::Tensor fwd_i32(torch::Tensor x, torch::Tensor idx, torch::Tensor w, int B_O, int B_M, int carveout) {
    int M = x.size(0), IN = x.size(1), O = idx.size(0), K = idx.size(1);
    auto out = torch::empty({M, O}, x.options());
    dim3 block(B_O, B_M), grid((O + B_O - 1) / B_O, (M + B_M - 1) / B_M);
    size_t shmem = (size_t)B_O * K * (sizeof(int) + sizeof(float));
    TORCH_CHECK(shmem <= 48 * 1024, "i32 tile > 48KB; lower B_O");
    // carveout in [0,100] = % of the unified L1/shared pool PREFERRED for shared memory.
    // Low value -> more L1 cache for the scattered x-gather. -1 => leave to the driver.
    if (carveout >= 0)
        cudaFuncSetAttribute(gather_fwd_i32_kernel,
                             cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    gather_fwd_i32_kernel<<<grid, block, shmem>>>(
        x.data_ptr<float>(), idx.data_ptr<int>(), w.data_ptr<float>(),
        out.data_ptr<float>(), M, O, IN, K, B_O, B_M);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fwd_i32 launch failed");
    return out;
}
"""


def _time(fn, iters: int, warmup: int = 15) -> float:
    """Median-ish mean ms per call over `iters`, after `warmup`."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--IN", type=int, default=10000)
    ap.add_argument("--O", type=int, default=90000)
    ap.add_argument("--K", type=int, default=40)
    ap.add_argument("--B_O", type=int, default=64)
    ap.add_argument("--B_M", type=int, default=16)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--carveout", type=int, default=-1,
                    help="0..100 = %% preferred for shared (low -> more L1). -1 = driver default.")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "need CUDA"
    dev = "cuda"
    torch.manual_seed(0)
    M, IN, O, K = args.M, args.IN, args.O, args.K
    x = torch.randn(M, IN, device=dev)
    idx64 = torch.randint(0, IN, (O, K), device=dev, dtype=torch.int64)
    idx32 = idx64.to(torch.int32)
    w = torch.randn(O, K, device=dev)

    print(f"gpu   : {torch.cuda.get_device_name(0)}")
    print(f"shape : M={M} IN={IN} O={O} K={K}  tile B_O={args.B_O} B_M={args.B_M}")
    print(f"idx shared/block : i64 {args.B_O*K*8/1024:.1f}KB  ->  i32 {args.B_O*K*4/1024:.1f}KB")
    print("compiling (first run ~80s)...", flush=True)
    ext = load_inline(name="fwd_int32_bench", cpp_sources=_CPP_SRC, cuda_sources=_CUDA_SRC,
                      functions=["fwd_i64", "fwd_i32"], verbose=False)

    # --- correctness vs eager reference -------------------------------------
    ref = (x[:, idx64] * w.unsqueeze(0)).sum(-1)            # [M,O]
    o64 = ext.fwd_i64(x, idx64, w, args.B_O, args.B_M)
    o32 = ext.fwd_i32(x, idx32, w, args.B_O, args.B_M, args.carveout)
    e64 = (o64 - ref).abs().max().item()
    e32 = (o32 - ref).abs().max().item()
    d = (o32 - o64).abs().max().item()
    print(f"\ncorrectness (max abs err vs eager):  i64 {e64:.2e}   i32 {e32:.2e}   i32-vs-i64 {d:.2e}")
    assert e32 < 1e-2 and d < 1e-3, "int32 variant diverged -- STOP"

    # --- timing A/B ----------------------------------------------------------
    t64 = _time(lambda: ext.fwd_i64(x, idx64, w, args.B_O, args.B_M), args.iters)
    t32 = _time(lambda: ext.fwd_i32(x, idx32, w, args.B_O, args.B_M, args.carveout), args.iters)
    print(f"\ntiming (ms/call, {args.iters} iters):")
    print(f"  i64 baseline : {t64:.3f}")
    print(f"  i32 variant  : {t32:.3f}   carveout={args.carveout}")
    print(f"  speedup      : {t64/t32:.3f}x  ({(t64/t32 - 1)*100:+.1f}%)")
    print("\nnote: to see WHY, profile L1 hit rate + L2 SOL with the ncu command in the docstring.")


if __name__ == "__main__":
    main()
