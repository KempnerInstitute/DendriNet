"""Forward experiment v2: stage x in shared, WARP-PER-OUTPUT (fixes the occupancy killer).

WHERE WE ARE (from cuda_gather_fwd_stagex.py + ncu on the Blackwell):
  Staging x[m,:] in shared collapsed L2 SOL 93% -> 17.3% (the L2 wall is GONE) and bank
  conflicts were FINE (239M, below v2's 302M). But it ran 18% slower for ONE reason: staging
  x + idx + w = 99KB shared forced 1 block/SM -> occupancy 15.8% (vs v2's 64%). Latency-bound.

THE FIX HERE: don't stage idx/w. Read them COALESCED from global via a warp-per-output layout:
  * Each WARP owns one output o. Its 32 lanes load idx[o*K + lane] / w[o*K + lane] -- these are
    CONSECUTIVE addresses -> fully coalesced global loads (no shared needed for idx/w).
  * Each lane gathers s_x[idx] from the shared x-row (on-chip, cheap), forms its product(s),
    then the warp __shfl_down-reduces the K=40 partials to out[m,o].
  * Shared = just the x-row = IN*4 = 40KB  (< 48KB default -> no opt-in, and ~2 blocks/SM).
  * Fat blocks (T=512) -> ~66% occupancy target, while L2 stays low from the staged gather.

Predicted: L2 stays ~low, occupancy recovers to ~64% -> should beat v2. idx/w now come from
global (coalesced) at ~115M sectors combined (BM=1, no reuse), so total L2 ~123M vs v2's 465M
(~3.8x fewer) -- the win rides on occupancy recovering. If it does, this is the Phase-4 forward.

A/B vs production v2, correctness vs eager. Kernel named stagex_wpo_kernel for ncu:
    ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
        -k "regex:stagex_wpo_kernel" --launch-skip 3 --launch-count 1 \
        python cuda_gather_fwd_stagex_wpo.py

Usage:  python cuda_gather_fwd_stagex_wpo.py [--P 512] [--T 512] [--iters 50]
        # sweep:  for P in 256 512 1024 2048; do python ... --P $P; done
        #         for T in 256 512 1024; do python ... --T $T; done
"""
from __future__ import annotations

import argparse
import torch
from torch.utils.cpp_extension import load_inline

_CPP_SRC = (
    "torch::Tensor fwd_v2(torch::Tensor x, torch::Tensor idx, torch::Tensor w, int B_O, int B_M);\n"
    "torch::Tensor fwd_wpo(torch::Tensor x, torch::Tensor idx, torch::Tensor w, int P, int T);"
)

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ===== baseline: production v2 forward, int64 idx =====
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

torch::Tensor fwd_v2(torch::Tensor x, torch::Tensor idx, torch::Tensor w, int B_O, int B_M) {
    int M = x.size(0), IN = x.size(1), O = idx.size(0), K = idx.size(1);
    auto out = torch::empty({M, O}, x.options());
    dim3 block(B_O, B_M), grid((O + B_O - 1) / B_O, (M + B_M - 1) / B_M);
    size_t shmem = (size_t)B_O * K * (sizeof(long) + sizeof(float));
    gather_fwd_i64_kernel<<<grid, block, shmem>>>(
        x.data_ptr<float>(), idx.data_ptr<long>(), w.data_ptr<float>(),
        out.data_ptr<float>(), M, O, IN, K, B_O, B_M);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fwd_v2 launch failed");
    return out;
}

// ===== experiment: stage x in shared, warp-per-output, coalesced global idx/w =====
__global__ void stagex_wpo_kernel(
        const float* __restrict__ x, const int* __restrict__ idx,
        const float* __restrict__ w, float* __restrict__ out,
        int M, int O, int IN, int K, int P) {
    extern __shared__ float s_x[];              // IN floats = the staged x-row
    int o_base = blockIdx.x * P, m = blockIdx.y;
    int T = blockDim.x, t = threadIdx.x;

    const float* x_m = x + (long)m * IN;
    for (int c = t; c < IN; c += T) s_x[c] = x_m[c];   // stage row (coalesced)
    __syncthreads();

    int warp = t >> 5, lane = t & 31, W = T >> 5;
    for (int po = warp; po < P; po += W) {
        int o = o_base + po;
        if (o >= O) continue;
        long base = (long)o * K;
        float p = 0.0f;
        int k1 = lane;                          // lanes 0..31 -> idx[base+0..31] (coalesced)
        if (k1 < K) { int ix = idx[base + k1]; p += s_x[ix] * w[base + k1]; }
        int k2 = lane + 32;                     // lanes 0..7  -> idx[base+32..39] (coalesced)
        if (k2 < K) { int ix = idx[base + k2]; p += s_x[ix] * w[base + k2]; }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) p += __shfl_down_sync(0xffffffffu, p, off);
        if (lane == 0) out[(long)m * O + o] = p;
    }
}

torch::Tensor fwd_wpo(torch::Tensor x, torch::Tensor idx, torch::Tensor w, int P, int T) {
    int M = x.size(0), IN = x.size(1), O = idx.size(0), K = idx.size(1);
    auto out = torch::empty({M, O}, x.options());
    dim3 grid((O + P - 1) / P, M), block(T);
    size_t shmem = (size_t)IN * sizeof(float);
    // 40KB for IN=10000 -> under the 48KB default, no opt-in needed. Guard anyway:
    if (shmem > 48 * 1024)
        cudaFuncSetAttribute(stagex_wpo_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    stagex_wpo_kernel<<<grid, block, shmem>>>(
        x.data_ptr<float>(), idx.data_ptr<int>(), w.data_ptr<float>(),
        out.data_ptr<float>(), M, O, IN, K, P);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fwd_wpo launch failed");
    return out;
}
"""


def _time(fn, iters: int, warmup: int = 15) -> float:
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
    ap.add_argument("--P", type=int, default=512, help="outputs per block")
    ap.add_argument("--T", type=int, default=512, help="threads per block (multiple of 32)")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "need CUDA"
    assert args.T % 32 == 0, "T must be a multiple of 32"
    dev = "cuda"
    torch.manual_seed(0)
    M, IN, O, K = args.M, args.IN, args.O, args.K
    x = torch.randn(M, IN, device=dev)
    idx64 = torch.randint(0, IN, (O, K), device=dev, dtype=torch.int64)
    idx32 = idx64.to(torch.int32)
    w = torch.randn(O, K, device=dev)

    print(f"gpu   : {torch.cuda.get_device_name(0)}")
    print(f"shape : M={M} IN={IN} O={O} K={K}")
    print(f"wpo   : P={args.P} T={args.T} ({args.T//32} warps/block)  shared/block={IN*4/1024:.1f}KB")
    print("compiling (first run ~90s)...", flush=True)
    ext = load_inline(name="fwd_wpo_bench", cpp_sources=_CPP_SRC, cuda_sources=_CUDA_SRC,
                      functions=["fwd_v2", "fwd_wpo"], verbose=False)

    # --- correctness vs eager ---
    ref = (x[:, idx64] * w.unsqueeze(0)).sum(-1)
    ov2 = ext.fwd_v2(x, idx64, w, 64, 16)
    owp = ext.fwd_wpo(x, idx32, w, args.P, args.T)
    ev2 = (ov2 - ref).abs().max().item()
    ewp = (owp - ref).abs().max().item()
    d = (owp - ov2).abs().max().item()
    print(f"\ncorrectness (max abs err vs eager):  v2 {ev2:.2e}   wpo {ewp:.2e}   wpo-vs-v2 {d:.2e}")
    assert ewp < 1e-2, "wpo diverged -- STOP"

    # --- timing A/B ---
    tv2 = _time(lambda: ext.fwd_v2(x, idx64, w, 64, 16), args.iters)
    twp = _time(lambda: ext.fwd_wpo(x, idx32, w, args.P, args.T), args.iters)
    print(f"\ntiming (ms/call, {args.iters} iters):")
    print(f"  v2 baseline  : {tv2:.3f}")
    print(f"  wpo          : {twp:.3f}")
    print(f"  speedup      : {tv2/twp:.3f}x  ({(tv2/twp - 1)*100:+.1f}%)")
    print("\nnext: profile occupancy + L2 SOL with the ncu command in the docstring.")


if __name__ == "__main__":
    main()
