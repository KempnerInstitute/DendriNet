"""Backward experiment (Phase-4 backward, step 1): shared-staged grad_x scatter.

WHERE WE ARE (ncu on the production v2 backward, cuda_gather_ncu_full.ncu-rep):
  L2-throughput-bound (L2 SOL 95.4%, L1 hit 0.78%), TWO ~equal L2 consumers:
    * global loads    467M sectors  -- uncoalesced x[m,idx] gather (recomputed for grad_w) + grad_out
    * atomic-reductions 460M sectors -- the grad_x global atomicAdd scatter
  grad_w is already cheap in v2 (shared privatization, ~28.8M atomics).

THE TILING CONFLICT: staging x / privatizing grad_x wants FEW m-rows per block; grad_w
privatization wants MANY. You cannot cheaply satisfy both in one kernel on a 99KB card.
Resolution here: SPLIT by gradient, because the two have opposite needs.

STEP 1 (this file): a dedicated grad_x kernel.
  grad_x[m,c] += grad_out[m,o] * w[o,k]  scattered by idx -- and crucially it DOES NOT NEED x.
  One m-row per block, stage grad_x[m,:] in shared (40KB, <48KB), warp-per-output, accumulate
  with fast SHARED atomicAdd, then flush (atomicAdd across the O/P blocks that share the row).
  This converts the 460M GLOBAL atomics into on-chip shared atomics + a coalesced flush -- with
  NO tiling conflict (grad_x needs neither x nor grad_w). grad_w stays the proven v2 path.
  Predicted: ~1.5x backward (fixes the atomic half; grad_w's x-gather load half is step 2).

THE RISK: shared atomicAdd contention on s_gx[ix] (3.6M adds into 10000 slots per row ~ 360x).
Shared atomics are fast but contention serializes -- exactly what ncu tells us. Measure it.

A/B: v2 backward (both grads, one kernel) vs SPLIT (grad_x staged + grad_w v2). Correctness of
BOTH grads vs an autograd reference. Kernels named for ncu (gradx_stage_kernel, gradw_only_kernel).

Usage:  python cuda_gather_bwd_stagex.py [--P 2048] [--T 512] [--B_O 64] [--B_M 16] [--iters 50]
"""
from __future__ import annotations

import argparse
import torch
from torch.utils.cpp_extension import load_inline

_CPP_SRC = (
    "std::vector<torch::Tensor> bwd_v2(torch::Tensor grad_out, torch::Tensor x, torch::Tensor idx, "
    "torch::Tensor w, int B_O, int B_M);\n"
    "torch::Tensor gradx_stage_into(torch::Tensor grad_out, torch::Tensor idx, torch::Tensor w, "
    "torch::Tensor grad_x, int P, int T);\n"
    "torch::Tensor gradw_v2(torch::Tensor grad_out, torch::Tensor x, torch::Tensor idx, int B_O, int B_M);\n"
    "torch::Tensor gradw_transpose(torch::Tensor goT, torch::Tensor xT, torch::Tensor idx, int P, int T);"
)

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// ===== baseline: production v2 backward (both grads, one kernel) =====
__global__ void gather_bwd_v2_kernel(
        const float* __restrict__ grad_out, const float* __restrict__ x,
        const long* __restrict__ idx, const float* __restrict__ w,
        float* __restrict__ grad_x, float* __restrict__ grad_w,
        int M, int O, int IN, int K, int B_O, int B_M) {
    extern __shared__ char smem[];
    long*  s_idx = reinterpret_cast<long*>(smem);
    float* s_w   = reinterpret_cast<float*>(smem + (size_t)B_O * K * sizeof(long));
    float* s_gw  = reinterpret_cast<float*>(smem + (size_t)B_O * K * (sizeof(long) + sizeof(float)));
    int o_base = blockIdx.x * B_O, m_base = blockIdx.y * B_M;
    int tx = threadIdx.x, ty = threadIdx.y, T = B_M * B_O, tid = ty * B_O + tx, tile = B_O * K;
    for (int i = tid; i < tile; i += T) {
        int oo = o_base + i / K, kk = i % K;
        if (oo < O) { s_idx[i] = idx[(long)oo * K + kk]; s_w[i] = w[(long)oo * K + kk]; }
        s_gw[i] = 0.0f;
    }
    __syncthreads();
    int o = o_base + tx, m = m_base + ty;
    if (o < O && m < M) {
        float go = grad_out[(long)m * O + o];
        const float* x_m = x + (long)m * IN;
        float* gx_m = grad_x + (long)m * IN;
        for (int k = 0; k < K; ++k) {
            long ix = s_idx[tx * K + k];
            atomicAdd(&gx_m[ix], go * s_w[tx * K + k]);
            atomicAdd(&s_gw[tx * K + k], go * x_m[ix]);
        }
    }
    __syncthreads();
    for (int i = tid; i < tile; i += T) {
        int oo = o_base + i / K, kk = i % K;
        if (oo < O) atomicAdd(&grad_w[(long)oo * K + kk], s_gw[i]);
    }
}

std::vector<torch::Tensor> bwd_v2(torch::Tensor grad_out, torch::Tensor x, torch::Tensor idx,
                                  torch::Tensor w, int B_O, int B_M) {
    int M = x.size(0), IN = x.size(1), O = idx.size(0), K = idx.size(1);
    auto grad_x = torch::zeros({M, IN}, x.options());
    auto grad_w = torch::zeros({O, K}, w.options());
    dim3 block(B_O, B_M), grid((O + B_O - 1) / B_O, (M + B_M - 1) / B_M);
    size_t shmem = (size_t)B_O * K * (sizeof(long) + 2 * sizeof(float));
    gather_bwd_v2_kernel<<<grid, block, shmem>>>(
        grad_out.data_ptr<float>(), x.data_ptr<float>(), idx.data_ptr<long>(), w.data_ptr<float>(),
        grad_x.data_ptr<float>(), grad_w.data_ptr<float>(), M, O, IN, K, B_O, B_M);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "bwd_v2 launch failed");
    return {grad_x, grad_w};
}

// ===== step 1a: grad_x via shared staging + warp-per-output (no x needed) =====
__global__ void gradx_stage_kernel(
        const float* __restrict__ grad_out, const int* __restrict__ idx,
        const float* __restrict__ w, float* __restrict__ grad_x,
        int M, int O, int IN, int K, int P) {
    extern __shared__ float s_gx[];             // IN floats = staged grad_x row
    int o_base = blockIdx.x * P, m = blockIdx.y;
    int T = blockDim.x, t = threadIdx.x;
    for (int c = t; c < IN; c += T) s_gx[c] = 0.0f;
    __syncthreads();
    int warp = t >> 5, lane = t & 31, W = T >> 5;
    const float* go_m = grad_out + (long)m * O;
    for (int po = warp; po < P; po += W) {
        int o = o_base + po;
        if (o >= O) continue;
        float go = go_m[o];
        long base = (long)o * K;
        for (int k = lane; k < K; k += 32) {
            int ix = idx[base + k];
            atomicAdd(&s_gx[ix], go * w[base + k]);   // SHARED atomic (fast)
        }
    }
    __syncthreads();
    float* gx_m = grad_x + (long)m * IN;              // flush: O/P blocks share this row
    for (int c = t; c < IN; c += T) atomicAdd(&gx_m[c], s_gx[c]);
}

// grad_x buffer supplied (pre-zeroed) so IN is known and flush accumulates correctly.
torch::Tensor gradx_stage_into(torch::Tensor grad_out, torch::Tensor idx, torch::Tensor w,
                               torch::Tensor grad_x, int P, int T) {
    int M = grad_out.size(0), O = idx.size(0), K = idx.size(1), IN = grad_x.size(1);
    dim3 grid((O + P - 1) / P, M), block(T);
    size_t shmem = (size_t)IN * sizeof(float);
    if (shmem > 48 * 1024)
        cudaFuncSetAttribute(gradx_stage_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    gradx_stage_kernel<<<grid, block, shmem>>>(
        grad_out.data_ptr<float>(), idx.data_ptr<int>(), w.data_ptr<float>(),
        grad_x.data_ptr<float>(), M, O, IN, K, P);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "gradx_stage launch failed");
    return grad_x;
}

// ===== step 1b: grad_w only (v2 privatization; grad_w needs idx + x, NOT w) =====
__global__ void gradw_only_kernel(
        const float* __restrict__ grad_out, const float* __restrict__ x,
        const long* __restrict__ idx, float* __restrict__ grad_w,
        int M, int O, int IN, int K, int B_O, int B_M) {
    extern __shared__ char smem[];
    long*  s_idx = reinterpret_cast<long*>(smem);
    float* s_gw  = reinterpret_cast<float*>(smem + (size_t)B_O * K * sizeof(long));
    int o_base = blockIdx.x * B_O, m_base = blockIdx.y * B_M;
    int tx = threadIdx.x, ty = threadIdx.y, T = B_M * B_O, tid = ty * B_O + tx, tile = B_O * K;
    for (int i = tid; i < tile; i += T) {
        int oo = o_base + i / K, kk = i % K;
        if (oo < O) s_idx[i] = idx[(long)oo * K + kk];
        s_gw[i] = 0.0f;
    }
    __syncthreads();
    int o = o_base + tx, m = m_base + ty;
    if (o < O && m < M) {
        float go = grad_out[(long)m * O + o];
        const float* x_m = x + (long)m * IN;
        for (int k = 0; k < K; ++k) {
            long ix = s_idx[tx * K + k];
            atomicAdd(&s_gw[tx * K + k], go * x_m[ix]);   // shared atomic; x_m[ix] still uncoalesced (step 2)
        }
    }
    __syncthreads();
    for (int i = tid; i < tile; i += T) {
        int oo = o_base + i / K, kk = i % K;
        if (oo < O) atomicAdd(&grad_w[(long)oo * K + kk], s_gw[i]);
    }
}

torch::Tensor gradw_v2(torch::Tensor grad_out, torch::Tensor x, torch::Tensor idx, int B_O, int B_M) {
    int M = x.size(0), IN = x.size(1), O = idx.size(0), K = idx.size(1);
    auto grad_w = torch::zeros({O, K}, x.options());
    dim3 block(B_O, B_M), grid((O + B_O - 1) / B_O, (M + B_M - 1) / B_M);
    size_t shmem = (size_t)B_O * K * (sizeof(long) + sizeof(float));
    gradw_only_kernel<<<grid, block, shmem>>>(
        grad_out.data_ptr<float>(), x.data_ptr<float>(), idx.data_ptr<long>(),
        grad_w.data_ptr<float>(), M, O, IN, K, B_O, B_M);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "gradw_v2 launch failed");
    return grad_w;
}

// ===== step 2: grad_w via transpose (coalesced M-reduction, NO atomics) =====
// grad_w[o,k] = sum_m goT[o,m] * xT[idx[o,k], m]. Transposing x->xT[IN,M] and grad_out->goT[O,M]
// turns the uncoalesced x[m,idx] gather into a COALESCED row read xT[ix,:] (M contiguous floats).
// Warp-per-output; goT[o,:] staged in registers (M<=128), reused across all K; each grad_w[o,k]
// is one warp's dot product -> written once, no atomics.
__global__ void gradw_transpose_kernel(
        const float* __restrict__ goT, const float* __restrict__ xT,
        const int* __restrict__ idx, float* __restrict__ grad_w,
        int M, int O, int IN, int K, int P) {
    int o_base = blockIdx.x * P;
    int T = blockDim.x, t = threadIdx.x;
    int warp = t >> 5, lane = t & 31, W = T >> 5;
    for (int po = warp; po < P; po += W) {
        int o = o_base + po;
        if (o >= O) continue;
        const float* go_o = goT + (long)o * M;                 // reused across all K
        float g0 = (lane      < M) ? go_o[lane]      : 0.0f;
        float g1 = (lane + 32 < M) ? go_o[lane + 32] : 0.0f;
        float g2 = (lane + 64 < M) ? go_o[lane + 64] : 0.0f;
        float g3 = (lane + 96 < M) ? go_o[lane + 96] : 0.0f;
        long base = (long)o * K;
        for (int k = 0; k < K; ++k) {
            const float* xt = xT + (long)idx[base + k] * M;    // COALESCED row read
            float p = 0.0f;
            if (lane      < M) p += g0 * xt[lane];
            if (lane + 32 < M) p += g1 * xt[lane + 32];
            if (lane + 64 < M) p += g2 * xt[lane + 64];
            if (lane + 96 < M) p += g3 * xt[lane + 96];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) p += __shfl_down_sync(0xffffffffu, p, off);
            if (lane == 0) grad_w[base + k] = p;               // one write, no atomic
        }
    }
}

torch::Tensor gradw_transpose(torch::Tensor goT, torch::Tensor xT, torch::Tensor idx, int P, int T) {
    int O = idx.size(0), K = idx.size(1), M = goT.size(1), IN = xT.size(0);
    TORCH_CHECK(M <= 128, "gradw_transpose (benchmark) stages goT[o,:] in 4 regs -> assumes M<=128; got M=", M);
    auto grad_w = torch::empty({O, K}, goT.options());
    dim3 grid((O + P - 1) / P), block(T);
    gradw_transpose_kernel<<<grid, block>>>(
        goT.data_ptr<float>(), xT.data_ptr<float>(), idx.data_ptr<int>(),
        grad_w.data_ptr<float>(), M, O, IN, K, P);
    { cudaError_t e = cudaGetLastError();
      TORCH_CHECK(e == cudaSuccess, "gradw_transpose launch failed: ", cudaGetErrorString(e)); }
    return grad_w;
}
"""


def _time(fn, iters, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=128); ap.add_argument("--IN", type=int, default=10000)
    ap.add_argument("--O", type=int, default=90000); ap.add_argument("--K", type=int, default=40)
    ap.add_argument("--P", type=int, default=2048); ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--B_O", type=int, default=64); ap.add_argument("--B_M", type=int, default=16)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    assert torch.cuda.is_available()
    dev = "cuda"; torch.manual_seed(0)
    M, IN, O, K = args.M, args.IN, args.O, args.K
    x = torch.randn(M, IN, device=dev)
    idx64 = torch.randint(0, IN, (O, K), device=dev, dtype=torch.int64)
    idx32 = idx64.to(torch.int32)
    w = torch.randn(O, K, device=dev)
    grad_out = torch.randn(M, O, device=dev)

    print(f"gpu   : {torch.cuda.get_device_name(0)}")
    print(f"shape : M={M} IN={IN} O={O} K={K}   split: P={args.P} T={args.T} | grad_w B_O={args.B_O} B_M={args.B_M}")
    print("compiling (first run ~90s)...", flush=True)
    ext = load_inline(name="bwd_stagex_bench", cpp_sources=_CPP_SRC, cuda_sources=_CUDA_SRC,
                      functions=["bwd_v2", "gradx_stage_into", "gradw_v2", "gradw_transpose"],
                      verbose=False)

    # --- autograd reference for both grads ---
    xr = x.clone().requires_grad_(True); wr = w.clone().requires_grad_(True)
    out = (xr[:, idx64] * wr.unsqueeze(0)).sum(-1)
    gx_ref, gw_ref = torch.autograd.grad(out, [xr, wr], grad_outputs=grad_out)

    # xT is x transposed -- production can save x pre-transposed in the forward, so it is
    # NOT charged to the backward. goT is grad_out transposed -- unavoidable each backward call,
    # so it IS charged (timed inside split_v2_bwd).
    xT = x.t().contiguous()                       # [IN, M]

    def split_v1_bwd():                           # step 1: grad_x staged + grad_w v2
        gx = torch.zeros(M, IN, device=dev)
        ext.gradx_stage_into(grad_out, idx32, w, gx, args.P, args.T)
        gw = ext.gradw_v2(grad_out, x, idx64, args.B_O, args.B_M)
        return gx, gw

    def split_v2_bwd():                           # step 2: grad_x staged + grad_w transpose
        goT = grad_out.t().contiguous()           # [O, M] -- charged to the backward
        gx = torch.zeros(M, IN, device=dev)
        ext.gradx_stage_into(grad_out, idx32, w, gx, args.P, args.T)
        gw = ext.gradw_transpose(goT, xT, idx32, args.P, args.T)
        return gx, gw

    gx_v2, gw_v2 = ext.bwd_v2(grad_out, x, idx64, w, args.B_O, args.B_M)
    gx_s1, gw_s1 = split_v1_bwd()
    gx_s2, gw_s2 = split_v2_bwd()

    def err(a, b): return (a - b).abs().max().item()
    print(f"\ncorrectness (max abs err vs autograd ref):")
    print(f"  v2       grad_x {err(gx_v2, gx_ref):.2e}   grad_w {err(gw_v2, gw_ref):.2e}")
    print(f"  split_v1 grad_x {err(gx_s1, gx_ref):.2e}   grad_w {err(gw_s1, gw_ref):.2e}")
    print(f"  split_v2 grad_x {err(gx_s2, gx_ref):.2e}   grad_w {err(gw_s2, gw_ref):.2e}")
    assert err(gx_s2, gx_ref) < 1e-2 and err(gw_s2, gw_ref) < 1e-2, "split_v2 diverged -- STOP"

    t_v2 = _time(lambda: ext.bwd_v2(grad_out, x, idx64, w, args.B_O, args.B_M), args.iters)
    t_s1 = _time(split_v1_bwd, args.iters)
    t_s2 = _time(split_v2_bwd, args.iters)
    print(f"\ntiming (ms/call, {args.iters} iters):")
    print(f"  v2 backward (1 kernel)          : {t_v2:.3f}")
    print(f"  split_v1 (gx stage + gw v2)      : {t_s1:.3f}   {t_v2/t_s1:.3f}x")
    print(f"  split_v2 (gx stage + gw transpose): {t_s2:.3f}   {t_v2/t_s2:.3f}x  <- step 2")
    print("\nnote: split_v2 charges the grad_out transpose (unavoidable); xT is precomputed")
    print("      (production saves x pre-transposed in forward). If step 2 wins, that is the")
    print("      new backward -- then productionize grad_x-stage + grad_w-transpose behind the flag.")


if __name__ == "__main__":
    main()
