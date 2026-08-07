"""CUDA fused sparse gather-multiply-sum: forward + backward in one Function.

Production form of the winning kernels:
  * forward  : Phase-4 warp-per-output with the x-row staged in shared (wpo) -- ~4.5x over v2
               on Blackwell (v2 was L2-throughput-bound; wpo drops L2 SOL 93%->67% and restores
               occupancy). Falls back to the Phase-3 v2 shared-tiled forward when the x-row does
               not fit shared (IN*4 > opt-in max).
  * backward : Phase-4 SPLIT backward -- ~2.6x over v2 on Blackwell. v2 was L2-bound with two
               ~equal costs (467M uncoalesced x-gather loads + 460M grad_x global atomics), which
               want opposite tilings, so we split by gradient: grad_x stages grad_x[m,:] in shared
               with fast atomics + coalesced flush (kills the global atomics); grad_w transposes
               x->xT, grad_out->goT so its m-reduction is a coalesced dot product (kills the
               scattered gather, and no atomics). Falls back to the v2 one-kernel backward when the
               grad_x row does not fit shared. (An earlier v3 grad_x scatter->gather transpose was
               tested and rejected: 2.2-2.4x slower -- that targeted grad_x's atomics, which were
               never the bottleneck; the win came from grad_w's gather + grad_x's atomic *venue*.)
Both beat the Triton path on both GPUs.

Drop-in for :func:`fused_sparse_gather` / :func:`triton_sparse_gather` (same signature).

LAZY COMPILE: the CUDA extension is compiled on FIRST USE only (then cached by torch across
processes), so importing this module never needs nvcc and never compiles anything. When the
``use_cuda_gather`` flag is off (the default), this path is never entered -> zero impact, no
toolchain required. Query :func:`cuda_gather_available` (which triggers the one-time lazy compile
attempt) before selecting this path; it returns False -- and the caller falls back to
triton/fused -- if there is no CUDA device, no ``nvcc``, or compilation fails.

AMP-safe: upcasts inputs to fp32 for the kernels and casts outputs/grads back to the caller's
dtypes (the indexed synapse only ever sees fp32 under this model's autocast anyway).
"""
from __future__ import annotations

import os

import torch

_EXT = None            # the compiled extension module (or None)
_COMPILE_TRIED = False  # so we attempt the (slow) compile at most once per process

_CPP_SRC = (
    "torch::Tensor gather_fwd_cuda(torch::Tensor x, torch::Tensor idx, torch::Tensor w, int B_O, int B_M);\n"
    "torch::Tensor gather_fwd_wpo_cuda(torch::Tensor x, torch::Tensor idx, torch::Tensor w, int P, int T);\n"
    "torch::Tensor gather_fwd_wpo_ns_cuda(torch::Tensor x, torch::Tensor idx, torch::Tensor w, int P, int T);\n"
    "torch::Tensor gather_fwd_wpo_hy_cuda(torch::Tensor x, torch::Tensor idx, torch::Tensor w, int P, int T, int S);\n"
    "std::vector<torch::Tensor> gather_bwd_cuda(torch::Tensor grad_out, torch::Tensor x, "
    "torch::Tensor idx, torch::Tensor w, int B_O, int B_M);\n"
    "std::vector<torch::Tensor> gather_bwd_wpo_cuda(torch::Tensor grad_out, torch::Tensor goT, "
    "torch::Tensor xT, torch::Tensor idx, torch::Tensor w, int P, int T);\n"
    "std::vector<torch::Tensor> gather_bwd_wpo_cuda2(torch::Tensor grad_out, torch::Tensor goT, "
    "torch::Tensor xT, torch::Tensor idx, torch::Tensor w, int P, int T, int gx_variant);\n"
    "int cuda_gather_max_optin_shared();"
)

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// ---- forward v2: shared-memory staging of idx/w -----------------------------
__global__ void gather_fwd_v2_kernel(
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

torch::Tensor gather_fwd_cuda(torch::Tensor x, torch::Tensor idx, torch::Tensor w, int B_O, int B_M) {
    int M = x.size(0), IN = x.size(1), O = idx.size(0), K = idx.size(1);
    auto out = torch::empty({M, O}, x.options());
    dim3 block(B_O, B_M), grid((O + B_O - 1) / B_O, (M + B_M - 1) / B_M);
    size_t shmem = (size_t)B_O * K * (sizeof(long) + sizeof(float));
    TORCH_CHECK(shmem <= 48 * 1024, "cuda_gather forward tile > 48KB; lower B_O");
    gather_fwd_v2_kernel<<<grid, block, shmem>>>(
        x.data_ptr<float>(), idx.data_ptr<long>(), w.data_ptr<float>(),
        out.data_ptr<float>(), M, O, IN, K, B_O, B_M);
    { cudaError_t e = cudaGetLastError();
      TORCH_CHECK(e == cudaSuccess, "gather_fwd_v2 launch failed: ", cudaGetErrorString(e),
                  " (B_O=", B_O, " B_M=", B_M, " shmem=", shmem, ")"); }
    return out;
}

// ---- backward v2: shared-memory privatization of grad_w ----------------------
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
            atomicAdd(&gx_m[ix], go * s_w[tx * K + k]);      // grad_x: global atomic (cheap)
            atomicAdd(&s_gw[tx * K + k], go * x_m[ix]);      // grad_w: shared atomic (fast)
        }
    }
    __syncthreads();
    for (int i = tid; i < tile; i += T) {
        int oo = o_base + i / K, kk = i % K;
        if (oo < O) atomicAdd(&grad_w[(long)oo * K + kk], s_gw[i]);  // one global atomic per [o,k]
    }
}

std::vector<torch::Tensor> gather_bwd_cuda(
        torch::Tensor grad_out, torch::Tensor x, torch::Tensor idx, torch::Tensor w, int B_O, int B_M) {
    int M = x.size(0), IN = x.size(1), O = idx.size(0), K = idx.size(1);
    auto grad_x = torch::zeros({M, IN}, x.options());
    auto grad_w = torch::zeros({O, K}, w.options());
    dim3 block(B_O, B_M), grid((O + B_O - 1) / B_O, (M + B_M - 1) / B_M);
    size_t shmem = (size_t)B_O * K * (sizeof(long) + 2 * sizeof(float));
    TORCH_CHECK(shmem <= 48 * 1024, "cuda_gather backward tile > 48KB; lower B_O");
    gather_bwd_v2_kernel<<<grid, block, shmem>>>(
        grad_out.data_ptr<float>(), x.data_ptr<float>(), idx.data_ptr<long>(), w.data_ptr<float>(),
        grad_x.data_ptr<float>(), grad_w.data_ptr<float>(), M, O, IN, K, B_O, B_M);
    { cudaError_t e = cudaGetLastError();
      TORCH_CHECK(e == cudaSuccess, "gather_bwd_v2 launch failed: ", cudaGetErrorString(e),
                  " (B_O=", B_O, " B_M=", B_M, " shmem=", shmem, ")"); }
    return {grad_x, grad_w};
}

// ---- forward wpo: stage the x-row in shared, warp-per-output, coalesced idx/w --------
// Profile-driven Phase-4 forward. v2 was L2-throughput-bound (L2 SOL 93%, ~29 sectors/req
// uncoalesced gather, L1 hit 5%). wpo stages x[m,:] into shared ONCE per block (coalesced),
// then each WARP owns one output: its 32 lanes read idx/w coalesced from global, gather from
// the on-chip x-row, and __shfl_down-reduce over K. Result on Blackwell: L2 SOL 93%->67%,
// occupancy back to ~64%, ~4.5x over v2 (and slightly better accuracy from the tree reduce).
// Requires only that the x-row fit shared (IN*4 <= optin max); caller falls back to v2 else.
__global__ void gather_fwd_wpo_kernel(
        const float* __restrict__ x, const int* __restrict__ idx,
        const float* __restrict__ w, float* __restrict__ out,
        int M, int O, int IN, int K, int P) {
    extern __shared__ float s_x[];                 // IN floats = the staged x-row
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
        // strided lane loop: each iteration reads idx[base+i*32 .. +31] coalesced. Handles any K.
        for (int k = lane; k < K; k += 32) { int ix = idx[base + k]; p += s_x[ix] * w[base + k]; }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) p += __shfl_down_sync(0xffffffffu, p, off);
        if (lane == 0) out[(long)m * O + o] = p;
    }
}

torch::Tensor gather_fwd_wpo_cuda(torch::Tensor x, torch::Tensor idx, torch::Tensor w, int P, int T) {
    int M = x.size(0), IN = x.size(1), O = idx.size(0), K = idx.size(1);
    auto out = torch::empty({M, O}, x.options());
    dim3 grid((O + P - 1) / P, M), block(T);
    size_t shmem = (size_t)IN * sizeof(float);
    if (shmem > 48 * 1024)   // opt into >48KB dynamic shared when the row is large
        cudaFuncSetAttribute(gather_fwd_wpo_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    gather_fwd_wpo_kernel<<<grid, block, shmem>>>(
        x.data_ptr<float>(), idx.data_ptr<int>(), w.data_ptr<float>(),
        out.data_ptr<float>(), M, O, IN, K, P);
    { cudaError_t e = cudaGetLastError();
      TORCH_CHECK(e == cudaSuccess, "gather_fwd_wpo launch failed: ", cudaGetErrorString(e),
                  " (P=", P, " T=", T, " shmem=", shmem, ")"); }
    return out;
}

// ---- forward wpo, HYBRID staging ----------------------------------------------------
// The staged kernel's IN*4-byte row caps Blackwell at 1 block/SM (33% occupancy;
// probe_shared_budget.py: 2 blocks would need 128,000 B against a 102,400 B/SM
// budget -- a hard limit, not a near-miss). The un-staged kernel gets full
// occupancy but pays ~5x more L2 requests and wins only 3/25 shapes at K=40.
//
// Hybrid: stage only the FIRST S elements of the x-row (S chosen so several
// blocks fit an SM); gathers with ix < S hit shared, the tail falls through to
// __ldg and the L2 that already runs at a 99.4% hit rate. With uniform indices,
// a fraction S/IN of gathers stays on-chip while occupancy triples.
//
// Accumulation order and (o,k) ownership are IDENTICAL to the staged and
// un-staged kernels, so outputs must be bit-identical -- the select changes
// where a value is read from, never what is summed or in what order.
__global__ void gather_fwd_wpo_hy_kernel(
        const float* __restrict__ x, const int* __restrict__ idx,
        const float* __restrict__ w, float* __restrict__ out,
        int M, int O, int IN, int K, int P, int S) {
    extern __shared__ float s_x[];                 // S floats: the staged PREFIX
    int o_base = blockIdx.x * P, m = blockIdx.y;
    int T = blockDim.x, t = threadIdx.x;
    const float* __restrict__ x_m = x + (long)m * IN;
    for (int c = t; c < S; c += T) s_x[c] = x_m[c];
    __syncthreads();
    int warp = t >> 5, lane = t & 31, W = T >> 5;
    for (int po = warp; po < P; po += W) {
        int o = o_base + po;
        if (o >= O) continue;
        long base = (long)o * K;
        float p = 0.0f;
        for (int k = lane; k < K; k += 32) {
            int ix = idx[base + k];
            float xv = (ix < S) ? s_x[ix] : __ldg(x_m + ix);
            p += xv * w[base + k];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) p += __shfl_down_sync(0xffffffffu, p, off);
        if (lane == 0) out[(long)m * O + o] = p;
    }
}

torch::Tensor gather_fwd_wpo_hy_cuda(torch::Tensor x, torch::Tensor idx, torch::Tensor w,
                                     int P, int T, int S) {
    int M = x.size(0), IN = x.size(1), O = idx.size(0), K = idx.size(1);
    TORCH_CHECK(S >= 0 && S <= IN, "hybrid stage prefix S out of range");
    auto out = torch::empty({M, O}, x.options());
    dim3 grid((O + P - 1) / P, M), block(T);
    size_t shmem = (size_t)S * sizeof(float);
    if (shmem > 48 * 1024)
        cudaFuncSetAttribute(gather_fwd_wpo_hy_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    gather_fwd_wpo_hy_kernel<<<grid, block, shmem>>>(
        x.data_ptr<float>(), idx.data_ptr<int>(), w.data_ptr<float>(),
        out.data_ptr<float>(), M, O, IN, K, P, S);
    { cudaError_t e = cudaGetLastError();
      TORCH_CHECK(e == cudaSuccess, "gather_fwd_wpo_hy launch failed: ", cudaGetErrorString(e),
                  " (P=", P, " T=", T, " S=", S, ")"); }
    return out;
}

// ---- forward wpo, NO shared staging -------------------------------------------------
// The staged variant above puts the whole x-row (IN floats) in shared memory, so at
// IN=16,000 it asks for 64KB and exactly ONE block fits per SM. ncu measured the
// consequence: 33.1% achieved occupancy, DRAM at 1.11% of peak, SM at 35.3% -- neither
// bandwidth- nor compute-bound, just starved of resident warps to hide latency with.
//
// The staging is also 71x redundant here: grid.x = ceil(O/P) blocks each re-stage the
// SAME x-row for a given m. And the L2 hit rate on that row is already 99.44%, so the
// data the staging protects is cache-resident anyway.
//
// This variant drops shared entirely and reads x through the read-only path, letting L2
// do the job it is already doing. It trades ~5x more L2 requests for full occupancy.
// Which wins is a measurement, not an argument -- see bench_gather_nostage.py.
//
// Bonus: with no shared requirement the IN <= optin/4 ceiling (25,344 on Blackwell)
// disappears, so this path also covers rows the staged kernel cannot run at all.
__global__ void gather_fwd_wpo_ns_kernel(
        const float* __restrict__ x, const int* __restrict__ idx,
        const float* __restrict__ w, float* __restrict__ out,
        int M, int O, int IN, int K, int P) {
    int o_base = blockIdx.x * P, m = blockIdx.y;
    int T = blockDim.x, t = threadIdx.x;
    const float* __restrict__ x_m = x + (long)m * IN;
    int warp = t >> 5, lane = t & 31, W = T >> 5;
    for (int po = warp; po < P; po += W) {
        int o = o_base + po;
        if (o >= O) continue;
        long base = (long)o * K;
        float p = 0.0f;
        for (int k = lane; k < K; k += 32) {
            int ix = idx[base + k];
            p += __ldg(x_m + ix) * w[base + k];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) p += __shfl_down_sync(0xffffffffu, p, off);
        if (lane == 0) out[(long)m * O + o] = p;
    }
}

torch::Tensor gather_fwd_wpo_ns_cuda(torch::Tensor x, torch::Tensor idx, torch::Tensor w, int P, int T) {
    int M = x.size(0), IN = x.size(1), O = idx.size(0), K = idx.size(1);
    auto out = torch::empty({M, O}, x.options());
    dim3 grid((O + P - 1) / P, M), block(T);
    gather_fwd_wpo_ns_kernel<<<grid, block, 0>>>(
        x.data_ptr<float>(), idx.data_ptr<int>(), w.data_ptr<float>(),
        out.data_ptr<float>(), M, O, IN, K, P);
    { cudaError_t e = cudaGetLastError();
      TORCH_CHECK(e == cudaSuccess, "gather_fwd_wpo_ns launch failed: ", cudaGetErrorString(e),
                  " (P=", P, " T=", T, ")"); }
    return out;
}

int cuda_gather_max_optin_shared() {
    int dev = 0; cudaGetDevice(&dev);
    int m = 0; cudaDeviceGetAttribute(&m, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    return m;
}

// ---- backward wpo: split by gradient (Phase-4 backward, ~2.6x over v2) ---------------
// v2 backward was L2-bound with TWO ~equal costs: 467M uncoalesced x-gather loads and 460M
// grad_x global atomics. They want opposite tilings, so we SPLIT into two kernels:
//   (a) grad_x: scatter go*w into grad_x[m,ix]; needs NO x. Stage grad_x[m,:] in shared,
//       accumulate with fast SHARED atomics, flush once -> kills the global atomics.
//   (b) grad_w: sum_m go*x[m,ix]. Transpose x->xT[IN,M], grad_out->goT[O,M] so the m-reduction
//       becomes a COALESCED dot product xT[ix,:].goT[o,:] -> the scattered gather is gone, and
//       each grad_w[o,k] is one warp's reduction (no atomics).

// (a) grad_x via shared staging + warp-per-output.
__global__ void gather_gradx_stage_kernel(
        const float* __restrict__ grad_out, const int* __restrict__ idx,
        const float* __restrict__ w, float* __restrict__ grad_x,
        int M, int O, int IN, int K, int P) {
    extern __shared__ float s_gx[];                 // IN floats = staged grad_x row
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
            atomicAdd(&s_gx[ix], go * w[base + k]);  // SHARED atomic
        }
    }
    __syncthreads();
    float* gx_m = grad_x + (long)m * IN;             // O/P blocks share this row -> atomic flush
    for (int c = t; c < IN; c += T) atomicAdd(&gx_m[c], s_gx[c]);
}

// (b) grad_w via transpose. Warp-per-output. For M<=128 goT[o,:] is register-staged (the
// benchmarked fast path); for M>128 it is re-read coalesced from global (correct for any batch).
__global__ void gather_gradw_transpose_kernel(
        const float* __restrict__ goT, const float* __restrict__ xT,
        const int* __restrict__ idx, float* __restrict__ grad_w,
        int M, int O, int IN, int K, int P) {
    int o_base = blockIdx.x * P;
    int T = blockDim.x, t = threadIdx.x;
    int warp = t >> 5, lane = t & 31, W = T >> 5;
    for (int po = warp; po < P; po += W) {
        int o = o_base + po;
        if (o >= O) continue;
        const float* go_o = goT + (long)o * M;
        long base = (long)o * K;
        if (M <= 128) {                              // register-staged fast path (batch<=128)
            float g0 = (lane      < M) ? go_o[lane]      : 0.0f;
            float g1 = (lane + 32 < M) ? go_o[lane + 32] : 0.0f;
            float g2 = (lane + 64 < M) ? go_o[lane + 64] : 0.0f;
            float g3 = (lane + 96 < M) ? go_o[lane + 96] : 0.0f;
            for (int k = 0; k < K; ++k) {
                const float* xt = xT + (long)idx[base + k] * M;   // COALESCED row read
                float p = 0.0f;
                if (lane      < M) p += g0 * xt[lane];
                if (lane + 32 < M) p += g1 * xt[lane + 32];
                if (lane + 64 < M) p += g2 * xt[lane + 64];
                if (lane + 96 < M) p += g3 * xt[lane + 96];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) p += __shfl_down_sync(0xffffffffu, p, off);
                if (lane == 0) grad_w[base + k] = p;
            }
        } else {                                     // general path (batch>128): re-read goT
            for (int k = 0; k < K; ++k) {
                const float* xt = xT + (long)idx[base + k] * M;
                float p = 0.0f;
                for (int mm = lane; mm < M; mm += 32) p += go_o[mm] * xt[mm];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) p += __shfl_down_sync(0xffffffffu, p, off);
                if (lane == 0) grad_w[base + k] = p;
            }
        }
    }
}

// grad_x WITHOUT the shared staging: accumulate straight into global with fp32
// atomics. The staged kernel is capped at 1 block/SM by its IN-float shared row
// (ncu: 33.26% occupancy, 21.01 ms -- 87% of the whole backward, vs grad_w's
// 3.16 ms), while registers would allow 4 blocks. This trades the shared atomics
// + row flush for global atomics at full occupancy.
//
// The v2 backward was originally slow partly because of global grad_x atomics,
// which is why the staged version exists -- so this is NOT obviously a win, and
// it is selected by measurement (bench_gradx_nostage.py), not by analogy with
// the forward. fp32 atomicAdd is native; do not let this path see bf16, where
// sm_80 emulates and collapses.
__global__ void gather_gradx_ns_kernel(
        const float* __restrict__ grad_out, const int* __restrict__ idx,
        const float* __restrict__ w, float* __restrict__ grad_x,
        int M, int O, int IN, int K, int P) {
    int o_base = blockIdx.x * P, m = blockIdx.y;
    int T = blockDim.x, t = threadIdx.x;
    int warp = t >> 5, lane = t & 31, W = T >> 5;
    const float* go_m = grad_out + (long)m * O;
    float* gx_m = grad_x + (long)m * IN;
    for (int po = warp; po < P; po += W) {
        int o = o_base + po;
        if (o >= O) continue;
        float go = go_m[o];
        long base = (long)o * K;
        for (int k = lane; k < K; k += 32)
            atomicAdd(&gx_m[idx[base + k]], go * w[base + k]);
    }
}

// gx_variant: 0 = staged (shared atomics + flush), 1 = un-staged (global atomics)
std::vector<torch::Tensor> gather_bwd_wpo_cuda2(
        torch::Tensor grad_out, torch::Tensor goT, torch::Tensor xT,
        torch::Tensor idx, torch::Tensor w, int P, int T, int gx_variant) {
    int M = grad_out.size(0), O = grad_out.size(1), IN = xT.size(0), K = idx.size(1);
    auto grad_x = torch::zeros({M, IN}, grad_out.options());
    auto grad_w = torch::empty({O, K}, grad_out.options());
    { dim3 grid((O + P - 1) / P, M), block(T);
      if (gx_variant == 1) {
          gather_gradx_ns_kernel<<<grid, block, 0>>>(
              grad_out.data_ptr<float>(), idx.data_ptr<int>(), w.data_ptr<float>(),
              grad_x.data_ptr<float>(), M, O, IN, K, P);
      } else {
          size_t sh = (size_t)IN * sizeof(float);
          if (sh > 48 * 1024)
              cudaFuncSetAttribute(gather_gradx_stage_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh);
          gather_gradx_stage_kernel<<<grid, block, sh>>>(
              grad_out.data_ptr<float>(), idx.data_ptr<int>(), w.data_ptr<float>(),
              grad_x.data_ptr<float>(), M, O, IN, K, P);
      }
      cudaError_t e = cudaGetLastError();
      TORCH_CHECK(e == cudaSuccess, "gather_gradx launch failed: ", cudaGetErrorString(e)); }
    { dim3 grid((O + P - 1) / P), block(T);
      gather_gradw_transpose_kernel<<<grid, block>>>(
          goT.data_ptr<float>(), xT.data_ptr<float>(), idx.data_ptr<int>(),
          grad_w.data_ptr<float>(), M, O, IN, K, P);
      cudaError_t e = cudaGetLastError();
      TORCH_CHECK(e == cudaSuccess, "gather_gradw_transpose launch failed: ", cudaGetErrorString(e)); }
    return {grad_x, grad_w};
}

std::vector<torch::Tensor> gather_bwd_wpo_cuda(
        torch::Tensor grad_out, torch::Tensor goT, torch::Tensor xT,
        torch::Tensor idx, torch::Tensor w, int P, int T) {
    int M = grad_out.size(0), O = grad_out.size(1), IN = xT.size(0), K = idx.size(1);
    auto grad_x = torch::zeros({M, IN}, grad_out.options());
    auto grad_w = torch::empty({O, K}, grad_out.options());
    { dim3 grid((O + P - 1) / P, M), block(T);
      size_t sh = (size_t)IN * sizeof(float);
      if (sh > 48 * 1024)
          cudaFuncSetAttribute(gather_gradx_stage_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh);
      gather_gradx_stage_kernel<<<grid, block, sh>>>(
          grad_out.data_ptr<float>(), idx.data_ptr<int>(), w.data_ptr<float>(),
          grad_x.data_ptr<float>(), M, O, IN, K, P);
      cudaError_t e = cudaGetLastError();
      TORCH_CHECK(e == cudaSuccess, "gather_gradx_stage launch failed: ", cudaGetErrorString(e)); }
    { dim3 grid((O + P - 1) / P), block(T);
      gather_gradw_transpose_kernel<<<grid, block>>>(
          goT.data_ptr<float>(), xT.data_ptr<float>(), idx.data_ptr<int>(),
          grad_w.data_ptr<float>(), M, O, IN, K, P);
      cudaError_t e = cudaGetLastError();
      TORCH_CHECK(e == cudaSuccess, "gather_gradw_transpose launch failed: ", cudaGetErrorString(e)); }
    return {grad_x, grad_w};
}
"""


def _load_ext():
    """Compile + cache the extension on first call. Returns the module or None."""
    global _EXT, _COMPILE_TRIED
    if _COMPILE_TRIED:
        return _EXT
    _COMPILE_TRIED = True
    try:
        from torch.utils.cpp_extension import CUDA_HOME, load_inline
        if not torch.cuda.is_available() or CUDA_HOME is None:
            return None
        _EXT = load_inline(
            name="dendrinet_cuda_gather",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["gather_fwd_cuda", "gather_fwd_wpo_cuda", "gather_fwd_wpo_ns_cuda",
                       "gather_fwd_wpo_hy_cuda",
                       "gather_bwd_cuda", "gather_bwd_wpo_cuda", "gather_bwd_wpo_cuda2",
                       "cuda_gather_max_optin_shared"],
            verbose=False,
        )
    except Exception:  # noqa: BLE001  (missing nvcc, compile error, arch mismatch, ...)
        _EXT = None
    return _EXT


def cuda_gather_available() -> bool:
    """Whether the CUDA gather path can be used. Triggers the one-time lazy compile."""
    return _load_ext() is not None


_TILE = None  # (B_O, B_M), queried once -- the device does not change mid-process


def _pick_tile() -> tuple[int, int]:
    """Per-arch tile for the v2 fallback forward/backward: B_M=16 is the sweet spot on both
    A100 and Blackwell; B_O by arch.

    Memoized: this ran ``get_device_capability()`` on every sparse-layer forward
    (60x/step on the stress config) to answer a question whose answer cannot
    change within a process. Negligible where the GPU is saturated; the batch-1
    RNN is 31-37% GPU-busy across 13k kernels, i.e. mostly CPU gap, so per-call
    host work there is paid T times per sequence.
    """
    global _TILE
    if _TILE is None:
        cc = torch.cuda.get_device_capability()
        # Hopper/Blackwell like wider tiles; Ampere prefers 32.
        _TILE = (64 if cc[0] >= 9 else 32, 16)
    return _TILE


_OPTIN_SHARED = None  # device max opt-in dynamic shared mem per block (bytes), queried once


def _optin_shared() -> int:
    global _OPTIN_SHARED
    if _OPTIN_SHARED is None:
        ext = _load_ext()
        _OPTIN_SHARED = ext.cuda_gather_max_optin_shared() if ext is not None else 0
    return _OPTIN_SHARED


def _wpo_ok(IN: int, K: int) -> bool:
    """Whether the Phase-4 warp-per-output forward applies: the x-row must fit shared memory
    (IN*4 <= device opt-in max). The strided lane loop handles any K, so K is unrestricted.
    Otherwise the caller falls back to the v2 shared-tiled forward, which has no IN limit."""
    return IN * 4 <= _optin_shared()


def _pick_wpo_params() -> tuple[int, int]:
    """(P outputs/block, T threads) for the STAGED forward. P=2048, T=512.

    WARNING -- this was tuned at in_features=10,000, where the staged x-row is
    40 KB, stays under the 48 KB default, and leaves ~2 blocks/SM for ~66%
    occupancy. That premise DOES NOT HOLD for the configs in use: at
    in_features=16,000 the row is 64 KB, exactly one block fits per SM, and ncu
    measures 33.1% occupancy with DRAM at 1.11% and SM at 35.3%. The kernel did
    not regress; its design point moved when the configs grew past ~12,288
    inputs (the 48 KB crossing).

    That is what `_pick_fwd_variant` exists to route around. Two things here are
    still open, deliberately not changed without measurement:
      * P=4096 measured ~1.5% faster than 2048 at four of five real shapes. The
        original ladder topped out at the shipped value, so it never bracketed
        the optimum from above. Not adopted pending an end-to-end check.
      * whether 2 blocks/SM at 64 KB is truly unreachable or a near-miss against
        the per-SM shared budget (a carveout may recover it, with no kernel
        change, on the path that wins 22 of 25 shapes). See
        profilings/dendrinet_harness/rnn_prof/probe_shared_budget.py.

    T=512 is unchanged and still the sweet spot. Safe on Ampere (row < 48 KB
    there at the design point).

    DENDRINET_STAGED_FWD_P overrides P for A/B measurement without a code flip.
    """
    import os
    override = os.environ.get("DENDRINET_STAGED_FWD_P")
    if override:
        return int(override), 512
    return 2048, 512


# Which forward kernel to use. "auto" applies the measured rule below; "staged" /
# "nostage" force one variant, for A/B and for reverting without a code change.
_FWD_VARIANT = os.environ.get("DENDRINET_GATHER_FWD_VARIANT", "auto").strip().lower()



def _pick_fwd_variant(IN: int, O: int, K: int) -> tuple[str, int, int]:
    """Choose the forward gather kernel: ("staged"|"nostage"|"hybrid", P, T).

    The staged kernel asks for IN*4 bytes of shared, so it fits
    ``optin // (IN*4)`` blocks per SM. Once that is 1, it is occupancy-capped
    (ncu: 33.1% achieved, DRAM 1.11%, SM 35.3%) and the un-staged kernel -- which
    uses no shared at all and lets L2 serve the gathers, at a 99.44% hit rate --
    can win. It does not *always* win, because it issues ~5x more L2 requests,
    and that only pays back when there are enough outputs to fill the grid.

    Every band is fitted per-architecture from measured sweeps -- see the
    branch comments below and profilings/dendrinet_harness/rnn_prof/logs/
    edges_{blackwell,h100,a100}. The reuse=O*K/IN heuristic that originally
    generalized the K=40 fit was RETIRED once the dense-K edge sweeps showed
    it selecting measurably slower kernels (0.58-0.89x cells at IN=24k,
    K=50-60): the bands are lookup tables over measured cells, not mechanism
    extrapolations. check_picker_dispatch.py asserts code-vs-measurement
    agreement per architecture.
    """
    ns_P = 64 if O >= 96_000 else 32
    if _FWD_VARIANT == "staged":
        return ("staged", *_pick_wpo_params())
    if _FWD_VARIANT == "nostage":
        return ("nostage", ns_P, 512)
    optin = _optin_shared()
    if IN * 4 > optin:
        # Above the ceiling the staged kernel cannot launch, but the answer is NOT
        # the un-staged one: measured against the v2 shared-tiled forward it
        # already fell back to, ns is 0.89-0.96x (bench_bwd_above_ceiling.py). v2
        # tiles over IN and keeps idx+w in shared, which still pays when the row is
        # huge. Returning "staged" here routes through _wpo_ok to that v2 path.
        # (The BACKWARD goes the other way above the ceiling -- ns beats v2
        # 1.11-1.20x there. Do not assume the two directions agree.)
        return ("staged", *_pick_wpo_params())
    blocks_per_sm = optin // max(IN * 4, 1)
    cc = torch.cuda.get_device_capability()
    if cc[0] >= 10:
        # Blackwell (measured on RTX PRO 6000 Server, sm_120, optin 101,376 B).
        # Fitted from 5x5 (IN, O) sweeps at K=10/40/80 plus the dense-K edge
        # sweep (logs/edges_blackwell: K=15..70 on the deciding shapes, batches
        # 64/128/512, hybrid re-swept at 128 and 512). The dominant factor is K
        # itself, not reuse: staging amortizes its IN*4-byte row over K gathers
        # per output, so low K starves it and high K feeds it. Measured bands
        # at 1 block/SM:
        #   K<=20: no-stage beats staged at every O (1.17-1.89x); hybrid beats
        #          no-stage where measured (1.03-1.09x) and takes the band.
        #   K>20, IN<=20k: O>=96k stays no-stage at EVERY K through 80
        #          (1.18-1.53x measured at 25/30/40/50/60/70/80 -- the old
        #          interpolated K<=60 cap forfeited 1.27-1.28x at 70/80);
        #          O<=96k is the hybrid's region (1.19-1.34x at K=40, held at
        #          batch 128).
        #   K>20, IN>20k: no-stage survives only at O>=96k AND K<=40
        #          (1.06-1.30x); the win dies between K=40 and 50 (0.89x at
        #          50, 0.69x at 60) and the old reuse>=120 rule shipped
        #          exactly those regressions plus (24k,48k,K60)=0.58x. The
        #          reuse term is RETIRED here -- every band below is a
        #          measured cell, not a mechanism extrapolation.
        # Batch: winner sets identical at 64/128/512 for every band cell.
        # Batch-64-only anomaly: no-stage wins tiny-O cells (IN=3-4k, O=2000,
        # 1.22-1.27x on ~30us kernels) that stay staged -- declined; the
        # picker has no batch input and the absolute cost is launch noise.
        # Hybrid at (24k, O>=96k, K<=20) is untested interior, as originally.
        # sm_100 (B200) is unmeasured: its ~232 KB optin pushes these INs to
        # >=3 blocks/SM, where the blk/SM predicate already keeps staged.
        hy_P = 4096 if O >= 96_000 else (2048 if O >= 24_000 else 1024)
        if blocks_per_sm <= 1:
            if K <= 20:
                return ("hybrid", hy_P, 512)
            if IN <= 20_000:
                if K <= 60 and O <= 96_000 and (K <= 50 or O < 40_000):
                    # Hybrid FIRST here: at (16k, 48k, K40) it beats no-stage
                    # 3.51 vs 4.27 ms -- dispatch order is part of the fit.
                    # K edge verified at shipped config (bench_shipped_bands):
                    # 1.21x at (48k, K50) but 0.94x at (48k, K60) -> that
                    # corner is staged; small O keeps hybrid through K=60.
                    return ("hybrid", hy_P, 512)
                if O >= 96_000:
                    return ("nostage", ns_P, 512)
            elif O >= 96_000 and K <= 40:
                return ("nostage", ns_P, 512)
    elif cc == (8, 0):
        # A100 (sm80, optin 167,936 B), REFIT from the dense edge sweep
        # (logs/edges_a100). The original K=40-only rule ("no-stage whenever
        # 1 blk/SM") shipped regressions at high K + small O: measured 0.94x
        # (K50), 0.81x (K60), 0.70x (K70), 0.58-0.70x (K80, O<=16k), all at
        # IN=24k. Measured bands:
        #   1 blk/SM (IN > ~21k): no-stage EVERYWHERE for K<=40 (1.02-2.06x;
        #     one accepted 0.99 at O=16k, K=40); for K>40 only O>=40k keeps
        #     it (1.29-1.69x at 48k/144k -- the small-O side flips to staged).
        #   2 blk/SM (IN ~14-21k): the whole row for K<=15 (1.11-1.24x);
        #     O>=40k up to K<=25 (1.05-1.09x at 48k); the largest O>=100k at
        #     any K (1.29x at K20 tapering to ~1.01 by K>=60, never below 1.0
        #     -- band kept).
        # Batch: every band cell holds at 64/128 (1.01-1.78x). Tiny-O
        # batch-64 anomalies (IN=3-4k, O=2k, 1.00-1.03x) declined as on
        # Blackwell. No-stage P is K-banded from the sweeps' per-cell best-P
        # modes (512 low-K/small-O, 64 mid, 32 high-K); the shipped-band
        # verify times exactly this dispatch against staged per cell.
        sm80_p = 512 if (K <= 20 or O < 40_000) else (64 if K <= 40 else 32)
        if blocks_per_sm == 2 and 40_000 <= O < 100_000 and K > 10:
            # (16k, 48k) column: shipped-band verify measured P=512/64 LOSING
            # the win there (0.81-0.95x vs staged); the sweeps' best P is 128
            # at K=15/20/25 and 128 restores it. K=10 stays 512 (verified).
            sm80_p = 128
        if blocks_per_sm <= 1:
            if K <= 40 or O >= 40_000:
                return ("nostage", sm80_p, 512)
        elif blocks_per_sm == 2:
            # Trimmed to the cells that win on BOTH sm80 variants: the 40GB
            # card (~20% less bandwidth, same cc so the picker cannot tell)
            # tips the marginal cells negative -- (48k,K25) and (144k,K>=50)
            # measured 0.93-0.95x there vs 0.99-1.05x on 80GB. Dropping them
            # costs ~nothing on 80GB and keeps the never-worse guarantee.
            if K <= 15 or (K <= 20 and O >= 40_000) or (K <= 40 and O >= 100_000):
                return ("nostage", sm80_p, 512)
    elif cc[0] == 9:
        # H100 (sm90, optin 232,448 B), from the full edge sweep
        # (logs/edges_h100: 5x5 grids at K=10/40/80, batches 64/128/512, dense
        # K=15..70 on the deciding shapes, hybrid at two batches). The 2.3x
        # larger shared budget changes the answer wholesale: staged keeps
        # >=3 blocks/SM up to IN~19k and wins EVERY such cell at every K and
        # every batch swept. No-stage wins only at 2 blocks/SM (IN ~19-29k):
        #   K<=15: the whole row, 1.05-1.09x;
        #   K=20..60: only the very largest O (the 144k cell, 1.05-1.30x,
        #             peak at K=30); K>=70 nothing (0.96x/0.85x).
        # Winner set is IDENTICAL at batch 64/128/512 (1.11-1.15x) -- no batch
        # dependence found. The hybrid kernel LOSES everywhere here
        # (0.54-0.94x, both batches): it exists to fix an occupancy collapse
        # sm90 does not have in this IN range; measured negative, no band.
        # 1 block/SM (IN 29-58k) is beyond the measured grid -> staged default.
        # Fitted on H100; H200 is also cc 9.0 with the same SM/shared budget
        # (faster HBM shifts margins, not mechanisms) and takes this branch.
        if blocks_per_sm == 2 and (K <= 15 or (K <= 60 and O >= 100_000)):
            # best no-stage P on sm90 is 512 in nearly every winning cell
            return ("nostage", 512, 512)
    # Unmeasured architectures keep the staged kernel -- the safe default.
    return ("staged", *_pick_wpo_params())


def _bwd_wpo_ok(IN: int) -> bool:
    """Whether the Phase-4 split backward applies: the grad_x row must fit shared
    (IN*4 <= optin max). The grad_w-transpose kernel uses no dynamic shared (register/loop),
    so only grad_x constrains it. Otherwise fall back to the v2 backward (no IN limit)."""
    return IN * 4 <= _optin_shared()


def _pick_bwd_wpo_params(out_features: int | None = None) -> tuple[int, int]:
    """(P outputs/block, T threads) for the split backward, adaptive in ``P``.

    The backward launches ``grid = ceil(out_features / P)``. A fixed P=2048 left
    the GPU mostly idle at real layer shapes: out=144,000 gives 71 blocks and
    out=48,000 gives 24, i.e. 38% and 13% of a 188-SM Blackwell. Nsight measured
    33.1% achieved occupancy against 100% theoretical with DRAM at 4.2% and SM at
    6.6% -- not bandwidth- or compute-bound, just under-filled. Measured gains
    from lowering P (batch 512, Blackwell): 7.78x at out=2,000 (one block!),
    2.39x at out=48,000, 1.37x at out=144,000; 1.32x over the whole backward
    weighted by the layer mix of the 12-layer stress config.

    Smaller P is not free -- more blocks re-stage more -- and the optimum also
    depends on in_features, so a closed form over (O, SM) does not fit: every
    shape disagreed with ``O/(2*SM)``. These thresholds are the measured optima
    (``rnn_prof/bench_gradw_tile.py``), interpolated conservatively and clamped
    to the shipped value above the measured range.

    NOT applied to the forward: sweeping it showed 1.00x, because it stages the
    x-row in shared per block, so fewer/larger blocks amortise that staging and
    the occupancy gain cancels. Only this kernel is under-filled.

    ``P`` merely partitions outputs across blocks and each (o, k) stays owned by
    a single warp, so results are unchanged -- see rnn_prof/check_gradw_tile_equiv.py.
    """
    if out_features is None:
        return 2048, 512
    o = int(out_features)
    if o <= 4_000:
        p = 96
    elif o <= 24_000:
        p = 256
    elif o <= 96_000:
        p = 384
    elif o <= 200_000:
        p = 1024
    else:
        p = 2048
    return p, 512


class _CudaSparseGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx, weight, B_O, B_M):
        ext = _load_ext()
        in_dtype, w_dtype = x.dtype, weight.dtype
        xf = x.contiguous().to(torch.float32)
        wf = weight.contiguous().to(torch.float32)
        # Indices are STORED int32 (the lever that halved the replicated idx
        # buffer for scaling). This used to upcast them to int64 here, cast
        # straight back to int32 to call the kernel, and then RETAIN the int64
        # copy for backward -- ~2x the necessary bytes, undoing that lever on
        # every call. int64 is now materialized only inside the v2 forward
        # branch, which is the sole consumer that needs it; both backward paths
        # take int32.
        idx32 = idx.contiguous().to(torch.int32)
        IN, K = xf.size(1), idx32.size(1)
        variant, P, T = _pick_fwd_variant(IN, idx32.size(0), K)
        if variant == "hybrid":                # prefix-staged: multi-block occupancy + L2 tail
            out = ext.gather_fwd_wpo_hy_cuda(xf, idx32, wf, P, T,
                                             8192 if K <= 20 else 12288)
        elif variant == "nostage":             # no shared -> not occupancy-capped, no IN ceiling
            out = ext.gather_fwd_wpo_ns_cuda(xf, idx32, wf, P, T)
        elif _wpo_ok(IN, K):                   # Phase-4 warp-per-output forward (~4.5x over v2)
            out = ext.gather_fwd_wpo_cuda(xf, idx32, wf, P, T)
        else:                                  # fall back to v2 (x-row too large for shared)
            out = ext.gather_fwd_cuda(xf, idx32.to(torch.int64), wf, B_O, B_M)
        ctx.save_for_backward(xf, idx32, wf)   # int32: both backward paths want it
        ctx.cfg = (B_O, B_M, in_dtype, w_dtype)
        return out.to(in_dtype)

    @staticmethod
    def backward(ctx, grad_out):
        ext = _load_ext()
        xf, idx, wf = ctx.saved_tensors
        B_O, B_M, in_dtype, w_dtype = ctx.cfg
        gof = grad_out.contiguous().to(torch.float32)
        IN = xf.size(1)
        if _bwd_wpo_ok(IN):                    # Phase-4 split backward (~2.6x over v2)
            # P adapts to out_features so the grid actually fills the GPU.
            P, T = _pick_bwd_wpo_params(gof.size(1))
            xT = xf.t().contiguous()           # [IN, M]  (x is small; transpose is cheap)
            goT = gof.t().contiguous()         # [O, M]   (grad_out transpose, charged per call)
            grad_x, grad_w = ext.gather_bwd_wpo_cuda(gof, goT, xT, idx.to(torch.int32), wf, P, T)
        else:
            # grad_x row does not fit shared, so the staged split backward cannot
            # run. The un-staged grad_x needs no shared and DOES run here, and it
            # beats the v2 fallback 1.11-1.20x above the ceiling (measured,
            # bench_bwd_above_ceiling.py) -- even though it loses 2.1x to staged
            # below it, where staged is available. Different comparison, different
            # winner.
            P, T = _pick_bwd_wpo_params(gof.size(1))
            xT = xf.t().contiguous()
            goT = gof.t().contiguous()
            grad_x, grad_w = ext.gather_bwd_wpo_cuda2(
                gof, goT, xT, idx.to(torch.int32), wf, P, T, 1)
        needs_x, _, needs_w = ctx.needs_input_grad[:3]
        return (
            grad_x.to(in_dtype) if needs_x else None,
            None,
            grad_w.to(w_dtype) if needs_w else None,
            None,
            None,
        )


def cuda_sparse_gather(
    x: torch.Tensor,
    connection_indices: torch.Tensor,
    weight: torch.Tensor,
    chunk_size: int = 2048,       # accepted for signature parity; unused (kernels tile internally)
    *,
    B_O: int | None = None,
    B_M: int | None = None,
) -> torch.Tensor:
    """Return ``out[m,o] = sum_k x[m, connection_indices[o,k]] * weight[o,k]`` via CUDA.

    Drop-in for :func:`fused_sparse_gather`. Requires a CUDA tensor and a working build
    toolchain (see :func:`cuda_gather_available`). ``chunk_size`` is ignored.
    """
    if not cuda_gather_available():
        raise RuntimeError(
            "cuda_sparse_gather unavailable (no CUDA device, no nvcc, or compilation failed). "
            "Use use_triton_gather / use_fused_gather instead."
        )
    if not x.is_cuda:
        raise RuntimeError("cuda_sparse_gather requires a CUDA tensor")
    if B_O is None or B_M is None:
        B_O, B_M = _pick_tile()
    return _CudaSparseGather.apply(x, connection_indices, weight, B_O, B_M)


__all__ = ["cuda_sparse_gather", "cuda_gather_available"]
