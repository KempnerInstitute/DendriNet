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

import torch

_EXT = None            # the compiled extension module (or None)
_COMPILE_TRIED = False  # so we attempt the (slow) compile at most once per process

_CPP_SRC = (
    "torch::Tensor gather_fwd_cuda(torch::Tensor x, torch::Tensor idx, torch::Tensor w, int B_O, int B_M);\n"
    "torch::Tensor gather_fwd_wpo_cuda(torch::Tensor x, torch::Tensor idx, torch::Tensor w, int P, int T);\n"
    "std::vector<torch::Tensor> gather_bwd_cuda(torch::Tensor grad_out, torch::Tensor x, "
    "torch::Tensor idx, torch::Tensor w, int B_O, int B_M);\n"
    "std::vector<torch::Tensor> gather_bwd_wpo_cuda(torch::Tensor grad_out, torch::Tensor goT, "
    "torch::Tensor xT, torch::Tensor idx, torch::Tensor w, int P, int T);\n"
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
            functions=["gather_fwd_cuda", "gather_fwd_wpo_cuda", "gather_bwd_cuda",
                       "gather_bwd_wpo_cuda", "cuda_gather_max_optin_shared"],
            verbose=False,
        )
    except Exception:  # noqa: BLE001  (missing nvcc, compile error, arch mismatch, ...)
        _EXT = None
    return _EXT


def cuda_gather_available() -> bool:
    """Whether the CUDA gather path can be used. Triggers the one-time lazy compile."""
    return _load_ext() is not None


def _pick_tile() -> tuple[int, int]:
    """Per-arch tile for the v2 fallback forward/backward: B_M=16 is the sweet spot on both
    A100 and Blackwell; B_O by arch."""
    cc = torch.cuda.get_device_capability()
    b_o = 64 if cc[0] >= 9 else 32     # Hopper/Blackwell like wider tiles; Ampere prefers 32
    return b_o, 16


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
    """(P outputs/block, T threads). Tuned on Blackwell (P=2048, T=512 -> ~4.5x over v2);
    T=512 (16 warps) is the occupancy sweet spot and 40KB-class rows leave ~2 blocks/SM.
    Safe on Ampere (row still < 48KB there)."""
    return 2048, 512


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
        idx = idx.contiguous().to(torch.int64)
        IN, K = xf.size(1), idx.size(1)
        if _wpo_ok(IN, K):                     # Phase-4 warp-per-output forward (~4.5x over v2)
            P, T = _pick_wpo_params()
            out = ext.gather_fwd_wpo_cuda(xf, idx.to(torch.int32), wf, P, T)
        else:                                  # fall back to v2 (x-row too large for shared)
            out = ext.gather_fwd_cuda(xf, idx, wf, B_O, B_M)
        ctx.save_for_backward(xf, idx, wf)     # int64 idx kept for the v2 backward
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
        else:                                  # fall back to v2 (grad_x row too large for shared)
            grad_x, grad_w = ext.gather_bwd_cuda(gof, xf, idx, wf, B_O, B_M)
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
