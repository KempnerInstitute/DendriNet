"""CUDA fused sparse gather-multiply-sum: forward + backward in one Function.

Production form of the Phase-3 winning kernels:
  * forward  : shared-memory tiling of idx/w (v2) -- 2.65x (A100) / 1.74x (Blackwell) vs eager
  * backward : shared-memory privatization of grad_w + cheap global-atomic grad_x (v2)
               -- 1.54x / 1.82x vs index_add_
Both beat the Triton path on both GPUs. (A v3 grad_x scatter->gather transpose was tested and
rejected: 2.2-2.4x slower, grad_x was never the bottleneck.)

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
    "std::vector<torch::Tensor> gather_bwd_cuda(torch::Tensor grad_out, torch::Tensor x, "
    "torch::Tensor idx, torch::Tensor w, int B_O, int B_M);"
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
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "gather_fwd_v2 launch failed");
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
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "gather_bwd_v2 launch failed");
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
            functions=["gather_fwd_cuda", "gather_bwd_cuda"],
            verbose=False,
        )
    except Exception:  # noqa: BLE001  (missing nvcc, compile error, arch mismatch, ...)
        _EXT = None
    return _EXT


def cuda_gather_available() -> bool:
    """Whether the CUDA gather path can be used. Triggers the one-time lazy compile."""
    return _load_ext() is not None


def _pick_tile() -> tuple[int, int]:
    """Per-arch tile: B_M=16 is the sweet spot on both A100 and Blackwell; B_O by arch."""
    cc = torch.cuda.get_device_capability()
    b_o = 64 if cc[0] >= 9 else 32     # Hopper/Blackwell like wider tiles; Ampere prefers 32
    return b_o, 16


class _CudaSparseGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx, weight, B_O, B_M):
        ext = _load_ext()
        in_dtype, w_dtype = x.dtype, weight.dtype
        xf = x.contiguous().to(torch.float32)
        wf = weight.contiguous().to(torch.float32)
        idx = idx.contiguous().to(torch.int64)
        out = ext.gather_fwd_cuda(xf, idx, wf, B_O, B_M)
        ctx.save_for_backward(xf, idx, wf)
        ctx.cfg = (B_O, B_M, in_dtype, w_dtype)
        return out.to(in_dtype)

    @staticmethod
    def backward(ctx, grad_out):
        ext = _load_ext()
        xf, idx, wf = ctx.saved_tensors
        B_O, B_M, in_dtype, w_dtype = ctx.cfg
        gof = grad_out.contiguous().to(torch.float32)
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
