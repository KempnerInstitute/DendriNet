"""Phase 3, Step 3: CUDA backward v2 -- shared-memory privatization of grad_w.

v1 was naive: ~460M global atomicAdds EACH for grad_x and grad_w -> lost to index_add_
(0.86x Blackwell) / tied (1.01x A100). v2 attacks grad_w only, to isolate the win:

  grad_w[o,k] = sum_m grad_out[m,o]*x[m,idx]   -- a reduction over the batch.

Instead of every one of the B_M rows in a block firing a GLOBAL atomic at grad_w[o,k],
the block keeps a __shared__ grad_w accumulator for its output tile. Threads atomicAdd
into SHARED (~10x faster), then flush ONE global atomic per [o,k] per block. Global
grad_w atomics drop by ~B_M.

grad_x is left as naive global atomics on purpose -- so any speedup here is attributable
to the grad_w privatization alone. (grad_x's scatter->gather transpose is v3.)

Run (after `source setup_cuda_env.sh`):
    python cuda_gather_bwd_v2.py --B_O 32 --B_M 16
"""
import os, argparse, statistics, time
import torch
from torch.utils.cpp_extension import CUDA_HOME, load_inline

assert torch.cuda.is_available() and CUDA_HOME is not None, "source setup_cuda_env.sh first"

p = argparse.ArgumentParser()
p.add_argument("--M", type=int, default=128)
p.add_argument("--in_features", type=int, default=10000)
p.add_argument("--O", type=int, default=90000)
p.add_argument("--K", type=int, default=40)
p.add_argument("--B_O", type=int, default=32)
p.add_argument("--B_M", type=int, default=16)
p.add_argument("--iters", type=int, default=50)
args = p.parse_args()

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"shape: M={args.M} IN={args.in_features} O={args.O} K={args.K}  "
      f"tile B_O={args.B_O} B_M={args.B_M}\n")

BUILD = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"cuda_gather_bwd_v2_{os.getuid()}")
os.makedirs(BUILD, exist_ok=True)

cpp_src = ("std::vector<torch::Tensor> gather_bwd_v2_cuda("
           "torch::Tensor grad_out, torch::Tensor x, torch::Tensor idx, torch::Tensor w, int B_O, int B_M);")
cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Tile = B_O outputs x B_M rows. Shared: [ idx | w | grad_w-accumulator ], each B_O*K.
// grad_w accumulates in shared (fast atomics), one global atomic per [o,k] per block.
// grad_x stays naive global atomics (isolated so the win is attributable to grad_w).
__global__ void gather_bwd_v2_kernel(
        const float* __restrict__ grad_out, const float* __restrict__ x,
        const long*  __restrict__ idx,      const float* __restrict__ w,
        float* __restrict__ grad_x, float* __restrict__ grad_w,
        int M, int O, int IN, int K, int B_O, int B_M) {
    extern __shared__ char smem[];
    long*  s_idx = reinterpret_cast<long*>(smem);
    float* s_w   = reinterpret_cast<float*>(smem + (size_t)B_O * K * sizeof(long));
    float* s_gw  = reinterpret_cast<float*>(smem + (size_t)B_O * K * (sizeof(long) + sizeof(float)));

    int o_base = blockIdx.x * B_O;
    int m_base = blockIdx.y * B_M;
    int tx = threadIdx.x, ty = threadIdx.y;
    int T = B_M * B_O, tid = ty * B_O + tx;
    int tile = B_O * K;

    // stage idx/w, zero the shared grad_w accumulator
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
        float*       gx_m = grad_x + (long)m * IN;
        for (int k = 0; k < K; ++k) {
            long ix = s_idx[tx * K + k];
            float wv = s_w[tx * K + k];
            atomicAdd(&gx_m[ix], go * wv);                 // grad_x: global (unchanged)
            atomicAdd(&s_gw[tx * K + k], go * x_m[ix]);    // grad_w: SHARED atomic (fast)
        }
    }
    __syncthreads();

    // flush shared grad_w -> global: one atomic per [o,k] per block (was B_M per block)
    for (int i = tid; i < tile; i += T) {
        int oo = o_base + i / K, kk = i % K;
        if (oo < O) atomicAdd(&grad_w[(long)oo * K + kk], s_gw[i]);
    }
}

std::vector<torch::Tensor> gather_bwd_v2_cuda(
        torch::Tensor grad_out, torch::Tensor x, torch::Tensor idx, torch::Tensor w,
        int B_O, int B_M) {
    TORCH_CHECK(x.is_cuda() && idx.is_cuda() && w.is_cuda() && grad_out.is_cuda(), "CUDA only");
    TORCH_CHECK(idx.scalar_type() == torch::kInt64, "idx must be int64");
    grad_out = grad_out.contiguous(); x = x.contiguous(); idx = idx.contiguous(); w = w.contiguous();
    int M = x.size(0), IN = x.size(1), O = idx.size(0), K = idx.size(1);
    auto grad_x = torch::zeros({M, IN}, x.options());
    auto grad_w = torch::zeros({O, K}, w.options());

    dim3 block(B_O, B_M);
    dim3 grid((O + B_O - 1) / B_O, (M + B_M - 1) / B_M);
    size_t shmem = (size_t)B_O * K * (sizeof(long) + 2 * sizeof(float));
    TORCH_CHECK(shmem <= 48 * 1024, "shared tile > 48KB; lower B_O");

    gather_bwd_v2_kernel<<<grid, block, shmem>>>(
        grad_out.data_ptr<float>(), x.data_ptr<float>(), idx.data_ptr<long>(),
        w.data_ptr<float>(), grad_x.data_ptr<float>(), grad_w.data_ptr<float>(),
        M, O, IN, K, B_O, B_M);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "gather_bwd_v2_kernel launch failed");
    return {grad_x, grad_w};
}
"""

print("=== compiling (first run ~60-80s) ===")
t0 = time.time()
mod = load_inline(name="cuda_gather_bwd_v2", cpp_sources=cpp_src, cuda_sources=cuda_src,
                  functions=["gather_bwd_v2_cuda"], build_directory=BUILD, verbose=True)
print(f"compiled in {time.time() - t0:.1f}s\n")

dev = "cuda"
torch.manual_seed(0)
x = torch.randn(args.M, args.in_features, device=dev)
idx = torch.randint(0, args.in_features, (args.O, args.K), device=dev)
w = torch.randn(args.O, args.K, device=dev)
grad_out = torch.randn(args.M, args.O, device=dev)


def ref_backward(grad_out, x, idx, w):
    M, IN = x.shape; O, K = idx.shape
    idx_flat = idx.reshape(-1)
    g = grad_out.unsqueeze(-1)
    grad_x = torch.zeros(M, IN, device=x.device)
    grad_x.index_add_(1, idx_flat, (g * w.unsqueeze(0)).reshape(M, O * K))
    sel = x.index_select(1, idx_flat).reshape(M, O, K)
    grad_w = (g * sel).sum(dim=0)
    return grad_x, grad_w


gx_ref, gw_ref = ref_backward(grad_out, x, idx, w)
gx, gw = mod.gather_bwd_v2_cuda(grad_out, x, idx, w, args.B_O, args.B_M)
print(f"grad_x: {'OK' if torch.allclose(gx, gx_ref, rtol=1e-3, atol=1e-3) else 'FAIL'}  "
      f"max_err={(gx-gx_ref).abs().max().item():.2e}")
print(f"grad_w: {'OK' if torch.allclose(gw, gw_ref, rtol=1e-3, atol=1e-3) else 'FAIL'}  "
      f"max_err={(gw-gw_ref).abs().max().item():.2e}\n")


def bench(fn):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(args.iters):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)


t_ref = bench(lambda: ref_backward(grad_out, x, idx, w))
t_v2 = bench(lambda: mod.gather_bwd_v2_cuda(grad_out, x, idx, w, args.B_O, args.B_M))
print(f"index_add_ backward : {t_ref:.3f} ms")
print(f"cuda v2 backward    : {t_v2:.3f} ms   ({t_ref / t_v2:.2f}x vs index_add_)")
print("\nread: v2 should improve on v1 (0.86x BW / 1.01x A100) by cutting grad_w global atomics")
print("      ~B_M-fold. If it beats index_add_ -> grad_w was a real chunk. If it still trails,")
print("      grad_x's naive scatter dominates -> v3 (transpose scatter->gather) is the fix.")
