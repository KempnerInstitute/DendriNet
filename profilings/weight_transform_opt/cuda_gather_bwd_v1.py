"""Phase 3, Step 2: CUDA backward scatter v1 -- naive global atomics (the baseline).

The backward computes both gradients of  out[m,o] = sum_k x[m,idx[o,k]]*w[o,k]:
    grad_x[m, idx[o,k]] += grad_out[m,o] * w[o,k]        (SCATTER -> contention)
    grad_w[o,k]         += sum_m grad_out[m,o]*x[m,idx]   (REDUCE over batch)

v1 is deliberately naive: one thread per (m,o), and every write is a global atomicAdd
-- exactly what index_add_ and the Triton backward do. This establishes correctness and
the baseline. v2 will privatize the accumulation in shared memory to cut global atomics.

The bar to clear is torch's index_add_ backward (already fast). Checks correctness
against an index_add_ reference and times both.

Run (after `source setup_cuda_env.sh`):
    python cuda_gather_bwd_v1.py
    python cuda_gather_bwd_v1.py --O 90000 --K 40 --block 256
"""
import os, argparse, statistics, time
import torch
from torch.utils.cpp_extension import CUDA_HOME, load_inline

assert torch.cuda.is_available(), "need CUDA"
assert CUDA_HOME is not None, "CUDA_HOME is None -- source setup_cuda_env.sh first"

p = argparse.ArgumentParser()
p.add_argument("--M", type=int, default=128)
p.add_argument("--in_features", type=int, default=10000)
p.add_argument("--O", type=int, default=90000)
p.add_argument("--K", type=int, default=40)
p.add_argument("--block", type=int, default=256, help="threads per block (over outputs)")
p.add_argument("--iters", type=int, default=50)
args = p.parse_args()

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"shape: M={args.M} IN={args.in_features} O={args.O} K={args.K} block={args.block}\n")

BUILD = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"cuda_gather_bwd_{os.getuid()}")
os.makedirs(BUILD, exist_ok=True)

cpp_src = ("std::vector<torch::Tensor> gather_bwd_v1_cuda("
           "torch::Tensor grad_out, torch::Tensor x, torch::Tensor idx, torch::Tensor w, int block);")
cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// v1: one thread per (m, o). Both gradients via GLOBAL atomicAdd (naive baseline).
__global__ void gather_bwd_v1_kernel(
        const float* __restrict__ grad_out,  // [M, O]
        const float* __restrict__ x,         // [M, IN]
        const long*  __restrict__ idx,       // [O, K]
        const float* __restrict__ w,         // [O, K]
        float* __restrict__ grad_x,          // [M, IN]  (pre-zeroed)
        float* __restrict__ grad_w,          // [O, K]   (pre-zeroed)
        int M, int O, int IN, int K) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if (o >= O || m >= M) return;

    float go = grad_out[(long)m * O + o];
    const long*  idx_o = idx + (long)o * K;
    const float* w_o   = w   + (long)o * K;
    const float* x_m   = x   + (long)m * IN;
    float*       gx_m  = grad_x + (long)m * IN;

    for (int k = 0; k < K; ++k) {
        long ix = idx_o[k];
        // grad_x scatter: many (o,k) in this row may hit the same ix -> contention
        atomicAdd(&gx_m[ix], go * w_o[k]);
        // grad_w reduce: all M rows (different m, same o,k) accumulate here
        atomicAdd(&grad_w[(long)o * K + k], go * x_m[ix]);
    }
}

std::vector<torch::Tensor> gather_bwd_v1_cuda(
        torch::Tensor grad_out, torch::Tensor x, torch::Tensor idx, torch::Tensor w, int block) {
    TORCH_CHECK(x.is_cuda() && idx.is_cuda() && w.is_cuda() && grad_out.is_cuda(), "CUDA only");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && w.scalar_type() == torch::kFloat32
                && grad_out.scalar_type() == torch::kFloat32, "float32 only");
    TORCH_CHECK(idx.scalar_type() == torch::kInt64, "idx must be int64");
    grad_out = grad_out.contiguous(); x = x.contiguous();
    idx = idx.contiguous(); w = w.contiguous();
    int M = x.size(0), IN = x.size(1), O = idx.size(0), K = idx.size(1);

    auto grad_x = torch::zeros({M, IN}, x.options());
    auto grad_w = torch::zeros({O, K}, w.options());

    dim3 threads(block);
    dim3 grid((O + block - 1) / block, M);
    gather_bwd_v1_kernel<<<grid, threads>>>(
        grad_out.data_ptr<float>(), x.data_ptr<float>(), idx.data_ptr<long>(),
        w.data_ptr<float>(), grad_x.data_ptr<float>(), grad_w.data_ptr<float>(),
        M, O, IN, K);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "gather_bwd_v1_kernel launch failed");
    return {grad_x, grad_w};
}
"""

print("=== compiling (first run ~60-80s) ===")
t0 = time.time()
mod = load_inline(name="cuda_gather_bwd", cpp_sources=cpp_src, cuda_sources=cuda_src,
                  functions=["gather_bwd_v1_cuda"], build_directory=BUILD, verbose=True)
print(f"compiled in {time.time() - t0:.1f}s\n")

dev = "cuda"
torch.manual_seed(0)
x = torch.randn(args.M, args.in_features, device=dev)
idx = torch.randint(0, args.in_features, (args.O, args.K), device=dev)
w = torch.randn(args.O, args.K, device=dev)
grad_out = torch.randn(args.M, args.O, device=dev)


def ref_backward(grad_out, x, idx, w):                 # the index_add_ path (bar to clear)
    M, IN = x.shape; O, K = idx.shape
    idx_flat = idx.reshape(-1)
    g = grad_out.unsqueeze(-1)
    grad_x = torch.zeros(M, IN, device=x.device)
    grad_x.index_add_(1, idx_flat, (g * w.unsqueeze(0)).reshape(M, O * K))
    sel = x.index_select(1, idx_flat).reshape(M, O, K)
    grad_w = (g * sel).sum(dim=0)
    return grad_x, grad_w


gx_ref, gw_ref = ref_backward(grad_out, x, idx, w)
gx, gw = mod.gather_bwd_v1_cuda(grad_out, x, idx, w, args.block)
gx_ok = torch.allclose(gx, gx_ref, rtol=1e-3, atol=1e-3)
gw_ok = torch.allclose(gw, gw_ref, rtol=1e-3, atol=1e-3)
print(f"grad_x: {'OK' if gx_ok else 'FAIL'}  max_err={(gx-gx_ref).abs().max().item():.2e}")
print(f"grad_w: {'OK' if gw_ok else 'FAIL'}  max_err={(gw-gw_ref).abs().max().item():.2e}\n")


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
t_cuda = bench(lambda: mod.gather_bwd_v1_cuda(grad_out, x, idx, w, args.block))
print(f"index_add_ backward : {t_ref:.3f} ms")
print(f"cuda v1 backward    : {t_cuda:.3f} ms   ({t_ref / t_cuda:.2f}x vs index_add_)")
print("\nread: correctness FIRST (both grads OK; grad_x ~1e-3 from atomic reorder is fine).")
print("      Then vs index_add_: v1 is NAIVE global atomics -- it may already win, or lose to")
print("      contention. Either way it's the baseline; v2 privatizes grad_w (and maybe grad_x)")
print("      in shared memory to cut global atomics -- the real CUDA experiment.")
