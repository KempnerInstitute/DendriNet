"""Phase 3, Step 1: CUDA forward gather v1 -- parity + baseline timing.

Goal is PARITY, not a speed record yet: establish the CUDA kernel + build + timing
scaffold with the EASY direction (the forward gather has no scatter contention), then
we tackle the hard backward (shared-memory scatter) on top of a working harness.

v1 kernel: one thread per output element (m, o); each thread loops K, gathering
x[m, idx[o,k]] * w[o,k]. idx/w are re-read per batch row (not amortized) -- the naive
baseline, mirroring Triton v1. v2 will tile outputs and stage idx/w in shared memory.

Compares against an eager reference and reports correctness + speedup at the real
large-synapse shape.

Run (on a GPU node, after `module load cuda`):
    python cuda_gather_fwd_v1.py
    python cuda_gather_fwd_v1.py --O 90000 --K 40 --block 256
"""
import os, argparse, statistics, time
import torch
from torch.utils.cpp_extension import CUDA_HOME, load_inline

assert torch.cuda.is_available(), "need CUDA"
assert CUDA_HOME is not None, "CUDA_HOME is None -- run `module load cuda` on a GPU node first"

p = argparse.ArgumentParser()
p.add_argument("--M", type=int, default=128)
p.add_argument("--in_features", type=int, default=10000)
p.add_argument("--O", type=int, default=90000)
p.add_argument("--K", type=int, default=40)
p.add_argument("--block", type=int, default=256, help="threads per block (over outputs)")
p.add_argument("--iters", type=int, default=50)
args = p.parse_args()

print(f"GPU: {torch.cuda.get_device_name(0)}   CUDA_HOME={CUDA_HOME}")
print(f"shape: M={args.M} IN={args.in_features} O={args.O} K={args.K} block={args.block}\n")

BUILD = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"cuda_gather_{os.getuid()}")
os.makedirs(BUILD, exist_ok=True)

cpp_src = "torch::Tensor gather_fwd_cuda(torch::Tensor x, torch::Tensor idx, torch::Tensor w, int block);"
cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// one thread per output element (m, o); loop K, gather-multiply-accumulate.
__global__ void gather_fwd_kernel(
        const float* __restrict__ x,      // [M, IN]
        const long*  __restrict__ idx,    // [O, K]
        const float* __restrict__ w,      // [O, K]
        float*       __restrict__ out,    // [M, O]
        int M, int O, int IN, int K) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;   // which output
    int m = blockIdx.y;                              // which batch row
    if (o >= O || m >= M) return;

    const long*  idx_o = idx + (long)o * K;
    const float* w_o   = w   + (long)o * K;
    const float* x_m   = x   + (long)m * IN;

    float acc = 0.0f;
    #pragma unroll 8
    for (int k = 0; k < K; ++k) {
        acc += x_m[idx_o[k]] * w_o[k];
    }
    out[(long)m * O + o] = acc;
}

torch::Tensor gather_fwd_cuda(torch::Tensor x, torch::Tensor idx, torch::Tensor w, int block) {
    TORCH_CHECK(x.is_cuda() && idx.is_cuda() && w.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w.scalar_type() == torch::kFloat32, "w must be float32");
    TORCH_CHECK(idx.scalar_type() == torch::kInt64, "idx must be int64");
    x = x.contiguous(); idx = idx.contiguous(); w = w.contiguous();
    int M = x.size(0), IN = x.size(1), O = idx.size(0), K = idx.size(1);
    auto out = torch::empty({M, O}, x.options());

    dim3 threads(block);
    dim3 grid((O + block - 1) / block, M);
    gather_fwd_kernel<<<grid, threads>>>(
        x.data_ptr<float>(), idx.data_ptr<long>(), w.data_ptr<float>(),
        out.data_ptr<float>(), M, O, IN, K);
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}
"""

print("=== compiling CUDA extension (first run invokes nvcc; ~30-60s) ===")
t0 = time.time()
mod = load_inline(name="cuda_gather_fwd", cpp_sources=cpp_src, cuda_sources=cuda_src,
                  functions=["gather_fwd_cuda"], build_directory=BUILD, verbose=True)
print(f"compiled in {time.time() - t0:.1f}s\n")

dev = "cuda"
torch.manual_seed(0)
x = torch.randn(args.M, args.in_features, device=dev)
idx = torch.randint(0, args.in_features, (args.O, args.K), device=dev)   # int64
w = torch.randn(args.O, args.K, device=dev)


def eager_forward(x, idx, w):
    M = x.shape[0]; O, K = idx.shape
    sel = x[:, idx.reshape(-1)].reshape(M, O, K)
    return (sel * w.unsqueeze(0)).sum(dim=-1)


ref = eager_forward(x, idx, w)
out = mod.gather_fwd_cuda(x, idx, w, args.block)
err = (out - ref).abs().max().item()
ok = torch.allclose(out, ref, rtol=1e-4, atol=1e-5)
print(f"correctness vs eager: {'OK' if ok else 'FAIL'}  (max_err={err:.2e})\n")


def bench(fn):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(args.iters):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)


t_eager = bench(lambda: eager_forward(x, idx, w))
t_cuda = bench(lambda: mod.gather_fwd_cuda(x, idx, w, args.block))
print(f"eager forward : {t_eager:.3f} ms")
print(f"cuda  forward : {t_cuda:.3f} ms   ({t_eager / t_cuda:.2f}x vs eager)")
print("\nread: v1 is the naive thread-per-output baseline -- expect it to BEAT eager (fused,")
print("      no [M,O,K] intermediate) but likely TRAIL Triton v2 (which amortizes idx/w across")
print("      the batch). If correct, we have a working CUDA harness; next we tile + use shared")
print("      memory for the forward, then build the real prize: the shared-memory scatter backward.")
