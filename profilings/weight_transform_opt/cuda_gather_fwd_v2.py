"""Phase 3, Step 1b: CUDA forward gather v2 -- shared-memory staging of idx/w.

v1 lesson: one thread per (m,o), grid (O/block, M). For a given output o, all M batch
rows re-read idx[o,:]/w[o,:] from GLOBAL memory independently -> ~M-fold redundant index
traffic. Only 1.21x vs eager, TRAILING Triton v2 (1.56x).

v2 fix (shared-memory 101): a block handles a tile of B_O outputs x B_M batch rows.
The block's threads COOPERATIVELY load that tile's idx/w into on-chip shared memory
ONCE, __syncthreads(), then every (m,o) thread reads idx/w from fast shared memory,
reused across all B_M rows. Redundant idx/w global traffic drops by ~B_M. (The
x[m,idx] gather stays uncoalesced -- random reads are inherent to sparse gather.)

Run (on a GPU node, after `source setup_cuda_env.sh`):
    python cuda_gather_fwd_v2.py
    python cuda_gather_fwd_v2.py --B_O 64 --B_M 8         # sweep the tile
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
p.add_argument("--B_O", type=int, default=32, help="outputs per block tile (blockDim.x)")
p.add_argument("--B_M", type=int, default=8,  help="batch rows per block tile (blockDim.y)")
p.add_argument("--iters", type=int, default=50)
args = p.parse_args()

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"shape: M={args.M} IN={args.in_features} O={args.O} K={args.K}  "
      f"tile B_O={args.B_O} B_M={args.B_M} ({args.B_O*args.B_M} threads/block)\n")

BUILD = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"cuda_gather_v2_{os.getuid()}")
os.makedirs(BUILD, exist_ok=True)

cpp_src = "torch::Tensor gather_fwd_v2_cuda(torch::Tensor x, torch::Tensor idx, torch::Tensor w, int B_O, int B_M);"
cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tile = B_O outputs x B_M batch rows. Shared memory stages this tile's idx/w once
// (they do not depend on the batch row m), reused by all B_M rows in the block.
// Dynamic shared memory layout: [ B_O*K longs (idx) | B_O*K floats (w) ].
__global__ void gather_fwd_v2_kernel(
        const float* __restrict__ x,      // [M, IN]
        const long*  __restrict__ idx,    // [O, K]
        const float* __restrict__ w,      // [O, K]
        float*       __restrict__ out,    // [M, O]
        int M, int O, int IN, int K, int B_O, int B_M) {
    extern __shared__ char smem_raw[];
    long*  s_idx = reinterpret_cast<long*>(smem_raw);
    float* s_w   = reinterpret_cast<float*>(smem_raw + (size_t)B_O * K * sizeof(long));

    int o_base = blockIdx.x * B_O;        // first output of this tile
    int m_base = blockIdx.y * B_M;        // first batch row of this tile
    int tx = threadIdx.x;                 // 0..B_O-1  (output within tile)
    int ty = threadIdx.y;                 // 0..B_M-1  (batch row within tile)
    int T   = B_M * B_O;                  // threads per block
    int tid = ty * B_O + tx;

    // --- cooperative load: all T threads fill B_O*K shared entries of idx/w once ---
    int tile = B_O * K;
    for (int i = tid; i < tile; i += T) {
        int oo = o_base + i / K;          // which output
        int kk = i % K;
        if (oo < O) {
            s_idx[i] = idx[(long)oo * K + kk];
            s_w[i]   = w[(long)oo * K + kk];
        }
    }
    __syncthreads();                      // shared idx/w now visible to the whole block

    // --- compute: each thread does one (m, o), reading idx/w from SHARED ---
    int o = o_base + tx;
    int m = m_base + ty;
    if (o < O && m < M) {
        const float* x_m = x + (long)m * IN;
        float acc = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < K; ++k) {
            long ix = s_idx[tx * K + k];      // fast: on-chip
            acc += x_m[ix] * s_w[tx * K + k]; // x gather still random/global
        }
        out[(long)m * O + o] = acc;
    }
}

torch::Tensor gather_fwd_v2_cuda(torch::Tensor x, torch::Tensor idx, torch::Tensor w,
                                 int B_O, int B_M) {
    TORCH_CHECK(x.is_cuda() && idx.is_cuda() && w.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w.scalar_type() == torch::kFloat32, "w must be float32");
    TORCH_CHECK(idx.scalar_type() == torch::kInt64, "idx must be int64");
    x = x.contiguous(); idx = idx.contiguous(); w = w.contiguous();
    int M = x.size(0), IN = x.size(1), O = idx.size(0), K = idx.size(1);
    auto out = torch::empty({M, O}, x.options());

    dim3 block(B_O, B_M);
    dim3 grid((O + B_O - 1) / B_O, (M + B_M - 1) / B_M);
    size_t shmem = (size_t)B_O * K * (sizeof(long) + sizeof(float));
    TORCH_CHECK(shmem <= 48 * 1024,
                "shared tile > 48KB; lower B_O or opt in to larger dynamic shared memory");

    gather_fwd_v2_kernel<<<grid, block, shmem>>>(
        x.data_ptr<float>(), idx.data_ptr<long>(), w.data_ptr<float>(),
        out.data_ptr<float>(), M, O, IN, K, B_O, B_M);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "gather_fwd_v2_kernel launch failed");
    return out;
}
"""

print("=== compiling (first run ~60-80s) ===")
t0 = time.time()
mod = load_inline(name="cuda_gather_v2", cpp_sources=cpp_src, cuda_sources=cuda_src,
                  functions=["gather_fwd_v2_cuda"], build_directory=BUILD, verbose=True)
print(f"compiled in {time.time() - t0:.1f}s\n")

dev = "cuda"
torch.manual_seed(0)
x = torch.randn(args.M, args.in_features, device=dev)
idx = torch.randint(0, args.in_features, (args.O, args.K), device=dev)
w = torch.randn(args.O, args.K, device=dev)


def eager_forward(x, idx, w):
    M = x.shape[0]; O, K = idx.shape
    sel = x[:, idx.reshape(-1)].reshape(M, O, K)
    return (sel * w.unsqueeze(0)).sum(dim=-1)


ref = eager_forward(x, idx, w)
out = mod.gather_fwd_v2_cuda(x, idx, w, args.B_O, args.B_M)
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
t_cuda = bench(lambda: mod.gather_fwd_v2_cuda(x, idx, w, args.B_O, args.B_M))
print(f"eager forward : {t_eager:.3f} ms")
print(f"cuda v2 forward: {t_cuda:.3f} ms   ({t_eager / t_cuda:.2f}x vs eager)")
print("\nread: v2 should BEAT v1's 1.21x and approach/beat Triton v2's 1.56x on Blackwell.")
print("      Sweep --B_O/--B_M: bigger B_M amortizes idx/w more but uses more shared memory")
print("      and registers. If v2 clears Triton, shared-memory staging was the lever; next we")
print("      apply the same idea to the HARD direction -- the atomic-scatter backward.")
