"""Phase 3, Step 4: CUDA backward v3 -- grad_x scatter -> atomic-free GATHER (the thesis).

v2 privatized grad_w in shared memory -> 1.54x (A100) / 1.82x (Blackwell) vs index_add_,
already past Triton (1.24/1.35). But grad_x was still naive global atomics. v3 removes
those atomics entirely by exploiting the FIXED connectivity.

grad_x[m,c] = sum_{(o,k): idx[o,k]=c} grad_out[m,o] * w[o,k]

is a SCATTER (contributions race to random targets c) only if you drive it from (o,k).
But connection_indices never change during training, so we precompute the TRANSPOSE once:
for each input c, the sorted list of contributions (o, position-in-w) that land on it.
Then each grad_x[m,c] is computed by ONE thread reading its own list -- a gather, no atomics,
no contention. The one-time build amortizes over every training step.

Two kernels: grad_w (v2 shared privatization) + grad_x (transpose gather). The transpose
(offsets/seg_o/seg_wpos) is built once from idx and excluded from per-step timing.

Run (after `source setup_cuda_env.sh`):
    python cuda_gather_bwd_v3.py --B_O 32 --B_M 16
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
p.add_argument("--B_O", type=int, default=32)   # grad_w tile
p.add_argument("--B_M", type=int, default=16)
p.add_argument("--tpb_x", type=int, default=256, help="threads/block for grad_x gather")
p.add_argument("--iters", type=int, default=50)
args = p.parse_args()

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"shape: M={args.M} IN={args.in_features} O={args.O} K={args.K}  "
      f"grad_w tile B_O={args.B_O} B_M={args.B_M}\n")

BUILD = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"cuda_gather_bwd_v3_{os.getuid()}")
os.makedirs(BUILD, exist_ok=True)

cpp_src = ("std::vector<torch::Tensor> gather_bwd_v3_cuda("
           "torch::Tensor grad_out, torch::Tensor x, torch::Tensor idx, torch::Tensor w, "
           "torch::Tensor offsets, torch::Tensor seg_o, torch::Tensor seg_wpos, int B_O, int B_M, int tpb_x);")
cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// grad_w: v2 shared-memory privatization (one global atomic per [o,k] per block).
__global__ void gradw_kernel(
        const float* __restrict__ grad_out, const float* __restrict__ x,
        const long* __restrict__ idx, const float* __restrict__ w, float* __restrict__ grad_w,
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
        for (int k = 0; k < K; ++k)
            atomicAdd(&s_gw[tx * K + k], go * x_m[s_idx[tx * K + k]]);
    }
    __syncthreads();
    for (int i = tid; i < tile; i += T) {
        int oo = o_base + i / K, kk = i % K;
        if (oo < O) atomicAdd(&grad_w[(long)oo * K + kk], s_gw[i]);
    }
}

// grad_x: transpose GATHER. One thread per (m, c). Reads c's precomputed contribution
// list [offsets[c], offsets[c+1]) and accumulates in a register -- NO atomics.
__global__ void gradx_gather_kernel(
        const float* __restrict__ grad_out,   // [M, O]
        const float* __restrict__ w_flat,     // [O*K]
        const long*  __restrict__ offsets,    // [IN+1]
        const int*   __restrict__ seg_o,      // [O*K]  output o of each sorted contribution
        const long*  __restrict__ seg_wpos,   // [O*K]  flat position into w
        float* __restrict__ grad_x,           // [M, IN]  (fully written)
        int M, int O, int IN) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;   // which input
    int m = blockIdx.y;                              // which batch row
    if (c >= IN || m >= M) return;
    long start = offsets[c], end = offsets[c + 1];
    const float* go_m = grad_out + (long)m * O;
    float acc = 0.0f;
    for (long j = start; j < end; ++j)
        acc += go_m[seg_o[j]] * w_flat[seg_wpos[j]];  // gather-reduce, contention-free
    grad_x[(long)m * IN + c] = acc;
}

std::vector<torch::Tensor> gather_bwd_v3_cuda(
        torch::Tensor grad_out, torch::Tensor x, torch::Tensor idx, torch::Tensor w,
        torch::Tensor offsets, torch::Tensor seg_o, torch::Tensor seg_wpos,
        int B_O, int B_M, int tpb_x) {
    TORCH_CHECK(idx.scalar_type() == torch::kInt64, "idx int64");
    TORCH_CHECK(seg_o.scalar_type() == torch::kInt32, "seg_o int32");
    TORCH_CHECK(offsets.scalar_type() == torch::kInt64 && seg_wpos.scalar_type() == torch::kInt64,
                "offsets/seg_wpos int64");
    grad_out = grad_out.contiguous(); x = x.contiguous(); w = w.contiguous();
    int M = x.size(0), IN = x.size(1), O = idx.size(0), K = idx.size(1);
    auto grad_w = torch::zeros({O, K}, w.options());
    auto grad_x = torch::empty({M, IN}, x.options());   // gather writes every element
    auto w_flat = w.reshape(-1).contiguous();

    // grad_w (shared privatization)
    dim3 b1(B_O, B_M), g1((O + B_O - 1) / B_O, (M + B_M - 1) / B_M);
    size_t shmem = (size_t)B_O * K * (sizeof(long) + 2 * sizeof(float));
    TORCH_CHECK(shmem <= 48 * 1024, "grad_w shared tile > 48KB; lower B_O");
    gradw_kernel<<<g1, b1, shmem>>>(grad_out.data_ptr<float>(), x.data_ptr<float>(),
        idx.data_ptr<long>(), w.data_ptr<float>(), grad_w.data_ptr<float>(), M, O, IN, K, B_O, B_M);

    // grad_x (transpose gather)
    dim3 b2(tpb_x), g2((IN + tpb_x - 1) / tpb_x, M);
    gradx_gather_kernel<<<g2, b2>>>(grad_out.data_ptr<float>(), w_flat.data_ptr<float>(),
        offsets.data_ptr<long>(), seg_o.data_ptr<int>(), seg_wpos.data_ptr<long>(),
        grad_x.data_ptr<float>(), M, O, IN);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "v3 kernel launch failed");
    return {grad_x, grad_w};
}
"""

print("=== compiling (first run ~60-90s) ===")
t0 = time.time()
mod = load_inline(name="cuda_gather_bwd_v3", cpp_sources=cpp_src, cuda_sources=cuda_src,
                  functions=["gather_bwd_v3_cuda"], build_directory=BUILD, verbose=True)
print(f"compiled in {time.time() - t0:.1f}s\n")

dev = "cuda"
torch.manual_seed(0)
x = torch.randn(args.M, args.in_features, device=dev)
idx = torch.randint(0, args.in_features, (args.O, args.K), device=dev)
w = torch.randn(args.O, args.K, device=dev)
grad_out = torch.randn(args.M, args.O, device=dev)
IN, O, K = args.in_features, args.O, args.K


def build_transpose(idx):
    """ONE-TIME (idx is fixed): reverse map input -> sorted contribution list."""
    idx_flat = idx.reshape(-1)                                  # [O*K]
    perm = torch.argsort(idx_flat)                              # positions sorted by target c
    offsets = torch.zeros(IN + 1, dtype=torch.int64, device=idx.device)
    offsets[1:] = torch.bincount(idx_flat, minlength=IN).cumsum(0)
    seg_o = (perm // K).to(torch.int32)                        # output of each contribution
    seg_wpos = perm.to(torch.int64)                            # flat position into w
    return offsets, seg_o, seg_wpos


t_build = time.time()
offsets, seg_o, seg_wpos = build_transpose(idx)
torch.cuda.synchronize()
print(f"transpose build (one-time, amortized): {(time.time()-t_build)*1e3:.2f} ms\n")


def ref_backward(grad_out, x, idx, w):
    M, IN = x.shape; O, K = idx.shape
    idx_flat = idx.reshape(-1); g = grad_out.unsqueeze(-1)
    grad_x = torch.zeros(M, IN, device=x.device)
    grad_x.index_add_(1, idx_flat, (g * w.unsqueeze(0)).reshape(M, O * K))
    sel = x.index_select(1, idx_flat).reshape(M, O, K)
    grad_w = (g * sel).sum(dim=0)
    return grad_x, grad_w


gx_ref, gw_ref = ref_backward(grad_out, x, idx, w)
gx, gw = mod.gather_bwd_v3_cuda(grad_out, x, idx, w, offsets, seg_o, seg_wpos,
                               args.B_O, args.B_M, args.tpb_x)
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
t_v3 = bench(lambda: mod.gather_bwd_v3_cuda(grad_out, x, idx, w, offsets, seg_o, seg_wpos,
                                            args.B_O, args.B_M, args.tpb_x))
print(f"index_add_ backward : {t_ref:.3f} ms")
print(f"cuda v3 backward    : {t_v3:.3f} ms   ({t_ref / t_v3:.2f}x vs index_add_)")
print("\nread: compare to v2 (1.54x A100 / 1.82x BW). If v3 > v2, killing grad_x atomics helped")
print("      -> the scatter->gather transpose is the win. If v3 ~= v2, grad_x wasn't the")
print("      bottleneck (grad_w was) and the gather's uncoalesced grad_out reads offset the gain.")
