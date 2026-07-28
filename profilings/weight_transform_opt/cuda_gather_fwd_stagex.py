"""Forward-kernel experiment: stage x[m,:] in SHARED memory, gather on-chip (Phase-4 forward).

MOTIVATION (from ncu_full.sh on the production v2 forward):
  * L2-throughput-bound: L2 SOL 93%, DRAM idle, L1 hit 5.4%, ~28.8 sectors/request.
  * Each x[m,c] is genuinely reused ~360x per row, but L1 catches only 5% -> the SAME
    L2-resident data is re-fetched hundreds of times. int32-idx (cuda_gather_fwd_int32.py)
    was flat, proving L1-capacity is NOT the lever.

THE STRUCTURAL FIX tested here: a block stages the x row(s) it needs into shared ONCE
(coalesced, 4 sectors/req), then every gather reads from shared instead of L2. That converts
~360 scattered L2 reads per element into 1 coalesced L2 load + 360 on-chip shared reads.

Sector math (why this is worth building, unlike v3): x-staging L2 traffic scales as ~1/P
where P = outputs computed per block (the 40KB row load is amortized over more outputs).
Predicted forward L2 sectors vs the current 465M:
    P=512,  B_M=1  -> ~145M  (~3x fewer)     P=512,  B_M=4 -> ~58M  (~8x)
    P=2048, B_M=4  -> ~37M   (~12x fewer)
B_M>1 keeps idx/w reuse (idx/w traffic ~ 1/B_M) but costs B_M*40KB of shared.

THE RISK we are measuring: gathering s_x[idx] with random idx hits SHARED-MEMORY BANK
CONFLICTS (32 banks, random -> ~2-4x serialization). Shared BW is ~10x L2 so it should still
net out ahead -- but "should" is why we profile, not assume.

Design of this kernel:
  * grid  = (ceil(O/P), ceil(M/B_M));  block = T threads (1D).
  * shared = s_x[B_M*IN] (the staged rows) + s_idx[P*K] (int32) + s_w[P*K].
  * stage x rows + idx/w cooperatively (all coalesced), __syncthreads, then each thread
    strides over its outputs; inner k-loop reads idx/w from shared and gathers x from shared,
    accumulating B_M rows at once (acc[B_M] kept in registers via templating on B_M).
  * out writes are coalesced (consecutive threads -> consecutive o).

A/B vs the production v2 (int64) forward, correctness vs eager. Kernel named stagex_kernel so
ncu can target it:
    ncu --metrics lts__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
lts__t_sectors_pipe_lsu_mem_global_op_ld.sum \
        -k "regex:stagex_kernel|gather_fwd_i64_kernel" --launch-skip 10 --launch-count 2 \
        python cuda_gather_fwd_stagex.py

Usage:  python cuda_gather_fwd_stagex.py [--P 512] [--BM 1] [--T 256] [--iters 50]
        # sweep:  for P in 256 512 1024 2048; do python ... --P $P; done
"""
from __future__ import annotations

import argparse
import torch
from torch.utils.cpp_extension import load_inline

_CPP_SRC = (
    "torch::Tensor fwd_v2(torch::Tensor x, torch::Tensor idx, torch::Tensor w, int B_O, int B_M);\n"
    "torch::Tensor fwd_stagex(torch::Tensor x, torch::Tensor idx, torch::Tensor w, int P, int BM, int T);\n"
    "int max_optin_shared();"
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

// ===== experiment: stage x rows in shared, gather on-chip =====
// Templated on BM so acc[BM] lives in registers.
template <int BM>
__global__ void stagex_kernel(
        const float* __restrict__ x, const int* __restrict__ idx,
        const float* __restrict__ w, float* __restrict__ out,
        int M, int O, int IN, int K, int P) {
    extern __shared__ char smem[];
    float* s_x   = reinterpret_cast<float*>(smem);                                  // BM*IN
    int*   s_idx = reinterpret_cast<int*>(smem + (size_t)BM * IN * sizeof(float));  // P*K
    float* s_w   = reinterpret_cast<float*>(smem + (size_t)BM * IN * sizeof(float)
                                                 + (size_t)P * K * sizeof(int));    // P*K
    int o_base = blockIdx.x * P, m_base = blockIdx.y * BM;
    int T = blockDim.x, t = threadIdx.x;

    // stage the BM x-rows (coalesced)
    for (int i = t; i < BM * IN; i += T) {
        int j = i / IN, c = i % IN, m = m_base + j;
        s_x[i] = (m < M) ? x[(long)m * IN + c] : 0.0f;
    }
    // stage idx/w for this block's P outputs (coalesced)
    int tile = P * K;
    for (int i = t; i < tile; i += T) {
        int oo = o_base + i / K, kk = i % K;
        if (oo < O) { s_idx[i] = idx[(long)oo * K + kk]; s_w[i] = w[(long)oo * K + kk]; }
    }
    __syncthreads();

    for (int p = t; p < P; p += T) {
        int o = o_base + p;
        if (o >= O) continue;
        float acc[BM];
        #pragma unroll
        for (int j = 0; j < BM; ++j) acc[j] = 0.0f;
        for (int k = 0; k < K; ++k) {
            int ix = s_idx[p * K + k];
            float wv = s_w[p * K + k];
            #pragma unroll
            for (int j = 0; j < BM; ++j) acc[j] += s_x[j * IN + ix] * wv;   // <-- on-chip gather
        }
        #pragma unroll
        for (int j = 0; j < BM; ++j) {
            int m = m_base + j;
            if (m < M) out[(long)m * O + o] = acc[j];
        }
    }
}

torch::Tensor fwd_stagex(torch::Tensor x, torch::Tensor idx, torch::Tensor w, int P, int BM, int T) {
    int M = x.size(0), IN = x.size(1), O = idx.size(0), K = idx.size(1);
    auto out = torch::empty({M, O}, x.options());
    dim3 grid((O + P - 1) / P, (M + BM - 1) / BM), block(T);
    size_t shmem = (size_t)BM * IN * sizeof(float) + (size_t)P * K * (sizeof(int) + sizeof(float));

    int dev = 0; cudaGetDevice(&dev);
    int maxsh = 0; cudaDeviceGetAttribute(&maxsh, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    TORCH_CHECK((int)shmem <= maxsh, "stagex shared ", shmem, "B > device optin max ", maxsh,
                "B; lower P or BM");

    #define LAUNCH(BMV) do {                                                                   \
        cudaFuncSetAttribute(stagex_kernel<BMV>, cudaFuncAttributeMaxDynamicSharedMemorySize,   \
                             (int)shmem);                                                       \
        stagex_kernel<BMV><<<grid, block, shmem>>>(                                             \
            x.data_ptr<float>(), idx.data_ptr<int>(), w.data_ptr<float>(),                      \
            out.data_ptr<float>(), M, O, IN, K, P);                                             \
    } while (0)
    if      (BM == 1) LAUNCH(1);
    else if (BM == 2) LAUNCH(2);
    else if (BM == 4) LAUNCH(4);
    else TORCH_CHECK(false, "BM must be 1, 2, or 4 (register-array templating)");
    #undef LAUNCH

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fwd_stagex launch failed");
    return out;
}

int max_optin_shared() {
    int dev = 0; cudaGetDevice(&dev);
    int m = 0; cudaDeviceGetAttribute(&m, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    return m;
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
    ap.add_argument("--P", type=int, default=512, help="outputs computed per block")
    ap.add_argument("--BM", type=int, default=1, help="m-rows staged per block (1, 2, or 4)")
    ap.add_argument("--T", type=int, default=256, help="threads per block")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "need CUDA"
    dev = "cuda"
    torch.manual_seed(0)
    M, IN, O, K = args.M, args.IN, args.O, args.K
    x = torch.randn(M, IN, device=dev)
    idx64 = torch.randint(0, IN, (O, K), device=dev, dtype=torch.int64)
    idx32 = idx64.to(torch.int32)
    w = torch.randn(O, K, device=dev)

    shmem_kb = (args.BM * IN * 4 + args.P * K * 8) / 1024
    print(f"gpu   : {torch.cuda.get_device_name(0)}")
    print(f"shape : M={M} IN={IN} O={O} K={K}")
    print(f"stagex: P={args.P} BM={args.BM} T={args.T}  shared/block={shmem_kb:.1f}KB")
    print("compiling (first run ~90s)...", flush=True)
    ext = load_inline(name="fwd_stagex_bench", cpp_sources=_CPP_SRC, cuda_sources=_CUDA_SRC,
                      functions=["fwd_v2", "fwd_stagex", "max_optin_shared"], verbose=False)

    # clamp P so (x rows + idx/w staging) fits this device's opt-in shared limit
    maxsh = ext.max_optin_shared()
    x_bytes = args.BM * IN * 4
    assert x_bytes < maxsh, f"BM={args.BM} needs {x_bytes/1024:.0f}KB for x alone > {maxsh/1024:.0f}KB optin; lower BM"
    max_P = (maxsh - x_bytes) // (K * 8)
    if args.P > max_P:
        print(f"NOTE: P={args.P} won't fit {maxsh/1024:.0f}KB optin shared; clamping to P={max_P}")
        args.P = int(max_P)
    shmem_kb = (args.BM * IN * 4 + args.P * K * 8) / 1024
    print(f"effective: P={args.P} BM={args.BM}  shared/block={shmem_kb:.1f}KB  (optin max {maxsh/1024:.0f}KB)")

    # --- correctness vs eager ---
    ref = (x[:, idx64] * w.unsqueeze(0)).sum(-1)
    ov2 = ext.fwd_v2(x, idx64, w, 64, 16)
    osx = ext.fwd_stagex(x, idx32, w, args.P, args.BM, args.T)
    ev2 = (ov2 - ref).abs().max().item()
    esx = (osx - ref).abs().max().item()
    d = (osx - ov2).abs().max().item()
    print(f"\ncorrectness (max abs err vs eager):  v2 {ev2:.2e}   stagex {esx:.2e}   stagex-vs-v2 {d:.2e}")
    assert esx < 1e-2, "stagex diverged -- STOP"

    # --- timing A/B ---
    tv2 = _time(lambda: ext.fwd_v2(x, idx64, w, 64, 16), args.iters)
    tsx = _time(lambda: ext.fwd_stagex(x, idx32, w, args.P, args.BM, args.T), args.iters)
    print(f"\ntiming (ms/call, {args.iters} iters):")
    print(f"  v2 baseline  : {tv2:.3f}")
    print(f"  stagex       : {tsx:.3f}")
    print(f"  speedup      : {tv2/tsx:.3f}x  ({(tv2/tsx - 1)*100:+.1f}%)")
    print("\nnext: profile L2 SOL + shared bank conflicts with the ncu command in the docstring.")


if __name__ == "__main__":
    main()
