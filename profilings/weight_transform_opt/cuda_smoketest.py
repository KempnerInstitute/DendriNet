"""CUDA build smoke test: can we compile + run a custom CUDA kernel on this node?

Phase 3 step 0 -- BEFORE writing the real gather/scatter kernel, prove the toolchain
works end to end: nvcc is found, CUDA_HOME resolves, torch can JIT-compile a CUDA
extension, and it runs correctly on this GPU. If this fails, the failure is
environment/toolchain (module load cuda, CUDA_HOME), NOT our kernel.

Needs a GPU node with the CUDA toolkit visible. Typical setup:
    salloc -p gpu --gres=gpu:1 -t 2:00:00 --mem=32G   # or your usual GPU alloc
    module load cuda                                   # puts nvcc on PATH, sets CUDA_HOME
    conda activate dendritic_env
    python cuda_smoketest.py

Run:  python cuda_smoketest.py
"""
import os, sys, time
import torch

print("=== environment ===")
print(f"torch        : {torch.__version__}")
print(f"cuda avail   : {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "no CUDA device visible"
print(f"gpu          : {torch.cuda.get_device_name(0)}")
cc = torch.cuda.get_device_capability(0)
print(f"compute cap  : sm_{cc[0]}{cc[1]}")
print(f"torch cuda   : {torch.version.cuda}")

from torch.utils.cpp_extension import CUDA_HOME, load_inline
print(f"CUDA_HOME    : {CUDA_HOME}")
if CUDA_HOME is None:
    print("\nCUDA_HOME is None -> nvcc/toolkit not visible. Run `module load cuda` on a GPU "
          "node (or set CUDA_HOME) and retry. Cannot build a CUDA extension without it.")
    raise SystemExit(1)

# keep build artifacts out of the repo
BUILD = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"cuda_smoke_{os.getuid()}")
os.makedirs(BUILD, exist_ok=True)

cpp_src = "torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);"
cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void add_kernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA");
    a = a.contiguous(); b = b.contiguous();
    auto out = torch::empty_like(a);
    int n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(),
                                    out.data_ptr<float>(), n);
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}
"""

print("\n=== compiling CUDA extension (first run is slow: nvcc invoked) ===")
t0 = time.time()
mod = load_inline(
    name="cuda_smoke",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["add_cuda"],
    build_directory=BUILD,
    verbose=True,
)
print(f"compiled in {time.time() - t0:.1f}s  (cached at {BUILD})")

print("\n=== run + correctness ===")
torch.manual_seed(0)
x = torch.randn(1_000_003, device="cuda")   # not a multiple of blockDim
y = torch.randn_like(x)
out = mod.add_cuda(x, y)
ref = x + y
ok = torch.allclose(out, ref, rtol=1e-6, atol=1e-6)
print(f"kernel ran   : OK")
print(f"correctness  : {'OK' if ok else 'FAIL'}  (max_err={(out-ref).abs().max().item():.2e})")

print("\nVERDICT:", "CUDA toolchain works -- clear to write the real gather/scatter kernel."
      if ok else "compiled + ran but WRONG result -- stop, investigate before proceeding.")
