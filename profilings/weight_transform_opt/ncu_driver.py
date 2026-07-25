"""Minimal driver for NSight Compute (ncu) to profile the PRODUCTION CUDA gather kernels.

Runs cuda_sparse_gather forward+backward a few times so ncu can capture
gather_fwd_v2_kernel and gather_bwd_v2_kernel on the real large-synapse shape.
Launched under ncu by ncu_quick.sh; also runs standalone as a sanity check.

Env: NCU_ITERS (default 4) -- number of fwd+bwd iterations to launch.
"""
import os, sys
REPO = "/n/home12/rwhalen/DendriNet"; sys.path.insert(0, os.path.join(REPO, "src"))
import torch
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.cuda_gather import (
    cuda_sparse_gather, cuda_gather_available,
)

assert torch.cuda.is_available(), "need CUDA"
assert cuda_gather_available(), "CUDA gather did not compile (source setup_cuda_env.sh first)"

dev = "cuda"
torch.manual_seed(0)
M, IN, O, K = 128, 10000, 90000, 40
x = torch.randn(M, IN, device=dev, requires_grad=True)
idx = torch.randint(0, IN, (O, K), device=dev)
w = torch.randn(O, K, device=dev, requires_grad=True)

iters = int(os.environ.get("NCU_ITERS", "4"))
for _ in range(iters):
    x.grad = None; w.grad = None
    out = cuda_sparse_gather(x, idx, w)          # -> gather_fwd_v2_kernel
    out.float().pow(2).sum().backward()          # -> gather_bwd_v2_kernel
    torch.cuda.synchronize()

print(f"ran {iters} fwd+bwd iters of the production CUDA gather on {torch.cuda.get_device_name(0)}")
