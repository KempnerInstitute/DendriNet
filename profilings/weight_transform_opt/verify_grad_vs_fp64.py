"""Decisive test: is the Triton gradient BIASED, or just reordered?

The training-equivalence test showed fused-vs-triton param divergence (~1e-2 after 50
steps) is ~1000x the fused-vs-fused floor (~1e-5). That means triton differs from
fused systematically -- but "differs from fused" is not "wrong". The real question is
whether triton is as close to the TRUE gradient as fused is.

This computes grad_x and grad_(pre_w) on a REAL synapse (real sampled connection
structure -> real contention) three ways:
    * fp64  : plain loop path in double  -> ground truth
    * fused : fused_sparse_gather (fp32)
    * triton: triton_sparse_gather (fp32)
and reports each fp32 path's error against the fp64 reference.

If  err(triton, fp64) ~= err(fused, fp64)  -> triton is as accurate as fused; the
50-step divergence is benign chaos amplification, shippable (default-off flag).
If  err(triton, fp64) >> err(fused, fp64)  -> triton is biased; real bug, investigate.

Run:  python verify_grad_vs_fp64.py
"""
import os, sys, copy
REPO = "/n/home12/rwhalen/DendriNet"
os.chdir(REPO); sys.path.insert(0, os.path.join(REPO, "src"))

import torch
from dendritic_modeling.config import load_config
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)

CONFIG = "configs/examples/sglimit5_h100_cifar10_ei_1layer_bf33_10000e1250i.yaml"
dev = "cuda"
assert torch.cuda.is_available()
print(f"GPU: {torch.cuda.get_device_name(0)}\n")

cfg = load_config(CONFIG)
model, _ = initialize_model(cfg.model); model = model.to(dev)
synapses = [m for m in model.modules() if isinstance(m, IndexedSparseLinear)]
S = synapses[0]
print(f"synapse: in={S.in_features} out={S.out_features} K={S.K}  "
      f"(one of {len(synapses)} indexed synapses)\n")

M = 128
torch.manual_seed(0)
x0 = torch.randn(M, S.in_features, device=dev)


def grads(module, x_in, flag):
    """Run module forward+backward with the chosen gather path; return (gx, g_prew)."""
    module.use_fused_gather = (flag == "fused")
    module.use_triton_gather = (flag == "triton")
    x = x_in.clone().detach().requires_grad_(True)
    if module.pre_w.grad is not None:
        module.pre_w.grad = None
    out = module(x)
    (out.float() * out.float()).sum().backward()
    return x.grad.detach().clone(), module.pre_w.grad.detach().clone()


# ---- fp64 ground truth: plain loop path (no fused, no triton) in double -------
S64 = copy.deepcopy(S).double().to(dev)
gx64, gpw64 = grads(S64, x0.double(), flag="loop")

# ---- fp32 fused and fp32 triton ----------------------------------------------
gx_fused, gpw_fused = grads(S, x0, flag="fused")
gx_tri, gpw_tri = grads(S, x0, flag="triton")


def rel_err(a, ref):
    ref = ref.float(); a = a.float()
    denom = ref.abs().max().clamp_min(1e-12)
    return (a - ref).abs().max().item() / denom.item()


print(f"{'gradient':>10}  {'err(fused, fp64)':>18}  {'err(triton, fp64)':>18}  {'ratio t/f':>10}")
for name, f, t, r in (("grad_x", gx_fused, gx_tri, gx64),
                      ("grad_pre_w", gpw_fused, gpw_tri, gpw64)):
    ef, et = rel_err(f, r), rel_err(t, r)
    ratio = et / ef if ef > 0 else float("inf")
    print(f"{name:>10}  {ef:>18.3e}  {et:>18.3e}  {ratio:>10.2f}")

print("\nread: both errors should be ~fp32 epsilon (1e-6..1e-3). If ratio ~1 (triton no")
print("      worse than fused vs the TRUE gradient), triton is as correct as fused -> the")
print("      50-step divergence is benign, ship default-off. If triton's error is orders")
print("      larger, the kernel is biased -> real bug.")
