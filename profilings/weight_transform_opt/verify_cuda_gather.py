"""GPU correctness gate for the PRODUCTION CUDA gather (src/.../synapse/cuda_gather.py).

Same decisive test that cleared the Triton path: is the CUDA gradient BIASED or just
reordered? Sweeps ALL indexed synapses and compares each fp32 path to the TRUE (fp64)
gradient computed by the plain loop in double:
    * fp64  : plain loop path in double  -> ground truth
    * fused : fused_sparse_gather (fp32)   (trusted production path, accuracy anchor)
    * cuda  : the CUDA v2 kernels (fp32)   (the new path under test)
Verdict is the WORST synapse's cuda/fused error ratio (ratio ~1 -> as accurate as fused
= unbiased; a real bug would be orders larger, not ~1x). Reports scatter CONTENTION
(max fan-in) per synapse so you can see accuracy holds where the atomics work hardest.

RUN ON A GPU NODE (after `source setup_cuda_env.sh`). The first synapse triggers the CUDA
extension's one-time lazy compile (~80s), then it's cached.

Run:  python verify_cuda_gather.py                 # all synapses, M=128, N=5
      python verify_cuda_gather.py --M 256 --N 10  # heavier stress
"""
import os, sys, copy, argparse
REPO = "/n/home12/rwhalen/DendriNet"
os.chdir(REPO); sys.path.insert(0, os.path.join(REPO, "src"))

import torch
from dendritic_modeling.config import load_config
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.cuda_gather import (
    cuda_gather_available,
)

CONFIG = "configs/examples/sglimit5_h100_cifar10_ei_1layer_bf33_10000e1250i.yaml"
dev = "cuda"
assert torch.cuda.is_available(), "need CUDA"
assert cuda_gather_available(), "CUDA gather did not compile (nvcc? toolchain? see setup_cuda_env.sh)"

p = argparse.ArgumentParser()
p.add_argument("--M", type=int, default=128)
p.add_argument("--N", type=int, default=5, help="trials per path (averages atomic jitter)")
p.add_argument("--ratio_thresh", type=float, default=5.0)
args = p.parse_args()
print(f"GPU: {torch.cuda.get_device_name(0)}   M={args.M}  N={args.N}\n")

cfg = load_config(CONFIG)
model, _ = initialize_model(cfg.model); model = model.to(dev)
synapses = [m for m in model.modules() if isinstance(m, IndexedSparseLinear)]
print(f"sweeping ALL {len(synapses)} indexed synapses (fp32 fused/cuda vs fp64 truth)\n")


def grads(module, x_in, flag):
    """flag in {loop, fused, cuda}: run fwd+bwd via that path; return (grad_x, grad_pre_w)."""
    module.use_fused_gather = (flag == "fused")
    module.use_triton_gather = False
    module.use_cuda_gather = (flag == "cuda")
    x = x_in.clone().detach().requires_grad_(True)
    if module.pre_w.grad is not None:
        module.pre_w.grad = None
    out = module(x)
    (out.float() * out.float()).sum().backward()
    return x.grad.detach().clone(), module.pre_w.grad.detach().clone()


def rel_err(a, ref):
    ref = ref.float(); a = a.float()
    return (a - ref).abs().max().item() / ref.abs().max().clamp_min(1e-12).item()


def mean_err(module, x0, gx_ref, gpw_ref, flag):
    exs, ews = [], []
    for _ in range(args.N):
        gx, gpw = grads(module, x0, flag)
        exs.append(rel_err(gx, gx_ref)); ews.append(rel_err(gpw, gpw_ref))
    return sum(exs) / len(exs), sum(ews) / len(ews)


hdr = (f"{'#':>3} {'out':>7} {'K':>3} {'max_fanin':>9}  "
       f"{'fused_gx':>9} {'cuda_gx':>9} {'r_gx':>5}  "
       f"{'fused_gw':>9} {'cuda_gw':>9} {'r_gw':>5}")
print(hdr); print("-" * len(hdr))

worst = {"r_gx": 0.0, "r_gw": 0.0, "gx_at": -1, "gw_at": -1}
skipped = []
for i, S in enumerate(synapses):
    torch.manual_seed(i)
    x0 = torch.randn(args.M, S.in_features, device=dev)
    counts = torch.bincount(S.connection_indices.reshape(-1), minlength=S.in_features)
    max_fanin = int(counts.max().item())
    try:
        S64 = copy.deepcopy(S).double().to(dev)
        gx64, gpw64 = grads(S64, x0.double(), "loop")
        f_gx, f_gw = mean_err(S, x0, gx64, gpw64, "fused")
        c_gx, c_gw = mean_err(S, x0, gx64, gpw64, "cuda")
    except RuntimeError as e:
        skipped.append((i, str(e)[:50]))
        print(f"{i:>3} {S.out_features:>7} {S.K:>3} {max_fanin:>9}  SKIPPED: {str(e)[:34]}")
        del x0; torch.cuda.empty_cache(); continue

    r_gx = c_gx / max(f_gx, 1e-30)
    r_gw = c_gw / max(f_gw, 1e-30)
    if r_gx > worst["r_gx"]: worst.update(r_gx=r_gx, gx_at=i)
    if r_gw > worst["r_gw"]: worst.update(r_gw=r_gw, gw_at=i)
    print(f"{i:>3} {S.out_features:>7} {S.K:>3} {max_fanin:>9}  "
          f"{f_gx:>9.2e} {c_gx:>9.2e} {r_gx:>5.2f}  "
          f"{f_gw:>9.2e} {c_gw:>9.2e} {r_gw:>5.2f}")
    del S64, gx64, gpw64, x0; torch.cuda.empty_cache()

print("-" * len(hdr))
print(f"\nWORST cuda/fused error ratio across all synapses:")
print(f"  grad_x     : {worst['r_gx']:.2f}  (@ synapse #{worst['gx_at']})")
print(f"  grad_pre_w : {worst['r_gw']:.2f}  (@ synapse #{worst['gw_at']})")
if skipped:
    print(f"  ({len(skipped)} skipped on fp64 OOM -- rerun those with a smaller --M)")

ok = worst["r_gx"] < args.ratio_thresh and worst["r_gw"] < args.ratio_thresh
print(f"\nVERDICT: {'UNBIASED' if ok else 'BIASED'} -- the CUDA path is "
      f"{'as accurate as fused vs the true gradient at every synapse shape' if ok else 'systematically off somewhere -- investigate'}"
      f" (worst ratio {max(worst['r_gx'], worst['r_gw']):.2f}, bar {args.ratio_thresh:g}).")
print("read: every row's fused_* and cuda_* both ~fp32 epsilon (1e-7..1e-4) and r_* ~1.")
print("      Watch high max_fanin rows -- if CUDA broke, it breaks there first. All r_* ~1")
print("      -> unbiased; combine with a training-equivalence run before flipping the flag on.")
