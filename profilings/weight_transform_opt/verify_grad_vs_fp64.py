"""Decisive correctness gate: is the Triton gradient BIASED, or just reordered?

"Differs from fused" is not "wrong" -- fused and triton round in different atomic
orders, so they disagree in the last fp32 digit even when both are correct. The only
question that matters is: does each fp32 path land as close to the TRUE (fp64)
gradient as the other? We compute grad_x and grad_(pre_w) three ways per synapse:
    * fp64  : plain loop path in double  -> ground truth
    * fused : fused_sparse_gather (fp32)   (existing, trusted production path)
    * triton: triton_sparse_gather (fp32)  (the new kernel)
and report each fp32 path's error vs the fp64 truth.

STRESS COVERAGE (for scaling): sweeps ALL indexed synapses, not just one. Reports
each synapse's scatter CONTENTION (max fan-in = most (o,k) pairs writing to a single
input index -> where the atomic backward works hardest). The verdict is the WORST
synapse's triton/fused error ratio, so a shape-dependent bug can't hide in an average.

  ratio ~1  -> triton as accurate as fused vs truth = UNBIASED (a real bug would be
               ~1e5x, not ~1x: it'd be off by O(0.1), not O(fp32 epsilon)).
  ratio >>1 -> triton systematically worse -> real bug, investigate.

Run:  python verify_grad_vs_fp64.py                 # all synapses, M=128, N=5
      python verify_grad_vs_fp64.py --M 256 --N 10  # heavier stress
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

CONFIG = "configs/examples/sglimit5_h100_cifar10_ei_1layer_bf33_10000e1250i.yaml"
dev = "cuda"
assert torch.cuda.is_available()

p = argparse.ArgumentParser()
p.add_argument("--M", type=int, default=128, help="batch rows (higher = more scatter traffic)")
p.add_argument("--N", type=int, default=5, help="trials per path (averages out atomic jitter)")
p.add_argument("--ratio_thresh", type=float, default=5.0, help="max triton/fused error ratio to pass")
args = p.parse_args()
print(f"GPU: {torch.cuda.get_device_name(0)}   M={args.M}  N={args.N}\n")

cfg = load_config(CONFIG)
model, _ = initialize_model(cfg.model); model = model.to(dev)
synapses = [m for m in model.modules() if isinstance(m, IndexedSparseLinear)]
print(f"sweeping ALL {len(synapses)} indexed synapses\n")


def grads(module, x_in, flag):
    """Forward+backward with the chosen gather path; return (grad_x, grad_pre_w)."""
    module.use_fused_gather = (flag == "fused")
    module.use_triton_gather = (flag == "triton")
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
       f"{'fused_gx':>9} {'triton_gx':>9} {'r_gx':>5}  "
       f"{'fused_gw':>9} {'triton_gw':>9} {'r_gw':>5}")
print(hdr); print("-" * len(hdr))

worst = {"r_gx": 0.0, "r_gw": 0.0, "gx_at": -1, "gw_at": -1}
skipped = []
for i, S in enumerate(synapses):
    torch.manual_seed(i)                       # distinct input per synapse
    x0 = torch.randn(args.M, S.in_features, device=dev)
    # scatter contention: most (o,k) pairs landing on one input index
    counts = torch.bincount(S.connection_indices.reshape(-1),
                            minlength=S.in_features)
    max_fanin = int(counts.max().item())
    try:
        S64 = copy.deepcopy(S).double().to(dev)
        gx64, gpw64 = grads(S64, x0.double(), "loop")
        f_gx, f_gw = mean_err(S, x0, gx64, gpw64, "fused")
        t_gx, t_gw = mean_err(S, x0, gx64, gpw64, "triton")
    except RuntimeError as e:                   # e.g. OOM on the largest shapes in fp64
        skipped.append((i, str(e)[:50]))
        print(f"{i:>3} {S.out_features:>7} {S.K:>3} {max_fanin:>9}  "
              f"{'SKIPPED (rerun with smaller --M): ' + str(e)[:30]}")
        del x0
        torch.cuda.empty_cache()
        continue

    r_gx = t_gx / max(f_gx, 1e-30)
    r_gw = t_gw / max(f_gw, 1e-30)
    if r_gx > worst["r_gx"]: worst.update(r_gx=r_gx, gx_at=i)
    if r_gw > worst["r_gw"]: worst.update(r_gw=r_gw, gw_at=i)
    print(f"{i:>3} {S.out_features:>7} {S.K:>3} {max_fanin:>9}  "
          f"{f_gx:>9.2e} {t_gx:>9.2e} {r_gx:>5.2f}  "
          f"{f_gw:>9.2e} {t_gw:>9.2e} {r_gw:>5.2f}")
    del S64, gx64, gpw64, x0
    torch.cuda.empty_cache()

print("-" * len(hdr))
print(f"\nWORST triton/fused error ratio across all synapses:")
print(f"  grad_x     : {worst['r_gx']:.2f}  (@ synapse #{worst['gx_at']})")
print(f"  grad_pre_w : {worst['r_gw']:.2f}  (@ synapse #{worst['gw_at']})")
if skipped:
    print(f"  ({len(skipped)} synapse(s) skipped on fp64 OOM -- rerun those with a smaller --M)")

ok = worst["r_gx"] < args.ratio_thresh and worst["r_gw"] < args.ratio_thresh
print(f"\nVERDICT: {'UNBIASED' if ok else 'BIASED'} -- even the WORST synapse keeps triton "
      f"within {args.ratio_thresh:g}x of fused's distance to the true gradient."
      if ok else
      f"\nVERDICT: BIASED -- synapse #{worst['gx_at'] if worst['r_gx']>=args.ratio_thresh else worst['gw_at']} "
      f"exceeds {args.ratio_thresh:g}x; the kernel is systematically off there -- investigate.")
print("read: every row's fused_* and triton_* should both be ~fp32 epsilon (1e-7..1e-4)")
print("      and r_* ~1. High max_fanin rows are the hardest atomic-scatter cases; if triton")
print("      were going to break, it breaks there first. All r_* ~1 -> unbiased at every scale.")
