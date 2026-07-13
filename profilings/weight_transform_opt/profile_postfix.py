"""Re-profile the model WITH the gather fix ON, to find the NEXT candidate.

The candidate list is dynamic: removing the 84% gather-backward reshuffles the
ranking. This measures the post-fix world and, crucially, answers whether the
model is now GPU-BOUND (-> look for the next GPU op) or LAUNCH-BOUND (-> the next
win is reducing kernel-launch count, not any single op).

Run:  python profile_postfix.py --synthetic --batch 128
"""
import os, sys, argparse, types
REPO = "/n/home12/rwhalen/DendriNet"
os.chdir(REPO); sys.path.insert(0, os.path.join(REPO, "src"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_value_
from torch.profiler import profile, ProfilerActivity

from dendritic_modeling.config import load_config
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)

CONFIG = "configs/examples/sglimit5_h100_cifar10_ei_1layer_bf33_10000e1250i.yaml"
p = argparse.ArgumentParser()
p.add_argument("--synthetic", action="store_true")
p.add_argument("--batch", type=int, default=128)
p.add_argument("--iters", type=int, default=10)
p.add_argument("--fused", type=int, default=1, help="1=gather fix ON (post-fix), 0=baseline")
args = p.parse_args()

assert torch.cuda.is_available()
dev = "cuda"
print(f"GPU: {torch.cuda.get_device_name(0)}  fused={bool(args.fused)} batch={args.batch}")

cfg = load_config(CONFIG)
model, _ = initialize_model(cfg.model); model = model.to(dev)

n = 0
for m in model.modules():
    if isinstance(m, IndexedSparseLinear):
        m.use_fused_gather = bool(args.fused); n += 1
print(f"use_fused_gather={bool(args.fused)} on {n} indexed synapses\n")

B = args.batch
x = torch.randn(B, 3072, device=dev)
y = torch.randint(0, 10, (B,), device=dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
ce, scaler = nn.CrossEntropyLoss(), GradScaler("cuda")

def step():
    with autocast("cuda"):
        loss = ce(model(x), y)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt); clip_grad_value_(model.parameters(), 5.0)
    scaler.step(opt); scaler.update()

for _ in range(5): step()
torch.cuda.synchronize()

# wall-clock per step
e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
e0.record()
for _ in range(args.iters): step()
e1.record(); torch.cuda.synchronize()
print(f"mean wall time/step: {e0.elapsed_time(e1)/args.iters:.1f} ms\n")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(args.iters): step()
    torch.cuda.synchronize()

ka = prof.key_averages()
print(ka.table(sort_by="cuda_time_total", row_limit=18))

# --- GPU-bound vs LAUNCH-bound verdict ---
def _dev(e):  # attribute name changed across torch versions
    return getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
# sum device time over KERNEL rows only (self_cpu==0); summing all rows double-counts
# the operator + kernel aggregation levels. this matches the profiler's Self CUDA total.
cuda_total = sum(_dev(e) for e in ka if e.self_cpu_time_total == 0)
cpu_total = sum(e.self_cpu_time_total for e in ka)
launches = sum(e.count for e in ka if e.key in ("cudaLaunchKernel",))
print(f"\nSelf CUDA total: {cuda_total/1e6:.2f} s   Self CPU total: {cpu_total/1e6:.2f} s")
print(f"cudaLaunchKernel calls over {args.iters} steps: {launches}  (~{launches//max(args.iters,1)}/step)")
print("\nverdict:")
if cpu_total > 1.3 * cuda_total:
    print("  LAUNCH-BOUND -> GPU is starved; next win = FEWER kernel launches")
    print("  (bigger chunks, fuse Python-loop ops, CUDA graphs / torch.compile) --")
    print("  a single GPU-op fix won't move wall-clock much.")
else:
    print("  GPU-BOUND -> next win = the new top GPU op in the table above.")
    print("  Pick the largest 'Self CUDA' row that is a shared primitive and A/B it.")
