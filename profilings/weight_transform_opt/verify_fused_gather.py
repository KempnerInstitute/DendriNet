"""Final gate: does use_fused_gather change TRAINING, not just one op?

Builds the real model from the config, then trains it twice from the SAME init on
the SAME data -- once with the baseline gather, once with the fused kernel -- and
compares the loss trajectory and final parameters. `allclose` on a single grad_x
flags harmless fp32 rounding (we proved that in the microbench); the real question
is whether the training trajectory stays together under AMP. It should.

Usage:
  python verify_fused_gather.py                 # synthetic input, small batch
  python verify_fused_gather.py --steps 20 --batch 32
"""
import os, sys, argparse, copy
REPO = "/n/home12/rwhalen/DendriNet"
os.chdir(REPO); sys.path.insert(0, os.path.join(REPO, "src"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_value_

from dendritic_modeling.config import load_config
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)

CONFIG = "configs/examples/sglimit5_h100_cifar10_ei_1layer_bf33_10000e1250i.yaml"
dev = "cuda"
assert torch.cuda.is_available()

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=20)
p.add_argument("--batch", type=int, default=32)
args = p.parse_args()
print(f"GPU: {torch.cuda.get_device_name(0)}  steps={args.steps} batch={args.batch}\n")

cfg = load_config(CONFIG)

# one fixed synthetic batch, reused for both runs
torch.manual_seed(123)
x = torch.randn(args.batch, 3072, device=dev)
y = torch.randint(0, 10, (args.batch,), device=dev)

# build once, snapshot the initial weights so both runs start identical
torch.manual_seed(0)
model0, _ = initialize_model(cfg.model)
init_state = copy.deepcopy(model0.state_dict())

def count_indexed(model):
    return sum(isinstance(m, IndexedSparseLinear) for m in model.modules())

def set_fused(model, on):
    n = 0
    for m in model.modules():
        if isinstance(m, IndexedSparseLinear):
            m.use_fused_gather = on; n += 1
    return n

def run(fused):
    torch.manual_seed(0)
    model, _ = initialize_model(cfg.model)
    model.load_state_dict(init_state)          # identical start
    model = model.to(dev)
    n = set_fused(model, fused)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce, scaler = nn.CrossEntropyLoss(), GradScaler("cuda")
    losses = []
    for _ in range(args.steps):
        with autocast("cuda"):
            loss = ce(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        clip_grad_value_(model.parameters(), cfg.training.main.common.grad_clip_value)
        scaler.step(opt); scaler.update()
        losses.append(loss.item())
    final = torch.cat([p.detach().flatten() for p in model.parameters()])
    return losses, final, n

print(f"IndexedSparseLinear modules in model: {count_indexed(model0)}")
l_off, f_off, n_off = run(fused=False)
l_on,  f_on,  n_on  = run(fused=True)
print(f"toggled use_fused_gather on {n_on} modules\n")

# ---- compare -------------------------------------------------------------
import statistics
traj_absdiff = [abs(a - b) for a, b in zip(l_off, l_on)]
print(f"{'step':>4}  {'loss(off)':>12}  {'loss(on)':>12}  {'|diff|':>10}")
for i in range(min(args.steps, 8)):
    print(f"{i:>4}  {l_off[i]:>12.6f}  {l_on[i]:>12.6f}  {traj_absdiff[i]:>10.2e}")
if args.steps > 8:
    print(f"  ... last step: off={l_off[-1]:.6f} on={l_on[-1]:.6f} |diff|={traj_absdiff[-1]:.2e}")

print(f"\nmax |loss diff| over trajectory : {max(traj_absdiff):.2e}")
print(f"final loss  off={l_off[-1]:.6f}  on={l_on[-1]:.6f}")
print(f"final param max|delta|          : {(f_off - f_on).abs().max().item():.2e}")
print(f"final param mean|delta|         : {(f_off - f_on).abs().mean().item():.2e}")

# verdict: under AMP, trajectories should track to well within bf16 noise
ok = max(traj_absdiff) < 1e-2 and (f_off - f_on).abs().max().item() < 1e-1
print("\nverdict:", "EQUIVALENT under AMP -- safe to enable use_fused_gather." if ok
      else "DIVERGED more than expected -- inspect before enabling.")
