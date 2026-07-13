"""A/B test: torch.compile (eager vs compiled) on the post-fix model.

Diffuse + mildly launch-bound profile -> torch.compile should help by fusing the
clustered elementwise ops and cutting kernel launches. Measures wall-time speedup
AND training-equivalence (loss trajectory), same discipline as every prior step.

Note: the custom autograd Function (_ChunkedSparseGatherMulSum) will cause graph
breaks; compile optimizes the regions AROUND it. That's expected, still useful.

Run:  python compile_ab.py --steps 20 --batch 128
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
p.add_argument("--warmup", type=int, default=10, help="includes torch.compile warmup")
p.add_argument("--batch", type=int, default=128)
p.add_argument("--mode", default="default", help="torch.compile mode (default / reduce-overhead)")
args = p.parse_args()
print(f"GPU: {torch.cuda.get_device_name(0)}  steps={args.steps} batch={args.batch} mode={args.mode}\n")

cfg = load_config(CONFIG)
torch.manual_seed(123)
x = torch.randn(args.batch, 3072, device=dev)
y = torch.randint(0, 10, (args.batch,), device=dev)

torch.manual_seed(0)
m0, _ = initialize_model(cfg.model)
init_state = copy.deepcopy(m0.state_dict())

def make():
    torch.manual_seed(0)
    m, _ = initialize_model(cfg.model)
    m.load_state_dict(init_state)
    m = m.to(dev)
    for mod in m.modules():
        if isinstance(mod, IndexedSparseLinear):
            mod.use_fused_gather = True          # post-fix world
    return m

def run(compiled):
    model = make()
    fwd = torch.compile(model, mode=args.mode) if compiled else model
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce, scaler = nn.CrossEntropyLoss(), GradScaler("cuda")
    losses = []

    def step(record):
        with autocast("cuda"):
            loss = ce(fwd(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); clip_grad_value_(model.parameters(), 5.0)
        scaler.step(opt); scaler.update()
        if record: losses.append(loss.item())

    for _ in range(args.warmup): step(False)     # compile happens here
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(args.steps): step(True)
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / args.steps, losses

# ---- diagnostic: SINGLE-STEP gradient equivalence (before any chaotic drift) ----
# same init + same input, one forward+backward, no optimizer step. If grads match
# here, compile is numerically faithful per-step and any trajectory drift is chaos.
def one_step_grads(compiled, gb=32):
    torch.cuda.empty_cache()
    model = make()
    fwd = torch.compile(model, mode=args.mode) if compiled else model
    ce = nn.CrossEntropyLoss()
    torch.manual_seed(999)                        # identical small batch for both
    xs = torch.randn(gb, 3072, device=dev)
    ys = torch.randint(0, 10, (gb,), device=dev)
    model.zero_grad(set_to_none=True)
    with autocast("cuda"):
        loss = ce(fwd(xs), ys)                     # first call compiles AND runs
    loss.backward()
    g = torch.cat([p.grad.flatten() for p in model.parameters()
                   if p.grad is not None]).detach().cpu()   # off-GPU to free memory
    lv = loss.item()
    del model, fwd, loss, xs, ys
    torch.cuda.empty_cache()
    return lv, g

print("single-step gradient check (eager vs compiled, batch 32) ...")
le, ge = one_step_grads(False)
lc, gc = one_step_grads(True)
gdiff = (ge - gc).abs()
print(f"  step-0 loss    eager={le:.6f}  compiled={lc:.6f}  |diff|={abs(le-lc):.2e}")
print(f"  step-0 grads   max|diff|={gdiff.max().item():.2e}  mean|diff|={gdiff.mean().item():.2e}")

# ---- timing + full trajectory ----
print("\nrunning eager ...")
t_eager, l_eager = run(False)
torch.cuda.empty_cache()
print("running compiled (first steps compile, be patient) ...")
t_comp, l_comp = run(True)

print(f"\nmean wall time/step  eager={t_eager:.1f} ms  compiled={t_comp:.1f} ms  "
      f"speedup={t_eager/t_comp:.2f}x\n")
print(f"{'step':>4}  {'eager':>11}  {'compiled':>11}  {'|diff|':>10}")
for i, (a, b) in enumerate(zip(l_eager, l_comp)):
    print(f"{i:>4}  {a:>11.6f}  {b:>11.6f}  {abs(a-b):>10.2e}")
maxdiff = max(abs(a - b) for a, b in zip(l_eager, l_comp))

print(f"\nspeedup={t_eager/t_comp:.2f}x   trajectory max|diff|={maxdiff:.2e}")
print("read: if step-0 grads match (~1e-3 or less) but trajectory drifts -> BENIGN chaos")
print("      (compile is faithful; drift is fp-amplification, both are valid runs).")
print("      if step-0 grads differ a lot -> compile changed the math -> inspect.")
