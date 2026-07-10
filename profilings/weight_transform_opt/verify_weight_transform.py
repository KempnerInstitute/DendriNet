"""Step 3: Verify the optimized apply_weight_transform -- CORRECTNESS first, then PERF.

Correctness is checked at three levels, cheapest->strongest:
  (1) op-level: forward + backward match the original for every transform type
  (2) cache-invalidation: after an in-place param update the cache must recompute
  (3) training-equivalence: identical models, identical input, identical grads and
      identical param trajectory over K real optimizer steps  <-- the one that matters

Only if ALL correctness checks pass does it run the performance comparison.

Usage:
  python verify_weight_transform.py \
      --opt-file src/dendritic_modeling/networks/utils/weight_transforms_optimized-rw.py \
      --opt-symbol apply_weight_transform
"""
import os, sys, types, argparse, importlib.util

REPO = "/n/home12/rwhalen/DendriNet"
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch
import torch.nn as nn

p = argparse.ArgumentParser()
p.add_argument("--opt-file", default="src/dendritic_modeling/networks/utils/weight_transforms_optimized-rw.py")
p.add_argument("--opt-symbol", default="apply_weight_transform")
p.add_argument("--steps", type=int, default=20)
p.add_argument("--synthetic", action="store_true")
args = p.parse_args()

CONFIG = "configs/examples/sglimit5_h100_cifar10_ei_1layer_bf33_10000e1250i.yaml"
dev = "cuda"
assert torch.cuda.is_available()

import dendritic_modeling.networks.utils.weight_transforms as wt
ORIG = wt.apply_weight_transform

# load the optimized function from the hyphenated file (not importable by name)
spec = importlib.util.spec_from_file_location("wt_opt", args.opt_file)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
OPT = getattr(mod, args.opt_symbol)
print(f"loaded optimized `{args.opt_symbol}` from {args.opt_file}")

def patch_all(fn):
    """swap apply_weight_transform in every dendritic_modeling module that bound it."""
    n = 0
    for m in list(sys.modules.values()):
        if not isinstance(m, types.ModuleType) or not m.__name__.startswith("dendritic_modeling"):
            continue
        if getattr(m, "apply_weight_transform", None) in (ORIG, fn):
            setattr(m, "apply_weight_transform", fn); n += 1
    return n

# =====================================================================
# (1) OP-LEVEL: forward + backward equivalence, all transform types
# =====================================================================
print("\n[1] op-level forward/backward equivalence")
ok = True
for t in ["softplus", "exp", "relu", "identity"]:
    torch.manual_seed(0)
    w1 = torch.randn(4096, 512, device=dev, requires_grad=True)
    w2 = w1.detach().clone().requires_grad_(True)
    o1 = ORIG(w1, t); o1.sum().backward()
    o2 = OPT(w2, t);  o2.sum().backward()
    f_ok = torch.allclose(o1, o2, rtol=1e-5, atol=1e-6)
    g_ok = torch.allclose(w1.grad, w2.grad, rtol=1e-5, atol=1e-6)
    ok &= f_ok and g_ok
    print(f"    {t:9s}  forward={'OK' if f_ok else 'FAIL'}  backward={'OK' if g_ok else 'FAIL'}")

# =====================================================================
# (2) CACHE INVALIDATION: stale weight must not survive a param update
# =====================================================================
print("\n[2] cache invalidation after in-place update")
w = torch.randn(1024, 256, device=dev, requires_grad=True)
a = OPT(w, "softplus")
with torch.no_grad():
    w.add_(1.0)                      # simulates optimizer step -> bumps w._version
b = OPT(w, "softplus")
ref = ORIG(w, "softplus")
inval_ok = torch.allclose(b, ref, rtol=1e-5, atol=1e-6)
stale = torch.allclose(a, b)         # if True, cache returned a stale tensor
ok &= inval_ok and not stale
print(f"    recomputed-correctly={'OK' if inval_ok else 'FAIL'}  served-stale={'YES(BAD)' if stale else 'no'}")

# =====================================================================
# (3) TRAINING EQUIVALENCE: identical trajectory over K real steps
# =====================================================================
print(f"\n[3] training-equivalence over {args.steps} steps (grads + param trajectory)")
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_value_
from dendritic_modeling.config import load_config
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

cfg = load_config(CONFIG)
B = cfg.training.main.common.batch_size
if args.synthetic:
    torch.manual_seed(1)
    x = torch.randn(B, 3072, device=dev); y = torch.randint(0, 10, (B,), device=dev)
else:
    from torch.utils.data import DataLoader
    from dendritic_modeling.datasets import get_unified_datasets
    tc = type("T", (), {"dataset": cfg.data.dataset_name, "data_path": cfg.data.base_dir,
        "train_valid_split": cfg.experiment.train_valid_split,
        "parameters": {"flatten": True, "normalize": True,
                       "split_seed": cfg.experiment.seed, "label_noise_seed": cfg.experiment.seed}})()
    ds, _, _ = get_unified_datasets(tc)
    x, y = next(iter(DataLoader(ds, batch_size=B, shuffle=False)))
    x, y = x.to(dev), y.to(dev)

def run(which_fn, seed=0):
    torch.manual_seed(seed)
    model, _ = initialize_model(cfg.model); model = model.to(dev)
    patch_all(which_fn)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce, scaler = nn.CrossEntropyLoss(), GradScaler()
    losses = []
    for _ in range(args.steps):
        with autocast():
            loss = ce(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        clip_grad_value_(model.parameters(), cfg.training.main.common.grad_clip_value)
        scaler.step(opt); scaler.update()
        losses.append(loss.item())
    final = torch.cat([p.detach().flatten() for p in model.parameters()])
    return losses, final

l_orig, f_orig = run(ORIG, seed=0)
l_opt,  f_opt  = run(OPT,  seed=0)
traj_ok = all(abs(a - b) < 1e-4 for a, b in zip(l_orig, l_opt))
param_ok = torch.allclose(f_orig, f_opt, rtol=1e-4, atol=1e-5)
ok &= traj_ok and param_ok
print(f"    loss trajectory match={'OK' if traj_ok else 'FAIL'}  (orig[-1]={l_orig[-1]:.5f} opt[-1]={l_opt[-1]:.5f})")
print(f"    final params match   ={'OK' if param_ok else 'FAIL'}  max|delta|={(f_orig-f_opt).abs().max().item():.2e}")

print("\n" + ("=" * 60))
if not ok:
    print("CORRECTNESS: FAILED -- do NOT proceed to perf. Fix the kernel first.")
    sys.exit(1)
print("CORRECTNESS: ALL PASSED. Proceeding to performance.")

# =====================================================================
# PERFORMANCE (only reached if correct): mean step time, orig vs opt
# =====================================================================
def timed(which_fn, warmup=5, iters=20):
    torch.manual_seed(0)
    model, _ = initialize_model(cfg.model); model = model.to(dev)
    patch_all(which_fn)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce, scaler = nn.CrossEntropyLoss(), GradScaler()
    def step():
        with autocast():
            loss = ce(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); clip_grad_value_(model.parameters(), 5.0)
        scaler.step(opt); scaler.update()
    for _ in range(warmup): step()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters): step()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters

t_orig = timed(ORIG); t_opt = timed(OPT)
print(f"\nmean step time  original={t_orig:.2f} ms  optimized={t_opt:.2f} ms  "
      f"speedup={t_orig/t_opt:.3f}x  ({(t_orig-t_opt):.2f} ms/step saved)")
