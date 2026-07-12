"""Step 1: Profile the PRE-optimized apply_weight_transform for this config.

Attributes every apply_weight_transform call into a single profiler row and
counts calls per step (to reveal whether there is redundant recompute to cache).
Original source files are NOT modified -- attribution is done by monkeypatching
the already-imported function reference in every module that bound it.

Run from anywhere:  python profile_weight_transform.py
"""
import os, sys, types, argparse

REPO = "/n/home12/rwhalen/DendriNet"
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_value_
from torch.profiler import profile, ProfilerActivity, record_function

CONFIG = "configs/examples/sglimit5_h100_cifar10_ei_1layer_bf33_10000e1250i.yaml"

p = argparse.ArgumentParser()
p.add_argument("--synthetic", action="store_true", help="skip disk, use random input")
p.add_argument("--warmup", type=int, default=5)
p.add_argument("--iters", type=int, default=10)
p.add_argument("--batch", type=int, default=None, help="override batch size (default: config's 512)")
args = p.parse_args()

# reduce allocator fragmentation on small MIG slices
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

assert torch.cuda.is_available(), "no CUDA device visible"
dev = "cuda"
print(f"GPU: {torch.cuda.get_device_name(0)}")

# ---- build the real model from the real config -----------------------------
from dendritic_modeling.config import load_config
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

cfg = load_config(CONFIG)
model, _ = initialize_model(cfg.model)
model = model.to(dev)
n_params = sum(p.numel() for p in model.parameters())
print(f"model built: {n_params/1e6:.1f}M params")

# ---- one real (or synthetic) batch -----------------------------------------
B = args.batch or cfg.training.main.common.batch_size  # default 512
if args.synthetic:
    x = torch.randn(B, 3072, device=dev)
    y = torch.randint(0, 10, (B,), device=dev)
    print(f"input: synthetic {tuple(x.shape)}")
else:
    from torch.utils.data import DataLoader
    from dendritic_modeling.datasets import get_unified_datasets
    task_cfg = type("T", (), {
        "dataset": cfg.data.dataset_name, "data_path": cfg.data.base_dir,
        "train_valid_split": cfg.experiment.train_valid_split,
        "parameters": {"flatten": True, "normalize": True,
                       "split_seed": cfg.experiment.seed,
                       "label_noise_seed": cfg.experiment.seed}})()
    train_ds, _, _ = get_unified_datasets(task_cfg)
    x, y = next(iter(DataLoader(train_ds, batch_size=B, shuffle=True)))
    x, y = x.to(dev), y.to(dev)
    print(f"input: real CIFAR-10 {tuple(x.shape)}")

# ---- monkeypatch apply_weight_transform: attribute + count -----------------
import dendritic_modeling.networks.utils.weight_transforms as wt
_orig = wt.apply_weight_transform
_stats = {"calls": 0, "shapes": {}}

def _wrapped(weights, transform_type="exp"):
    _stats["calls"] += 1
    key = (tuple(weights.shape), str(weights.dtype))
    _stats["shapes"][key] = _stats["shapes"].get(key, 0) + 1
    with record_function("apply_weight_transform"):
        return _orig(weights, transform_type)

# patch the definition module AND every module that did `from ... import apply_weight_transform`
patched = 0
for m in list(sys.modules.values()):
    if isinstance(m, types.ModuleType) and getattr(m, "apply_weight_transform", None) is _orig:
        setattr(m, "apply_weight_transform", _wrapped); patched += 1
print(f"patched apply_weight_transform in {patched} module namespace(s)")

# ---- training step matching the real `standard` strategy (AMP) -------------
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
ce, scaler = nn.CrossEntropyLoss(), GradScaler()

def step():
    with autocast():
        loss = ce(model(x), y)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    clip_grad_value_(model.parameters(), cfg.training.main.common.grad_clip_value)
    scaler.step(opt); scaler.update()

# ---- warmup ----------------------------------------------------------------
for _ in range(args.warmup):
    step()
torch.cuda.synchronize()

# ---- stable per-step wall time (CUDA events) -------------------------------
_stats["calls"] = 0
ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
ev0.record()
for _ in range(args.iters):
    step()
ev1.record(); torch.cuda.synchronize()
step_ms = ev0.elapsed_time(ev1) / args.iters
calls_per_step = _stats["calls"] / args.iters
print(f"\nmean step time: {step_ms:.2f} ms  |  apply_weight_transform calls/step: {calls_per_step:.0f}")
print("weight shapes seen (shape -> calls over run):")
for k, v in sorted(_stats["shapes"].items(), key=lambda kv: -kv[1]):
    print(f"    {k[0]} {k[1]}: {v}")

# ---- attributed profiler pass ----------------------------------------------
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=False) as prof:
    for _ in range(args.iters):
        step()
    torch.cuda.synchronize()

print("\n" + "=" * 90)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

# ---- pull out the attributed row + verdict ---------------------------------
row = next((e for e in prof.key_averages() if e.key == "apply_weight_transform"), None)
if row is not None:
    total_cuda_us = sum(e.self_cuda_time_total for e in prof.key_averages())
    share = 100.0 * row.cuda_time_total / max(total_cuda_us, 1)
    print(f"\napply_weight_transform: {row.cuda_time_total/1e3:.2f} ms CUDA over "
          f"{args.iters} steps ({row.cuda_time_total/1e3/args.iters:.3f} ms/step), "
          f"~{share:.1f}% of total GPU time, {row.count} calls")
    print("VERDICT HEURISTIC:")
    print(f"  * If share < ~2-3% -> caching cannot help meaningfully; stop here.")
    print(f"  * If calls/step ~= number of distinct synapse/blocklinear weights -> no")
    print(f"    redundant recompute in this (feedforward) config; caching is a no-op.")
    print(f"  * Redundancy exists only if calls/step exceeds the distinct-weight count.")
else:
    print("\n(apply_weight_transform row not found -- patch may have missed a binding)")
