"""Sweep output_chunk_size for the fused gather to test the launch-bound hypothesis.

Bigger chunks => fewer loop iterations => fewer kernel launches, at the cost of a
larger transient `selected` tensor (more memory). If wall-time drops as chunk grows,
the model is launch-bound and Triton fusion (which also cuts launches) will help.
If wall-time is flat, we're GPU-bound on the diffuse ops.

Chunk size is a pure loop-tiling knob: output is bit-identical for every value, so
this only measures speed/memory, not correctness.

Run:  python bench_chunksize.py --batch 128
"""
import os, sys, argparse
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
p.add_argument("--batch", type=int, default=128)
p.add_argument("--iters", type=int, default=15)
p.add_argument("--warmup", type=int, default=5)
p.add_argument("--chunks", default="2048,4096,8192,16384,131072",
               help="comma list; a value >= max out_features means one-shot (no chunking)")
args = p.parse_args()
chunks = [int(c) for c in args.chunks.split(",")]
print(f"GPU: {torch.cuda.get_device_name(0)}  batch={args.batch}\n")

cfg = load_config(CONFIG)
model, _ = initialize_model(cfg.model); model = model.to(dev)
idx_layers = [m for m in model.modules() if isinstance(m, IndexedSparseLinear)]
for m in idx_layers:
    m.use_fused_gather = True
print(f"use_fused_gather=True on {len(idx_layers)} indexed synapses")
print(f"max out_features among them: {max(m.out_features for m in idx_layers)}\n")

x = torch.randn(args.batch, 3072, device=dev)
y = torch.randint(0, 10, (args.batch,), device=dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
ce, scaler = nn.CrossEntropyLoss(), GradScaler("cuda")

def step():
    with autocast("cuda"):
        loss = ce(model(x), y)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt); clip_grad_value_(model.parameters(), 5.0)
    scaler.step(opt); scaler.update()

def timed(chunk):
    for m in idx_layers:
        m.output_chunk_size = chunk
    for _ in range(args.warmup): step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(args.iters): step()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / args.iters, torch.cuda.max_memory_allocated() / 1e9

print(f"{'chunk':>10}  {'ms/step':>9}  {'peak GB':>9}  {'vs 2048':>8}")
base = None
for c in chunks:
    try:
        ms, gb = timed(c)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"{c:>10}  {'OOM':>9}  {'--':>9}  {'--':>8}")
        continue
    if base is None: base = ms
    label = f"{c}" + ("*" if c >= max(m.out_features for m in idx_layers) else "")
    print(f"{label:>10}  {ms:>9.1f}  {gb:>9.2f}  {base/ms:>7.2f}x")
    torch.cuda.empty_cache()

print("\n* = one-shot (chunk >= out_features, no tiling)")
print("read: speedup as chunk grows  -> LAUNCH-BOUND (Triton fusion will help; also just")
print("      raise indexed_output_chunk_size in YAML for a free win).")
print("      flat or worse              -> GPU-bound on diffuse ops (chunk won't help).")
