"""End-to-end A/B of the positive-tanh sigmoid identity on a real config.

bench_positive_tanh.py measures the activation in isolation, which overstates
what the model sees: the activation is one op among many, and the gather still
dominates. This times a full forward+backward on the actual stress config with
the flag off and on, which is the number worth quoting.

    python bench_positive_tanh_e2e.py --config <cfg.yaml> --batch 512
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time

import torch

from dendritic_modeling.networks.activations.parametric import (
    set_positive_tanh_sigmoid,
)


def _load_profiler():
    spec = importlib.util.spec_from_file_location("pd", "profile_dendrinet.py")
    pd = importlib.util.module_from_spec(spec)
    sys.modules["pd"] = pd
    spec.loader.exec_module(pd)
    return pd


def measure(pd, config, batch, path, enabled, iters=8):
    set_positive_tanh_sigmoid(enabled)
    torch.manual_seed(0)
    core, dim, _ = pd.build_core_from_config(config)
    pd.apply_path(core, path)
    pd.freeze_rewiring(core)
    core = core.to("cuda").train()
    x = torch.randn(batch, dim, device="cuda").abs().requires_grad_(True)

    def step():
        core.zero_grad(set_to_none=True)
        out = core(x)
        out = out[0] if isinstance(out, (tuple, list)) else out
        out.float().pow(2).mean().backward()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000.0
    peak = torch.cuda.max_memory_allocated() / 2**20

    # Same input, same seed -> outputs are comparable across the two paths.
    with torch.no_grad():
        out = core(x)
        out = out[0] if isinstance(out, (tuple, list)) else out
        sig = out.float().clone()

    del core, x
    torch.cuda.empty_cache()
    return ms, peak, sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--path", default="fused")
    ap.add_argument("--iters", type=int, default=8)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("needs CUDA")
        return 1
    pd = _load_profiler()

    print(f"GPU {torch.cuda.get_device_name(0)}   config {a.config}")
    print(f"path={a.path}  batch={a.batch}  iters={a.iters}\n")

    off_ms, off_mb, off_sig = measure(pd, a.config, a.batch, a.path, False, a.iters)
    on_ms, on_mb, on_sig = measure(pd, a.config, a.batch, a.path, True, a.iters)

    print(f"{'':>14}{'step (ms)':>12}{'peak (MB)':>12}")
    print(f"{'identity off':>14}{off_ms:>12.2f}{off_mb:>12.1f}")
    print(f"{'identity on':>14}{on_ms:>12.2f}{on_mb:>12.1f}")
    print(f"{'speedup':>14}{off_ms / on_ms:>11.3f}x{off_mb / on_mb:>11.3f}x")

    denom = off_sig.abs().max().item() or 1.0
    rel = (off_sig - on_sig).abs().max().item() / denom
    print(f"\n  max relative output deviation through the whole network: {rel:.2e}")
    set_positive_tanh_sigmoid(False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
