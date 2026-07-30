"""End-to-end A/B of the forward-gather variant picker on a real config.

The micro-benchmark says the un-staged kernel is 1.50x at (16,000 -> 144,000)
and slower at most other shapes, so the picker is supposed to take the wins and
leave the losses. That claim is only worth anything measured on the real layer
mix, where the gather is one op among many and the backward is untouched.

Also asserts the whole-network output is UNCHANGED. The two kernels accumulate
in identical order, so this should be exact equality, not a tolerance.

    python bench_gather_variant_e2e.py --config <cfg.yaml> --batch 512
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    cuda_gather as cg,
)


def _load_profiler():
    spec = importlib.util.spec_from_file_location("pd", "profile_dendrinet.py")
    pd = importlib.util.module_from_spec(spec)
    sys.modules["pd"] = pd
    spec.loader.exec_module(pd)
    return pd


def measure(pd, config, batch, variant, iters):
    cg._FWD_VARIANT = variant
    torch.manual_seed(0)
    core, dim, _ = pd.build_core_from_config(config)
    pd.apply_path(core, "cuda")
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
    ms = (time.perf_counter() - t0) / iters * 1e3
    peak = torch.cuda.max_memory_allocated() / 2**20

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
    ap.add_argument("--iters", type=int, default=8)
    a = ap.parse_args()
    if not (torch.cuda.is_available() and cg.cuda_gather_available()):
        print("needs CUDA + built extension")
        return 1
    pd = _load_profiler()
    print(f"GPU {torch.cuda.get_device_name(0)}   {a.config}   batch={a.batch}\n")

    s_ms, s_mb, s_sig = measure(pd, a.config, a.batch, "staged", a.iters)
    a_ms, a_mb, a_sig = measure(pd, a.config, a.batch, "auto", a.iters)
    cg._FWD_VARIANT = "auto"

    print(f"{'':>18}{'step (ms)':>12}{'peak (MB)':>12}")
    print(f"{'staged (before)':>18}{s_ms:>12.2f}{s_mb:>12.1f}")
    print(f"{'auto picker':>18}{a_ms:>12.2f}{a_mb:>12.1f}")
    print(f"{'speedup':>18}{s_ms / a_ms:>11.3f}x{s_mb / a_mb:>11.3f}x")

    same = torch.equal(s_sig, a_sig)
    print(f"\n  network output identical: {same}"
          f"   (max|d| {(s_sig - a_sig).abs().max().item():.3e})")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
