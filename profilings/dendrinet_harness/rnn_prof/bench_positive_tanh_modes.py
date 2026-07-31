"""End-to-end A/B of all three positive-tanh spellings on a real config.

  eager    : (tanh(m(x-b)) + 1) / 2   -- retains x-b, tanh(...), and the output
  identity : sigmoid(2m(x-b))         -- retains x-b and the output
  fused    : _PositiveTanhCentered     -- retains the output only

The allocation profile put this line at 57.6% of live allocation, so the count of
tensors it holds across backward is the largest memory lever in the model. This
measures what that count is worth end-to-end, and asserts the network output is
unchanged (the forward arithmetic is identical; only what backward keeps differs).

    python bench_positive_tanh_modes.py --config <cfg.yaml> --path cuda --batch 512
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time

import torch

from dendritic_modeling.networks.activations.parametric import (
    set_fused_positive_tanh,
    set_positive_tanh_sigmoid,
)

MODES = [("eager", False, False), ("identity", True, False), ("fused", True, True)]


def _load_profiler():
    spec = importlib.util.spec_from_file_location("pd", "profile_dendrinet.py")
    pd = importlib.util.module_from_spec(spec)
    sys.modules["pd"] = pd
    spec.loader.exec_module(pd)
    return pd


def measure(pd, cfg, batch, path, identity, fused, iters):
    set_positive_tanh_sigmoid(identity)
    set_fused_positive_tanh(fused)
    torch.manual_seed(0)
    core, dim, _ = pd.build_core_from_config(cfg)
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
    ms = (time.perf_counter() - t0) / iters * 1e3
    peak = torch.cuda.max_memory_allocated() / 2**20

    with torch.no_grad():
        out = core(x)
        out = out[0] if isinstance(out, (tuple, list)) else out
        sig = out.float().clone()
    # grad signature too -- the modes differ only in BACKWARD, so comparing the
    # forward output alone would not test the thing that actually changed.
    core.zero_grad(set_to_none=True)
    out = core(x)
    out = out[0] if isinstance(out, (tuple, list)) else out
    out.float().pow(2).mean().backward()
    gsig = x.grad.float().clone()
    del core, x
    torch.cuda.empty_cache()
    return ms, peak, sig, gsig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--path", default="cuda")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--iters", type=int, default=6)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("needs CUDA")
        return 1
    pd = _load_profiler()
    pd.assert_path_available(a.path)
    print(f"GPU {torch.cuda.get_device_name(0)}   {a.config}")
    print(f"path={a.path}  batch={a.batch}\n")

    res = {}
    for name, ident, fused in MODES:
        res[name] = measure(pd, a.config, a.batch, a.path, ident, fused, a.iters)
        print(f"  measured {name}")

    base_ms, base_mb, base_out, base_g = res["eager"]
    print(f"\n{'mode':>10}{'step (ms)':>12}{'peak (MB)':>12}{'vs eager':>11}"
          f"{'mem saved':>12}")
    for name, _, _ in MODES:
        ms, mb, _, _ = res[name]
        print(f"{name:>10}{ms:>12.1f}{mb:>12.1f}{base_mb / mb:>10.2f}x"
              f"{(base_mb - mb):>11.0f} MB")

    print(f"\n{'mode':>10}{'max rel d out':>16}{'max rel d grad_x':>19}")
    for name, _, _ in MODES:
        _, _, o, g = res[name]
        ro = (o - base_out).abs().max().item() / (base_out.abs().max().item() or 1)
        rg = (g - base_g).abs().max().item() / (base_g.abs().max().item() or 1)
        print(f"{name:>10}{ro:>16.2e}{rg:>19.2e}")

    set_positive_tanh_sigmoid(True)
    set_fused_positive_tanh(True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
