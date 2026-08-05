"""Sweep batch size x memory-optimization mode, recording speed, peak, and OOM.

Feeds the six-figure set. The question is what the reactivation/shunt work bought
in the two currencies that matter -- step time and peak memory -- and, at the top
of the batch range, whether a configuration runs at all.

Modes are the cumulative stack, so each row isolates one change:

    eager        (tanh(m(x-b))+1)/2                 retains x-b, tanh(...), y
    identity     sigmoid(2m(x-b))                   retains x-b, y
    fused-act    _PositiveTanhCentered              retains y
    fused-shunt  _ShuntDividePositiveTanh           retains d+eps, y

The model is built ONCE and the mode flags toggled per measurement -- they are
read at call time, not construction time -- because building the 12-layer 16k
config takes minutes and 24 rebuilds would dominate the run.

**OOM is a result, not a crash.** At the top of the range some cells will not fit,
and where each mode dies is exactly what the figures need to show. Each cell is
guarded, recovered, and recorded with status="oom".

    python sweep_batch_modes.py --config <cfg.yaml> --out sweep.json
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time

import torch

from dendritic_modeling.networks.activations.parametric import (
    set_fused_positive_tanh,
    set_fused_shunt,
    set_positive_tanh_sigmoid,
)

# (name, sigmoid identity, fused activation, fused shunt) -- cumulative
MODES = [
    ("eager", False, False, False),
    ("identity", True, False, False),
    ("fused-act", True, True, False),
    ("fused-shunt", True, True, True),
]
BATCHES = [128, 256, 512, 1024, 2048, 4096]


def _load_profiler():
    spec = importlib.util.spec_from_file_location("pd", "profile_dendrinet.py")
    pd = importlib.util.module_from_spec(spec)
    sys.modules["pd"] = pd
    spec.loader.exec_module(pd)
    return pd


def _clear():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def measure(core, dim, batch, iters, warmup=2):
    """Return (step_ms, peak_mb) or raise torch.OutOfMemoryError."""
    x = torch.randn(batch, dim, device="cuda").abs().requires_grad_(True)

    def step():
        core.zero_grad(set_to_none=True)
        o = core(x)
        o = o[0] if isinstance(o, (tuple, list)) else o
        o.float().pow(2).mean().backward()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    _clear()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    peak = torch.cuda.max_memory_allocated() / 2**20
    del x
    _clear()
    return ms, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--path", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", type=int, nargs="*", default=BATCHES)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("needs CUDA")
        return 1

    props = torch.cuda.get_device_properties(0)
    gpu = props.name
    capacity = props.total_memory / 2**20
    print(f"GPU {gpu}  {capacity:,.0f} MB  path={a.path}\n", flush=True)

    pd = _load_profiler()
    pd.assert_path_available(a.path)
    core, dim, _ = pd.build_core_from_config(a.config)
    pd.apply_path(core, a.path)
    pd.freeze_rewiring(core)
    core = core.to("cuda").train()
    print("model built; sweeping\n", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    rows = []

    def flush():
        # Written after every cell. This sweep takes ~20 min on a live
        # allocation; losing it to a missing directory or a preemption at the
        # last line is not an acceptable failure mode.
        with open(a.out, "w") as fh:
            json.dump({"gpu": gpu, "capacity_mb": capacity, "config": a.config,
                       "path": a.path, "rows": rows}, fh, indent=1)

    print(f"{'mode':>12}{'batch':>7}{'step ms':>11}{'peak MB':>11}  status", flush=True)
    for name, ident, fused, shunt in MODES:
        set_positive_tanh_sigmoid(ident)
        set_fused_positive_tanh(fused)
        set_fused_shunt(shunt)
        oomed = False
        for b in a.batches:
            if oomed:  # once a mode OOMs, every larger batch will too
                rows.append({"gpu": gpu, "capacity_mb": capacity, "mode": name,
                             "batch": b, "status": "oom", "step_ms": None,
                             "peak_mb": None})
                print(f"{name:>12}{b:>7}{'—':>11}{'—':>11}  oom (skipped)", flush=True)
                flush()
                continue
            iters = 5 if b <= 1024 else 3
            try:
                ms, peak = measure(core, dim, b, iters)
                rows.append({"gpu": gpu, "capacity_mb": capacity, "mode": name,
                             "batch": b, "status": "ok", "step_ms": ms,
                             "peak_mb": peak})
                print(f"{name:>12}{b:>7}{ms:>11.1f}{peak:>11.1f}  ok", flush=True)
                flush()
            except torch.OutOfMemoryError:
                oomed = True
                _clear()
                rows.append({"gpu": gpu, "capacity_mb": capacity, "mode": name,
                             "batch": b, "status": "oom", "step_ms": None,
                             "peak_mb": None})
                print(f"{name:>12}{b:>7}{'—':>11}{'—':>11}  OOM", flush=True)
                flush()

    set_positive_tanh_sigmoid(True)
    set_fused_positive_tanh(True)
    set_fused_shunt(True)
    flush()
    print(f"\nwrote {a.out}  ({sum(r['status'] == 'ok' for r in rows)}/{len(rows)} ok)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
