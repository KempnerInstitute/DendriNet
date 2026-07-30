#!/usr/bin/env python
"""Batch-1 recurrent-RNN gather profiler: gather-bound or launch-bound?

MEASUREMENT ONLY -- nothing here changes model numerics or the shipped defaults.

Why this exists
---------------
The sparse-gather kernels (``use_cuda_gather`` / ``use_triton_gather`` /
``use_fused_gather`` in ``indexed_sparse.py``) were tuned for large-batch
feedforward cores, where they are a large win. In the batch-1 recurrent regime
(dendritic RNN, ~58 timesteps, batch 1) the A/B says something different, and the
winner flips with the GPU. Deciding issue #383 needs to know *which resource* the
step is actually spending its time on:

  * gather-bound   -> the GPU is busy inside the gather kernels
                      => the fix is a better (recurrent-native) kernel.
  * launch-bound   -> the GPU is idle waiting on CPU dispatch
                      => the fix is fewer launches (CUDA-graph the cell; the
                         connection indices are fixed across timesteps, so the
                         unrolled cell is a static graph).

``enable_profiling: true`` is not wired into the FSDP trainer, so this script
builds the same core directly from the config and profiles fwd+bwd standalone.

Two independent diagnostics
---------------------------
1. GPU-BUSY FRACTION (``--mode profile``). Sum every CUDA kernel duration from
   the profiler trace and divide by the un-profiled wall clock for the same
   region. High busy fraction => genuinely gather-bound. Low busy fraction, with
   ``cudaLaunchKernel`` CPU time filling the gap, => launch-bound. Also reports
   the kernel-count and kernel-duration histogram, and per-op self-CUDA time
   split cleanly into forward and backward passes.

2. BATCH SCALING (``--mode sweep``). Launch overhead is per-call and does not
   care about batch size; gather work is proportional to it. So a step time that
   stays ~flat from batch 1 to batch 32 is launch-bound, and one that scales
   ~linearly is gather-bound. This is independent of any profiler attribution.

Examples
--------
    python profile_rnn_gather.py --config <cfg.yaml> --path loop  --mode profile
    python profile_rnn_gather.py --config <cfg.yaml> --path fused --mode sweep
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import traceback
from collections import defaultdict

import torch

from dendritic_modeling.config import load_config
from dendritic_modeling.config.conversion import to_plain_dict
from dendritic_modeling.networks.architectures.factory import get_architecture

# The three gather flags, in the precedence order indexed_sparse.py applies them
# (cuda -> triton -> fused -> original advanced-index loop).
GATHER_FLAGS = ("use_cuda_gather", "use_triton_gather", "use_fused_gather")
PATH_FLAGS = {
    "loop": (),
    "fused": ("use_fused_gather",),
    "triton": ("use_triton_gather",),
    "cuda": ("use_cuda_gather",),
}
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


# --------------------------------------------------------------------------- #
# Build                                                                        #
# --------------------------------------------------------------------------- #
def build_core(config_path: str, input_dim_override: int | None):
    """Build the core module exactly as the training factory does."""
    config = load_config(config_path)
    core_cfg = to_plain_dict(config.model.core)
    core_type = (core_cfg.get("type") or "").lower()

    input_dim = input_dim_override
    if input_dim is None:
        input_dim = core_cfg.get("input_dim")
    if input_dim is None:
        # Recurrent configs carry the real input width on the (identity) encoder.
        try:
            enc = to_plain_dict(config.model.encoder) or {}
            input_dim = (enc.get("params") or {}).get("input_dim")
        except Exception:
            input_dim = None
    if input_dim is None:
        raise ValueError(
            f"Could not resolve input_dim for {config_path}; pass --input-dim."
        )

    core = get_architecture(
        core_type, core_cfg, input_dim=int(input_dim), suffix_input_dim=None
    )
    meta = {
        "config": config_path,
        "core_type": core_type,
        "input_dim": int(input_dim),
        "is_recurrent": bool(getattr(core, "is_recurrent", False)),
        "n_params": int(sum(p.numel() for p in core.parameters())),
    }
    return core, int(input_dim), meta


def apply_path(core, path: str) -> dict:
    """Force one gather path on every IndexedSparseLinear-like module.

    Every flag is set explicitly (not just the selected one) so the measured
    path is unambiguous regardless of what the config asked for.
    """
    on = set(PATH_FLAGS[path])
    n = 0
    shapes = []
    for m in core.modules():
        if not all(hasattr(m, f) for f in GATHER_FLAGS):
            continue
        for f in GATHER_FLAGS:
            setattr(m, f, f in on)
        n += 1
        shapes.append(
            {
                "in_features": int(getattr(m, "in_features", -1)),
                "out_features": int(getattr(m, "out_features", -1)),
                "K": int(getattr(m, "K", -1)),
                "chunk": int(getattr(m, "output_chunk_size", -1)),
            }
        )
    return {"layers_configured": n, "path": path, "layer_shapes": shapes}


def make_step(core, batch, seq_len, input_dim, device, dtype):
    """Return a closure running one forward and the loss for it."""
    x = (
        torch.randn(batch, seq_len, input_dim, device=device, dtype=dtype)
        .abs()  # shunting cores require non-negative drive
        .requires_grad_(True)
    )

    def step():
        out = core(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out, out.float().pow(2).mean()

    return step, x


# --------------------------------------------------------------------------- #
# Wall-clock timing (authoritative; measured with the profiler OFF)            #
# --------------------------------------------------------------------------- #
def time_step(core, step, x, warmup, iters):
    for _ in range(warmup):
        core.zero_grad(set_to_none=True)
        x.grad = None
        _, loss = step()
        loss.backward()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    fwd, bwd = [], []
    for _ in range(iters):
        core.zero_grad(set_to_none=True)
        x.grad = None
        f0, f1 = _ev(), _ev()
        b0, b1 = _ev(), _ev()
        f0.record()
        _, loss = step()
        f1.record()
        b0.record()
        loss.backward()
        b1.record()
        fwd.append((f0, f1))
        bwd.append((b0, b1))
    torch.cuda.synchronize()

    fwd_ms = [a.elapsed_time(b) for a, b in fwd]
    bwd_ms = [a.elapsed_time(b) for a, b in bwd]
    return {
        "fwd_ms_mean": statistics.mean(fwd_ms),
        "fwd_ms_cv": _cv(fwd_ms),
        "bwd_ms_mean": statistics.mean(bwd_ms),
        "bwd_ms_cv": _cv(bwd_ms),
        "step_ms_mean": statistics.mean(fwd_ms) + statistics.mean(bwd_ms),
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 1e6,
    }


def _ev():
    return torch.cuda.Event(enable_timing=True)


def _cv(xs):
    if len(xs) < 2:
        return 0.0
    m = statistics.mean(xs)
    return (statistics.pstdev(xs) / m) if m else 0.0


# --------------------------------------------------------------------------- #
# Profiler pass: forward and backward captured in separate traces so per-op    #
# self-CUDA time is cleanly attributable to one phase.                         #
# --------------------------------------------------------------------------- #
def _self_cuda_us(e):
    for name in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(e, name, None)
        if v is not None:
            return float(v)
    return 0.0


def _trace_stats(trace_path):
    """Kernel-level facts from a chrome trace: GPU busy time, launches, sizes."""
    with open(trace_path) as fh:
        trace = json.load(fh)
    events = trace.get("traceEvents", [])

    kernel_us = 0.0
    kernels = []
    memcpy_us = 0.0
    runtime_us = 0.0
    launch_us = 0.0
    launch_n = 0
    per_kernel = defaultdict(lambda: {"n": 0, "us": 0.0})

    for e in events:
        if e.get("ph") != "X":
            continue
        cat = (e.get("cat") or "").lower()
        dur = float(e.get("dur") or 0.0)
        name = e.get("name") or "?"
        if cat == "kernel":
            kernel_us += dur
            kernels.append(dur)
            per_kernel[name]["n"] += 1
            per_kernel[name]["us"] += dur
        elif cat in ("gpu_memcpy", "gpu_memset"):
            memcpy_us += dur
            kernels.append(dur)
            per_kernel[name]["n"] += 1
            per_kernel[name]["us"] += dur
        elif cat in ("cuda_runtime", "runtime"):
            runtime_us += dur
            if "launch" in name.lower():
                launch_us += dur
                launch_n += 1

    kernels.sort()
    all_kernels = sorted(
        (
            {"kernel": k, "n": v["n"], "total_us": v["us"]}
            for k, v in per_kernel.items()
        ),
        key=lambda r: -r["total_us"],
    )

    return {
        "all_kernels": all_kernels,
        "gpu_kernel_us": kernel_us + memcpy_us,
        "n_kernels": len(kernels),
        "kernel_us_median": statistics.median(kernels) if kernels else 0.0,
        "kernel_us_mean": statistics.mean(kernels) if kernels else 0.0,
        "n_kernels_under_5us": sum(1 for d in kernels if d < 5.0),
        "n_kernels_under_20us": sum(1 for d in kernels if d < 20.0),
        "cuda_runtime_us": runtime_us,
        "cuda_launch_us": launch_us,
        "n_cuda_launches": launch_n,
        "top_kernels": all_kernels[:25],
    }


def profile_phases(core, step, x, outdir, tag, n_iters):
    """Profile forward and backward separately; return per-phase op/kernel stats.

    Each iteration opens two independent profiler sessions -- one spanning only
    the forward call, one spanning only ``loss.backward()`` -- so a kernel is
    attributed to exactly one phase with no scope-nesting ambiguity. Only one
    autograd graph is alive at a time, which keeps peak memory at the same level
    as the untraced timing loop.
    """
    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    os.makedirs(outdir, exist_ok=True)

    ops = {p: defaultdict(lambda: {"n": 0, "us": 0.0}) for p in ("forward", "backward")}
    traces = {p: [] for p in ("forward", "backward")}

    # Warm the allocator so the traced iterations are steady-state, not first-call.
    core.zero_grad(set_to_none=True)
    x.grad = None
    _, warm = step()
    warm.backward()
    torch.cuda.synchronize()
    del warm

    def collect(prof, phase, i):
        for e in prof.key_averages():
            us = _self_cuda_us(e)
            if us <= 0:
                continue
            ops[phase][e.key]["n"] += int(e.count)
            ops[phase][e.key]["us"] += us
        path = os.path.join(outdir, f"trace_{tag}_{phase}_{i}.json")
        prof.export_chrome_trace(path)
        traces[phase].append(path)

    for i in range(n_iters):
        core.zero_grad(set_to_none=True)
        x.grad = None

        with profile(activities=acts, record_shapes=False) as pf:
            _, loss = step()
            torch.cuda.synchronize()
        collect(pf, "forward", i)

        with profile(activities=acts, record_shapes=False) as pb:
            loss.backward()
            torch.cuda.synchronize()
        collect(pb, "backward", i)

        del loss
        gc.collect()

    out = {}
    for phase in ("forward", "backward"):
        stats = _merge_trace_stats(traces[phase])
        total = sum(v["us"] for v in ops[phase].values()) or 1.0
        stats["top_ops"] = sorted(
            (
                {
                    "op": k,
                    "count": v["n"],
                    "self_cuda_us": v["us"],
                    "pct": 100 * v["us"] / total,
                }
                for k, v in ops[phase].items()
            ),
            key=lambda r: -r["self_cuda_us"],
        )[:25]
        stats["n_iters"] = n_iters
        stats["traces"] = traces[phase]
        out[phase] = stats

    torch.cuda.empty_cache()
    return out


def _merge_trace_stats(paths):
    """Sum per-iteration trace stats into one record."""
    merged = None
    per_kernel = defaultdict(lambda: {"n": 0, "us": 0.0})
    scalar_keys = (
        "gpu_kernel_us", "n_kernels", "n_kernels_under_5us",
        "n_kernels_under_20us", "cuda_runtime_us", "cuda_launch_us",
        "n_cuda_launches",
    )
    medians, means = [], []
    for p in paths:
        s = _trace_stats(p)
        if merged is None:
            merged = {k: 0 for k in scalar_keys}
        for k in scalar_keys:
            merged[k] += s[k]
        medians.append(s["kernel_us_median"])
        means.append(s["kernel_us_mean"])
        for row in s["all_kernels"]:
            per_kernel[row["kernel"]]["n"] += row["n"]
            per_kernel[row["kernel"]]["us"] += row["total_us"]
    merged["kernel_us_median"] = statistics.mean(medians) if medians else 0.0
    merged["kernel_us_mean"] = statistics.mean(means) if means else 0.0
    merged["top_kernels"] = sorted(
        ({"kernel": k, "n": v["n"], "total_us": v["us"]} for k, v in per_kernel.items()),
        key=lambda r: -r["total_us"],
    )[:25]
    return merged


# --------------------------------------------------------------------------- #
# Modes                                                                        #
# --------------------------------------------------------------------------- #
def run_profile(args, device, dtype):
    core, input_dim, meta = build_core(args.config, args.input_dim)
    info = apply_path(core, args.path)
    core = core.to(device=device, dtype=dtype).train()
    step, x = make_step(core, args.batch, args.seq_len, input_dim, device, dtype)

    result = {**meta, **info, "batch": args.batch, "seq_len": args.seq_len,
              "dtype": args.dtype, "gpu": torch.cuda.get_device_name(0)}
    result["layer_shapes"] = info["layer_shapes"][:12]  # keep the JSON readable

    wall = time_step(core, step, x, args.warmup, args.iters)
    result["wall"] = wall

    tag = f"{os.path.basename(args.config).replace('.yaml', '')}_{args.path}_b{args.batch}"
    phases = profile_phases(core, step, x, args.outdir, tag, args.prof_iters)
    result["phases"] = phases

    # The headline: GPU busy fraction per phase, profiler kernel time over the
    # profiler-free wall clock. <~0.5 means the GPU is idle waiting on the CPU.
    for phase, wall_key in (("forward", "fwd_ms_mean"), ("backward", "bwd_ms_mean")):
        p = phases[phase]
        per_iter_gpu_ms = p["gpu_kernel_us"] / p["n_iters"] / 1000.0
        wall_ms = wall[wall_key]
        p["gpu_ms_per_iter"] = per_iter_gpu_ms
        p["wall_ms_per_iter"] = wall_ms
        p["gpu_busy_frac"] = per_iter_gpu_ms / wall_ms if wall_ms else 0.0
        p["kernels_per_iter"] = p["n_kernels"] / p["n_iters"]
        p["launch_us_per_iter"] = p["cuda_launch_us"] / p["n_iters"]
    return result


def run_cudagraph(args, device, dtype):
    """Measure the launch-overhead ceiling: eager forward vs CUDA-graph replay.

    The connection indices are fixed for the life of the model and the batch and
    sequence length are fixed, so the unrolled cell is a static graph and the
    whole forward is capturable. Both sides run under ``no_grad`` so the only
    difference is who issues the kernels: the CPU dispatcher (eager) or the graph
    replay. The speedup is therefore a direct, hard upper bound on what
    CUDA-graphing the recurrent cell can recover.
    """
    core, input_dim, meta = build_core(args.config, args.input_dim)
    info = apply_path(core, args.path)
    core = core.to(device=device, dtype=dtype).train()

    x = torch.randn(
        args.batch, args.seq_len, input_dim, device=device, dtype=dtype
    ).abs()

    out = {**meta, "path": args.path, "batch": args.batch, "seq_len": args.seq_len,
           "dtype": args.dtype, "gpu": torch.cuda.get_device_name(0),
           "layers_configured": info["layers_configured"]}

    def timed(fn, n):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        evs = []
        for _ in range(n):
            a, b = _ev(), _ev()
            a.record()
            fn()
            b.record()
            evs.append((a, b))
        torch.cuda.synchronize()
        return [a.elapsed_time(b) for a, b in evs]

    with torch.no_grad():
        eager = timed(lambda: core(x), args.iters)
    out["eager_ms_mean"] = statistics.mean(eager)
    out["eager_ms_cv"] = _cv(eager)

    # Warm up on a side stream first: capture requires the allocator and every
    # lazily-initialized kernel to already be in their steady state.
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):
                core(x)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph):
                static_out = core(x)
        torch.cuda.synchronize()

        replay = timed(graph.replay, args.iters)
        out["graph_ms_mean"] = statistics.mean(replay)
        out["graph_ms_cv"] = _cv(replay)
        out["speedup"] = out["eager_ms_mean"] / out["graph_ms_mean"]
        out["launch_overhead_ms"] = out["eager_ms_mean"] - out["graph_ms_mean"]
        out["launch_overhead_frac"] = out["launch_overhead_ms"] / out["eager_ms_mean"]
        out["capture"] = "OK"
        out["static_out_shape"] = tuple(static_out.shape)
    except Exception as e:
        out["capture"] = "FAILED"
        out["capture_error"] = f"{type(e).__name__}: {str(e).splitlines()[0]}"
        out["capture_traceback"] = traceback.format_exc()
    return out


def run_sweep(args, device, dtype):
    """Batch scaling: flat => launch-bound, linear => gather-bound."""
    rows = []
    for batch in args.batches:
        core, input_dim, meta = build_core(args.config, args.input_dim)
        info = apply_path(core, args.path)
        core = core.to(device=device, dtype=dtype).train()
        try:
            step, x = make_step(core, batch, args.seq_len, input_dim, device, dtype)
            wall = time_step(core, step, x, args.warmup, args.iters)
            rows.append({"batch": batch, "status": "OK", **wall})
        except RuntimeError as e:
            msg = str(e).splitlines()[0]
            rows.append(
                {
                    "batch": batch,
                    "status": "OOM" if "out of memory" in msg.lower() else "ERROR",
                    "error": msg,
                }
            )
        del core
        gc.collect()
        torch.cuda.empty_cache()

    base = next((r for r in rows if r["status"] == "OK"), None)
    if base:
        for r in rows:
            if r["status"] == "OK":
                r["step_ms_rel"] = r["step_ms_mean"] / base["step_ms_mean"]
                r["batch_rel"] = r["batch"] / base["batch"]
                # 0 => perfectly flat (launch-bound); 1 => linear (work-bound).
                r["scaling_exponent"] = (
                    (r["step_ms_rel"] - 1) / (r["batch_rel"] - 1)
                    if r["batch_rel"] > 1
                    else 0.0
                )
    return {
        "config": args.config,
        "path": args.path,
        "seq_len": args.seq_len,
        "dtype": args.dtype,
        "gpu": torch.cuda.get_device_name(0),
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
def print_profile(r):
    print("=" * 78)
    print(f"CONFIG {r['config']}")
    print(f"GPU    {r['gpu']}   path={r['path']}  dtype={r['dtype']}  "
          f"batch={r['batch']}  T={r['seq_len']}")
    print(f"       {r['layers_configured']} indexed-sparse layers, "
          f"{r['n_params'] / 1e6:.1f}M params")
    w = r["wall"]
    print(f"\nWALL   fwd {w['fwd_ms_mean']:8.2f} ms (cv {w['fwd_ms_cv']:.3f})   "
          f"bwd {w['bwd_ms_mean']:8.2f} ms (cv {w['bwd_ms_cv']:.3f})   "
          f"step {w['step_ms_mean']:8.2f} ms   peak {w['peak_mem_mb']:.0f} MB")
    for phase in ("forward", "backward"):
        p = r["phases"][phase]
        print(f"\n--- {phase.upper()} " + "-" * (74 - len(phase)))
        print(f"  GPU busy {p['gpu_busy_frac'] * 100:5.1f}%   "
              f"({p['gpu_ms_per_iter']:.2f} ms GPU / {p['wall_ms_per_iter']:.2f} ms wall)")
        print(f"  kernels/iter {p['kernels_per_iter']:9.0f}   "
              f"median {p['kernel_us_median']:.1f} us   mean {p['kernel_us_mean']:.1f} us   "
              f"<5us: {100 * p['n_kernels_under_5us'] / max(p['n_kernels'], 1):.0f}%")
        print(f"  cudaLaunchKernel CPU time {p['launch_us_per_iter'] / 1000:.2f} ms/iter")
        print("  top ops by self-CUDA:")
        for row in p["top_ops"][:8]:
            print(f"    {row['pct']:5.1f}%  {row['self_cuda_us'] / p['n_iters'] / 1000:8.2f} ms/it  "
                  f"n={row['count']:6d}  {row['op'][:52]}")
        print("  top kernels:")
        for row in p["top_kernels"][:6]:
            print(f"           {row['total_us'] / p['n_iters'] / 1000:8.2f} ms/it  "
                  f"n={row['n']:6d}  {row['kernel'][:56]}")


def print_cudagraph(r):
    print("=" * 78)
    print(f"CUDA-GRAPH CEILING  {r['gpu']}  path={r['path']}  batch={r['batch']}  "
          f"T={r['seq_len']}  dtype={r['dtype']}")
    print(f"  eager forward (no_grad)  {r['eager_ms_mean']:8.2f} ms "
          f"(cv {r['eager_ms_cv']:.3f})")
    if r["capture"] != "OK":
        print(f"  CAPTURE FAILED: {r['capture_error']}")
        print("  -> the cell is not capturable as-is; see traceback in the JSON.")
        return
    print(f"  graph replay             {r['graph_ms_mean']:8.2f} ms "
          f"(cv {r['graph_ms_cv']:.3f})")
    print(f"  speedup                  {r['speedup']:8.2f}x")
    print(f"  launch overhead removed  {r['launch_overhead_ms']:8.2f} ms "
          f"({r['launch_overhead_frac'] * 100:.1f}% of eager forward)")


def print_sweep(s):
    print("=" * 78)
    print(f"BATCH SWEEP  {s['gpu']}  path={s['path']}  T={s['seq_len']}  dtype={s['dtype']}")
    print(f"{'batch':>6} {'step ms':>10} {'fwd ms':>10} {'bwd ms':>10} "
          f"{'rel':>7} {'scaling':>8}")
    for r in s["rows"]:
        if r["status"] != "OK":
            print(f"{r['batch']:>6} {r['status']:>10}  {r.get('error', '')[:40]}")
            continue
        print(f"{r['batch']:>6} {r['step_ms_mean']:>10.2f} {r['fwd_ms_mean']:>10.2f} "
              f"{r['bwd_ms_mean']:>10.2f} {r.get('step_ms_rel', 1):>7.2f} "
              f"{r.get('scaling_exponent', 0):>8.2f}")
    print("\n  scaling exponent: ~0.0 = flat (LAUNCH-BOUND), ~1.0 = linear (WORK-BOUND)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--path", choices=list(PATH_FLAGS), default="loop")
    ap.add_argument("--mode", choices=["profile", "sweep", "cudagraph"],
                    default="profile")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--seq-len", type=int, default=58,
                    help="romo default: 20 stim + 10 delay + 20 stim + 8 response")
    ap.add_argument("--input-dim", type=int, default=None)
    ap.add_argument("--dtype", choices=list(DTYPES), default="bf16",
                    help="bf16 matches the FSDP param_dtype used in the A/B runs")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--prof-iters", type=int, default=3)
    ap.add_argument("--outdir", default="profilings/dendrinet_harness/rnn_prof")
    ap.add_argument("--json", default=None, help="write the result dict here")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr)
        return 2
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda")
    dtype = DTYPES[args.dtype]
    os.makedirs(args.outdir, exist_ok=True)

    try:
        if args.mode == "profile":
            out = run_profile(args, device, dtype)
            print_profile(out)
        elif args.mode == "cudagraph":
            out = run_cudagraph(args, device, dtype)
            print_cudagraph(out)
        else:
            out = run_sweep(args, device, dtype)
            print_sweep(out)
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1

    path = args.json or os.path.join(
        args.outdir,
        f"{args.mode}_{os.path.basename(args.config).replace('.yaml', '')}_"
        f"{args.path}_b{args.batch}.json",
    )
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
