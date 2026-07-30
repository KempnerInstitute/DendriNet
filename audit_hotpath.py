#!/usr/bin/env python
"""Passes B/C/D of the optimization sweep: syncs, anti-patterns, allocations.

Pass A (scale_diff.py) ranks kernels by how their cost GROWS. These three find
defects that a kernel-level profile cannot see at all:

  B. SYNC AUDIT -- every .item()/.cpu()/bool(tensor) in a hot path stalls the
     CPU until the GPU drains, serialising a pipeline that should overlap. These
     do not show up as kernel time; they show up as gaps between kernels.
  C. ANTI-PATTERN SCAN -- static scan for the four defect classes this codebase
     has actually shipped: dense materialisation of a sparse structure,
     autograd-retained intermediates, loop-invariant recompute, and Python loops
     over tensors. Ranked by whether the enclosing function is on the per-step
     path.
  D. ALLOCATION PROFILE -- attributes peak memory to the source line that
     allocated it, which is how you find the tensor that actually costs you
     rather than the one you assume does.

B and D need a GPU and a config; C is static and runs anywhere.

    python audit_hotpath.py --pass C
    python audit_hotpath.py --pass B --config <cfg.yaml> --batch 64
    python audit_hotpath.py --pass D --config <cfg.yaml> --batch 64
"""

from __future__ import annotations

import argparse
import ast
import os
import pathlib
import sys

SRC = pathlib.Path("src/dendritic_modeling")

# Functions that run once per training step (or per timestep). A defect inside
# one of these is paid every step; the same code in __init__ is paid once.
HOT = {
    "forward", "step", "_forward_recurrent", "_forward_feedforward",
    "compute_raw_currents", "compute_branch_raw_currents",
    "_stateful_raw_currents", "backward", "apply_weight_transform",
    "sparse_weight", "_normalized_sparse_weight", "_transformed_weight",
}
SYNC_CALLS = {"item", "tolist", "cpu", "numpy", "nonzero_static"}


# --------------------------------------------------------------------------- #
# Pass C -- static anti-pattern scan                                           #
# --------------------------------------------------------------------------- #
def enclosing_functions(tree):
    """Map each node to the nearest enclosing FunctionDef name."""
    owner = {}

    def walk(node, fn):
        for child in ast.iter_child_nodes(node):
            f = child.name if isinstance(child, ast.FunctionDef) else fn
            owner[child] = f
            walk(child, f)

    walk(tree, None)
    return owner


def scan_file(path):
    try:
        tree = ast.parse(path.read_text())
    except Exception:
        return []
    owner = enclosing_functions(tree)
    out = []

    for node in ast.walk(tree):
        fn = owner.get(node)
        hot = fn in HOT

        # (i) device syncs
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in SYNC_CALLS and not node.args:
                out.append(("sync", node.lineno, fn, f".{node.func.attr}()", hot))

        # (ii) dense allocation shaped [out_features, in_features]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"zeros", "ones", "empty", "full"}:
                names = [
                    a.attr if isinstance(a, ast.Attribute) else
                    (a.id if isinstance(a, ast.Name) else "")
                    for a in node.args
                ]
                if any("out_features" in n for n in names) and any(
                    "in_features" in n for n in names
                ):
                    out.append(("dense", node.lineno, fn,
                                f"torch.{node.func.attr}(out_features, in_features)",
                                hot))

        # (iii) Python loop over a tensor dimension inside a hot function
        if isinstance(node, (ast.For, ast.While)) and hot:
            out.append(("pyloop", node.lineno, fn, type(node).__name__.lower(), hot))

    return out


def pass_c():
    rows = []
    for p in sorted(SRC.rglob("*.py")):
        if any(s in str(p) for s in ("/tests/", "/analysis/", "/scripts/")):
            continue
        for kind, line, fn, what, hot in scan_file(p):
            rows.append((hot, kind, str(p), line, fn, what))
    rows.sort(key=lambda r: (not r[0], r[1], r[2]))

    LABEL = {"sync": "DEVICE SYNC", "dense": "DENSE ALLOC", "pyloop": "PYTHON LOOP"}
    for hot_only in (True, False):
        sel = [r for r in rows if r[0] is hot_only]
        head = ("ON THE PER-STEP PATH (paid every step)" if hot_only
                else "off the hot path (init/setup -- paid once)")
        print(f"\n{'=' * 78}\n{head}: {len(sel)} finding(s)\n{'=' * 78}")
        for _, kind, path, line, fn, what in sel[:40 if hot_only else 12]:
            print(f"  {LABEL[kind]:<12} {path.replace('src/dendritic_modeling/', '')}"
                  f":{line}  in {fn}()  -- {what}")
        if len(sel) > (40 if hot_only else 12):
            print(f"  ... and {len(sel) - (40 if hot_only else 12)} more")
    return 0


# --------------------------------------------------------------------------- #
# Pass B -- runtime sync audit                                                 #
# --------------------------------------------------------------------------- #
def pass_b(config, batch):
    import torch
    from torch.profiler import ProfilerActivity, profile

    core, dim, step = _build(config, batch)
    for _ in range(2):
        core.zero_grad(set_to_none=True)
        step().backward()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        core.zero_grad(set_to_none=True)
        step().backward()
        torch.cuda.synchronize()

    tot = {}
    for e in prof.key_averages():
        k = e.key
        if any(s in k for s in ("cudaStreamSynchronize", "cudaDeviceSynchronize",
                                "cudaMemcpyAsync", "Memcpy DtoH", "item")):
            tot[k] = (int(e.count), float(getattr(e, "cpu_time_total", 0)) / 1000)
    print(f"\n{'=' * 78}\nPASS B -- device syncs in one fwd+bwd (batch {batch})\n{'=' * 78}")
    if not tot:
        print("  no explicit sync/DtoH events recorded")
    for k, (n, ms) in sorted(tot.items(), key=lambda kv: -kv[1][1]):
        print(f"  {n:>7} x  {ms:9.2f} ms CPU   {k[:56]}")
    print("\n  A sync stalls the CPU until the GPU drains; it costs wall-clock")
    print("  that never appears as kernel time.")
    return 0


# --------------------------------------------------------------------------- #
# Pass D -- allocation profile by call site                                    #
# --------------------------------------------------------------------------- #
def pass_d(config, batch, topn=15):
    import torch

    core, dim, step = _build(config, batch)
    for _ in range(2):
        core.zero_grad(set_to_none=True)
        step().backward()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    torch.cuda.memory._record_memory_history(max_entries=200_000)
    core.zero_grad(set_to_none=True)
    step().backward()
    torch.cuda.synchronize()
    snap = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)

    by_site = {}
    for seg in snap.get("segments", []):
        for blk in seg.get("blocks", []):
            frames = blk.get("frames") or []
            if not frames or blk.get("state") != "active_allocated":
                continue
            f = next((fr for fr in frames
                      if "dendritic_modeling" in (fr.get("filename") or "")), frames[0])
            key = (f"{os.path.basename(f.get('filename','?'))}:{f.get('line','?')}"
                   f"  {f.get('name','?')}")
            sz, n = by_site.get(key, (0, 0))
            by_site[key] = (sz + blk.get("size", 0), n + 1)

    print(f"\n{'=' * 78}\nPASS D -- live allocations by source line (batch {batch})\n{'=' * 78}")
    if not by_site:
        print("  no attributed allocations captured")
    tot = sum(v[0] for v in by_site.values()) or 1
    print(f"{'MB':>10}{'blocks':>9}{'share':>8}  site")
    for k, (sz, n) in sorted(by_site.items(), key=lambda kv: -kv[1][0])[:topn]:
        print(f"{sz / 2**20:>10,.0f}{n:>9}{100 * sz / tot:>7.1f}%  {k[:60]}")
    return 0


def _build(config, batch):
    import torch
    sys.path.insert(0, ".")
    import importlib.util
    spec = importlib.util.spec_from_file_location("pd", "profile_dendrinet.py")
    pd = importlib.util.module_from_spec(spec)
    sys.modules["pd"] = pd
    spec.loader.exec_module(pd)
    core, dim, meta = pd.build_core_from_config(config)
    pd.apply_path(core, "cuda" if pd.assert_path_available and False else "fused")
    pd.freeze_rewiring(core)
    core = core.to("cuda").train()
    x = torch.randn(batch, dim, device="cuda").abs().requires_grad_(True)

    def step():
        out = core(x)
        out = out[0] if isinstance(out, (tuple, list)) else out
        return out.float().pow(2).mean()

    return core, dim, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="which", choices=["B", "C", "D"], required=True)
    ap.add_argument("--config")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    if a.which == "C":
        return pass_c()
    if not a.config:
        raise SystemExit(f"pass {a.which} needs --config")
    return pass_b(a.config, a.batch) if a.which == "B" else pass_d(a.config, a.batch)


if __name__ == "__main__":
    raise SystemExit(main())
