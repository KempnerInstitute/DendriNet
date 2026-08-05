#!/usr/bin/env python
"""Rank operations by how their cost GROWS, not by how much it is.

Pass A of the optimization sweep. Profiling one configuration tells you what is
expensive today; profiling two scales and differencing tells you what will be
expensive at the scale you actually care about. That distinction is how the
sparse gather was originally found -- its grad_x staging went from 3% to 32% of
backward as in_features grew, which no single-point profile would have surfaced.

For each device kernel it fits a growth exponent between two scale points:

    exponent = log(t2 / t1) / log(s2 / s1)

    ~1.0  cost is proportional to scale -- the expected, healthy case
    >1.2  SUPER-LINEAR: this op is a scaling hazard, it will dominate later
    <0.8  sub-linear: fixed overhead being amortised away

The scale axis can be batch (`--axis batch`, using two batch sizes of one
config) or model size (`--axis model`, using two configs whose contact counts
differ). Batch answers "what breaks as I feed more data"; model answers "what
breaks as I grow the network".

Usage:
    python scale_diff.py --root profilings/dendrinet_harness/ff_stress/blackwell \
                         --path cuda --axis batch --scales 64 512
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict

WRAPPERS = ("_TritonSparseGather", "_CudaSparseGather")


def is_kernel(op: str) -> bool:
    """True for real device kernels only.

    The profiler emits an aten:: dispatch row and its kernel row for the same
    work, and the custom paths add an autograd.Function wrapper whose time is the
    sum of its own kernels. Counting all three triples the total.
    """
    if op.startswith("aten::") or "Command Buffer" in op:
        return False
    return not op.startswith(WRAPPERS)


def load_cells(root):
    cells = {}
    for f in glob.glob(os.path.join(root, "**", "results_*.json"), recursive=True):
        for r in json.load(open(f)):
            if r.get("status") == "OK" and r.get("split_fb"):
                cells[(r.get("path"), r.get("batch"))] = r
    return cells


def op_times(rec):
    """{kernel: ms} summed over forward and backward, deduplicated."""
    out = defaultdict(float)
    for phase in ("forward", "backward"):
        for o in rec["split_fb"][phase]["ops"]:
            if is_kernel(o["op"]):
                out[o["op"]] += o["self_cuda_us"] / 1000.0
    return out


def short(op, n=54):
    s = op
    for junk in ("void ", "at::native::", "(anonymous namespace)::", "at_cuda_detail::"):
        s = s.replace(junk, "")
    s = s.split("(")[0].split("<")[0].split("::")[-1]
    return s[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--path", default="cuda")
    ap.add_argument("--axis", choices=["batch", "model"], default="batch")
    ap.add_argument("--scales", nargs=2, type=float, required=True,
                    help="the two scale points (batch sizes, or contact counts)")
    ap.add_argument("--root2", default=None,
                    help="second result tree, for --axis model")
    ap.add_argument("--min-ms", type=float, default=5.0,
                    help="ignore kernels below this at the larger scale")
    args = ap.parse_args()

    s1, s2 = args.scales
    if args.axis == "batch":
        cells = load_cells(args.root)
        try:
            r1, r2 = cells[(args.path, int(s1))], cells[(args.path, int(s2))]
        except KeyError:
            have = sorted(b for p, b in cells if p == args.path)
            raise SystemExit(f"need per-op data for {args.path} at {int(s1)} and "
                             f"{int(s2)}; have {have}")
    else:
        c1, c2 = load_cells(args.root), load_cells(args.root2 or args.root)
        r1 = next(v for k, v in sorted(c1.items()) if k[0] == args.path)
        r2 = next(v for k, v in sorted(c2.items()) if k[0] == args.path)

    t1, t2 = op_times(r1), op_times(r2)
    ratio = s2 / s1
    rows = []
    for op in set(t1) | set(t2):
        a, b = t1.get(op, 0.0), t2.get(op, 0.0)
        if b < args.min_ms:
            continue
        # A kernel absent at the small scale cannot be fitted; report it apart.
        exp = math.log(b / a) / math.log(ratio) if a > 0 else float("inf")
        rows.append((exp, b, a, op))
    rows.sort(key=lambda r: (-r[0], -r[1]))

    tot1, tot2 = sum(t1.values()), sum(t2.values())
    whole = math.log(tot2 / tot1) / math.log(ratio)
    print(f"scale axis : {args.axis}   {s1:g} -> {s2:g}  ({ratio:g}x)")
    print(f"path       : {args.path}")
    print(f"whole step : {tot1:,.0f} -> {tot2:,.0f} ms   exponent {whole:.2f}"
          f"   ({'super-linear' if whole > 1.1 else 'linear' if whole > 0.9 else 'sub-linear'})")
    print(f"\n{'exp':>6}{'ms @' + str(int(s2)):>12}{'ms @' + str(int(s1)):>12}"
          f"{'share':>8}  kernel")
    print(f"{'-'*6}{'-'*12}{'-'*12}{'-'*8}  {'-'*54}")
    for exp, b, a, op in rows:
        flag = "  <-- SUPER-LINEAR" if exp > 1.2 else ""
        e = "  new" if exp == float("inf") else f"{exp:6.2f}"
        print(f"{e}{b:>12,.0f}{a:>12,.0f}{100 * b / tot2:>7.1f}%  {short(op)}{flag}")
    haz = [r for r in rows if r[0] > 1.2]
    print(f"\n{len(haz)} kernel(s) growing super-linearly"
          f" ({sum(r[1] for r in haz) / tot2 * 100:.1f}% of time at the larger scale)"
          if haz else "\nNo super-linear kernels: every op scales at or below the axis.")


if __name__ == "__main__":
    main()
