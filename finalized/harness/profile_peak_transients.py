"""Decompose the transient gap: what is live AT the peak, not at the top of the step.

Every allocation figure in this work so far came from `_snapshot()` taken after
forward, which sees only tensors RETAINED for backward. Peak is higher, and the
difference is transients -- allocated and freed inside the step, never present
when the snapshot fires:

    batch  512 : retained 10,097 MB   peak 13,370 MB   gap 3,273 MB (24.5%)
    batch 2048 : retained 38,931 MB   peak 48,810 MB   gap 9,880 MB (20.2%)

The gap barely moved all day because every fix targeted the retained side, so as a
share of peak it roughly tripled. It is now the largest unattributed term.

Method: record the allocator's full event trace, replay it, track the running
total, and dump the live set at the moment the total is maximal. That is the peak
composition -- transients included -- which a point-in-time snapshot cannot give.

Two suspects going in, both stated before measuring:
  * `_ShuntDividePositiveTanh.backward` allocates ~8 full-size intermediates
    (gs, gv, logit, v, grad_n, grad_d and temporaries). At [512, 144000] that is
    281 MB each. Cutting retained memory may have raised transient memory.
  * the gather backward materializes `xT` and `goT`; `goT` alone is 295 MB per
    layer. Measured at 5.2% of backward TIME earlier; its memory was never checked.

    python profile_peak_transients.py --config <cfg.yaml> --batch 512
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import os
import sys

import torch


def _load_profiler():
    spec = importlib.util.spec_from_file_location("pd", "profile_dendrinet.py")
    pd = importlib.util.module_from_spec(spec)
    sys.modules["pd"] = pd
    spec.loader.exec_module(pd)
    return pd


def _site(frames):
    if not frames:
        return "<no frames>"
    f = next((q for q in frames
              if "dendritic_modeling" in (q.get("filename") or "")), frames[0])
    name = f.get("name", "?")
    return f"{os.path.basename(f.get('filename', '?'))}:{f.get('line', '?')}  {name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--path", default="cuda")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--top", type=int, default=16)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("needs CUDA")
        return 1

    pd = _load_profiler()
    core, dim, _ = pd.build_core_from_config(a.config)
    pd.assert_path_available(a.path)
    pd.apply_path(core, a.path)
    pd.freeze_rewiring(core)
    core = core.to("cuda").train()
    x = torch.randn(a.batch, dim, device="cuda").abs().requires_grad_(True)
    print(f"GPU {torch.cuda.get_device_name(0)}\npath={a.path}  batch={a.batch}\n")

    def step():
        core.zero_grad(set_to_none=True)
        o = core(x)
        o = o[0] if isinstance(o, (tuple, list)) else o
        o.float().pow(2).mean().backward()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        torch.cuda.memory._record_memory_history(
            enabled="all", context="all", stacks="python", max_entries=3_000_000
        )
    except TypeError:                      # older signature
        torch.cuda.memory._record_memory_history(max_entries=3_000_000)
    step()
    torch.cuda.synchronize()
    snap = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    reported_peak = torch.cuda.max_memory_allocated() / 2**20

    traces = snap.get("device_traces") or []
    if not traces or not traces[0]:
        print("no allocator trace captured -- this torch build may not support "
              "context/stacks recording")
        return 1
    events = traces[0]

    # Replay: track live blocks by address, find the moment `total` is maximal.
    live: dict[int, tuple[int, object]] = {}
    total = 0
    best_total = -1
    best_live = None
    best_i = 0
    for i, e in enumerate(events):
        act = e.get("action")
        if act == "alloc":
            live[e["addr"]] = (e["size"], e.get("frames"))
            total += e["size"]
            if total > best_total:
                best_total, best_live, best_i = total, dict(live), i
        elif act in ("free_completed", "free_requested"):
            got = live.pop(e["addr"], None)
            if got is not None:
                total -= got[0]

    if best_live is None:
        print("trace had no alloc events")
        return 1

    by_site = collections.Counter()
    for size, frames in best_live.values():
        by_site[_site(frames)] += size

    peak_mb = best_total / 2**20
    print("=" * 84)
    print(f"LIVE AT PEAK  ({len(events):,} events, peak at event {best_i:,})")
    print("=" * 84)
    print(f"{'MB':>10}{'share':>8}  site")
    for k, sz in by_site.most_common(a.top):
        print(f"{sz / 2**20:>10,.1f}{100 * sz / best_total:>7.1f}%  {k[:64]}")
    print(f"\n  trace peak {peak_mb:,.1f} MB   torch-reported peak "
          f"{reported_peak:,.1f} MB")

    # Same attribution restricted to blocks still live at the END of the trace,
    # i.e. the retained set. The difference is what is transient.
    end_by_site = collections.Counter()
    for size, frames in live.values():
        end_by_site[_site(frames)] += size
    print("\n" + "=" * 84)
    print("TRANSIENT CONTRIBUTION  (live at peak, minus what survives the step)")
    print("=" * 84)
    print(f"{'MB':>10}  site")
    diff = collections.Counter()
    for k, v in by_site.items():
        d = v - end_by_site.get(k, 0)
        if d > 0:
            diff[k] = d
    for k, sz in diff.most_common(a.top):
        print(f"{sz / 2**20:>10,.1f}  {k[:70]}")
    print(f"\n  transient total {sum(diff.values()) / 2**20:,.1f} MB "
          f"of a {peak_mb:,.1f} MB peak")
    return 0


if __name__ == "__main__":
    sys.exit(main())
