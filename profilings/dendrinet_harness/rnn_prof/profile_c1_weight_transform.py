"""C1 profile: what does materializing the transformed weight actually cost?

`indexed_sparse.py:501` calls `_transformed_weight()` -> `softplus(pre_w)`, producing
a full [out_features, K] tensor per layer, BEFORE the gather path is chosen (:514) --
so it happens on loop, fused, triton and cuda alike.

Pass D previously put `weight_transforms.py:119` at 28.9% of live allocation, but
that run was forced to the `fused` path at batch 64 by a harness bug. This
re-measures it where it matters and cross-checks the profiler against arithmetic:

  [1] ARITHMETIC -- sum of [out_features, K] over every indexed layer. What the
      transform MUST cost if every output is held.
  [2] PROFILER   -- bytes actually attributed to weight_transforms.py during a
      real step, and the share of live allocation.
  [3] CONTEXT    -- the other big retained tensors, so C1 gets ranked against
      them rather than quoted alone. On the cuda path `ctx.save_for_backward`
      holds x (fp32) and idx (int64) too, and int64 idx is 2x the weight.

If [1] and [2] disagree the attribution is wrong and neither number should be
trusted.

    python profile_c1_weight_transform.py --config <cfg.yaml> --path cuda --batch 512
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--path", default="cuda")
    ap.add_argument("--batch", type=int, default=512)
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

    print(f"GPU {torch.cuda.get_device_name(0)}")
    print(f"config {a.config}\npath={a.path}  batch={a.batch}\n")

    # ---------------- [1] arithmetic --------------------------------------- #
    print("=" * 78)
    print("[1] ARITHMETIC -- what the transformed weights must cost")
    print("=" * 78)
    shapes = collections.Counter()
    total_w = 0
    n_layers = 0
    for m in core.modules():
        pw = getattr(m, "pre_w", None)
        if pw is None or not hasattr(m, "connection_indices"):
            continue
        n_layers += 1
        shapes[tuple(pw.shape)] += 1
        total_w += pw.numel() * 4          # fp32 transform output
    for shp, n in sorted(shapes.items(), key=lambda kv: -kv[0][0] * kv[0][1]):
        mb = shp[0] * shp[1] * 4 / 2**20
        print(f"  {n:>3} layers of {str(shp):>18}  {mb:>8.1f} MB each"
              f"  {n * mb:>9.1f} MB total")
    print(f"\n  {n_layers} indexed layers, transformed weight total: "
          f"{total_w / 2**20:,.1f} MB")

    # ---------------- [2] + [3] profiler ----------------------------------- #
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

    torch.cuda.memory._record_memory_history(max_entries=400_000)
    core.zero_grad(set_to_none=True)
    out = core(x)
    out = out[0] if isinstance(out, (tuple, list)) else out
    loss = out.float().pow(2).mean()
    # Snapshot at the TOP of the step -- after forward, before backward frees
    # anything. That is when the retained set is largest and when C1's tensor is
    # still held for backward.
    snap = torch.cuda.memory._snapshot()
    loss.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    torch.cuda.memory._record_memory_history(enabled=None)

    by_site = collections.Counter()
    for seg in snap.get("segments", []):
        for blk in seg.get("blocks", []):
            if blk.get("state") != "active_allocated":
                continue
            frames = blk.get("frames") or []
            if not frames:
                continue
            f = next((fr for fr in frames
                      if "dendritic_modeling" in (fr.get("filename") or "")), frames[0])
            key = (f"{os.path.basename(f.get('filename', '?'))}:{f.get('line', '?')}"
                   f"  {f.get('name', '?')}")
            by_site[key] += blk.get("size", 0)

    live = sum(by_site.values()) or 1
    print("\n" + "=" * 78)
    print("[2]+[3] PROFILER -- live allocation at the top of the step")
    print("=" * 78)
    print(f"{'MB':>10}{'share':>8}  site")
    for k, sz in by_site.most_common(14):
        print(f"{sz / 2**20:>10,.1f}{100 * sz / live:>7.1f}%  {k[:58]}")

    wt = sum(v for k, v in by_site.items() if "weight_transforms" in k)
    print(f"\n  live total {live / 2**20:,.1f} MB   step peak {peak / 2**20:,.1f} MB")
    print(f"  weight_transforms attributed: {wt / 2**20:,.1f} MB "
          f"({100 * wt / live:.1f}% of live, {100 * wt / peak:.1f}% of peak)")
    print(f"  arithmetic said: {total_w / 2**20:,.1f} MB  -> "
          f"{'AGREE' if abs(wt - total_w) < 0.25 * max(total_w, 1) else 'DISAGREE (do not trust either)'}")
    print("\n  C1 is only worth doing if it is large as a share of PEAK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
