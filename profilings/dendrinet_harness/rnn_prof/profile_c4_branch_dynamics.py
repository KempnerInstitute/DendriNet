"""C4 profile: how many full-size tensors does the shunt keep, and why?

`branch_dynamics` is 59.7% of live allocation after the reactivation fix --
9,538.9 MB across three lines:

    213  shunt_numerator = shunt_numerator + excitation
    245  shunt_numerator = shunt_numerator + branch_current
    285  voltage = shunt_numerator / (denominator + owner.epsilon)

The accumulation chain looks like the culprit, but `a + b` retains NOTHING for
backward (the gradient passes through unchanged), so every intermediate in the
chain is freed the moment it is superseded. The suspect is instead the divide:
`DivBackward` needs BOTH operands, so it pins the final numerator AND the
`denominator + epsilon` tensor that line 285 allocates.

That predicts a specific number, which is the point of this script: if the model
is right, the three lines together hold exactly **2** full-size tensors, and the
reactivation holds a third. Fusing the divide with the reactivation that consumes
it (line 323, same function) could then drop 3 -> 2, because `voltage` is
recoverable from the activation output.

  [1] ARITHMETIC -- record the true [batch, units] shape at every reactivation
      call by hooking it, and sum. That is what ONE full-size tensor costs.
  [2] PROFILER   -- bytes actually attributed to each line.
  [3] RATIO      -- measured / one-tensor. Should come out near-integer. If it
      does not, the model is wrong and no fix should be designed on it.

    python profile_c4_branch_dynamics.py --config <cfg.yaml> --path cuda --batch 512
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import os
import sys

import torch

from dendritic_modeling.networks.activations.parametric import (
    ParametricActivation,
    fused_shunt_enabled,
    set_fused_shunt,
)


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
    print(f"GPU {torch.cuda.get_device_name(0)}\npath={a.path}  batch={a.batch}\n")

    # ---------------- [1] arithmetic, from real shapes ---------------------- #
    seen = []
    handles = []

    def rec(_mod, inp, _out):
        if inp and torch.is_tensor(inp[0]):
            seen.append(tuple(inp[0].shape))

    for m in core.modules():
        if isinstance(m, ParametricActivation):
            handles.append(m.register_forward_hook(rec))
    # The shape pass MUST run unfused. When the shunt fusion is active it calls
    # the activation's parameters directly and never invokes the module, so the
    # forward hooks record nothing and "one tensor" comes out as 0 MB. Force the
    # plain path here, then restore -- the shapes are identical either way.
    was_fused = fused_shunt_enabled()
    set_fused_shunt(False)
    with torch.no_grad():
        core(x)
    set_fused_shunt(was_fused)
    for h in handles:
        h.remove()
    if not seen:
        print("no reactivation calls recorded -- cannot size a tensor")
        return 1

    shapes = collections.Counter(seen)
    one_tensor = sum(
        n * torch.Size(s).numel() * 4 for s, n in shapes.items()
    )
    print("=" * 78)
    print("[1] ARITHMETIC -- what ONE full-size tensor costs, from hooked shapes")
    print("=" * 78)
    for s, n in sorted(shapes.items(), key=lambda kv: -kv[1] * torch.Size(kv[0]).numel()):
        mb = torch.Size(s).numel() * 4 / 2**20
        print(f"  {n:>3} calls of {str(tuple(s)):>22}  {mb:>8.1f} MB each"
              f"  {n * mb:>9.1f} MB")
    print(f"\n  {len(seen)} reactivation calls; ONE full-size tensor = "
          f"{one_tensor / 2**20:,.1f} MB")

    # ---------------- [2] profiler ------------------------------------------ #
    def step():
        core.zero_grad(set_to_none=True)
        out = core(x)
        out = out[0] if isinstance(out, (tuple, list)) else out
        out.float().pow(2).mean().backward()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    torch.cuda.memory._record_memory_history(max_entries=400_000)
    core.zero_grad(set_to_none=True)
    out = core(x)
    out = out[0] if isinstance(out, (tuple, list)) else out
    loss = out.float().pow(2).mean()
    snap = torch.cuda.memory._snapshot()
    loss.backward()
    torch.cuda.synchronize()
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
            by_site[f"{os.path.basename(f.get('filename','?'))}:{f.get('line','?')}"] += \
                blk.get("size", 0)

    live = sum(by_site.values()) or 1
    bd = sum(v for k, v in by_site.items() if k.startswith("branch_dynamics"))
    pa = sum(v for k, v in by_site.items() if k.startswith("parametric"))

    print("\n" + "=" * 78)
    print("[2]+[3] PROFILER, and tensors-per-site")
    print("=" * 78)
    print(f"{'MB':>10}{'share':>8}{'tensors':>10}  site")
    for k, sz in by_site.most_common(10):
        print(f"{sz / 2**20:>10,.1f}{100 * sz / live:>7.1f}%"
              f"{sz / one_tensor:>10.2f}  {k}")

    print(f"\n  branch_dynamics total : {bd / 2**20:>9,.1f} MB "
          f"= {bd / one_tensor:.2f} full-size tensors")
    print(f"  parametric total      : {pa / 2**20:>9,.1f} MB "
          f"= {pa / one_tensor:.2f} full-size tensors")
    print(f"  combined              : {(bd + pa) / 2**20:>9,.1f} MB "
          f"= {(bd + pa) / one_tensor:.2f} full-size tensors")
    print("\n  Model predicts branch_dynamics=2 (numerator, denominator+eps),")
    print("  parametric=1 (y), combined=3. Fusing the divide into the")
    print("  reactivation should reach 2 -- worth ~1 tensor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
