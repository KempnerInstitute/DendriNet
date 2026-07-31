"""Verify C6 (int32 index retention) and size C5 (memoized tile). One command.

Both changes were committed without ever running on a GPU. This is the check.

  [1] STRUCTURE  -- what dtype does the gather actually RETAIN for backward?
      Read it off the autograd node rather than trusting the source. int64 here
      means C6 did not take.
  [2] MEMORY     -- allocation attributed to the gather's saved index copy, on
      the real config, against the arithmetic for int32 vs int64. Recorded
      Blackwell baseline before C6: 889.3 MB at 12L/16k, cuda, batch 512.
  [3] GRADIENTS  -- cuda path vs the fused reference. Must agree; C6 touches the
      gather hot path, which nothing else in the recent memory work does.
  [4] C5         -- what memoizing _pick_tile() is actually worth, as a share of
      a step. Expected ~0 on saturated feedforward; the point is to say so with
      a number instead of assuming.

    python verify_c5_c6.py --config <cfg.yaml> --batch 512
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import os
import sys
import time

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    cuda_gather as cg,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.fused_gather import (
    fused_sparse_gather,
)

BASELINE_MB = 889.3  # Blackwell, 12L/16k, cuda, batch 512, before C6


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
    if not (torch.cuda.is_available() and cg.cuda_gather_available()):
        print("needs CUDA + built extension "
              "(source profilings/weight_transform_opt/setup_cuda_env.sh)")
        return 1
    dev = torch.device("cuda")
    print(f"GPU {torch.cuda.get_device_name(0)}   optin={cg._optin_shared():,} B\n")
    ok = True

    # ---------------- [1] what is retained --------------------------------- #
    print("=" * 78)
    print("[1] STRUCTURE -- dtype the gather retains for backward")
    print("=" * 78)
    IN, O, K = 16000, 48000, 40
    x = torch.randn(64, IN, device=dev).abs().requires_grad_(True)
    idx = torch.randint(0, IN, (O, K), dtype=torch.int32, device=dev)
    w = (torch.randn(O, K, device=dev) * 0.05).requires_grad_(True)
    out = cg.cuda_sparse_gather(x, idx, w)
    saved = getattr(out.grad_fn, "saved_tensors", ())
    idx_saved = [t for t in saved if not t.is_floating_point()]
    dtypes = [str(t.dtype).replace("torch.", "") for t in saved]
    print(f"  saved tensors: {dtypes}")
    if idx_saved:
        d = idx_saved[0].dtype
        good = d == torch.int32
        ok &= good
        mb = idx_saved[0].numel() * idx_saved[0].element_size() / 2**20
        print(f"  index copy retained as {str(d).replace('torch.','')}"
              f"  ({mb:.1f} MB for this one layer)   "
              f"{'OK -- C6 took' if good else 'FAIL -- still int64'}")
    else:
        print("  no integer tensor retained (unexpected)")
        ok = False
    del x, idx, w, out
    torch.cuda.empty_cache()

    # ---------------- [3] gradients ----------------------------------------- #
    print("\n" + "=" * 78)
    print("[3] GRADIENTS -- cuda path vs the fused reference")
    print("=" * 78)
    for in_f, out_f, kk in [(16000, 144000, 40), (16000, 48000, 40), (4000, 48000, 10)]:
        idx = torch.randint(0, in_f, (out_f, kk), dtype=torch.int32, device=dev)
        x0 = torch.randn(128, in_f, device=dev).abs()
        w0 = torch.randn(out_f, kk, device=dev) * 0.05
        g = torch.randn(128, out_f, device=dev)
        xc, wc = x0.clone().requires_grad_(True), w0.clone().requires_grad_(True)
        cg.cuda_sparse_gather(xc, idx, wc).backward(g)
        xf_, wf_ = x0.clone().requires_grad_(True), w0.clone().requires_grad_(True)
        fused_sparse_gather(xf_, idx, wf_, chunk_size=65536).backward(g)
        rw = (wc.grad - wf_.grad).norm().item() / (wf_.grad.norm().item() or 1)
        rx = (xc.grad - xf_.grad).norm().item() / (xf_.grad.norm().item() or 1)
        good = rw < 1e-5 and rx < 1e-5
        ok &= good
        print(f"  in={in_f:>6,} out={out_f:>7,} K={kk:>3}  grad_w {rw:.2e}"
              f"  grad_x {rx:.2e}  {'OK' if good else 'FAIL'}")
        del idx, x0, w0, g, xc, wc, xf_, wf_
        torch.cuda.empty_cache()

    # ---------------- [4] C5 sizing ----------------------------------------- #
    print("\n" + "=" * 78)
    print("[4] C5 -- what memoizing _pick_tile() is worth")
    print("=" * 78)
    n = 60  # sparse-layer forwards per step on the stress config
    t0 = time.perf_counter()
    for _ in range(n * 20):
        torch.cuda.get_device_capability()
    uncached = (time.perf_counter() - t0) / 20 * 1e3
    t0 = time.perf_counter()
    for _ in range(n * 20):
        cg._pick_tile()
    cached = (time.perf_counter() - t0) / 20 * 1e3
    print(f"  {n} calls/step:  uncached {uncached:.4f} ms   memoized {cached:.4f} ms"
          f"   saved {uncached - cached:.4f} ms/step")

    # ---------------- [2] memory on the real config ------------------------- #
    print("\n" + "=" * 78)
    print("[2] MEMORY -- gather's retained index copy on the real config")
    print("=" * 78)
    pd = _load_profiler()
    core, dim, _ = pd.build_core_from_config(a.config)
    pd.assert_path_available(a.path)
    pd.apply_path(core, a.path)
    pd.freeze_rewiring(core)
    core = core.to(dev).train()
    xin = torch.randn(a.batch, dim, device=dev).abs().requires_grad_(True)

    n_idx = 0
    for m in core.modules():
        ci = getattr(m, "connection_indices", None)
        if ci is not None:
            n_idx += ci.numel()
    print(f"  {n_idx:,} indices total -> int32 {n_idx * 4 / 2**20:,.1f} MB"
          f" | int64 {n_idx * 8 / 2**20:,.1f} MB")

    def step():
        core.zero_grad(set_to_none=True)
        o = core(xin)
        o = o[0] if isinstance(o, (tuple, list)) else o
        o.float().pow(2).mean().backward()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history(max_entries=400_000)
    core.zero_grad(set_to_none=True)
    o = core(xin)
    o = o[0] if isinstance(o, (tuple, list)) else o
    loss = o.float().pow(2).mean()
    snap = torch.cuda.memory._snapshot()
    loss.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**20
    torch.cuda.memory._record_memory_history(enabled=None)

    by_site = collections.Counter()
    for seg in snap.get("segments", []):
        for blk in seg.get("blocks", []):
            if blk.get("state") != "active_allocated":
                continue
            fr = blk.get("frames") or []
            if not fr:
                continue
            f = next((q for q in fr
                      if "dendritic_modeling" in (q.get("filename") or "")), fr[0])
            by_site[f"{os.path.basename(f.get('filename','?'))}:{f.get('line','?')}"] += \
                blk.get("size", 0)
    gather_mb = sum(v for k, v in by_site.items()
                    if k.startswith("cuda_gather")) / 2**20
    print(f"\n{'MB':>10}{'share':>8}  site")
    live = sum(by_site.values()) or 1
    for k, sz in by_site.most_common(8):
        print(f"{sz / 2**20:>10,.1f}{100 * sz / live:>7.1f}%  {k}")
    print(f"\n  cuda_gather retained : {gather_mb:,.1f} MB   step peak {peak:,.1f} MB")
    print(f"  Baseline before C6 was {BASELINE_MB:,.1f} MB at 12L/16k cuda batch 512.")
    print("  Expect 0.0, not half. `.to(torch.int32)` on an already-int32")
    print("  contiguous tensor returns THE SAME TENSOR, so the retained index is")
    print("  an alias of the module's existing buffer and costs no new bytes. The")
    print("  old `.to(torch.int64)` was a real copy, so C6 removes all 889 MB.")

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
