"""Validate the recurrent weight-transform hoist: numerics + memory scaling.

`IndexedSparseLinear` recomputes `apply_weight_transform(pre_w)` on every
timestep of the recurrent unroll, and autograd saves each copy for backward. That
costs `T * out_features * K` elements of activation -- at T=58 the single largest
memory term in the model, and one that does NOT shard under FSDP (every rank runs
the full forward). `cache_recurrent_weight` hoists it out of the loop.

Checks, in order:
  1. the flag reaches the layers built from a real config;
  2. forward is bit-identical with the cache on vs off;
  3. gradients agree to fp tolerance (mathematically identical, but the shared
     node accumulates T gradients in a different order, so not bit-identical);
  4. peak memory scales with T when off, and is ~flat when on.
"""
import argparse
import gc
import sys

import torch

from dendritic_modeling.config import load_config
from dendritic_modeling.config.conversion import to_plain_dict
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)


def build(config_path, cache_on):
    cfg = load_config(config_path)
    core_cfg = to_plain_dict(cfg.model.core)
    for layer in core_cfg.get("unified_ei", {}).get("layers", []) or []:
        for pop in ("excitatory", "inhibitory"):
            if isinstance(layer.get(pop), dict):
                layer[pop]["indexed_cache_recurrent_weight"] = cache_on
    enc = to_plain_dict(cfg.model.encoder) or {}
    in_dim = (enc.get("params") or {}).get("input_dim")
    core = get_architecture(
        (core_cfg.get("type") or "").lower(), core_cfg,
        input_dim=int(in_dim), suffix_input_dim=None,
    )
    layers = [m for m in core.modules() if isinstance(m, IndexedSparseLinear)]
    got = {bool(m.cache_recurrent_weight) for m in layers}
    return core, layers, got


def run(core, T, in_dim, device, dtype, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(1, T, in_dim, device=device, dtype=dtype).abs().requires_grad_(True)
    torch.cuda.reset_peak_memory_stats()
    out = core(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    loss = out.float().pow(2).mean()
    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 2**20
    grads = {n: p.grad.detach().float().clone()
             for n, p in core.named_parameters() if p.grad is not None}
    return out.detach().float().clone(), loss.item(), grads, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seq-lens", nargs="+", type=int, default=[8, 16, 29, 58])
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = ap.parse_args()
    dev = torch.device("cuda")
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    print(f"GPU: {torch.cuda.get_device_name(0)}   config={args.config}  "
          f"dtype={args.dtype}")

    # --- 1. flag propagation -------------------------------------------------
    core_on, layers_on, got_on = build(args.config, True)
    core_off, layers_off, got_off = build(args.config, False)
    print(f"\n[1] flag propagation: {len(layers_on)} IndexedSparseLinear layers; "
          f"cache_on={got_on} cache_off={got_off}")
    if got_on != {True} or got_off != {False}:
        print("    FAIL: flag did not reach every layer")
        return 1
    print("    OK")

    enc_dim = core_on.input_projection[0].in_features if hasattr(
        core_on, "input_projection") else None
    cfg = load_config(args.config)
    in_dim = int((to_plain_dict(cfg.model.encoder).get("params") or {})["input_dim"])

    core_on = core_on.to(dev, dtype).train()
    core_off = core_off.to(dev, dtype).train()
    core_on.load_state_dict(core_off.state_dict())  # identical weights

    # --- 2/3. numerics at full T --------------------------------------------
    T = args.seq_lens[-1]
    o_off, l_off, g_off, p_off = run(core_off, T, in_dim, dev, dtype)
    o_on, l_on, g_on, p_on = run(core_on, T, in_dim, dev, dtype)
    fwd_max = (o_off - o_on).abs().max().item()
    print(f"\n[2] forward  (T={T}): max|off-on| = {fwd_max:.3e}   "
          f"loss off={l_off:.6f} on={l_on:.6f}")
    print("    OK (bit-identical)" if fwd_max == 0 else "    NOTE: not bit-identical")

    worst, worst_n = 0.0, ""
    for n in g_off:
        a, b = g_off[n], g_on[n]
        d = (a - b).norm().item() / (b.norm().item() or 1.0)
        if d > worst:
            worst, worst_n = d, n
    print(f"[3] gradients (T={T}): worst relative L2 diff = {worst:.3e}  ({worst_n})")
    tol = 5e-2 if dtype is torch.bfloat16 else 1e-4
    print(f"    {'OK' if worst < tol else 'FAIL'} (tol {tol:.0e} for {args.dtype})")

    # --- 4. memory vs T ------------------------------------------------------
    print(f"\n[4] peak memory (MB) vs sequence length")
    print(f"{'T':>5} {'cache OFF':>11} {'cache ON':>10} {'saved':>9} {'ratio':>7}")
    rows = []
    for T in args.seq_lens:
        _, _, _, poff = run(core_off, T, in_dim, dev, dtype)
        gc.collect(); torch.cuda.empty_cache()
        _, _, _, pon = run(core_on, T, in_dim, dev, dtype)
        gc.collect(); torch.cuda.empty_cache()
        rows.append((T, poff, pon))
        print(f"{T:>5} {poff:>11.0f} {pon:>10.0f} {poff - pon:>9.0f} "
              f"{poff / pon:>6.2f}x")
    if len(rows) >= 2:
        (t0, a0, b0), (t1, a1, b1) = rows[0], rows[-1]
        print(f"\n    OFF grows {(a1 - a0) / (t1 - t0):7.2f} MB per timestep")
        print(f"    ON  grows {(b1 - b0) / (t1 - t0):7.2f} MB per timestep")
    return 0


if __name__ == "__main__":
    sys.exit(main())
