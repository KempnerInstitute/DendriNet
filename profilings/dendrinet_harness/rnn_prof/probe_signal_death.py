"""Where does input-dependence die? Measure it layer by layer.

The stress config converges onto EXACTLY chance (2.3027 vs ln(10)=2.30259) and
pins there -- the readout learns to ignore its input. A first hypothesis, that the
positive-tanh's gain (m/2 = 0.75 at m=1.5) contracts the signal over 12 layers,
predicted that raising m to 2.5 would fix it. It did not: epoch-1 loss was
identical to four decimals (2.5557 both), so the reactivation gain is not the
mechanism. The signal is already gone before it matters.

So measure instead of theorise. The quantity that matters is variance ACROSS THE
BATCH: if two different inputs produce the same activations, the layer has stopped
carrying information, and everything downstream is decoration.

For each layer this reports:

  batch_std   mean over units of the per-unit standard deviation across inputs.
              This is the signal. If it goes to ~0, input-dependence is dead.
  unit_mean   mean activation, to distinguish "collapsed to a constant" from
              "saturated at a rail".
  rel         batch_std / |unit_mean|, i.e. signal relative to offset -- the
              readout can only use the varying part.

Hooks sit on the reactivation modules, which are the layer outputs.

    python probe_signal_death.py --config <cfg.yaml> --batch 64
"""

from __future__ import annotations

import argparse
import importlib.util
import sys

import torch

from dendritic_modeling.networks.activations.parametric import (
    ParametricActivation,
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
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--sweep-m", type=float, nargs="*", default=None,
                    help="Override every reactivation's m and report only the "
                         "end-to-end signal ratio, once per value. Bypasses the "
                         "config entirely -- reactivation_init_m is a dead knob "
                         "under init_policy=analytical, which is how one 'test' "
                         "of it silently tested nothing.")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("needs CUDA")
        return 1

    pd = _load_profiler()
    core, dim, _ = pd.build_core_from_config(a.config)
    pd.assert_path_available(a.path)
    pd.apply_path(core, a.path)
    pd.freeze_rewiring(core)
    core = core.to("cuda").eval()

    rows = []

    def hook(name):
        def fn(_m, inp, out):
            t = out.detach().float()
            if t.dim() < 2:
                return
            # std across the BATCH for each unit, then averaged over units.
            bs = t.std(dim=0)
            rows.append((name, t.shape[-1], bs.mean().item(),
                         t.mean().item(), inp[0].detach().float().std(dim=0).mean().item()))
        return fn

    handles = [m.register_forward_hook(hook(n))
               for n, m in core.named_modules() if isinstance(m, ParametricActivation)]

    # The fused shunt path calls the activation's parameters directly and never
    # invokes the module, so forward hooks record nothing -- the probe's own
    # subject bypasses its instrument (same trap as the C4 profiler). Force the
    # plain module path for the probe; activations are identical either way.
    set_fused_shunt(False)
    torch.manual_seed(0)
    x = torch.randn(a.batch, dim, device="cuda").abs()

    if a.sweep_m:
        import math
        acts = [m for m in core.modules()
                if isinstance(m, ParametricActivation) and hasattr(m, "log_m")]
        print(f"m-SWEEP over {len(acts)} reactivation modules, "
              f"batch={a.batch}, input batch_std={x.std(dim=0).mean().item():.3e}\n")
        print(f"{'m':>7}{'gain m/2':>10}{'last-layer batch_std':>22}"
              f"{'vs input':>12}")
        for mv in a.sweep_m:
            for act in acts:
                act.log_m.data.fill_(math.log(mv))
            rows.clear()
            with torch.no_grad():
                core(x)
            last = rows[-1][2] if rows else float("nan")
            print(f"{mv:>7.1f}{mv/2:>10.2f}{last:>22.3e}"
                  f"{last / x.std(dim=0).mean().item():>12.2e}")
        set_fused_shunt(True)
        for h in handles:
            h.remove()
        print("\n  A ratio near 1 means the signal survives all 12 layers at that m.")
        return 0

    with torch.no_grad():
        core(x)
    set_fused_shunt(True)
    for h in handles:
        h.remove()
    if not rows:
        print("no activations recorded -- hooks never fired")
        return 1

    print(f"GPU {torch.cuda.get_device_name(0)}   {a.config}")
    print(f"batch={a.batch}   input batch_std = {x.std(dim=0).mean().item():.4e}\n")
    print(f"{'#':>3}{'units':>9}{'in batch_std':>15}{'out batch_std':>15}"
          f"{'out mean':>11}{'rel':>10}  module")
    prev = x.std(dim=0).mean().item()
    for i, (name, units, bstd, mean, in_bstd) in enumerate(rows):
        rel = bstd / (abs(mean) or 1e-30)
        short = name.replace("core_network.layers.", "L").replace(
            ".populations.", ".").replace(".branch_layers.", ".b").replace(
            ".reactivation", "")
        print(f"{i:>3}{units:>9,}{in_bstd:>15.3e}{bstd:>15.3e}{mean:>11.4f}"
              f"{rel:>10.2e}  {short[:34]}")
        prev = bstd

    if rows:
        first, last = rows[0][2], rows[-1][2]
        print(f"\n  first layer out batch_std {first:.3e}  ->  last {last:.3e}"
              f"   ({last/(first or 1e-30):.2e}x)")
        print("  A ratio near zero means the network's output does not depend on")
        print("  its input, which is exactly what 'converges to chance' looks like.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
