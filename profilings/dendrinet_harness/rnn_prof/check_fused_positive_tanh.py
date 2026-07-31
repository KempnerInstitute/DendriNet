"""Equivalence for the fused positive-tanh (retains only y).

Compares the three spellings through the real factory, on value and every
gradient, at several input spreads -- including ones that saturate the sigmoid
hard, which is where the logit reconstruction inside the fused backward is most
likely to break.

  eager    : (tanh(m(x-b)) + 1) / 2
  identity : sigmoid(2m(x-b))            -- retains x-b and y
  fused    : _PositiveTanhCentered        -- retains y only

fp64 is the honest test of the algebra: if fused were a different function the
gap would survive the precision increase.

    python check_fused_positive_tanh.py
"""

from __future__ import annotations

import sys

import torch

from dendritic_modeling.networks.activations.factory import ActivationFactory
from dendritic_modeling.networks.activations.parametric import (
    set_fused_positive_tanh,
    set_positive_tanh_sigmoid,
)

NAMES = ["param_tanh", "param_tanh_only_m"]
SPREADS = [0.5, 2.0, 6.0, 20.0]
TOL = 1e-5


def run(name, dtype, spread, identity, fused, dev):
    set_positive_tanh_sigmoid(identity)
    set_fused_positive_tanh(fused)
    torch.manual_seed(7)
    act = ActivationFactory.create(name, output_dim=512).to(dev).to(dtype)
    torch.manual_seed(11)
    x = (torch.randn(1024, 512, device=dev, dtype=dtype) * spread + 1.0
         ).requires_grad_(True)
    out = act(x)
    out.float().pow(2).sum().backward()
    return (out.detach().float(), x.grad.float(),
            [p.grad.float() for p in act.parameters() if p.grad is not None])


def rel(u, v):
    return ((u - v).abs().max() / (u.abs().max() or 1.0)).item()


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}\n")
    ok = True
    for dtype in (torch.float32, torch.float64):
        print(f"{'activation':<20}{'dtype':>9}{'spread':>8}"
              f"{'sat%':>7}{'d out':>11}{'d grad_x':>11}{'d grad_p':>11}")
        for name in NAMES:
            for spread in SPREADS:
                base = run(name, dtype, spread, True, False, dev)   # identity
                fus = run(name, dtype, spread, True, True, dev)     # fused
                y = base[0]
                sat = 100 * ((y >= 1.0) | (y <= 0.0)).float().mean().item()
                d_out = rel(base[0], fus[0])
                d_gx = rel(base[1], fus[1])
                d_gp = max([rel(u, v) for u, v in zip(base[2], fus[2])] or [0.0])
                good = max(d_out, d_gx, d_gp) < TOL
                ok &= good
                print(f"{name:<20}{str(dtype).replace('torch.', ''):>9}"
                      f"{spread:>8.1f}{sat:>7.1f}{d_out:>11.2e}{d_gx:>11.2e}"
                      f"{d_gp:>11.2e}{'' if good else '  FAIL'}")
        print()

    # The fused path must also agree with the ORIGINAL eager tanh, not just with
    # the identity -- otherwise two changes could cancel and hide a real drift.
    print("fused vs the original eager tanh spelling (fp64):")
    for name in NAMES:
        e = run(name, torch.float64, 2.0, False, False, dev)
        f = run(name, torch.float64, 2.0, True, True, dev)
        d = max(rel(e[0], f[0]), rel(e[1], f[1]),
                max([rel(u, v) for u, v in zip(e[2], f[2])] or [0.0]))
        ok &= d < TOL
        print(f"  {name:<22} max rel {d:.2e}{'' if d < TOL else '  FAIL'}")

    set_positive_tanh_sigmoid(True)
    set_fused_positive_tanh(True)
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
