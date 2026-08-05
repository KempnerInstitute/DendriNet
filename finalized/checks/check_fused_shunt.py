"""Equivalence for the fused shunt-divide + positive-tanh.

Compares `_ShuntDividePositiveTanh` against the eager composition it replaces,
on value and on EVERY gradient (numerator, denominator, log_m, b), across
saturation regimes.

The fused backward reconstructs two things it does not store -- `z = logit(y)`
for grad_m, and `v = c + z/(2m)` for grad_d. Both are exact in real arithmetic
and both go singular exactly where the fp32 sigmoid saturates, so the interesting
rows are the ones with a high saturation fraction, not the well-behaved ones.

fp64 is the test of the algebra: a wrong derivation stays wrong at higher
precision, while a rounding difference collapses.

    python check_fused_shunt.py
"""

from __future__ import annotations

import math
import sys

import torch

from dendritic_modeling.networks.activations.parametric import (
    _ShuntDividePositiveTanh,
)

SPREADS = [0.5, 2.0, 6.0, 20.0]
TOL = 1e-5


def eager(n, d, eps, log_m, b):
    m = log_m.exp()
    v = n / (d + eps)
    return torch.sigmoid((2.0 * m) * (v - b))


def run(fn, dtype, spread, dev, dim=512, rows=1024):
    torch.manual_seed(3)
    n = (torch.randn(rows, dim, dtype=dtype, device=dev) * spread
         ).requires_grad_(True)
    # denominator is a conductance sum: starts at 1 and accumulates positives.
    d = (1.0 + torch.rand(rows, dim, dtype=dtype, device=dev) * 3.0
         ).requires_grad_(True)
    log_m = torch.full((dim,), math.log(1.5), dtype=dtype,
                       device=dev).requires_grad_(True)
    b = torch.full((dim,), 0.5, dtype=dtype, device=dev).requires_grad_(True)
    torch.manual_seed(5)
    g = torch.randn(rows, dim, dtype=dtype, device=dev)
    y = fn(n, d, 1e-6, log_m, b)
    y.backward(g)
    return y.detach(), n.grad, d.grad, log_m.grad, b.grad


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}\n")
    ok = True
    for dtype in (torch.float32, torch.float64):
        print(f"{'dtype':>9}{'spread':>8}{'sat%':>7}{'d y':>11}{'d grad_n':>11}"
              f"{'d grad_d':>11}{'d grad_m':>11}{'d grad_b':>11}")
        for spread in SPREADS:
            ref = run(eager, dtype, spread, dev)
            fus = run(lambda n, d, e, lm, bb: _ShuntDividePositiveTanh.apply(
                n, d, e, lm.exp(), bb), dtype, spread, dev)
            sat = 100 * ((ref[0] >= 1.0) | (ref[0] <= 0.0)).float().mean().item()

            def rel(u, v):
                return ((u - v).abs().max() / (u.abs().max() or 1.0)).item()

            ds = [rel(a, b_) for a, b_ in zip(ref, fus)]
            good = max(ds) < TOL and not any(
                torch.isnan(t).any() or torch.isinf(t).any() for t in fus[1:]
            )
            ok &= good
            print(f"{str(dtype).replace('torch.', ''):>9}{spread:>8.1f}{sat:>7.1f}"
                  + "".join(f"{v:>11.2e}" for v in ds)
                  + ("" if good else "  FAIL"))
        print()

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
