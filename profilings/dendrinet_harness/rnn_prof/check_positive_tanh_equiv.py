"""Equivalence check for the positive-tanh sigmoid identity, via the real factory.

Covers the activations the fast path touches, constructed the way configs
construct them (ActivationFactory.create), and compares value + every gradient.

param_linear_tanh is included as a CONTROL: it is not on the fast path, so it
must be bit-identical in every dtype. It was on the fast path briefly, and this
check is why it is not -- its (1 - exp(-3x)) factor reaches ~1e5 and amplified
the identity's 1e-7 rounding to 4e-4 on parameter gradients.

Comparison is RELATIVE, deliberately. That same factor pushes its gradients to
~1e12, where an absolute tolerance is meaningless: a 1e-13 relative deviation
shows up as an absolute one of 0.1, making a correct rewrite look broken.

fp64 is the real test of the rewrite. If the two spellings were different
functions the gap would survive the precision increase; if it is rounding, it
goes to exactly zero.
"""

from __future__ import annotations

import sys

import torch

from dendritic_modeling.networks.activations.factory import ActivationFactory
from dendritic_modeling.networks.activations.parametric import (
    set_positive_tanh_sigmoid,
)

NAMES = ["param_tanh", "param_tanh_only_m", "param_linear_tanh"]
TOL = 1e-5  # relative


def run(name, dtype):
    res = {}
    for enabled in (False, True):
        set_positive_tanh_sigmoid(enabled)
        torch.manual_seed(7)
        act = ActivationFactory.create(name, output_dim=512).to(dtype)
        torch.manual_seed(11)
        x = torch.randn(64, 512, dtype=dtype, requires_grad=True)
        out = act(x)
        out.float().pow(2).sum().backward()
        res[enabled] = (
            out.detach().float(),
            x.grad.float(),
            [p.grad.float() for p in act.parameters() if p.grad is not None],
        )
    a, b = res[False], res[True]

    def rel(u, v):
        return ((u - v).abs().max() / (u.abs().max() or 1.0)).item()

    return (rel(a[0], b[0]), rel(a[1], b[1]),
            max([rel(u, v) for u, v in zip(a[2], b[2])] or [0.0]))


def main():
    print(f"{'activation':<24}{'dtype':>10}{'rel out':>12}{'rel grad_x':>12}"
          f"{'rel grad_p':>12}{'':>8}")
    ok = True
    for dtype in (torch.float32, torch.float64):
        for name in NAMES:
            d_out, d_gx, d_gp = run(name, dtype)
            # fp64 is the honest test of the rewrite itself: if the two forms
            # were structurally different, the gap would NOT shrink with
            # precision. If it does shrink, the gap is rounding.
            good = max(d_out, d_gx, d_gp) < TOL
            ok &= good
            print(f"{name:<24}{str(dtype).replace('torch.', ''):>10}"
                  f"{d_out:>12.2e}{d_gx:>12.2e}{d_gp:>12.2e}"
                  f"{'  OK' if good else '  FAIL':>8}")
    set_positive_tanh_sigmoid(False)
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
