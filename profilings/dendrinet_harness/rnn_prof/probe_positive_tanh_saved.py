"""What does the reactivation actually RETAIN, and can we retain less?

`parametric.py:57` is 57.6% of live allocation (14,943 MB, cuda batch 512). The
sigmoid identity removed transient temporaries and bought 1.19x. The tensors held
across backward are untouched, and those are the bulk.

`sigmoid((2m)*(x-b))` builds three full-size tensors:
    t1 = x - b        <- retained: MulBackward needs it for grad w.r.t. m
    t2 = (2m) * t1    <- transient, freed once sigmoid runs
    y  = sigmoid(t2)  <- retained: SigmoidBackward needs the output
So **two** full-size tensors per call survive into backward.

A fused autograd Function could hold fewer, because of an asymmetry in what each
gradient needs (dy/dz = y(1-y), z = 2m(x-b)):

    grad_x = g * y(1-y) * 2m              <- needs ONLY y
    grad_b = -2m * sum(g * y(1-y))        <- needs ONLY y
    grad_m = (1/m) * sum(g * y(1-y) * z)  <- needs z

Only grad_m needs anything besides y. And z is recoverable from y alone:
z = logit(y). If that reconstruction is sound we retain ONE tensor instead of
two -- a 2x cut on the largest line in the model.

It is not obviously sound. Once |z| is large, y saturates to exactly 1.0 (or 0.0)
in fp32 and logit(y) becomes +-inf; multiplied by y(1-y) = 0 that is 0*inf = nan.
So the question is not "is the algebra right" (it is) but "how often does the
float hit the wall, and does it poison grad_m".

[1] and [2] are pure numerics and run on CPU. [3] needs a GPU.

    python probe_positive_tanh_saved.py
"""

from __future__ import annotations

import argparse
import math
import sys

import torch


def true_grads(x, log_m, b, g):
    """Reference: plain autograd through the shipped expression."""
    x = x.clone().requires_grad_(True)
    lm = log_m.clone().requires_grad_(True)
    bb = b.clone().requires_grad_(True)
    m = lm.exp()
    y = torch.sigmoid((2.0 * m) * (x - bb))
    y.backward(g)
    return x.grad, lm.grad, bb.grad, y.detach()


def from_y_only(y, log_m, b, g, clamp=False):
    """Reconstruct every gradient from y alone, via z = logit(y).

    ``clamp`` bounds ONLY the logit argument, never the ``y`` used for
    ``y*(1-y)``. That matters: where the sigmoid has saturated, ``y*(1-y)``
    evaluates to exactly 0 in floating point -- and PyTorch's own sigmoid
    backward computes the same ``g*y*(1-y)``, so the reference drops those terms
    too. Their true contribution is therefore 0, and multiplying a *finite*
    stand-in z by that 0 reproduces it exactly. Leaving z as inf instead turns
    the same term into 0*inf = nan.
    """
    m = log_m.exp()
    s = y * (1.0 - y)                       # dy/dz -- exactly 0 at saturation
    gs = g * s
    grad_x = gs * (2.0 * m)
    grad_b = -(gs * (2.0 * m)).sum(dim=0)
    if clamp:
        eps = torch.finfo(y.dtype).eps
        z = torch.logit(y.clamp(eps, 1.0 - eps))
    else:
        z = torch.logit(y)
    grad_m_raw = (gs * z).sum(dim=0) / m    # d/dm, then chain to log_m
    grad_log_m = grad_m_raw * m
    return grad_x, grad_log_m, grad_b, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--dim", type=int, default=1024)
    a = ap.parse_args()
    torch.manual_seed(0)

    print("=" * 78)
    print("[1] SATURATION -- where does fp32 sigmoid lose z?")
    print("=" * 78)
    zs = torch.tensor([4.0, 8.0, 12.0, 16.0, 17.0, 18.0, 20.0, 30.0])
    print(f"{'z':>8}{'sigmoid(z)':>16}{'logit(sigmoid(z))':>20}{'y*(1-y)':>12}")
    for z in zs:
        y = torch.sigmoid(z)
        print(f"{z.item():>8.1f}{y.item():>16.10f}{torch.logit(y).item():>20.4f}"
              f"{(y * (1 - y)).item():>12.3e}")
    print("\n  sigmoid saturates to exactly 1.0 in fp32 near z=17; logit then")
    print("  returns inf and inf*0 = nan. The question is how often z gets there.")

    print("\n" + "=" * 78)
    print("[2] AGREEMENT -- reconstructed vs true gradients, by input spread")
    print("=" * 78)
    print(f"{'x spread':>10}{'|z|max':>9}{'sat frac':>10}"
          f"{'rel grad_x':>13}{'rel grad_log_m':>16}{'rel grad_b':>13}{'nan':>6}")
    log_m = torch.full((a.dim,), math.log(1.5))
    b = torch.full((a.dim,), 1.0)
    for spread in (0.5, 1.0, 2.0, 3.0, 6.0, 12.0):
        x = torch.randn(a.n, a.dim) * spread + 1.0
        g = torch.randn(a.n, a.dim)
        gx0, gm0, gb0, y = true_grads(x, log_m, b, g)
        gx1, gm1, gb1, z = from_y_only(y, log_m, b, g)
        sat = ((y >= 1.0) | (y <= 0.0)).float().mean().item()
        has_nan = bool(torch.isnan(gm1).any() or torch.isinf(gm1).any())

        def rel(u, v):
            d = (u - v).abs().max().item()
            return d / (u.abs().max().item() or 1.0)

        zmax = z[torch.isfinite(z)].abs().max().item() if torch.isfinite(z).any() else float("inf")
        print(f"{spread:>10.1f}{zmax:>9.1f}{100*sat:>9.2f}%"
              f"{rel(gx0, gx1):>13.2e}{rel(gm0, gm1):>16.2e}{rel(gb0, gb1):>13.2e}"
              f"{'YES' if has_nan else 'no':>6}")

    print("\n  grad_x and grad_b need only y, so they should be clean throughout.")
    print("  grad_log_m is the one that rides on logit(y).")

    print("\n" + "=" * 78)
    print("[2b] CLAMPED logit argument -- does it recover grad_log_m exactly?")
    print("=" * 78)
    print(f"{'x spread':>10}{'sat frac':>10}{'rel grad_log_m':>16}{'nan':>6}")
    ok = True
    for spread in (0.5, 1.0, 2.0, 3.0, 6.0, 12.0, 30.0):
        x = torch.randn(a.n, a.dim) * spread + 1.0
        g = torch.randn(a.n, a.dim)
        _, gm0, _, y = true_grads(x, log_m, b, g)
        _, gm1, _, _ = from_y_only(y, log_m, b, g, clamp=True)
        sat = ((y >= 1.0) | (y <= 0.0)).float().mean().item()
        bad = bool(torch.isnan(gm1).any() or torch.isinf(gm1).any())
        r = (gm0 - gm1).abs().max().item() / (gm0.abs().max().item() or 1.0)
        ok &= (not bad) and r < 1e-5
        print(f"{spread:>10.1f}{100*sat:>9.2f}%{r:>16.2e}{'YES' if bad else 'no':>6}")

    print("\n" + ("  VERDICT: y alone suffices -- one retained tensor instead of two."
                  if ok else
                  "  VERDICT: clamping does NOT recover it; keep a second tensor."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
