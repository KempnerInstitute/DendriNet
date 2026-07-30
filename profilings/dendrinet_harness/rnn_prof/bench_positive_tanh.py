"""Measure and verify the (tanh(z)+1)/2 == sigmoid(2z) fast path.

Pass D of the hot-path audit attributed 39.9% of live allocation on the 12-layer
stress config to ParametricTanh.forward. The cause is spelling, not maths: the
naive form runs four full-size elementwise kernels and allocates a temporary for
each, and two of those temporaries (the +1 and the /2) do nothing that sigmoid
does not do for free.

The identity is exact in real arithmetic but NOT bit-identical in floating
point, so this reports three things and not just a speedup:

  [1] agreement -- max abs/rel deviation of value AND of every gradient, at the
      real activation dtypes, including near z = 0 where the curve is steepest.
  [2] cost      -- forward, backward and peak memory, at real layer widths.
  [3] drift     -- 200 optimizer steps on a fixed problem down both paths, to
      show whether the 1e-7 compounds or stays put. A per-op tolerance says
      nothing about a trajectory; this does.

    python bench_positive_tanh.py [--steps 200]
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from dendritic_modeling.networks.activations.parametric import (
    ParametricTanh,
    set_positive_tanh_sigmoid,
)

# (batch, width) drawn from the shapes the stress config actually runs.
SHAPES = [(512, 144_000), (512, 48_000), (128, 144_000), (64, 16_000)]
DTYPES = [torch.float32, torch.bfloat16]


def _timeit(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def _run(act, x, backward: bool):
    out = act(x)
    if backward:
        out.pow(2).mean().backward()
    return out


# --------------------------------------------------------------------------- #
# [1] agreement                                                                #
# --------------------------------------------------------------------------- #
def check_agreement(dev):
    print(f"\n{'=' * 78}\n[1] AGREEMENT -- value and gradients, both paths\n{'=' * 78}")
    print(f"{'dtype':>10}{'regime':>14}{'d value':>12}{'d grad_x':>12}"
          f"{'d grad_m':>12}{'d grad_b':>12}")
    worst = 0.0
    for dtype in DTYPES:
        # "steep" concentrates x near the midpoint b, where the curve has its
        # largest derivative and any rewrite is most likely to show up.
        for regime, sampler in (
            ("wide", lambda: torch.randn(4096, 2048, device=dev) * 3.0),
            ("steep", lambda: torch.randn(4096, 2048, device=dev) * 0.01 + 1.0),
        ):
            x0 = sampler().to(dtype)
            grads = {}
            for enabled in (False, True):
                set_positive_tanh_sigmoid(enabled)
                act = ParametricTanh(2048).to(dev).to(dtype)
                torch.manual_seed(0)
                with torch.no_grad():
                    act.log_m.normal_(0.4, 0.1)
                    act.b.normal_(1.0, 0.1)
                x = x0.clone().requires_grad_(True)
                out = act(x)
                out.pow(2).mean().backward()
                grads[enabled] = (out.detach().float(), x.grad.float(),
                                  act.log_m.grad.float(), act.b.grad.float())
            devs = []
            for a, b in zip(grads[False], grads[True]):
                denom = a.abs().max().item() or 1.0
                devs.append((a - b).abs().max().item() / denom)
            worst = max(worst, max(devs))
            print(f"{str(dtype).replace('torch.', ''):>10}{regime:>14}"
                  + "".join(f"{d:>12.2e}" for d in devs))
    print(f"\n  worst relative deviation anywhere: {worst:.2e}")
    return worst


# --------------------------------------------------------------------------- #
# [2] cost                                                                     #
# --------------------------------------------------------------------------- #
def check_cost(dev):
    print(f"\n{'=' * 78}\n[2] COST -- fwd / fwd+bwd / peak memory\n{'=' * 78}")
    print(f"{'batch':>7}{'width':>9}{'dtype':>10}  "
          f"{'fwd off':>9}{'fwd on':>9}{'x':>6}   "
          f"{'bwd off':>9}{'bwd on':>9}{'x':>6}   "
          f"{'MB off':>9}{'MB on':>9}{'x':>6}")
    rows = []
    for batch, width in SHAPES:
        for dtype in DTYPES:
            res = {}
            for enabled in (False, True):
                set_positive_tanh_sigmoid(enabled)
                act = ParametricTanh(width).to(dev).to(dtype)
                x = torch.randn(batch, width, device=dev, dtype=dtype)
                xg = x.clone().requires_grad_(True)
                with torch.no_grad():
                    fwd = _timeit(lambda: _run(act, x, False))
                bwd = _timeit(lambda: (act.zero_grad(set_to_none=True),
                                       xg.grad is not None and xg.grad.zero_(),
                                       _run(act, xg, True)))
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                out = act(xg)
                out.pow(2).mean().backward()
                peak = (torch.cuda.max_memory_allocated() - base) / 2**20
                res[enabled] = (fwd, bwd, peak)
                del out, act, x, xg
                torch.cuda.empty_cache()
            (f0, b0, m0), (f1, b1, m1) = res[False], res[True]
            rows.append((f0 / f1, b0 / b1, m0 / max(m1, 1e-9)))
            print(f"{batch:>7}{width:>9,}{str(dtype).replace('torch.', ''):>10}  "
                  f"{f0:>9.3f}{f1:>9.3f}{f0/f1:>6.2f}   "
                  f"{b0:>9.3f}{b1:>9.3f}{b0/b1:>6.2f}   "
                  f"{m0:>9.1f}{m1:>9.1f}{m0/max(m1,1e-9):>6.2f}")
    if rows:
        n = len(rows)
        print(f"\n  mean over {n} shape/dtype combinations:  "
              f"fwd {sum(r[0] for r in rows)/n:.2f}x   "
              f"bwd {sum(r[1] for r in rows)/n:.2f}x   "
              f"peak {sum(r[2] for r in rows)/n:.2f}x")
    return rows


# --------------------------------------------------------------------------- #
# [3] drift                                                                    #
# --------------------------------------------------------------------------- #
def check_drift(dev, steps):
    print(f"\n{'=' * 78}\n[3] DRIFT -- {steps} optimizer steps down each path\n{'=' * 78}")
    traj = {}
    for enabled in (False, True):
        set_positive_tanh_sigmoid(enabled)
        torch.manual_seed(1234)
        act = ParametricTanh(256).to(dev)
        x = torch.randn(1024, 256, device=dev)
        target = torch.rand(1024, 256, device=dev)
        opt = torch.optim.Adam(act.parameters(), lr=1e-2)
        losses = []
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = (act(x) - target).pow(2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        traj[enabled] = (losses, act.log_m.detach().clone(), act.b.detach().clone())
    l0, l1 = traj[False][0], traj[True][0]
    print(f"{'step':>8}{'loss off':>16}{'loss on':>16}{'rel diff':>12}")
    for i in (0, steps // 10, steps // 4, steps // 2, steps - 1):
        rel = abs(l0[i] - l1[i]) / (abs(l0[i]) or 1.0)
        print(f"{i:>8}{l0[i]:>16.10f}{l1[i]:>16.10f}{rel:>12.2e}")
    dm = (traj[False][1] - traj[True][1]).abs().max().item()
    db = (traj[False][2] - traj[True][2]).abs().max().item()
    final = abs(l0[-1] - l1[-1]) / (abs(l0[-1]) or 1.0)
    print(f"\n  final loss rel diff {final:.2e}   "
          f"max |d log_m| {dm:.2e}   max |d b| {db:.2e}")
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("needs CUDA")
        return 1
    print(f"GPU: {torch.cuda.get_device_name(0)}   torch {torch.__version__}")

    worst = check_agreement(torch.device("cuda"))
    check_cost(torch.device("cuda"))
    final = check_drift(torch.device("cuda"), args.steps)
    set_positive_tanh_sigmoid(False)

    print(f"\n{'=' * 78}")
    print("The identity is exact; the deviation is rounding. What matters is")
    print("whether it compounds -- see [3], not [1].")
    return 0


if __name__ == "__main__":
    sys.exit(main())
