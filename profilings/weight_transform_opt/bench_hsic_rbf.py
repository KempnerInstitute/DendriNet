"""Isolated microbenchmark for the RBF HSIC auxiliary gradient (S8 candidate).

Answers, for ONE op in a controlled environment:
  (1) is there room?  -> the current path builds an (n, n, d) intermediate; how much
                         latency / peak memory does removing it save?
  (2) what metrics    -> latency (self + target grads) and peak memory, current vs optimized
  (3) which matters   -> current is MEMORY-bound (materializes z_diff); optimized routes the
                         work through W @ z (a cuBLAS matmul) -> compute-bound
  (4) an optimization -> distribute the pairwise sum:
                            sum_j W_ij (z_i - z_j) = z_i * rowsum(W)_i - (W z)_i
                         so grad = scale * (z * W.sum(1,keepdim) - W @ z), no (n,n,d) tensor.

Unlike the gather kernel, compute_hsic_gradient returns the gradient ANALYTICALLY (it is not
an autograd.Function), so there is nothing to .backward() -- we time the gradient call itself.

Baseline "Kernel A" is the REAL production function, imported, so the before-number is exactly
what training runs today. "Kernel B" is the matmul-form rewrite, checked for numerical identity.

Usage:
  python bench_hsic_rbf.py                      # default sweep of (n, d)
  python bench_hsic_rbf.py --n 256 --d 512      # single shape
  python bench_hsic_rbf.py --device cpu         # force CPU (login node)
"""
import argparse
import statistics
import time
from types import SimpleNamespace

import torch

from dendritic_modeling.training.strategies.local_learning_parts.local_learning_hsic import (
    compute_hsic_gradient,
)

# ------------------------------------------------------------------ config obj
# compute_hsic_gradient reads local_cfg.hsic.{kernel,sigma,weight,degree,coef0}.
# weight=1.0 and per-call weight=1.0 keep the scaling neutral so A and B match cleanly.
def make_cfg(sigma: float) -> SimpleNamespace:
    return SimpleNamespace(
        hsic=SimpleNamespace(
            kernel="rbf", sigma=sigma, weight=1.0, degree=2, coef0=1.0
        )
    )


# ------------------------------------------------- Kernel A: current impl (real)
# imported production function; RBF branch builds z_diff = (n, n, d) explicitly.
def kernel_A(cfg, z, y, grad_type):
    return compute_hsic_gradient(cfg, z, y, weight=1.0, grad_type=grad_type)


# ----------------------------------- Kernel B: distribute the sum (no (n,n,d))
# identical math; the pairwise-difference sum collapses to a row-sum and a matmul.
def _kernel_matrix_rbf(X, Y, gamma):
    X_norm = (X**2).sum(dim=1, keepdim=True)
    Y_norm = (Y**2).sum(dim=1, keepdim=True)
    dist_sq = X_norm + Y_norm.t() - 2.0 * (X @ Y.t())
    return torch.exp(-gamma * dist_sq)


def kernel_B(cfg, z, y, grad_type):
    n = z.size(0)
    sigma = cfg.hsic.sigma
    gamma = 1.0 / (2.0 * sigma**2) if sigma > 0 else 1.0
    K_zz = _kernel_matrix_rbf(z, z, gamma)
    if grad_type == "self":
        W = K_zz
        sign = -2.0 * gamma
    else:  # target
        L_yy = _kernel_matrix_rbf(y, y, gamma)
        Lc = (
            L_yy
            - L_yy.mean(dim=0, keepdim=True)
            - L_yy.mean(dim=1, keepdim=True)
            + L_yy.mean()
        )
        W = K_zz * Lc
        sign = 2.0 * gamma
    row = W.sum(dim=1, keepdim=True)          # (n, 1)
    grad_z = sign * (z * row - W @ z)         # (n, d) -- no (n, n, d) tensor
    return grad_z * cfg.hsic.weight * 1.0 / float(n)


# ------------------------------------------------------------------ inputs
def make_inputs(n, d, dy, dev, dtype):
    torch.manual_seed(0)
    z = torch.randn(n, d, device=dev, dtype=dtype)
    y = torch.randn(n, dy, device=dev, dtype=dtype)
    return z, y


# ------------------------------------------------------------ timing helper
def bench(fn, cfg, z, y, grad_type, iters, warmup, dev):
    cuda = dev == "cuda"
    for _ in range(warmup):
        fn(cfg, z, y, grad_type)
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(iters):
        if cuda:
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            fn(cfg, z, y, grad_type)
            e1.record()
            torch.cuda.synchronize()
            times.append(e0.elapsed_time(e1))
        else:
            t0 = time.perf_counter()
            fn(cfg, z, y, grad_type)
            times.append((time.perf_counter() - t0) * 1e3)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if cuda else float("nan")
    return statistics.median(times), peak_gb


# ------------------------------------------------------------ correctness
def check_equiv(cfg, z, y, grad_type):
    # fp64 for an exact check that the rewrite is algebra, not approximation.
    zd, yd = z.double(), y.double()
    a = kernel_A(cfg, zd, yd, grad_type)
    b = kernel_B(cfg, zd, yd, grad_type)
    return torch.allclose(a, b, rtol=1e-10, atol=1e-12), (a - b).abs().max().item()


# ------------------------------------------------------------ run
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=None, help="batch size (single-shape mode)")
    p.add_argument("--d", type=int, default=None, help="layer width (single-shape mode)")
    p.add_argument("--dy", type=int, default=10, help="target dim (e.g. #classes)")
    p.add_argument("--sigma", type=float, default=1.0, help="RBF bandwidth")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = p.parse_args()

    dev = (
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda"
        else "cpu"
    )
    if dev == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("device: CPU (login node -- GPU numbers come from sbatch)")
    print(f"dtype: fp32 | dy={args.dy} | sigma={args.sigma}\n")

    if args.n and args.d:
        shapes = [(args.n, args.d)]
    else:
        shapes = [(128, 256), (256, 512), (512, 512), (256, 1024)]

    cfg = make_cfg(args.sigma)
    hdr = (
        f"{'shape (n,d)':>14}{'grad':>8}"
        f"{'A ms':>10}{'B ms':>10}{'speedup':>9}"
        f"{'A peakGB':>10}{'B peakGB':>10}{'z_diff MB':>11}{'equiv':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for n, d in shapes:
        z, y = make_inputs(n, d, args.dy, dev, torch.float32)
        zdiff_mb = n * n * d * 4 / 1e6  # the intermediate Kernel A builds, fp32
        for gt in ("self", "target"):
            ok, maxdiff = check_equiv(cfg, z, y, gt)
            a_ms, a_pk = bench(kernel_A, cfg, z, y, gt, args.iters, args.warmup, dev)
            b_ms, b_pk = bench(kernel_B, cfg, z, y, gt, args.iters, args.warmup, dev)
            speed = a_ms / b_ms if b_ms > 0 else float("inf")
            tag = "OK" if ok else f"FAIL({maxdiff:.1e})"
            print(
                f"{f'({n},{d})':>14}{gt:>8}"
                f"{a_ms:>10.3f}{b_ms:>10.3f}{speed:>8.1f}x"
                f"{a_pk:>10.3f}{b_pk:>10.3f}{zdiff_mb:>11.1f}{tag:>8}"
            )

    print("\nread me:")
    print("  * speedup = A/B latency. B routes work through W @ z (cuBLAS matmul).")
    print("  * z_diff MB is the (n,n,d) tensor Kernel A materializes and B never builds;")
    print("    A peakGB should track it, B peakGB should be far lower (the memory win).")
    print("  * equiv is an fp64 allclose (rtol 1e-10): the rewrite must be exact algebra.")
    print("  * scaling: z_diff grows LINEARLY in d and QUADRATICALLY in n -> the gap widens.")


if __name__ == "__main__":
    main()
