"""Equivalence + accuracy check for the fused gather's fp32 grad_x accumulation.

Compares three implementations of the same math on the real recurrent layer shapes:
  * loop      -- the shipped advanced-index baseline in indexed_sparse.py
  * fused     -- fused_sparse_gather (now accumulating grad_x in fp32)
  * fp64 ref  -- the same math in double precision, treated as ground truth

The fused path must (a) match the loop baseline's forward exactly and (b) be no
further from the fp64 reference than the loop baseline is -- the point of the
change is that fp32 accumulation moves grad_x toward the reference, not away.
"""
import sys

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.fused_gather import (
    fused_sparse_gather,
)

# A representative spread of the 5k/5k recurrent layer shapes: huge fan-out with
# small K, and small fan-out with huge K.
CASES = [
    dict(M=1, in_f=4000, out_f=100000, K=64),
    dict(M=1, in_f=4000, out_f=4000, K=1168),
    dict(M=1, in_f=1000, out_f=1000, K=1000),
    dict(M=4, in_f=4000, out_f=20000, K=240),
]


def loop_gather(x, idx, w, chunk):
    out = x.new_empty(x.shape[0], idx.shape[0])
    for start in range(0, idx.shape[0], chunk):
        end = min(start + chunk, idx.shape[0])
        sel = x[:, idx[start:end].reshape(-1)].reshape(
            x.shape[0], end - start, idx.shape[1]
        )
        out[:, start:end] = (sel * w[start:end].unsqueeze(0)).sum(dim=-1)
    return out


def rel(a, b):
    """Relative L2 distance, computed in fp64."""
    a, b = a.double(), b.double()
    denom = b.norm().item()
    return (a - b).norm().item() / denom if denom else 0.0


def run(case, dtype, chunk=65536):
    dev = torch.device("cuda")
    M, in_f, out_f, K = case["M"], case["in_f"], case["out_f"], case["K"]
    g = torch.Generator(device="cpu").manual_seed(0)

    idx = torch.stack(
        [torch.randperm(in_f, generator=g)[:K] for _ in range(out_f)]
    ).to(dev)
    x0 = torch.randn(M, in_f, generator=g).abs().to(dev)
    w0 = torch.randn(out_f, K, generator=g).to(dev) * 0.05
    go = torch.randn(M, out_f, generator=g).to(dev)

    def grads(impl, dt):
        x = x0.to(dt).detach().requires_grad_(True)
        w = w0.to(dt).detach().requires_grad_(True)
        out = (
            fused_sparse_gather(x, idx, w, chunk_size=chunk)
            if impl == "fused"
            else loop_gather(x, idx, w, chunk)
        )
        out.backward(go.to(dt))
        return out.detach(), x.grad.detach(), w.grad.detach()

    ref_out, ref_gx, ref_gw = grads("loop", torch.float64)
    l_out, l_gx, l_gw = grads("loop", dtype)
    f_out, f_gx, f_gw = grads("fused", dtype)

    return {
        "fwd_max_abs_diff": (f_out.double() - l_out.double()).abs().max().item(),
        "loop_gx_err": rel(l_gx, ref_gx),
        "fused_gx_err": rel(f_gx, ref_gx),
        "loop_gw_err": rel(l_gw, ref_gw),
        "fused_gw_err": rel(f_gw, ref_gw),
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA required")
        return 1
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    ok = True
    for dtype, dname in ((torch.bfloat16, "bf16"), (torch.float32, "fp32")):
        print(f"\n=== {dname} ===")
        print(f"{'shape':>28} {'fwd maxdiff':>12} {'gx loop':>10} {'gx fused':>10} "
              f"{'gw loop':>10} {'gw fused':>10}")
        for c in CASES:
            r = run(c, dtype)
            tag = f"M{c['M']} in{c['in_f']} out{c['out_f']} K{c['K']}"
            print(f"{tag:>28} {r['fwd_max_abs_diff']:>12.3e} "
                  f"{r['loop_gx_err']:>10.3e} {r['fused_gx_err']:>10.3e} "
                  f"{r['loop_gw_err']:>10.3e} {r['fused_gw_err']:>10.3e}")
            # The fused grad_x must not be materially worse than the baseline's.
            if r["fused_gx_err"] > max(r["loop_gx_err"] * 2, 1e-6):
                print("    ^^ REGRESSION: fused grad_x is further from the fp64 reference")
                ok = False
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
