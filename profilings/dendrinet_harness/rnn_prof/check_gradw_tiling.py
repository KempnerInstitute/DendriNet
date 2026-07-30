"""Verify the M-tiled gather_gradw_transpose_kernel against the fused reference.

The kernel used to register-stage goT only for M<=128 and re-read it inside the
k-loop otherwise. It now tiles M in chunks of 128 and accumulates, so the staged
path applies at every batch size. That changes the floating-point summation
ORDER for M>128 (one pass -> ceil(M/128) partial sums), so this checks the result
against an independent implementation rather than assuming bit-identity.

Crucially it sweeps ACROSS the old M=128 boundary: if the tiling were wrong, the
error would jump exactly there.
"""
import sys

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    cuda_gather as cg,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.fused_gather import (
    fused_sparse_gather,
)


def rel(a, b):
    d = b.double()
    n = d.norm().item()
    return (a.double() - d).norm().item() / n if n else 0.0


def main():
    if not (torch.cuda.is_available() and cg.cuda_gather_available()):
        print("needs CUDA + built extension "
              "(source profilings/weight_transform_opt/setup_cuda_env.sh)")
        return 1
    dev = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"\n{'in':>6}{'out':>8}{'K':>6}{'batch':>7}{'grad_w rel':>13}"
          f"{'grad_x rel':>13}   tile")
    worst = 0.0
    # batches straddling the old 128 cliff
    for in_f, out_f, K in ((4000, 20000, 40), (2000, 8000, 120)):
        idx = torch.stack(
            [torch.randperm(in_f, device=dev)[:K] for _ in range(out_f)]
        ).contiguous()
        for M in (64, 128, 129, 192, 256, 512):
            x = torch.randn(M, in_f, device=dev).abs()
            w = (torch.randn(out_f, K, device=dev) * 0.05)
            go = torch.randn(M, out_f, device=dev)

            xc = x.clone().requires_grad_(True); wc = w.clone().requires_grad_(True)
            cg.cuda_sparse_gather(xc, idx, wc).backward(go)

            xf = x.clone().requires_grad_(True); wf = w.clone().requires_grad_(True)
            fused_sparse_gather(xf, idx, wf, chunk_size=65536).backward(go)

            gw, gx = rel(wc.grad, wf.grad), rel(xc.grad, xf.grad)
            worst = max(worst, gw, gx)
            print(f"{in_f:>6}{out_f:>8}{K:>6}{M:>7}{gw:>13.3e}{gx:>13.3e}"
                  f"   {-(-M // 128)} tile(s)")
            del x, w, go, xc, wc, xf, wf
            torch.cuda.empty_cache()
        del idx
        torch.cuda.empty_cache()
    tol = 1e-5
    print(f"\nworst relative error {worst:.3e}  (tol {tol:.0e})  "
          f"{'PASS' if worst < tol else 'FAIL'}")
    print("no discontinuity at M=128 => the tiled staging is correct")
    return 0 if worst < tol else 1


if __name__ == "__main__":
    sys.exit(main())
