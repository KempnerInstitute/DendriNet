"""Verify that the backward tile P changes speed but not results.

`_pick_bwd_wpo_params` now returns a P that depends on out_features, so different
layers launch different grids. P only partitions outputs across blocks and each
(o, k) stays owned by a single warp, so gradients should be BIT-identical across
P -- unlike a change to accumulation order, which would only be identical to
within rounding. This asserts the strong property, and also checks the adaptive
picker against the fused reference end to end.
"""
import sys

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    cuda_gather as cg,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.fused_gather import (
    fused_sparse_gather,
)

SHAPES = [(16000, 144000, 40), (16000, 48000, 40), (16000, 2000, 40),
          (4000, 48000, 10), (3072, 2000, 40)]
P_VALUES = [2048, 1024, 384, 256, 96, 32]


def main():
    if not (torch.cuda.is_available() and cg.cuda_gather_available()):
        print("needs CUDA + built extension")
        return 1
    ext = cg._load_ext()
    dev = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    ok = True
    print("\n[1] grad_w bit-identical across P; grad_x within atomic tolerance")
    print(f"{'in':>6}{'out':>8}{'K':>5}  {'chosen P':>9}  {'max|dgrad_w|':>14}{'max|dgrad_x|':>14}")
    for in_f, out_f, K in SHAPES:
        idx = torch.stack([torch.randperm(in_f, device=dev)[:K]
                           for _ in range(out_f)]).to(torch.int32).contiguous()
        M = 256
        gof = torch.randn(M, out_f, device=dev).contiguous()
        goT = gof.t().contiguous()
        xT = torch.randn(in_f, M, device=dev).abs().contiguous()
        wf = (torch.randn(out_f, K, device=dev) * 0.05).contiguous()
        ref = None
        worst_w = worst_x = 0.0
        for P in P_VALUES:
            gx, gw = ext.gather_bwd_wpo_cuda(gof, goT, xT, idx, wf, P, 512)
            if ref is None:
                ref = (gx.clone(), gw.clone())
            else:
                worst_x = max(worst_x, (gx - ref[0]).abs().max().item())
                worst_w = max(worst_w, (gw - ref[1]).abs().max().item())
        chosen = cg._pick_bwd_wpo_params(out_f)[0]
        # grad_w: bit-identical.  grad_x: atomics, so tolerance only (it varies
        # this much between repeated runs at one P -- verified separately).
        bad = worst_w != 0.0 or worst_x > 1e-4
        ok &= not bad
        print(f"{in_f:>6}{out_f:>8}{K:>5}  {chosen:>9}  {worst_w:>14.1e}{worst_x:>14.1e}"
              f"{'   <-- FAIL' if bad else ''}")
        del idx, gof, goT, xT, wf, ref
        torch.cuda.empty_cache()

    print("\n[2] adaptive picker vs the fused reference, end to end")
    for in_f, out_f, K in SHAPES[:3]:
        idx = torch.stack([torch.randperm(in_f, device=dev)[:K]
                           for _ in range(out_f)]).to(torch.int32).contiguous()
        x0 = torch.randn(128, in_f, device=dev).abs()
        w0 = torch.randn(out_f, K, device=dev) * 0.05
        go = torch.randn(128, out_f, device=dev)
        xc, wc = x0.clone().requires_grad_(True), w0.clone().requires_grad_(True)
        cg.cuda_sparse_gather(xc, idx, wc).backward(go)
        xf, wf2 = x0.clone().requires_grad_(True), w0.clone().requires_grad_(True)
        fused_sparse_gather(xf, idx, wf2, chunk_size=65536).backward(go)
        rw = (wc.grad - wf2.grad).norm().item() / (wf2.grad.norm().item() or 1)
        rx = (xc.grad - xf.grad).norm().item() / (xf.grad.norm().item() or 1)
        good = rw < 1e-5 and rx < 1e-5
        ok &= good
        print(f"  in={in_f:>6} out={out_f:>7} K={K:>3}  P={cg._pick_bwd_wpo_params(out_f)[0]:>5}"
              f"  grad_w rel {rw:.2e}  grad_x rel {rx:.2e}  {'OK' if good else 'FAIL'}")
        del idx, x0, w0, go, xc, wc, xf, wf2
        torch.cuda.empty_cache()

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
