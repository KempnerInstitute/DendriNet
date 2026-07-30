"""Cost of the Python wrapper around the CUDA gather, separate from the kernel.

The kernel has been tuned (wpo forward, adaptive backward tile). The Python in
_CudaSparseGather around it has not, and reading it turns up two things that the
kernel work would never have surfaced:

  1. An int32 -> int64 -> int32 round trip. Indices are STORED int32 (the
     scaling lever that halved the replicated idx buffer), then forward line 420
     upcasts them to int64, line 424 casts them straight back to int32 to call
     the kernel, line 427 saves the int64 for backward, and backward line 443
     casts to int32 a third time. int64 is genuinely needed only by the v2
     fallback, which the stress shapes never take.

  2. Two full transposes materialized per backward call (xT, goT). The comment
     calls the x transpose cheap; goT is [O, M] and at out=144,000 x batch 512
     that is a 295 MB copy, charged per layer per step.

This times both against the kernel they wrap, so the wrapper's share is measured
rather than assumed.

    python bench_gather_wrapper.py [--batch 512]
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    cuda_gather as cg,
)

SHAPES = [(16000, 144000, 40, 11), (16000, 48000, 40, 11),
          (4000, 144000, 10, 11), (16000, 2000, 40, 11)]


def timeit(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()
    if not (torch.cuda.is_available() and cg.cuda_gather_available()):
        print("needs CUDA + built extension")
        return 1
    ext = cg._load_ext()
    dev = torch.device("cuda")
    M = a.batch
    print(f"GPU {torch.cuda.get_device_name(0)}  batch={M}\n")

    print(f"{'in':>7}{'out':>8}{'K':>4}  {'bwd kernel':>11}{'xT+goT':>9}"
          f"{'idx casts':>11}{'wrapper %':>11}   {'idx MB':>9}")
    tot_k = tot_w = 0.0
    for in_f, out_f, K, count in SHAPES:
        idx32 = torch.randint(0, in_f, (out_f, K), device=dev,
                              dtype=torch.int32).contiguous()
        wf = (torch.randn(out_f, K, device=dev) * 0.05).contiguous()
        xf = torch.randn(M, in_f, device=dev).abs().contiguous()
        gof = torch.randn(M, out_f, device=dev).contiguous()
        P, T = cg._pick_bwd_wpo_params(out_f)

        xT = xf.t().contiguous()
        goT = gof.t().contiguous()
        kern = timeit(lambda: ext.gather_bwd_wpo_cuda(gof, goT, xT, idx32, wf, P, T))
        trans = timeit(lambda: (xf.t().contiguous(), gof.t().contiguous()))
        # what the wrapper actually does to idx: store int32 -> int64 (fwd 420)
        # -> int32 (fwd 424) -> int32 again (bwd 443)
        casts = timeit(lambda: idx32.contiguous().to(torch.int64).to(torch.int32))

        share = 100 * (trans + casts) / (kern + trans + casts)
        idx_mb = out_f * K * 8 / 2**20  # the int64 copy retained per layer
        tot_k += kern * count
        tot_w += (trans + casts) * count
        print(f"{in_f:>7,}{out_f:>8,}{K:>4}  {kern:>11.2f}{trans:>9.2f}"
              f"{casts:>11.2f}{share:>10.1f}%   {idx_mb:>9.1f}")
        del idx32, wf, xf, gof, xT, goT
        torch.cuda.empty_cache()

    print(f"\n  weighted by the stress config's layer mix (60 sparse layers):")
    print(f"    backward kernels : {tot_k:8.1f} ms")
    print(f"    wrapper overhead : {tot_w:8.1f} ms   "
          f"-> {100*tot_w/(tot_k+tot_w):.1f}% of the backward")
    print(f"\n  The int64 idx copy is also RETAINED for backward, so it is peak")
    print(f"  memory as well as time -- and the stored buffer was already int32.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
