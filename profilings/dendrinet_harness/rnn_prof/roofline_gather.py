"""Is the CUDA gather at hardware roofline, or merely faster than the alternatives?

"Fastest path we have" and "no headroom left" are different claims, and only the
second one closes the optimization question. This computes the achieved DRAM
bandwidth of the gather at the dominant feedforward shape and states it as a
fraction of the card's peak -- the one number that says whether a better kernel
is possible.

Bytes moved per forward (the compulsory traffic, ignoring reuse):
  read  idx : out*K*4                (int32)
  read  w   : out*K*4
  read  x   : batch*in*4             (each x row is read by many outputs; with
                                      perfect caching this is the floor)
  write out : batch*out*4

If achieved/peak is high, the kernel is done and only algorithmic change helps.
If it is low, the kernel is latency- or occupancy-bound and a better one exists.

    python roofline_gather.py [--batch 512]
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    cuda_gather as cg,
)

# The shapes the 12-layer stress config actually spends its time in, and how
# many layers of each it has.
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
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()
    if not (torch.cuda.is_available() and cg.cuda_gather_available()):
        print("needs CUDA + built extension "
              "(source profilings/weight_transform_opt/setup_cuda_env.sh)")
        return 1
    ext = cg._load_ext()
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)

    # Measured peak, not the datasheet number: a large device-to-device copy is
    # the honest ceiling for a bandwidth-bound kernel on this card.
    n = 1 << 28
    src = torch.empty(n, dtype=torch.float32, device=dev)
    dst = torch.empty_like(src)
    t = timeit(lambda: dst.copy_(src))
    peak = 2 * n * 4 / t / 1e9
    print(f"GPU {props.name}  SMs={props.multi_processor_count}")
    print(f"measured peak bandwidth (d2d copy): {peak:,.0f} GB/s   batch={a.batch}\n")

    print(f"{'in':>7}{'out':>8}{'K':>4}  {'phase':>5}{'ms':>9}{'GB/s':>10}"
          f"{'% of peak':>11}")
    for in_f, out_f, K, _ in SHAPES:
        idx = torch.randint(0, in_f, (out_f, K), device=dev,
                            dtype=torch.int32).contiguous()
        wf = (torch.randn(out_f, K, device=dev) * 0.05).contiguous()
        xf = torch.randn(a.batch, in_f, device=dev).abs().contiguous()
        P, T = cg._pick_wpo_params()
        fwd = timeit(lambda: ext.gather_fwd_wpo_cuda(xf, idx, wf, P, T))
        bytes_fwd = (out_f * K * 4 * 2) + (a.batch * in_f * 4) + (a.batch * out_f * 4)

        gof = torch.randn(a.batch, out_f, device=dev).contiguous()
        goT = gof.t().contiguous()
        xT = xf.t().contiguous()
        Pb, Tb = cg._pick_bwd_wpo_params(out_f)
        bwd = timeit(lambda: ext.gather_bwd_wpo_cuda(gof, goT, xT, idx, wf, Pb, Tb))
        # bwd reads grad_out and x, writes grad_x and grad_w
        bytes_bwd = ((a.batch * out_f * 4) + (a.batch * in_f * 4)
                     + (a.batch * in_f * 4) + (out_f * K * 4) + (out_f * K * 4))

        for label, sec, nbytes in (("fwd", fwd, bytes_fwd), ("bwd", bwd, bytes_bwd)):
            gbs = nbytes / sec / 1e9
            print(f"{in_f:>7,}{out_f:>8,}{K:>4}  {label:>5}{sec*1e3:>9.2f}"
                  f"{gbs:>10,.0f}{100*gbs/peak:>10.1f}%")
        del idx, wf, xf, gof, goT, xT
        torch.cuda.empty_cache()

    print("\n  A kernel near peak is finished -- only an algorithmic change helps.")
    print("  A kernel far below peak is latency/occupancy bound: a better one exists.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
