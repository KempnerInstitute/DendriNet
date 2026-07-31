"""Above the shared-memory ceiling, what should the BACKWARD do?

The un-staged forward lifted the forward's IN <= optin/4 ceiling (25,344 on
Blackwell). The backward's ceiling was NOT lifted: `_bwd_wpo_ok(IN)` gates on the
same IN*4 <= optin, and above it the path falls back to the v2 one-kernel
backward -- documented as ~2.6x slower than the split backward.

So the claim "the ceiling is lifted" is only half true today: forward runs, then
backward silently degrades.

But gather_gradx_ns_kernel needs NO shared memory, so unlike the staged kernel it
CAN run above the ceiling. It lost 2.1x to staged where staged is available --
that verdict stands. The open question is different: above the ceiling staged is
not an option, and the real comparison is ns vs v2.

If ns wins here, the rejected kernel is not dead code -- it is the correct
fallback, and the backward ceiling lifts too.

    python bench_bwd_above_ceiling.py [--batch 256]
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    cuda_gather as cg,
)


def timeit(fn, iters=5, warmup=2):
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
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    if not (torch.cuda.is_available() and cg.cuda_gather_available()):
        print("needs CUDA + built extension")
        return 1
    ext = cg._load_ext()
    if not hasattr(ext, "gather_bwd_wpo_cuda2"):
        print("stale build cache -- no gather_bwd_wpo_cuda2")
        return 1
    dev = torch.device("cuda")
    M = a.batch
    optin = cg._optin_shared()
    ceiling = optin // 4
    print(f"GPU {torch.cuda.get_device_name(0)}  optin={optin:,} B  "
          f"IN ceiling={ceiling:,}  batch={M}\n")

    shapes = [(ceiling + 8000, 48000, 40), (ceiling + 8000, 16000, 40),
              (2 * ceiling, 16000, 40)]
    B_O, B_M = cg._pick_tile()

    print(f"{'IN':>8}{'out':>8}{'K':>4}   {'v2 (current)':>15}{'ns grad_x':>15}"
          f"{'speedup':>10}   {'grad_x rel':>12}{'grad_w rel':>12}")
    for in_f, out_f, K in shapes:
        idx64 = torch.randint(0, in_f, (out_f, K), device=dev,
                              dtype=torch.int64).contiguous()
        idx32 = idx64.to(torch.int32).contiguous()
        w = (torch.randn(out_f, K, device=dev) * 0.05).contiguous()
        x = torch.randn(M, in_f, device=dev).abs().contiguous()
        gof = torch.randn(M, out_f, device=dev).contiguous()
        goT, xT = gof.t().contiguous(), x.t().contiguous()

        assert not cg._bwd_wpo_ok(in_f), "shape is not actually above the ceiling"
        P, T = cg._pick_bwd_wpo_params(out_f)

        t_v2 = timeit(lambda: ext.gather_bwd_cuda(gof, x, idx64, w, B_O, B_M))
        t_ns = timeit(lambda: ext.gather_bwd_wpo_cuda2(gof, goT, xT, idx32, w, P, T, 1))

        gx0, gw0 = ext.gather_bwd_cuda(gof, x, idx64, w, B_O, B_M)
        gx1, gw1 = ext.gather_bwd_wpo_cuda2(gof, goT, xT, idx32, w, P, T, 1)
        rx = (gx0 - gx1).abs().max().item() / (gx0.abs().max().item() or 1)
        rw = (gw0 - gw1).abs().max().item() / (gw0.abs().max().item() or 1)

        print(f"{in_f:>8,}{out_f:>8,}{K:>4}   {t_v2:>12.2f} ms{t_ns:>12.2f} ms"
              f"{t_v2/t_ns:>9.2f}x   {rx:>12.2e}{rw:>12.2e}")
        del idx64, idx32, w, x, gof, goT, xT, gx0, gw0, gx1, gw1
        torch.cuda.empty_cache()

    print("\n  Above the ceiling the staged kernel cannot run at all, so this is")
    print("  ns vs v2 -- a different question from the ns-vs-staged verdict.\n")

    # The forward picker already routes to ns above the ceiling, but that was
    # asserted from the mechanism, not measured against the v2 forward it
    # replaces. Same comparison, forward side.
    print("=" * 78)
    print("FORWARD above the ceiling: v2 (what it used to do) vs ns")
    print("=" * 78)
    print(f"{'IN':>8}{'out':>8}{'K':>4}   {'v2':>13}{'ns':>13}{'speedup':>10}"
          f"{'max rel':>11}")
    for in_f, out_f, K in shapes:
        idx64 = torch.randint(0, in_f, (out_f, K), device=dev,
                              dtype=torch.int64).contiguous()
        idx32 = idx64.to(torch.int32).contiguous()
        w = (torch.randn(out_f, K, device=dev) * 0.05).contiguous()
        x = torch.randn(M, in_f, device=dev).abs().contiguous()
        # The picker now routes here to v2 (via _wpo_ok), because ns LOSES above
        # the ceiling -- the opposite of the backward. This keeps that measured.
        _, P, T = cg._pick_fwd_variant(in_f, out_f, K)
        P, T = (64 if out_f >= 96_000 else 32), 512   # ns's own best tile
        t_v2 = timeit(lambda: ext.gather_fwd_cuda(x, idx64, w, B_O, B_M))
        t_ns = timeit(lambda: ext.gather_fwd_wpo_ns_cuda(x, idx32, w, P, T))
        o0 = ext.gather_fwd_cuda(x, idx64, w, B_O, B_M)
        o1 = ext.gather_fwd_wpo_ns_cuda(x, idx32, w, P, T)
        rel = (o0 - o1).abs().max().item() / (o0.abs().max().item() or 1)
        print(f"{in_f:>8,}{out_f:>8,}{K:>4}   {t_v2:>10.2f} ms{t_ns:>10.2f} ms"
              f"{t_v2/t_ns:>9.2f}x{rel:>11.2e}")
        del idx64, idx32, w, x, o0, o1
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
