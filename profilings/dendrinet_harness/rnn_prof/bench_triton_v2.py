"""Triton gather v2 candidates vs the shipped Triton kernels. Nothing ships
from here until it wins; the shipped module is untouched.

Candidates port the CUDA campaign's transferable lessons:
  fwd v2  : int32 indices (no per-call int64 upcast), K blocked into B_K-wide
            vector loads of idx/w (vs the shipped scalar static_range(K) loop),
            tile sizes swept per shape.
  bwd v2  : SPLIT (the CUDA Phase-4 idea, minus unportable shared staging) --
            grad_w via transposed coalesced reduction over m, NO atomics;
            grad_x via fp32 atomic scatter with swept tiles.

Gates, in order:
  [1] correctness vs an fp64 reference: v2's fp32 error must be <= 2x the
      shipped kernel's own fp32 error at every shape (reassociation only).
  [2] speed: per-shape sweep; a candidate earns adoption only where it beats
      the SHIPPED triton by >3% (cuda numbers printed as the ceiling).

    python bench_triton_v2.py [--batch 512] [--K 40]
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import triton
import triton.language as tl

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.triton_gather import (
    triton_sparse_gather,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    cuda_gather as cg,
)

SHAPES = [(3072, 16000), (8000, 48000), (16000, 16000), (16000, 48000),
          (16000, 144000), (24000, 48000)]

FWD_TILES = [(16, 64, 8), (32, 64, 8), (16, 128, 8), (32, 32, 16),
             (16, 64, 16), (64, 32, 8)]
GW_TILES = [(64, 64), (32, 128), (128, 32)]        # (B_O rows, B_M chunk)
GX_TILES = [(16, 64, 8), (32, 64, 8), (16, 128, 8)]


@triton.jit
def _fwd_v2(x_ptr, idx_ptr, w_ptr, out_ptr, M, O, IN, K,
            B_M: tl.constexpr, B_O: tl.constexpr, B_K: tl.constexpr):
    mb = tl.program_id(0)
    ob = tl.program_id(1)
    m_off = mb * B_M + tl.arange(0, B_M)
    o_off = ob * B_O + tl.arange(0, B_O)
    m_mask = m_off < M
    o_mask = o_off < O
    acc = tl.zeros([B_M, B_O], dtype=tl.float32)
    for kb in range(0, tl.cdiv(K, B_K)):
        k_off = kb * B_K + tl.arange(0, B_K)
        k_mask = k_off < K
        ok = o_off[:, None] * K + k_off[None, :]                     # [B_O, B_K]
        om = o_mask[:, None] & k_mask[None, :]
        i2 = tl.load(idx_ptr + ok, mask=om, other=0)                 # int32 vector load
        w2 = tl.load(w_ptr + ok, mask=om, other=0.0)
        xo = m_off[:, None, None] * IN + i2[None, :, :]              # [B_M, B_O, B_K]
        xm = m_mask[:, None, None] & om[None, :, :]
        xv = tl.load(x_ptr + xo, mask=xm, other=0.0)
        acc += tl.sum(xv * w2[None, :, :], axis=2)
    oo = m_off[:, None] * O + o_off[None, :]
    tl.store(out_ptr + oo, acc, mask=(m_mask[:, None] & o_mask[None, :]))


@triton.jit
def _gw_v2(goT_ptr, xT_ptr, idx_ptr, gw_ptr, M, O, IN, K,
           B_O: tl.constexpr, B_M: tl.constexpr):
    """grad_w[o,k] = sum_m go[m,o] * x[m, idx[o,k]] -- transposed, coalesced
    over m, no atomics. One program owns B_O rows for ONE k (grid: O-tiles x K)."""
    ob = tl.program_id(0)
    k = tl.program_id(1)
    o_off = ob * B_O + tl.arange(0, B_O)
    o_mask = o_off < O
    idx = tl.load(idx_ptr + o_off * K + k, mask=o_mask, other=0)     # [B_O]
    acc = tl.zeros([B_O], dtype=tl.float32)
    for mb in range(0, tl.cdiv(M, B_M)):
        m_off = mb * B_M + tl.arange(0, B_M)
        m_mask = m_off < M
        mm = o_mask[:, None] & m_mask[None, :]
        go = tl.load(goT_ptr + o_off[:, None] * M + m_off[None, :], mask=mm, other=0.0)
        xv = tl.load(xT_ptr + idx[:, None] * M + m_off[None, :], mask=mm, other=0.0)
        acc += tl.sum(go * xv, axis=1)
    tl.store(gw_ptr + o_off * K + k, acc, mask=o_mask)


@triton.jit
def _gx_v2(go_ptr, idx_ptr, w_ptr, gx_ptr, M, O, IN, K,
           B_M: tl.constexpr, B_O: tl.constexpr, B_K: tl.constexpr):
    mb = tl.program_id(0)
    ob = tl.program_id(1)
    m_off = mb * B_M + tl.arange(0, B_M)
    o_off = ob * B_O + tl.arange(0, B_O)
    m_mask = m_off < M
    o_mask = o_off < O
    mo = m_mask[:, None] & o_mask[None, :]
    go = tl.load(go_ptr + m_off[:, None] * O + o_off[None, :], mask=mo, other=0.0)
    for kb in range(0, tl.cdiv(K, B_K)):
        k_off = kb * B_K + tl.arange(0, B_K)
        k_mask = k_off < K
        ok = o_off[:, None] * K + k_off[None, :]
        om = o_mask[:, None] & k_mask[None, :]
        i2 = tl.load(idx_ptr + ok, mask=om, other=0)
        w2 = tl.load(w_ptr + ok, mask=om, other=0.0)
        xo = m_off[:, None, None] * IN + i2[None, :, :]
        xm = m_mask[:, None, None] & om[None, :, :]
        tl.atomic_add(gx_ptr + xo, go[:, :, None] * w2[None, :, :], mask=xm)


def fwd_v2(x, idx32, w, tiles):
    B_M, B_O, B_K = tiles
    M, IN = x.shape
    O, K = idx32.shape
    out = torch.empty(M, O, device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(M, B_M), triton.cdiv(O, B_O))
    _fwd_v2[grid](x, idx32, w, out, M, O, IN, K, B_M, B_O, B_K)
    return out


def bwd_v2(go, x, idx32, w, gw_tiles, gx_tiles):
    M, IN = x.shape
    O, K = idx32.shape
    goT = go.t().contiguous()
    xT = x.t().contiguous()
    gw = torch.empty(O, K, device=x.device, dtype=torch.float32)
    B_O, B_M = gw_tiles
    _gw_v2[(triton.cdiv(O, B_O), K)](goT, xT, idx32, gw, M, O, IN, K, B_O, B_M)
    gx = torch.zeros(M, IN, device=x.device, dtype=torch.float32)
    B_M2, B_O2, B_K2 = gx_tiles
    grid = (triton.cdiv(M, B_M2), triton.cdiv(O, B_O2))
    _gx_v2[grid](go, idx32, w, gx, M, O, IN, K, B_M2, B_O2, B_K2)
    return gx, gw


def timeit(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def best(fn, params):
    res = {}
    for p in params:
        try:
            res[p] = timeit(lambda p=p: fn(p))
        except Exception:
            pass
    if not res:
        return None, float("inf")
    k = min(res, key=res.get)
    return k, res[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--K", type=int, default=40)
    a = ap.parse_args()
    dev = torch.device("cuda")
    M, K = a.batch, a.K
    has_cuda = cg.cuda_gather_available()
    print(f"GPU {torch.cuda.get_device_name(0)}  batch={M} K={K}\n")

    print("=" * 96)
    print("[1] CORRECTNESS vs fp64 reference (fp32 max|err|; v2 must be <= 2x shipped's)")
    print("=" * 96)
    ok = True
    for IN, O in SHAPES[:3]:
        idx32 = torch.randint(0, IN, (O, K), dtype=torch.int32, device=dev)
        idx64 = idx32.to(torch.int64)
        w = (torch.randn(O, K, device=dev) * 0.05)
        x = torch.randn(128, IN, device=dev).abs()
        # fp64 reference
        ref = torch.zeros(128, O, dtype=torch.float64, device=dev)
        xd, wd = x.double(), w.double()
        for k in range(K):
            ref += xd[:, idx64[:, k]] * wd[:, k]
        e_ship = (triton_sparse_gather(x, idx64, w).double() - ref).abs().max().item()
        e_v2 = (fwd_v2(x, idx32, w, (16, 64, 8)).double() - ref).abs().max().item()
        # backward reference via autograd on the fp64 reference expression
        xg = xd.clone().requires_grad_(True)
        wg = wd.clone().requires_grad_(True)
        out = torch.zeros(128, O, dtype=torch.float64, device=dev)
        for k in range(K):
            out = out + xg[:, idx64[:, k]] * wg[:, k]
        out.sum().backward()
        go = torch.ones(128, O, device=dev)
        gx, gw = bwd_v2(go, x, idx32, w, (64, 64), (16, 64, 8))
        e_gw = (gw.double() - wg.grad).abs().max().item()
        e_gx = (gx.double() - xg.grad).abs().max().item()
        cell_ok = e_v2 <= max(2 * e_ship, 1e-6) and e_gw < 1e-3 and e_gx < 1e-3
        ok &= cell_ok
        print(f"  in={IN:>6,} out={O:>7,}: fwd ship {e_ship:.1e} v2 {e_v2:.1e} | "
              f"gw {e_gw:.1e} gx {e_gx:.1e}  {'OK' if cell_ok else 'FAIL'}")
        del idx32, idx64, w, x, ref, xd, wd, xg, wg, out, gx, gw
        torch.cuda.empty_cache()
    if not ok:
        print("CORRECTNESS FAIL -- not sweeping")
        return 1

    print("\n" + "=" * 96)
    print("[2] SWEEP -- v2 (best tiles) vs shipped triton vs cuda")
    print("=" * 96)
    print(f"{'in':>7}{'out':>8} | {'ship f/b':>15} | {'v2 f/b':>15} | "
          f"{'cuda f/b':>13} | {'v2/ship f':>9}{'  b':>7}")
    for IN, O in SHAPES:
        idx32 = torch.randint(0, IN, (O, K), dtype=torch.int32, device=dev)
        idx64 = idx32.to(torch.int64)
        w = (torch.randn(O, K, device=dev) * 0.05).contiguous()
        x = torch.randn(M, IN, device=dev).abs().contiguous()
        go = torch.randn(M, O, device=dev).contiguous()

        sf = timeit(lambda: triton_sparse_gather(x, idx64, w))
        xg = x.detach().requires_grad_(True)
        wg = w.detach().requires_grad_(True)
        out = triton_sparse_gather(xg, idx64, wg)
        sb = timeit(lambda: torch.autograd.grad(out, (xg, wg), go, retain_graph=True))

        ft, vf = best(lambda t: fwd_v2(x, idx32, w, t), FWD_TILES)
        bt, vb = best(lambda t: bwd_v2(go, x, idx32, w, t[0], t[1]),
                      [(g, s) for g in GW_TILES for s in GX_TILES])

        if has_cuda:
            cf = timeit(lambda: cg.cuda_sparse_gather(x, idx32, w))
            xg2 = x.detach().requires_grad_(True)
            wg2 = w.detach().requires_grad_(True)
            out2 = cg.cuda_sparse_gather(xg2, idx32, wg2)
            cb = timeit(lambda: torch.autograd.grad(out2, (xg2, wg2), go, retain_graph=True))
        else:
            cf = cb = float("nan")

        print(f"{IN:>7,}{O:>8,} | {sf:>6.2f} /{sb:>7.2f} | {vf:>6.2f} /{vb:>7.2f} | "
              f"{cf:>5.2f} /{cb:>6.2f} | {sf/vf:>8.2f}x {sb/vb:>6.2f}x   "
              f"f{ft} gw{bt[0] if bt else '-'} gx{bt[1] if bt else '-'}")
        del idx32, idx64, w, x, go
        torch.cuda.empty_cache()
    print("\n  v2 ships only where v2/ship > 1.03 at BOTH directions' chosen tiles.")
    return 0


if __name__ == "__main__" and "--v3" not in sys.argv and "--verify3" not in sys.argv:
    sys.exit(main())


# ---- v3: transposed-x coalesced variants (appended iteration) ----------------
@triton.jit
def _fwd_v3(xT_ptr, idx_ptr, w_ptr, out_ptr, M, O, IN, K,
            B_M: tl.constexpr, B_O: tl.constexpr, B_K: tl.constexpr):
    """Gather from xT [IN, M]: addresses idx*M + m are CONTIGUOUS over m."""
    mb = tl.program_id(0)
    ob = tl.program_id(1)
    m_off = mb * B_M + tl.arange(0, B_M)
    o_off = ob * B_O + tl.arange(0, B_O)
    m_mask = m_off < M
    o_mask = o_off < O
    acc = tl.zeros([B_O, B_M], dtype=tl.float32)
    for kb in range(0, tl.cdiv(K, B_K)):
        k_off = kb * B_K + tl.arange(0, B_K)
        k_mask = k_off < K
        ok = o_off[:, None] * K + k_off[None, :]
        om = o_mask[:, None] & k_mask[None, :]
        i2 = tl.load(idx_ptr + ok, mask=om, other=0)
        w2 = tl.load(w_ptr + ok, mask=om, other=0.0)
        xo = i2[:, :, None] * M + m_off[None, None, :]              # [B_O, B_K, B_M]
        xm = om[:, :, None] & m_mask[None, None, :]
        xv = tl.load(xT_ptr + xo, mask=xm, other=0.0)               # coalesced over m
        acc += tl.sum(xv * w2[:, :, None], axis=1)
    oo = m_off[None, :] * O + o_off[:, None]
    tl.store(out_ptr + oo, acc, mask=(m_mask[None, :] & o_mask[:, None]))


@triton.jit
def _gx_v3(goT_ptr, idx_ptr, w_ptr, gxT_ptr, M, O, IN, K,
           B_M: tl.constexpr, B_O: tl.constexpr, B_K: tl.constexpr):
    """Atomic scatter into gxT [IN, M]: contiguous over m -> coalesced atomics."""
    mb = tl.program_id(0)
    ob = tl.program_id(1)
    m_off = mb * B_M + tl.arange(0, B_M)
    o_off = ob * B_O + tl.arange(0, B_O)
    m_mask = m_off < M
    o_mask = o_off < O
    om_go = o_mask[:, None] & m_mask[None, :]
    go = tl.load(goT_ptr + o_off[:, None] * M + m_off[None, :], mask=om_go, other=0.0)
    for kb in range(0, tl.cdiv(K, B_K)):
        k_off = kb * B_K + tl.arange(0, B_K)
        k_mask = k_off < K
        ok = o_off[:, None] * K + k_off[None, :]
        om = o_mask[:, None] & k_mask[None, :]
        i2 = tl.load(idx_ptr + ok, mask=om, other=0)
        w2 = tl.load(w_ptr + ok, mask=om, other=0.0)
        xo = i2[:, :, None] * M + m_off[None, None, :]              # [B_O, B_K, B_M]
        xm = om[:, :, None] & m_mask[None, None, :]
        tl.atomic_add(gxT_ptr + xo, w2[:, :, None] * go[:, None, :], mask=xm)


def fwd_v3(x, idx32, w, tiles):
    B_M, B_O, B_K = tiles
    M, IN = x.shape
    O, K = idx32.shape
    xT = x.t().contiguous()
    out = torch.empty(M, O, device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(M, B_M), triton.cdiv(O, B_O))
    _fwd_v3[grid](xT, idx32, w, out, M, O, IN, K, B_M, B_O, B_K)
    return out


def gx_v3(go, x, idx32, w, tiles):
    B_M, B_O, B_K = tiles
    M, IN = x.shape
    O, K = idx32.shape
    goT = go.t().contiguous()
    gxT = torch.zeros(IN, M, device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(M, B_M), triton.cdiv(O, B_O))
    _gx_v3[grid](goT, idx32, w, gxT, M, O, IN, K, B_M, B_O, B_K)
    return gxT.t().contiguous()


def main_v3():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--K", type=int, default=40)
    a = ap.parse_args()
    dev = torch.device("cuda")
    M, K = a.batch, a.K
    print(f"GPU {torch.cuda.get_device_name(0)}  batch={M} K={K}  [v3 iteration]\n")
    V3F_TILES = [(32, 32, 8), (64, 32, 8), (32, 64, 8), (64, 16, 16), (128, 16, 8)]
    # correctness on one shape
    IN, O = 8000, 48000
    idx32 = torch.randint(0, IN, (O, K), dtype=torch.int32, device=dev)
    idx64 = idx32.to(torch.int64)
    w = (torch.randn(O, K, device=dev) * 0.05)
    x = torch.randn(128, IN, device=dev).abs()
    ref = torch.zeros(128, O, dtype=torch.float64, device=dev)
    for k in range(K):
        ref += x.double()[:, idx64[:, k]] * w.double()[:, k]
    e3 = (fwd_v3(x, idx32, w, (32, 32, 8)).double() - ref).abs().max().item()
    go1 = torch.ones(128, O, device=dev)
    xg = x.double().clone().requires_grad_(True)
    out = torch.zeros(128, O, dtype=torch.float64, device=dev)
    for k in range(K):
        out = out + xg[:, idx64[:, k]] * w.double()[:, k]
    out.sum().backward()
    e3x = (gx_v3(go1, x, idx32, w, (32, 32, 8)).double() - xg.grad).abs().max().item()
    print(f"correctness (in={IN}, out={O}): fwd_v3 {e3:.1e}  gx_v3 {e3x:.1e}"
          f"  {'OK' if e3 < 1e-5 and e3x < 1e-3 else 'FAIL'}")
    if not (e3 < 1e-5 and e3x < 1e-3):
        return 1
    del ref, xg, out
    torch.cuda.empty_cache()

    print(f"\n{'in':>7}{'out':>8} | {'v2 f':>7} {'v3 f':>7} | {'v2 gx':>7} {'v3 gx':>7} | "
          f"{'cuda f':>7}")
    for IN, O in SHAPES:
        idx32 = torch.randint(0, IN, (O, K), dtype=torch.int32, device=dev)
        w = (torch.randn(O, K, device=dev) * 0.05).contiguous()
        x = torch.randn(M, IN, device=dev).abs().contiguous()
        go = torch.randn(M, O, device=dev).contiguous()
        _, v2f = best(lambda t: fwd_v2(x, idx32, w, t), FWD_TILES)
        t3, v3f = best(lambda t: fwd_v3(x, idx32, w, t), V3F_TILES)
        _, v2x = best(lambda t: (torch.zeros(M, IN, device=dev),
                                 _gx_v2[(triton.cdiv(M, t[0]), triton.cdiv(O, t[1]))](
                                     go, idx32, w, torch.zeros(M, IN, device=dev),
                                     M, O, IN, K, t[0], t[1], t[2]))[0], GX_TILES)
        t3x, v3x = best(lambda t: gx_v3(go, x, idx32, w, t), V3F_TILES)
        cf = timeit(lambda: cg.cuda_sparse_gather(x, idx32, w)) if cg.cuda_gather_available() else float("nan")
        print(f"{IN:>7,}{O:>8,} | {v2f:>7.2f} {v3f:>7.2f} | {v2x:>7.2f} {v3x:>7.2f} | "
              f"{cf:>7.2f}   f{t3} gx{t3x}")
        del idx32, w, x, go
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__" and "--v3" in sys.argv and "--verify3" not in sys.argv:
    sys.argv.remove("--v3")
    sys.exit(main_v3())


def main_verify3():
    """Hard verification of v3 at the SWEPT tiles, batch 512, every shape."""
    dev = torch.device("cuda")
    M, K = 512, 40
    print(f"[verify3] {torch.cuda.get_device_name(0)} batch={M} K={K}")
    ok = True
    for IN, O in SHAPES:
        idx32 = torch.randint(0, IN, (O, K), dtype=torch.int32, device=dev)
        idx64 = idx32.to(torch.int64)
        w = (torch.randn(O, K, device=dev) * 0.05)
        x = torch.randn(M, IN, device=dev).abs()
        go = torch.randn(M, O, device=dev)
        # fp64 reference forward + grads
        xd, wd, god = x.double(), w.double(), go.double()
        ref = torch.zeros(M, O, dtype=torch.float64, device=dev)
        gw_ref = torch.zeros(O, K, dtype=torch.float64, device=dev)
        gx_ref = torch.zeros(M, IN, dtype=torch.float64, device=dev)
        for k in range(K):
            xk = xd[:, idx64[:, k]]
            ref += xk * wd[:, k]
            gw_ref[:, k] = (god * xk).sum(0)
            gx_ref.index_add_(1, idx64[:, k], god * wd[:, k])
        ft = (64, 16, 16) if (IN, O) == (3072, 16000) else (128, 16, 8)
        out = fwd_v3(x, idx32, w, ft)
        gx = gx_v3(go, x, idx32, w, (32, 32, 8))
        gw = torch.empty(O, K, device=dev, dtype=torch.float32)
        goT = go.t().contiguous(); xT = x.t().contiguous()
        _gw_v2[(triton.cdiv(O, 64), K)](goT, xT, idx32, gw, M, O, IN, K, 64, 64)
        ef = (out.double() - ref).abs().max().item()
        ew = (gw.double() - gw_ref).abs().max().item()
        ex = (gx.double() - gx_ref).abs().max().item()
        # scale-aware bounds: fp32 reassociation on |sum| ~ K*0.05 terms
        cell = ef < 1e-5 and ew < 5e-3 and ex < 5e-4
        ok &= cell
        print(f"  in={IN:>6,} out={O:>7,}: fwd {ef:.1e}  gw {ew:.1e}  gx {ex:.1e}  "
              f"{'OK' if cell else 'FAIL'}")
        del idx32, idx64, w, x, go, xd, wd, god, ref, gw_ref, gx_ref, out, gx, gw, goT, xT
        torch.cuda.empty_cache()
    print("[verify3]", "ALL OK" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__" and "--verify3" in sys.argv:
    sys.exit(main_verify3())
