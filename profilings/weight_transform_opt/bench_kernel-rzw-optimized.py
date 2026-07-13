"""OPTIMIZED sparse-gather module + benchmark  (route 2: custom autograd backward).

  Baseline / revert-to file:  bench_kernel.py       (unmodified reference)
  This file:                  the production candidate + its A/B benchmark

If anything goes wrong, bench_kernel.py is the known-good baseline. This file is the
ONLY place the optimized kernel lives, so it is unambiguous what to inspect/roll back.

The module (top half) is importable:
    from importlib import util
    m = util.spec_from_file_location("opt", ".../bench_kernel-rzw-optimized.py")...
    m.fused_sparse_gather(x, connection_indices, weight, chunk_size=2048)

The benchmark (bottom half, under __main__) races the baseline advanced-index gather
against fused_sparse_gather for identical output + gradients, then times fwd/bwd.

Key differences from the 20-line Kernel B in bench_kernel.py:
  * CHUNKED forward AND backward  -> bounded transient memory, fits out=90000 @ N=512
  * `selected` is NOT saved for backward (it is the 7.4 GB tensor at real scale);
    it is recomputed per-chunk in backward from x + indices (one cheap re-gather)
  * backward scatter uses index_add_ (grouped) instead of the generic
    indexing_backward_kernel (the measured 6.6x win)
"""
import torch

# =====================================================================
# THE OPTIMIZED MODULE  (importable; mirrors indexed_sparse.py semantics)
# out[m, o] = sum_k  x[m, connection_indices[o, k]] * weight[o, k]
# =====================================================================
class ChunkedSparseGatherMulSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx_flat, weight, out_features, K, chunk_size):
        # x: [M, IN]   idx_flat: [out_features*K] long   weight: [out_features, K]
        M = x.shape[0]
        out = x.new_empty(M, out_features)
        for start in range(0, out_features, chunk_size):
            end = min(start + chunk_size, out_features)
            idx_chunk = idx_flat[start * K:end * K]
            selected = x.index_select(1, idx_chunk).reshape(M, end - start, K)
            out[:, start:end] = (selected * weight[start:end].unsqueeze(0)).sum(dim=-1)
        # save only small tensors; recompute `selected` in backward to bound memory
        ctx.save_for_backward(x, idx_flat, weight)
        ctx.meta = (out_features, K, chunk_size)
        return out

    @staticmethod
    def backward(ctx, grad_out):                        # grad_out: [M, out_features]
        x, idx_flat, weight = ctx.saved_tensors
        out_features, K, chunk_size = ctx.meta
        M = x.shape[0]
        grad_x = torch.zeros_like(x)
        grad_w = torch.empty_like(weight)
        for start in range(0, out_features, chunk_size):
            end = min(start + chunk_size, out_features)
            idx_chunk = idx_flat[start * K:end * K]
            g = grad_out[:, start:end].unsqueeze(-1)                    # [M, chunk, 1]
            selected = x.index_select(1, idx_chunk).reshape(M, end - start, K)
            grad_w[start:end] = (g * selected).sum(dim=0)               # [chunk, K]
            grad_selected = (g * weight[start:end].unsqueeze(0)).reshape(M, (end - start) * K)
            grad_x.index_add_(1, idx_chunk, grad_selected)             # grouped scatter
        return grad_x, None, grad_w, None, None, None


def fused_sparse_gather(x, connection_indices, weight, chunk_size=2048):
    """Drop-in for indexed_sparse.py's gather+mul+sum. `weight` is the already-
    normalized sparse weight [out_features, K]; `connection_indices` is [out_features, K] long."""
    out_features, K = connection_indices.shape
    idx_flat = connection_indices.reshape(-1)
    return ChunkedSparseGatherMulSum.apply(x, idx_flat, weight, out_features, K, chunk_size)


# =====================================================================
# BENCHMARK  (only runs when executed directly, not on import)
# =====================================================================
if __name__ == "__main__":
    import argparse, statistics
    Event = torch.cuda.Event

    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=64)
    p.add_argument("--in_features", type=int, default=3072)
    p.add_argument("--out", type=int, default=10000)
    p.add_argument("--K", type=int, default=40)
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=8)
    args = p.parse_args()

    assert torch.cuda.is_available()
    dev = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"shape: B={args.B} in={args.in_features} out={args.out} K={args.K} "
          f"chunk={args.chunk} (fp32)\n")

    def make_inputs():
        torch.manual_seed(0)
        x = torch.randn(args.B, args.in_features, device=dev, requires_grad=True)
        idx = torch.randint(0, args.in_features, (args.out, args.K), device=dev)
        w = torch.randn(args.out, args.K, device=dev, requires_grad=True)
        return x, idx, w

    # baseline: current advanced-index gather (autograd's generic backward)
    def baseline(x, idx, w):
        B = x.shape[0]
        selected = x[:, idx.reshape(-1)].reshape(B, args.out, args.K)
        return (selected * w.unsqueeze(0)).sum(dim=-1)

    # optimized: the module above
    def optimized(x, idx, w):
        return fused_sparse_gather(x, idx, w, chunk_size=args.chunk)

    def measure_peak_bw(nbytes=256 * 1024 * 1024):
        n = nbytes // 4
        a, b = torch.empty(n, device=dev), torch.empty(n, device=dev)
        for _ in range(5): b.copy_(a)
        torch.cuda.synchronize()
        e0, e1 = Event(True), Event(True); e0.record()
        for _ in range(30): b.copy_(a)
        e1.record(); torch.cuda.synchronize()
        return 2 * nbytes / (e0.elapsed_time(e1) / 30 / 1e3) / 1e9

    def bench(fn):
        x, idx, w = make_inputs()
        for _ in range(args.warmup):
            x.grad = w.grad = None
            fn(x, idx, w).sum().backward()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        fwd, bwd = [], []
        for _ in range(args.iters):
            x.grad = w.grad = None
            e0, e1 = Event(True), Event(True)
            e0.record(); out = fn(x, idx, w); e1.record(); torch.cuda.synchronize()
            fwd.append(e0.elapsed_time(e1))
            loss = out.sum(); torch.cuda.synchronize()
            e0, e1 = Event(True), Event(True)
            e0.record(); loss.backward(); e1.record(); torch.cuda.synchronize()
            bwd.append(e0.elapsed_time(e1))
        return statistics.median(fwd), statistics.median(bwd), \
            torch.cuda.max_memory_allocated() / 1e9

    def _report(tag, a, b):
        diff = (a - b).abs()
        max_abs = diff.max().item()
        max_rel = (diff / b.abs().clamp_min(1e-12)).max().item()
        ok = torch.allclose(a, b, rtol=1e-4, atol=1e-5)
        print(f"    {tag:8s} {'OK  ' if ok else 'FAIL'}  max_abs={max_abs:.2e}  max_rel={max_rel:.2e}")
        return ok

    def _run_pair(dtype, B, IN, OUT, K):
        torch.manual_seed(0)
        x = torch.randn(B, IN, device=dev, dtype=dtype)
        idx = torch.randint(0, IN, (OUT, K), device=dev)
        w = torch.randn(OUT, K, device=dev, dtype=dtype)
        xa = x.clone().requires_grad_(True); wa = w.clone().requires_grad_(True)
        xb = x.clone().requires_grad_(True); wb = w.clone().requires_grad_(True)
        # baseline (advanced-index) vs optimized (fused_sparse_gather), same inputs
        sel = xa[:, idx.reshape(-1)].reshape(B, OUT, K)
        oa = (sel * wa.unsqueeze(0)).sum(-1); oa.sum().backward()
        ob = fused_sparse_gather(xb, idx, wb, chunk_size=args.chunk); ob.sum().backward()
        return (oa, ob), (xa.grad, xb.grad), (wa.grad, wb.grad)

    def check_equiv():
        # (1) fp32 at the requested shape, with per-tensor diffs
        print("  fp32 @ requested shape:")
        (oa, ob), (gxa, gxb), (gwa, gwb) = _run_pair(
            torch.float32, args.B, args.in_features, args.out, args.K)
        ok32 = _report("out", oa, ob) & _report("grad_x", gxa, gxb) & _report("grad_w", gwa, gwb)

        # (2) fp64 at a small fixed shape: proves the ALGORITHM (no logic bug),
        #     small so it never OOMs regardless of --out
        print("  fp64 @ small fixed shape (algorithmic check):")
        (oa, ob), (gxa, gxb), (gwa, gwb) = _run_pair(torch.float64, 16, 1024, 4096, 32)
        ok64 = _report("out", oa, ob) & _report("grad_x", gxa, gxb) & _report("grad_w", gwa, gwb)

        if ok32:
            print("  verdict: identical within fp32 tolerance.")
        elif ok64:
            print("  verdict: fp32 diff is ACCUMULATION-ORDER noise (fp64 is exact) -- NOT a bug.")
            print("           check max_rel above: ~1e-3 or less = rounding. Real gate is the")
            print("           training-equivalence test under AMP, far below bf16 precision.")
        else:
            print("  verdict: REAL DISCREPANCY (fails even in fp64) -- logic bug, investigate.")
        return ok64

    peak_bw = measure_peak_bw()
    print(f"empirical peak HBM bandwidth (memcpy): {peak_bw:.0f} GB/s\n")
    check_equiv()

    bytes_fwd = 2 * args.B * args.out * args.K * 4
    bytes_bwd = 3 * args.B * args.out * args.K * 4
    print(f"\n{'kernel':<12}{'fwd ms':>10}{'bwd ms':>10}{'fwd GB/s':>11}{'%peak':>8}"
          f"{'bwd GB/s':>11}{'%peak':>8}{'peak GB':>10}")
    for name, fn in [("baseline", baseline), ("optimized", optimized)]:
        f_ms, b_ms, pk = bench(fn)
        f_bw = bytes_fwd / (f_ms / 1e3) / 1e9
        b_bw = bytes_bwd / (b_ms / 1e3) / 1e9
        print(f"{name:<12}{f_ms:>10.3f}{b_ms:>10.3f}{f_bw:>11.0f}{100*f_bw/peak_bw:>7.0f}%"
              f"{b_bw:>11.0f}{100*b_bw/peak_bw:>7.0f}%{pk:>10.2f}")
