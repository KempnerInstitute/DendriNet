"""Triton fused sparse-gather FORWARD kernel, v1 (to benchmark, not a proven optimum).

Computes, without ever materializing `selected = [M, O, K]`:
    out[m, o] = sum_k  x[m, idx[o, k]] * w[o, k]

Architecture (per the design decision):
  * one batch row m per program  (B_M = 1)
  * P output partitions per row   (grid = (M, P))  -> M*P programs for occupancy
  * each program walks its output partition in B_O-sized tiles
  * the K loop is unrolled (K fixed & small)
  * the input row x[m,:] is reused via L2 caching (Triton has no explicit shared
    staging for dynamic gathers) -- so P mainly controls occupancy here, and the
    optimum may trend higher than CUDA intuition suggests. Sweep decides.

Backward is intentionally NOT fused yet -- forward-first, verify, then extend.

Run:  python triton_gather_v1.py                 # sweep P in {1,2,4,8}
      python triton_gather_v1.py --in_features 3072
"""
import argparse, statistics
import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_gather_fwd(
    x_ptr, idx_ptr, w_ptr, out_ptr,
    M, O, IN,
    K: tl.constexpr, P: tl.constexpr, B_O: tl.constexpr,
):
    m = tl.program_id(0)              # which batch row
    p = tl.program_id(1)             # which output partition
    o_per_p = tl.cdiv(O, P)
    o_start = p * o_per_p
    o_end = tl.minimum(o_start + o_per_p, O)
    x_row = x_ptr + m * IN            # base pointer of this batch row

    # walk this partition's outputs in tiles of B_O
    for o_base in tl.range(o_start, o_end, B_O):
        offs = o_base + tl.arange(0, B_O)        # [B_O] output ids
        mask = offs < o_end
        base_ok = offs * K                        # row offset into idx/w
        acc = tl.zeros([B_O], dtype=tl.float32)
        for k in tl.static_range(K):              # unrolled (K fixed)
            idx = tl.load(idx_ptr + base_ok + k, mask=mask, other=0)   # [B_O] long
            wk = tl.load(w_ptr + base_ok + k, mask=mask, other=0.0)    # [B_O]
            xv = tl.load(x_row + idx, mask=mask, other=0.0)            # [B_O] gather (L2)
            acc += xv * wk
        tl.store(out_ptr + m * O + offs, acc, mask=mask)


def triton_gather(x, idx, w, P=4, B_O=256):
    M, IN = x.shape
    O, K = idx.shape
    x = x.contiguous()
    idx = idx.contiguous().to(torch.int64)
    w = w.contiguous()
    out = torch.empty(M, O, device=x.device, dtype=x.dtype)
    _sparse_gather_fwd[(M, P)](x, idx, w, out, M, O, IN, K, P, B_O)
    return out


def eager_forward(x, idx, w):
    M = x.shape[0]
    O, K = idx.shape
    sel = x[:, idx.reshape(-1)].reshape(M, O, K)
    return (sel * w.unsqueeze(0)).sum(dim=-1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=128)
    p.add_argument("--in_features", type=int, default=10000)
    p.add_argument("--O", type=int, default=90000)
    p.add_argument("--K", type=int, default=40)
    p.add_argument("--B_O", type=int, default=256)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--Ps", default="1,2,4,8")
    args = p.parse_args()

    dev = "cuda"
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"shape: M={args.M} IN={args.in_features} O={args.O} K={args.K} "
          f"B_O={args.B_O} (fp32)\n")

    torch.manual_seed(0)
    x = torch.randn(args.M, args.in_features, device=dev)
    idx = torch.randint(0, args.in_features, (args.O, args.K), device=dev)
    w = torch.randn(args.O, args.K, device=dev)

    ref = eager_forward(x, idx, w)

    def bench(fn):
        for _ in range(5): fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(args.iters):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); fn(); e1.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        return statistics.median(ts)

    t_eager = bench(lambda: eager_forward(x, idx, w))
    print(f"eager forward: {t_eager:.3f} ms\n")

    print(f"{'P':>4}  {'progs':>7}  {'ms':>8}  {'vs eager':>9}  {'correct':>8}  {'max_err':>10}")
    for P in [int(v) for v in args.Ps.split(",")]:
        out = triton_gather(x, idx, w, P=P, B_O=args.B_O)   # first call compiles
        err = (out - ref).abs().max().item()
        ok = torch.allclose(out, ref, rtol=1e-4, atol=1e-5)
        t = bench(lambda P=P: triton_gather(x, idx, w, P=P, B_O=args.B_O))
        print(f"{P:>4}  {args.M*P:>7}  {t:>8.3f}  {t_eager/t:>8.2f}x  "
              f"{'OK' if ok else 'FAIL':>8}  {err:>10.2e}")

    print("\nread: pick the fastest P that is still 'OK'. If speedup > ~1.5x and correct,")
    print("      the fusion works -> wire it into the forward behind a flag, verify, extend")
    print("      to backward. If P keeps winning as it grows, occupancy (not reloads) rules.")
