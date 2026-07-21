"""Triton fused sparse-gather FORWARD kernel, v2: batch tiling to amortize idx/w.

v1 finding: with one batch row per program, idx/w were re-read M times (they don't
depend on the batch row) -> ~5.5 GB redundant index traffic, capping speedup at 1.33x.
v2 tiles the batch (B_M rows per program): load idx[o,k]/w[o,k] ONCE per output tile,
reuse across B_M rows. Parallelism now comes from the output-block grid dim (no P knob).

Sweep B_M to find the amortization sweet spot (register pressure eventually bites:
acc is [B_M, B_O] in registers, so keep B_M*B_O modest).

Run:  python triton_gather_v2.py                       # sweep B_M in {1,2,4,8,16}
      python triton_gather_v2.py --B_O 64 --B_Ms 4,8,16,32
"""
import argparse, statistics
import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_gather_fwd_v2(
    x_ptr, idx_ptr, w_ptr, out_ptr,
    M, O, IN,
    K: tl.constexpr, B_M: tl.constexpr, B_O: tl.constexpr,
):
    mb = tl.program_id(0)                      # which batch tile
    ob = tl.program_id(1)                      # which output tile
    m_off = mb * B_M + tl.arange(0, B_M)       # [B_M] batch rows
    o_off = ob * B_O + tl.arange(0, B_O)       # [B_O] outputs
    m_mask = m_off < M
    o_mask = o_off < O
    base_ok = o_off * K                         # [B_O] into idx/w rows

    acc = tl.zeros([B_M, B_O], dtype=tl.float32)
    for k in tl.static_range(K):                # unrolled, K fixed
        idx = tl.load(idx_ptr + base_ok + k, mask=o_mask, other=0)    # [B_O]  read ONCE
        wk = tl.load(w_ptr + base_ok + k, mask=o_mask, other=0.0)     # [B_O]  read ONCE
        x_off = m_off[:, None] * IN + idx[None, :]                    # [B_M, B_O]
        xmask = m_mask[:, None] & o_mask[None, :]
        xv = tl.load(x_ptr + x_off, mask=xmask, other=0.0)           # [B_M, B_O] gather
        acc += xv * wk[None, :]                                       # reuse idx/w across B_M
    out_off = m_off[:, None] * O + o_off[None, :]
    tl.store(out_ptr + out_off, acc, mask=(m_mask[:, None] & o_mask[None, :]))


def triton_gather_v2(x, idx, w, B_M=8, B_O=128):
    M, IN = x.shape
    O, K = idx.shape
    x = x.contiguous()
    idx = idx.contiguous().to(torch.int64)
    w = w.contiguous()
    out = torch.empty(M, O, device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, B_M), triton.cdiv(O, B_O))
    _sparse_gather_fwd_v2[grid](x, idx, w, out, M, O, IN, K, B_M, B_O)
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
    p.add_argument("--B_O", type=int, default=128)
    p.add_argument("--B_Ms", default="1,2,4,8,16")
    p.add_argument("--iters", type=int, default=50)
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

    print(f"{'B_M':>5}  {'progs':>8}  {'ms':>8}  {'vs eager':>9}  {'correct':>8}  {'max_err':>10}")
    for B_M in [int(v) for v in args.B_Ms.split(",")]:
        try:
            out = triton_gather_v2(x, idx, w, B_M=B_M, B_O=args.B_O)   # compiles
        except Exception as e:  # noqa: BLE001
            print(f"{B_M:>5}  {'--':>8}  {'ERR':>8}  {str(e)[:40]}")
            continue
        err = (out - ref).abs().max().item()
        ok = torch.allclose(out, ref, rtol=1e-4, atol=1e-5)
        progs = ((args.M + B_M - 1) // B_M) * ((args.O + args.B_O - 1) // args.B_O)
        t = bench(lambda B_M=B_M: triton_gather_v2(x, idx, w, B_M=B_M, B_O=args.B_O))
        print(f"{B_M:>5}  {progs:>8}  {t:>8.3f}  {t_eager/t:>8.2f}x  "
              f"{'OK' if ok else 'FAIL':>8}  {err:>10.2e}")

    print("\nread: expect speedup to climb with B_M (idx/w amortized) then flatten/drop when")
    print("      registers spill (acc is B_M*B_O floats). Best B_M is the amortization sweet")
    print("      spot. If v2 clearly beats v1's 1.33x, batch tiling was the real lever.")
