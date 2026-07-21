"""Triton fused sparse-gather BACKWARD kernel, v1 (to benchmark).

Computes both gradients of  out[m,o] = sum_k x[m, idx[o,k]] * w[o,k]  in one pass:
  grad_w[o,k] = sum_m grad_out[m,o] * x[m, idx[o,k]]      (reduce over batch)
  grad_x[m,c] = sum_{(o,k): idx=c} grad_out[m,o] * w[o,k] (SCATTER -> tl.atomic_add)

grad_x is the hard part: many outputs write to one input, so it's an atomic
scatter -- the transpose of the forward gather. The bar to clear is torch's
index_add_ (the current, already-fast backward). This benchmark checks correctness
against an index_add_ reference and times both.

Run:  python triton_gather_bwd_v1.py            # sweep B_M in {8,16,32}
"""
import argparse, statistics
import torch
import triton
import triton.language as tl


@triton.jit
def _bwd(grad_out_ptr, x_ptr, idx_ptr, w_ptr, grad_x_ptr, grad_w_ptr,
         M, O, IN, K: tl.constexpr, B_M: tl.constexpr, B_O: tl.constexpr):
    mb = tl.program_id(0)
    ob = tl.program_id(1)
    m_off = mb * B_M + tl.arange(0, B_M)
    o_off = ob * B_O + tl.arange(0, B_O)
    m_mask = m_off < M
    o_mask = o_off < O
    mo = m_mask[:, None] & o_mask[None, :]
    go = tl.load(grad_out_ptr + m_off[:, None] * O + o_off[None, :], mask=mo, other=0.0)  # [B_M,B_O]
    base_ok = o_off * K
    for k in tl.static_range(K):
        idx = tl.load(idx_ptr + base_ok + k, mask=o_mask, other=0)      # [B_O]
        wk = tl.load(w_ptr + base_ok + k, mask=o_mask, other=0.0)       # [B_O]
        xoff = m_off[:, None] * IN + idx[None, :]                        # [B_M,B_O]
        # grad_x scatter:  grad_x[m, idx] += grad_out[m,o] * w[o,k]
        tl.atomic_add(grad_x_ptr + xoff, go * wk[None, :], mask=mo)
        # grad_w reduce:   grad_w[o,k] += sum_m grad_out[m,o] * x[m,idx]
        xv = tl.load(x_ptr + xoff, mask=mo, other=0.0)                   # [B_M,B_O]
        tl.atomic_add(grad_w_ptr + base_ok + k, tl.sum(go * xv, axis=0), mask=o_mask)


def triton_backward(grad_out, x, idx, w, B_M=16, B_O=64):
    M, IN = x.shape
    O, K = idx.shape
    grad_x = torch.zeros(M, IN, device=x.device, dtype=torch.float32)
    grad_w = torch.zeros(O, K, device=x.device, dtype=torch.float32)
    _bwd[(triton.cdiv(M, B_M), triton.cdiv(O, B_O))](
        grad_out.contiguous(), x.contiguous(), idx.contiguous().to(torch.int64),
        w.contiguous(), grad_x, grad_w, M, O, IN, K, B_M, B_O)
    return grad_x, grad_w


def ref_backward(grad_out, x, idx, w):                # the current index_add_ path
    M, IN = x.shape
    O, K = idx.shape
    idx_flat = idx.reshape(-1)
    g = grad_out.unsqueeze(-1)
    grad_x = torch.zeros(M, IN, device=x.device)
    grad_x.index_add_(1, idx_flat, (g * w.unsqueeze(0)).reshape(M, O * K))
    sel = x.index_select(1, idx_flat).reshape(M, O, K)
    grad_w = (g * sel).sum(dim=0)
    return grad_x, grad_w


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=128)
    p.add_argument("--in_features", type=int, default=10000)
    p.add_argument("--O", type=int, default=90000)
    p.add_argument("--K", type=int, default=40)
    p.add_argument("--B_O", type=int, default=64)
    p.add_argument("--B_Ms", default="8,16,32")
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()

    dev = "cuda"
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"shape: M={args.M} IN={args.in_features} O={args.O} K={args.K} B_O={args.B_O} (fp32)\n")

    torch.manual_seed(0)
    x = torch.randn(args.M, args.in_features, device=dev)
    idx = torch.randint(0, args.in_features, (args.O, args.K), device=dev)
    w = torch.randn(args.O, args.K, device=dev)
    grad_out = torch.randn(args.M, args.O, device=dev)

    gx_ref, gw_ref = ref_backward(grad_out, x, idx, w)

    def bench(fn):
        for _ in range(5): fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(args.iters):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); fn(); e1.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        return statistics.median(ts)

    t_ref = bench(lambda: ref_backward(grad_out, x, idx, w))
    print(f"index_add_ backward (reference): {t_ref:.3f} ms\n")

    print(f"{'B_M':>5}  {'ms':>8}  {'vs index_add':>13}  {'gx ok':>6}  {'gw ok':>6}  "
          f"{'gx_err':>10}  {'gw_err':>10}")
    for B_M in [int(v) for v in args.B_Ms.split(",")]:
        try:
            gx, gw = triton_backward(grad_out, x, idx, w, B_M=B_M, B_O=args.B_O)
        except Exception as e:  # noqa: BLE001
            print(f"{B_M:>5}  {'ERR':>8}  {str(e)[:44]}")
            continue
        # scatter accumulates in varying order -> compare with a modest tolerance
        gx_ok = torch.allclose(gx, gx_ref, rtol=1e-3, atol=1e-3)
        gw_ok = torch.allclose(gw, gw_ref, rtol=1e-3, atol=1e-3)
        gx_err = (gx - gx_ref).abs().max().item()
        gw_err = (gw - gw_ref).abs().max().item()
        t = bench(lambda B_M=B_M: triton_backward(grad_out, x, idx, w, B_M=B_M, B_O=args.B_O))
        print(f"{B_M:>5}  {t:>8.3f}  {t_ref/t:>12.2f}x  "
              f"{'OK' if gx_ok else 'FAIL':>6}  {'OK' if gw_ok else 'FAIL':>6}  "
              f"{gx_err:>10.2e}  {gw_err:>10.2e}")

    print("\nread: correctness FIRST (both gx & gw OK). Then vs index_add: >1x means the")
    print("      fused atomic-scatter beats torch's scatter -> promising. <1x means atomic")
    print("      contention loses -> keep index_add_ backward, or go CUDA/smem for the scatter.")
