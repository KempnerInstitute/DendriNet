"""Isolated microbenchmark for the indexed-sparse gather kernel (the 84% hotspot).

Answers, for ONE op in a controlled environment:
  (1) is there room?  -> achieved bandwidth vs empirical PEAK, and current-vs-optimized race
  (2) what metrics    -> latency (fwd/bwd separately), bytes moved, GB/s, peak memory
  (3) which matters   -> this op is ~0 FLOP -> MEMORY-BOUND -> achieved GB/s / peak is the number
  (4) an optimization -> custom autograd.Function with explicit index_add_ backward (Kernel B)

No model, no data, no autocast noise -- just the op, at realistic shapes.

Usage:
  python bench_kernel.py                       # default shape
  python bench_kernel.py --B 128 --out 20000 --K 40   # scale to taste (watch memory)
"""
import argparse, statistics
import torch

Event = torch.cuda.Event

p = argparse.ArgumentParser()
p.add_argument("--B", type=int, default=64, help="batch")
p.add_argument("--in_features", type=int, default=3072)
p.add_argument("--out", type=int, default=10000, help="output neurons*branches")
p.add_argument("--K", type=int, default=40, help="synapses per output (fan-in)")
p.add_argument("--iters", type=int, default=30)
p.add_argument("--warmup", type=int, default=8)
args = p.parse_args()

assert torch.cuda.is_available()
dev = "cuda"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"shape: B={args.B} in={args.in_features} out={args.out} K={args.K} (fp32)\n")

# ------------------------------------------------------------------ inputs
def make_inputs():
    torch.manual_seed(0)
    x = torch.randn(args.B, args.in_features, device=dev, requires_grad=True)
    idx = torch.randint(0, args.in_features, (args.out, args.K), device=dev)
    w = torch.randn(args.out, args.K, device=dev, requires_grad=True)
    return x, idx, w

# ------------------------------------------------- Kernel A: current impl
# mirrors indexed_sparse.py:267-270 (advanced-index gather -> mul -> sum);
# backward is torch's generic indexing_backward_kernel (the 84% cost)
def kernel_A(x, idx, w):
    B = x.shape[0]
    selected = x[:, idx.reshape(-1)].reshape(B, args.out, args.K)
    return (selected * w.unsqueeze(0)).sum(dim=-1)

# ----------------------------------- Kernel B: custom autograd optimization
# identical math; backward for grad_x uses an explicit index_add_ on a
# pre-zeroed buffer instead of the generic advanced-indexing scatter path.
class SparseGatherMulSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx, w):
        B, IN = x.shape
        OUT, K = idx.shape
        idx_flat = idx.reshape(-1)
        selected = x.index_select(1, idx_flat).reshape(B, OUT, K)
        out = (selected * w.unsqueeze(0)).sum(dim=-1)
        ctx.save_for_backward(idx_flat, w, selected)
        ctx.dims = (B, IN, OUT, K)
        return out

    @staticmethod
    def backward(ctx, grad_out):                 # grad_out: [B, OUT]
        idx_flat, w, selected = ctx.saved_tensors
        B, IN, OUT, K = ctx.dims
        g = grad_out.unsqueeze(-1)               # [B, OUT, 1]
        grad_w = (g * selected).sum(dim=0)       # [OUT, K]
        grad_selected = (g * w.unsqueeze(0)).reshape(B, OUT * K)
        grad_x = torch.zeros(B, IN, device=grad_out.device, dtype=grad_out.dtype)
        grad_x.index_add_(1, idx_flat, grad_selected)   # explicit scatter-accumulate
        return grad_x, None, grad_w

def kernel_B(x, idx, w):
    return SparseGatherMulSum.apply(x, idx, w)

# ---------------------------------------------- empirical peak HBM bandwidth
def measure_peak_bw(nbytes=256 * 1024 * 1024):
    n = nbytes // 4
    a, b = torch.empty(n, device=dev), torch.empty(n, device=dev)
    for _ in range(5): b.copy_(a)
    torch.cuda.synchronize()
    e0, e1 = Event(True), Event(True); e0.record()
    for _ in range(30): b.copy_(a)
    e1.record(); torch.cuda.synchronize()
    t_s = e0.elapsed_time(e1) / 30 / 1e3
    return 2 * nbytes / t_s / 1e9                # read+write, GB/s

# ------------------------------------------------------------ timing helper
def bench(fn):
    x, idx, w = make_inputs()
    # correctness reference against pure-autograd A is checked by caller
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
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    return statistics.median(fwd), statistics.median(bwd), peak_gb

# ------------------------------------------------------------ correctness
def check_equiv():
    xa, idx, wa = make_inputs()
    xb = xa.detach().clone().requires_grad_(True)
    wb = wa.detach().clone().requires_grad_(True)
    oa = kernel_A(xa, idx, wa); oa.sum().backward()
    ob = kernel_B(xb, idx, wb); ob.sum().backward()
    ok = (torch.allclose(oa, ob, rtol=1e-4, atol=1e-5) and
          torch.allclose(xa.grad, xb.grad, rtol=1e-4, atol=1e-5) and
          torch.allclose(wa.grad, wb.grad, rtol=1e-4, atol=1e-5))
    print(f"correctness A vs B (out+grad_x+grad_w): {'OK' if ok else 'FAIL'}")
    return ok

# ------------------------------------------------------------ run
peak_bw = measure_peak_bw()
print(f"empirical peak HBM bandwidth (memcpy): {peak_bw:.0f} GB/s\n")

if not check_equiv():
    print("!! Kernel B is not numerically equivalent -- fix before trusting timings.")

# effective memory traffic per forward: materialize `selected` (write) then read it
# for mul+sum. dominant term ~= 2 * B*OUT*K * 4 bytes.
bytes_fwd = 2 * args.B * args.out * args.K * 4
bytes_bwd = 3 * args.B * args.out * args.K * 4   # grad_selected write + read + scatter

print(f"\n{'kernel':<10}{'fwd ms':>10}{'bwd ms':>10}{'fwd GB/s':>11}{'%peak':>8}"
      f"{'bwd GB/s':>11}{'%peak':>8}{'peak GB':>10}")
for name, fn in [("A current", kernel_A), ("B index_add", kernel_B)]:
    f_ms, b_ms, pk = bench(fn)
    f_bw = bytes_fwd / (f_ms / 1e3) / 1e9
    b_bw = bytes_bwd / (b_ms / 1e3) / 1e9
    print(f"{name:<10}{f_ms:>10.3f}{b_ms:>10.3f}{f_bw:>11.0f}{100*f_bw/peak_bw:>7.0f}%"
          f"{b_bw:>11.0f}{100*b_bw/peak_bw:>7.0f}%{pk:>10.2f}")

print("\nread me:")
print("  * backward dominates -> that's your 84%. Compare A.bwd vs B.bwd for the win.")
print("  * %peak on backward = how close to the memory ceiling. <20% => lots of room;")
print("    >70% => near the wall, stop optimizing this op.")
print("  * if B.bwd %peak is still low, the index_add_ path is ALSO memory-bound/serialized")
print("    -> next lever is reducing traffic (don't materialize `selected`) or a fused kernel.")
