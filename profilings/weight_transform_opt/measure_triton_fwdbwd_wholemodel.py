"""Step (b): whole-model impact of the Triton FORWARD + BACKWARD gather kernels.

Follow-up to measure_triton_wholemodel.py (forward only, gave +12-14% whole-step).
Here the wrapper autograd.Function uses Triton for BOTH directions:
  * forward : _fwd  (v2 batch-tiled gather)         -- proven +12-14% whole-step
  * backward: _bwd  (fused atomic-scatter, one pass) -- proven 1.24-1.35x vs index_add_

Three A/B legs, all fp32, no autocast, apples-to-apples on the real model:
  1. correctness: full-model forward AND a full backward (grad match) vs the
     current index_select/index_add_ path
  2. whole-step timing: current fused path  vs  triton fwd-only  vs  triton fwd+bwd

Run:  python measure_triton_fwdbwd_wholemodel.py --B_M 16 --B_O 64
"""
import os, sys, argparse, statistics
REPO = "/n/home12/rwhalen/DendriNet"
os.chdir(REPO); sys.path.insert(0, os.path.join(REPO, "src"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import triton
import triton.language as tl

from dendritic_modeling.config import load_config
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
import dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse as ixs
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)

CONFIG = "configs/examples/sglimit5_h100_cifar10_ei_1layer_bf33_10000e1250i.yaml"
dev = "cuda"
assert torch.cuda.is_available()

p = argparse.ArgumentParser()
p.add_argument("--B_M", type=int, default=16)          # bwd sweet spot on both GPUs
p.add_argument("--B_O", type=int, default=64)
p.add_argument("--B_M_fwd", type=int, default=0)        # 0 -> reuse --B_M for the forward too
p.add_argument("--batch", type=int, default=128)
p.add_argument("--iters", type=int, default=20)
args = p.parse_args()
B_M_FWD = args.B_M_fwd or args.B_M
print(f"GPU: {torch.cuda.get_device_name(0)}  B_M(bwd)={args.B_M} "
      f"B_M(fwd)={B_M_FWD} B_O={args.B_O} batch={args.batch} (fp32)\n")


# ---- forward kernel (v2 batch-tiled gather) ----------------------------------
@triton.jit
def _fwd(x_ptr, idx_ptr, w_ptr, out_ptr, M, O, IN,
         K: tl.constexpr, B_M: tl.constexpr, B_O: tl.constexpr):
    mb = tl.program_id(0); ob = tl.program_id(1)
    m_off = mb * B_M + tl.arange(0, B_M)
    o_off = ob * B_O + tl.arange(0, B_O)
    m_mask = m_off < M; o_mask = o_off < O
    base_ok = o_off * K
    acc = tl.zeros([B_M, B_O], dtype=tl.float32)
    for k in tl.static_range(K):
        idx = tl.load(idx_ptr + base_ok + k, mask=o_mask, other=0)
        wk = tl.load(w_ptr + base_ok + k, mask=o_mask, other=0.0)
        x_off = m_off[:, None] * IN + idx[None, :]
        xmask = m_mask[:, None] & o_mask[None, :]
        xv = tl.load(x_ptr + x_off, mask=xmask, other=0.0)
        acc += xv * wk[None, :]
    out_off = m_off[:, None] * O + o_off[None, :]
    tl.store(out_ptr + out_off, acc, mask=(m_mask[:, None] & o_mask[None, :]))


# ---- backward kernel (fused atomic-scatter, one pass) ------------------------
@triton.jit
def _bwd(grad_out_ptr, x_ptr, idx_ptr, w_ptr, grad_x_ptr, grad_w_ptr,
         M, O, IN, K: tl.constexpr, B_M: tl.constexpr, B_O: tl.constexpr):
    mb = tl.program_id(0); ob = tl.program_id(1)
    m_off = mb * B_M + tl.arange(0, B_M)
    o_off = ob * B_O + tl.arange(0, B_O)
    m_mask = m_off < M; o_mask = o_off < O
    mo = m_mask[:, None] & o_mask[None, :]
    go = tl.load(grad_out_ptr + m_off[:, None] * O + o_off[None, :], mask=mo, other=0.0)
    base_ok = o_off * K
    for k in tl.static_range(K):
        idx = tl.load(idx_ptr + base_ok + k, mask=o_mask, other=0)
        wk = tl.load(w_ptr + base_ok + k, mask=o_mask, other=0.0)
        xoff = m_off[:, None] * IN + idx[None, :]
        tl.atomic_add(grad_x_ptr + xoff, go * wk[None, :], mask=mo)
        xv = tl.load(x_ptr + xoff, mask=mo, other=0.0)
        tl.atomic_add(grad_w_ptr + base_ok + k, tl.sum(go * xv, axis=0), mask=o_mask)


class TritonGatherFwdBwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, conn_idx, weight, B_M_fwd, B_M_bwd, B_O):
        x = x.contiguous(); idx = conn_idx.contiguous().to(torch.int64); w = weight.contiguous()
        M, IN = x.shape; O, K = idx.shape
        out = torch.empty(M, O, device=x.device, dtype=x.dtype)
        _fwd[(triton.cdiv(M, B_M_fwd), triton.cdiv(O, B_O))](x, idx, w, out, M, O, IN, K, B_M_fwd, B_O)
        ctx.save_for_backward(x, idx, w)
        ctx.cfg = (M, O, IN, K, B_M_bwd, B_O)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, idx, w = ctx.saved_tensors
        M, O, IN, K, B_M, B_O = ctx.cfg
        grad_out = grad_out.contiguous()
        needs_x, _, needs_w = ctx.needs_input_grad[:3]
        grad_x = torch.zeros(M, IN, device=x.device, dtype=torch.float32)
        grad_w = torch.zeros(O, K, device=x.device, dtype=torch.float32)
        _bwd[(triton.cdiv(M, B_M), triton.cdiv(O, B_O))](
            grad_out, x, idx, w, grad_x, grad_w, M, O, IN, K, B_M, B_O)
        return (grad_x if needs_x else None), None, (grad_w if needs_w else None), None, None, None


# forward-only wrapper (index_add_ backward) for the middle A/B leg
class TritonGatherFwdOnly(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, conn_idx, weight, B_M_fwd, B_O):
        x = x.contiguous(); idx = conn_idx.contiguous().to(torch.int64); w = weight.contiguous()
        M, IN = x.shape; O, K = idx.shape
        out = torch.empty(M, O, device=x.device, dtype=x.dtype)
        _fwd[(triton.cdiv(M, B_M_fwd), triton.cdiv(O, B_O))](x, idx, w, out, M, O, IN, K, B_M_fwd, B_O)
        ctx.save_for_backward(x, idx, w); ctx.O = O; ctx.K = K
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, idx, w = ctx.saved_tensors; O, K = ctx.O, ctx.K; M = x.shape[0]
        idx_flat = idx.reshape(-1)
        needs_x, _, needs_w = ctx.needs_input_grad[:3]
        grad_x = torch.zeros_like(x) if needs_x else None
        grad_w = torch.empty_like(w) if needs_w else None
        g = grad_out.unsqueeze(-1)
        if needs_w:
            sel = x.index_select(1, idx_flat).reshape(M, O, K)
            grad_w = (g * sel).sum(dim=0)
        if needs_x:
            gs = (g * w.unsqueeze(0)).reshape(M, O * K)
            grad_x.index_add_(1, idx_flat, gs)
        return grad_x, None, grad_w, None, None


def triton_fwdbwd(x, connection_indices, weight, chunk_size=2048):
    return TritonGatherFwdBwd.apply(x, connection_indices, weight, B_M_FWD, args.B_M, args.B_O)

def triton_fwdonly(x, connection_indices, weight, chunk_size=2048):
    return TritonGatherFwdOnly.apply(x, connection_indices, weight, B_M_FWD, args.B_O)


# ---- build model, enable the (existing) fused path on all indexed synapses ----
cfg = load_config(CONFIG)
model, _ = initialize_model(cfg.model); model = model.to(dev)
n = 0
for m in model.modules():
    if isinstance(m, IndexedSparseLinear):
        m.use_fused_gather = True; n += 1
print(f"use_fused_gather=True on {n} indexed synapses")

x = torch.randn(args.batch, 3072, device=dev)
y = torch.randint(0, 10, (args.batch,), device=dev)
ce = nn.CrossEntropyLoss()
ORIG = ixs.fused_sparse_gather                      # index_select fwd + index_add_ bwd


# ---- 1. correctness: full-model forward AND backward (grad match) ------------
def full_grads(which):
    ixs.fused_sparse_gather = which
    model.zero_grad(set_to_none=True)
    out = model(x)
    loss = ce(out, y)
    loss.backward()
    grads = {name: pr.grad.detach().clone() for name, pr in model.named_parameters()
             if pr.grad is not None}
    return out.detach(), grads

o_ref, g_ref = full_grads(ORIG)
o_tri, g_tri = full_grads(triton_fwdbwd)
fwd_ok = torch.allclose(o_ref, o_tri, rtol=1e-3, atol=1e-4)
fwd_err = (o_ref - o_tri).abs().max().item()
# grad match across all params
gmax, gname, all_ok = 0.0, "", True
for name in g_ref:
    if name not in g_tri:
        all_ok = False; continue
    e = (g_ref[name] - g_tri[name]).abs().max().item()
    if e > gmax: gmax, gname = e, name
    if not torch.allclose(g_ref[name], g_tri[name], rtol=1e-2, atol=1e-3):
        all_ok = False
print(f"full-model forward match : {'OK' if fwd_ok else 'FAIL'}  max_err={fwd_err:.2e}")
print(f"full-model backward match: {'OK' if all_ok else 'FAIL'}  "
      f"max_grad_err={gmax:.2e} (@ {gname})\n")


# ---- 2. whole-step timing: orig  vs  fwd-only  vs  fwd+bwd --------------------
def step(opt):
    opt.zero_grad(set_to_none=True)
    loss = ce(model(x), y)
    loss.backward(); opt.step()

def timed(which):
    ixs.fused_sparse_gather = which
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(5): step(opt)
    torch.cuda.synchronize()
    ts = []
    for _ in range(args.iters):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); step(opt); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)

t_orig = timed(ORIG)
t_fwd = timed(triton_fwdonly)
t_both = timed(triton_fwdbwd)
print(f"whole step/iter:")
print(f"  orig (index_select + index_add_) : {t_orig:.1f} ms   (baseline)")
print(f"  triton fwd  +  index_add_ bwd    : {t_fwd:.1f} ms   "
      f"{t_orig/t_fwd:.3f}x  ({100*(t_orig-t_fwd)/t_orig:+.1f}%)")
print(f"  triton fwd  +  triton bwd        : {t_both:.1f} ms   "
      f"{t_orig/t_both:.3f}x  ({100*(t_orig-t_both)/t_orig:+.1f}%)")
print(f"\ndelta fwd->fwd+bwd: {t_fwd - t_both:+.1f} ms/step "
      f"({100*(t_fwd-t_both)/t_orig:+.1f}% of baseline from adding the triton backward)")
print("\nread: the fwd+bwd row is the full prize. If it clears the forward-only row by a")
print("      meaningful margin, productionize BOTH kernels in one autograd.Function. If the")
print("      backward barely moves the whole step, keep index_add_ and ship forward only.")
