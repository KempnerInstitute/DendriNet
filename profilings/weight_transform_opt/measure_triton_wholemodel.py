"""Step (a): whole-model impact of the Triton FORWARD gather kernel.

Wires the Triton forward (v2, batch-tiled) into the real model via a wrapper
autograd.Function [Triton forward + the existing index_add_ backward], WITHOUT
touching production source, then:
  1. correctness: full-model forward output, current-fused vs Triton, must match
  2. whole-step timing A/B: current fused path vs Triton-forward path

fp32, no autocast, for a clean apples-to-apples signal. bf16/AMP integration is a
follow-up once we decide the whole-model % justifies it.

Run:  python measure_triton_wholemodel.py --B_M 16 --B_O 64
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
p.add_argument("--B_M", type=int, default=16)
p.add_argument("--B_O", type=int, default=64)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--iters", type=int, default=20)
p.add_argument("--bwd_chunk", type=int, default=8192)
args = p.parse_args()
print(f"GPU: {torch.cuda.get_device_name(0)}  B_M={args.B_M} B_O={args.B_O} "
      f"batch={args.batch} (fp32)\n")


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


class TritonGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, conn_idx, weight, B_M, B_O, bwd_chunk):
        x = x.contiguous(); idx = conn_idx.contiguous().to(torch.int64); w = weight.contiguous()
        M, IN = x.shape; O, K = idx.shape
        out = torch.empty(M, O, device=x.device, dtype=x.dtype)
        _fwd[(triton.cdiv(M, B_M), triton.cdiv(O, B_O))](x, idx, w, out, M, O, IN, K, B_M, B_O)
        ctx.save_for_backward(x, idx, w)
        ctx.cfg = (O, K, bwd_chunk)
        return out

    @staticmethod
    def backward(ctx, grad_out):                    # index_add_ backward (the proven one)
        x, idx, w = ctx.saved_tensors
        O, K, chunk = ctx.cfg
        M = x.shape[0]
        idx_flat = idx.reshape(-1)
        needs_x, _, needs_w = ctx.needs_input_grad[:3]
        grad_x = torch.zeros_like(x) if needs_x else None
        grad_w = torch.empty_like(w) if needs_w else None
        for start in range(0, O, chunk):
            end = min(start + chunk, O)
            idxc = idx_flat[start * K:end * K]
            g = grad_out[:, start:end].unsqueeze(-1)
            if needs_w:
                sel = x.index_select(1, idxc).reshape(M, end - start, K)
                grad_w[start:end] = (g * sel).sum(dim=0)
            if needs_x:
                gs = (g * w[start:end].unsqueeze(0)).reshape(M, (end - start) * K)
                grad_x.index_add_(1, idxc, gs)
        return grad_x, None, grad_w, None, None, None


def triton_fused_gather(x, connection_indices, weight, chunk_size=2048):
    return TritonGather.apply(x, connection_indices, weight, args.B_M, args.B_O, args.bwd_chunk)


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
ORIG = ixs.fused_sparse_gather                      # index_select forward

# ---- 1. correctness: full-model forward, current-fused vs Triton-forward ------
ixs.fused_sparse_gather = ORIG
o_ref = model(x)
ixs.fused_sparse_gather = triton_fused_gather
o_tri = model(x)
err = (o_ref - o_tri).abs().max().item()
print(f"full-model forward match: {'OK' if torch.allclose(o_ref, o_tri, rtol=1e-3, atol=1e-4) else 'FAIL'}"
      f"  max_err={err:.2e}\n")

# ---- 2. whole-step timing A/B -------------------------------------------------
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

t_fused = timed(ORIG)
t_triton = timed(triton_fused_gather)
print(f"whole step/iter  fused(index_select)={t_fused:.1f} ms   "
      f"triton_fwd={t_triton:.1f} ms   speedup={t_fused/t_triton:.3f}x")
print(f"delta: {t_fused - t_triton:+.1f} ms/step "
      f"({100*(t_fused-t_triton)/t_fused:+.1f}% whole-step)")
print("\nread: this % is the REAL prize of the Triton forward. If it's a few %, the")
print("      forward win is marginal end-to-end (backward/other ops dominate) -> the")
print("      bigger lever is the Triton backward next.")
