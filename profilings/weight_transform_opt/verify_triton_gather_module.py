"""Correctness gate for the PRODUCTION module (src/.../synapse/triton_gather.py).

Validates the real triton_sparse_gather() we'd ship -- not the microbench copies --
against the existing fused_sparse_gather baseline, in fp32 AND under bf16 autocast
(the dtype-safety path that's new to the production wrapper). Checks:
  * forward output match
  * grad_x and grad_w match (via autograd, exactly how training will call it)
  * dtype of outputs/grads is preserved (bf16 in -> bf16 out/grads)

Run:  python verify_triton_gather_module.py
"""
import os, sys
REPO = "/n/home12/rwhalen/DendriNet"
sys.path.insert(0, os.path.join(REPO, "src"))

import torch
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.fused_gather import (
    fused_sparse_gather,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.triton_gather import (
    triton_sparse_gather, HAS_TRITON,
)

assert torch.cuda.is_available(), "need CUDA"
assert HAS_TRITON, "Triton not available in this env"
dev = "cuda"
print(f"GPU: {torch.cuda.get_device_name(0)}   triton available: {HAS_TRITON}\n")

M, IN, O, K = 128, 10000, 90000, 40
torch.manual_seed(0)
idx = torch.randint(0, IN, (O, K), device=dev)


def run(fn, x0, w0):
    """Forward+backward through `fn`; return (out, grad_x, grad_w), dtypes preserved.

    Inputs are passed in their real dtypes DIRECTLY -- we do NOT rely on
    torch.autocast, which does not cast the inputs of a custom autograd.Function
    (it only casts registered ops). To exercise the bf16 path we hand the synapse a
    genuine bf16 activation, exactly as an upstream autocast op would.
    """
    x = x0.clone().detach().requires_grad_(True)
    w = w0.clone().detach().requires_grad_(True)
    out = fn(x, idx, w)
    (out.float() * out.float()).sum().backward()   # loss in fp32 so bf16 grad is well-defined
    return out.detach(), x.grad.detach(), w.grad.detach()


def eager_ref(x0, w0):
    """Plain fp32 eager gather-mul-sum reference (works for any input dtype).

    Used for the bf16 row: the fused baseline CANNOT run on bf16-x + fp32-w (its
    index_add_ throws on the dtype mismatch), so we reference against fp32 eager math
    instead -- which also documents that the Triton path is strictly more robust.
    """
    def fn(x, idx_, w):
        xf = x.float(); wf = w.float()
        sel = xf[:, idx_.reshape(-1)].reshape(x.shape[0], O, K)
        return ((sel * wf.unsqueeze(0)).sum(-1)).to(x.dtype)
    return run(fn, x0, w0)


def compare(tag, x_dtype, w_dtype, rtol, atol, ref="fused"):
    # autocast state = bf16 activations, fp32 params -> pass x as bf16, w as fp32
    x0 = torch.randn(M, IN, device=dev).to(x_dtype)
    w0 = torch.randn(O, K, device=dev).to(w_dtype)
    o_ref, gx_ref, gw_ref = (run(fused_sparse_gather, x0, w0) if ref == "fused"
                             else eager_ref(x0, w0))
    o_tri, gx_tri, gw_tri = run(triton_sparse_gather, x0, w0)

    def chk(a, b):
        return (torch.allclose(a, b, rtol=rtol, atol=atol),
                (a.float() - b.float()).abs().max().item(), a.dtype == b.dtype)

    print(f"[{tag}]")
    for name, ref, tri in (("out", o_ref, o_tri), ("grad_x", gx_ref, gx_tri),
                           ("grad_w", gw_ref, gw_tri)):
        ok, err, dtype_ok = chk(ref, tri)
        print(f"  {name:>7}: {'OK' if ok else 'FAIL':>4}  max_err={err:.2e}  "
              f"dtype ref={ref.dtype} tri={tri.dtype} {'ok' if dtype_ok else 'MISMATCH'}")
    print()


# fp32 vs the fused baseline: this IS the AMP-safety proof -- the synapse only ever
# sees fp32 under this model's autocast (confirmed: check_synapse_input_dtype.py).
compare("fp32 (vs fused baseline)", torch.float32, torch.float32, rtol=1e-3, atol=1e-3)
# bf16 robustness bonus: the fused baseline CANNOT run this (index_add_ dtype crash),
# so reference against fp32 eager. Proves the Triton path is strictly more robust.
compare("bf16 x / fp32 w (vs eager, robustness bonus)",
        torch.bfloat16, torch.float32, rtol=3e-2, atol=3e-2, ref="eager")

print("read: fp32 row is the real gate (out/grad_w near-exact, grad_x within ~1e-3 atomic).")
print("      bf16 row is a bonus: out & grad_x must be bf16, grad_w fp32 (dtypes preserved),")
print("      all within bf16 precision -- the baseline can't even run this case.")
