import torch
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)

torch.manual_seed(0)
allok = True
devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
for dev in devs:
    for dtype in [torch.float32, torch.float64]:
        m = IndexedSparseLinear(64, 128, 8, seed=1, output_chunk_size=32).to(dev, dtype)
        x = torch.randn(5, 64, device=dev, dtype=dtype).abs().requires_grad_(True)
        outs, gxs, gws = [], [], []
        for fused in (False, True):
            m.use_fused_gather = fused
            m.zero_grad(set_to_none=True)
            x.grad = None
            o = m(x)
            o.pow(2).sum().backward()
            outs.append(o.detach().clone())
            gxs.append(x.grad.clone())
            gws.append(m.pre_w.grad.clone())
        fd = (outs[0] - outs[1]).abs().max().item()
        xd = (gxs[0] - gxs[1]).abs().max().item()
        wd = (gws[0] - gws[1]).abs().max().item()
        tol = 1e-10 if dtype == torch.float64 else 1e-4
        ok = fd == 0 and xd < tol and wd < tol
        allok &= ok
        status = "OK" if ok else "FAIL"
        print(f"{dev:5} {str(dtype):16} fwd={fd:.2e} gx={xd:.2e} gw={wd:.2e}  {status}")
print("PASS" if allok else "FAIL")
