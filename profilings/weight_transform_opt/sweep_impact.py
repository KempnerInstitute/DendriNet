"""Phase 2 - IMPACT: for every config whose model contains IndexedSparseLinear layers, verify the
kernels are equivalence-preserving and show their effect.

For each config: build the model, take every DISTINCT (in_features, out_features, K) sparse-layer
shape it contains, and on each shape run the layer forward+backward with the flags OFF (original
loop) vs ON (cuda->triton->fused), on a fixed synthetic batch. Report the worst |off-on| across
output + both gradients (equivalence), and time OFF vs ON on the largest layer (effect). This
exercises the exact kernel code paths that run inside these models, with no dataset dependency."""
import os, sys, glob
REPO = "/n/home12/rwhalen/DendriNet"
os.chdir(REPO); sys.path.insert(0, os.path.join(REPO, "src"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from dendritic_modeling.config import load_config
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear)

DEV = "cuda"; B = 128
OFF = (False, False, False); ON = (True, True, True)


def fwd_bwd(m, x):
    xx = x.clone().requires_grad_(True)
    if m.pre_w.grad is not None: m.pre_w.grad = None
    out = m(xx)
    out.sum().backward()
    return out.detach(), xx.grad.detach(), m.pre_w.grad.detach().clone()


def set_flags(m, f): m.use_fused_gather, m.use_triton_gather, m.use_cuda_gather = f


def time_layer(m, x, f, iters=30):
    set_flags(m, f)
    for _ in range(5):                              # warmup
        xx = x.clone().requires_grad_(True); m.pre_w.grad = None
        m(xx).sum().backward()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        xx = x.clone().requires_grad_(True); m.pre_w.grad = None
        m(xx).sum().backward()
    t1.record(); torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


def test_config(path):
    cfg = load_config(path)
    model, _ = initialize_model(cfg.model)
    layers = [m for m in model.modules() if isinstance(m, IndexedSparseLinear)]
    if not layers:
        return {"n": 0}
    uniq = {}
    for m in layers:
        uniq.setdefault((m.in_features, m.out_features, m.K), m)
    worst_o = worst_gx = worst_gw = 0.0
    for (inf, outf, K), m in uniq.items():
        m.to(DEV)
        x = torch.randn(B, inf, device=DEV)
        set_flags(m, OFF); o0, gx0, gw0 = fwd_bwd(m, x)
        set_flags(m, ON);  o1, gx1, gw1 = fwd_bwd(m, x)
        worst_o = max(worst_o, (o0 - o1).abs().max().item())
        worst_gx = max(worst_gx, (gx0 - gx1).abs().max().item())
        worst_gw = max(worst_gw, (gw0 - gw1).abs().max().item())
    big = max(uniq.values(), key=lambda m: m.in_features * m.out_features)
    big.to(DEV); xb = torch.randn(B, big.in_features, device=DEV)
    ms_off = time_layer(big, xb, OFF); ms_on = time_layer(big, xb, ON)
    del model; torch.cuda.empty_cache()
    return {"n": len(layers), "shapes": len(uniq), "do": worst_o, "dgx": worst_gx, "dgw": worst_gw,
            "big": (big.in_features, big.out_features, big.K), "off": ms_off, "on": ms_on}


def main():
    assert torch.cuda.is_available()
    print(f"gpu: {torch.cuda.get_device_name(0)}  batch={B}\n")
    hdr = (f"{'config':<50}{'#L':>4}{'shp':>4}  {'max|Δout|':>10}{'max|Δgx|':>10}{'max|Δgw|':>10}   "
           f"{'biggest layer':>18}{'off ms':>9}{'on ms':>8}{'x':>6}")
    print(hdr); print("-" * len(hdr))
    for path in sorted(glob.glob("configs/examples/*.yaml")):
        name = os.path.basename(path)
        try:
            r = test_config(path)
        except Exception as e:
            print(f"{name:<50}  BUILD/RUN FAIL: {str(e).splitlines()[-1][:60]}"); continue
        if r["n"] == 0:
            print(f"{name:<50}{0:>4}   (no IndexedSparseLinear - kernels do not apply)"); continue
        big = f"{r['big'][0]}x{r['big'][1]}k{r['big'][2]}"
        sp = r["off"] / r["on"] if r["on"] > 0 else 0
        print(f"{name:<50}{r['n']:>4}{r['shapes']:>4}  {r['do']:>10.2e}{r['dgx']:>10.2e}{r['dgw']:>10.2e}   "
              f"{big:>18}{r['off']:>9.3f}{r['on']:>8.3f}{sp:>5.1f}x")


if __name__ == "__main__":
    main()
