"""Phase 1 - DISCOVERY: which configs in configs/examples/ do our gather kernels actually apply to?

Builds every config's model and introspects it for IndexedSparseLinear layers (the layer whose
use_fused_gather/use_triton_gather/use_cuda_gather flags our kernels control). A config is
"impacted" iff building it yields >=1 such layer. Also reports, per layer, the gather input width
(in_features) and whether the CUDA warp-per-output path engages (guard: in_features*4 <= optin_shared).
No data needed -- this is purely structural. Build failures are reported (also informative)."""
import os, sys, glob, yaml
REPO = "/n/home12/rwhalen/DendriNet"
os.chdir(REPO); sys.path.insert(0, os.path.join(REPO, "src"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from dendritic_modeling.config import load_config
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear)

optin = None
try:
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import cuda_gather
    if hasattr(cuda_gather, "_optin_shared"):
        optin = int(cuda_gather._optin_shared())
except Exception as e:
    print("optin query failed:", str(e).splitlines()[0])
if optin is None:
    optin = 99 * 1024  # RTX PRO 6000 Blackwell fallback
print(f"optin_shared = {optin} bytes  -> CUDA wpo path engages when in_features <= {optin // 4}\n")

configs = sorted(glob.glob("configs/examples/*.yaml"))
hdr = f"{'config':<54}{'dataset':<28}{'#sparse':>8}{'in_features':>16}{'CUDA':>7}"
print(hdr); print("-" * len(hdr))
rows = []
for path in configs:
    name = os.path.basename(path)
    raw = yaml.safe_load(open(path)) or {}
    ds = (raw.get("data", {}) or {}).get("dataset_name", "-")
    try:
        cfg = load_config(path)
        model, _ = initialize_model(cfg.model)
        sparse = [m for m in model.modules() if isinstance(m, IndexedSparseLinear)]
        if not sparse:
            print(f"{name:<54}{str(ds):<28}{0:>8}{'-':>16}{'-':>7}   (no IndexedSparseLinear)")
            rows.append((name, ds, 0, None, None)); continue
        ins = sorted({m.in_features for m in sparse})
        rng = f"{min(ins)}..{max(ins)}" if len(ins) > 1 else f"{ins[0]}"
        cuda_all = all(i * 4 <= optin for i in ins)
        cuda_any = any(i * 4 <= optin for i in ins)
        tag = "all" if cuda_all else ("some" if cuda_any else "none")
        print(f"{name:<54}{str(ds):<28}{len(sparse):>8}{rng:>16}{tag:>7}")
        rows.append((name, ds, len(sparse), rng, tag))
        del model
    except Exception as e:
        msg = str(e).splitlines()[-1][:44]
        print(f"{name:<54}{str(ds):<28}{'BUILD FAIL':>8}{'':>16}{'':>7}   {msg}")
        rows.append((name, ds, "FAIL", None, None))

n_ok = sum(1 for r in rows if isinstance(r[2], int) and r[2] > 0)
print(f"\nsummary: {n_ok}/{len(rows)} configs contain IndexedSparseLinear layers (our kernels apply)")
