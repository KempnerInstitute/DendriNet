"""Step (b) final gate: TRAINING-equivalence of the Triton path vs the fused path.

One-step grad match (measure_triton_fwdbwd_wholemodel.py) already passed at ~2e-8.
This is the stronger test: from an identical initial state, run K optimizer steps
twice -- once on the fused path, once on the Triton path -- under bf16 autocast (the
real training regime) and compare:
  * the per-step loss trajectory   (max abs difference across all K steps)
  * the final parameters           (max abs difference across every tensor)

Both runs use the same seed and the same fixed batch each step, so any residual
stochasticity is shared and differences isolate the gather kernel. Divergence should
stay at fp32-accumulation / atomic-reorder scale (~1e-5..1e-3), NOT grow into a
different trajectory.

Run:  python train_equivalence_triton.py --steps 50
"""
import os, sys, argparse, copy
REPO = "/n/home12/rwhalen/DendriNet"
os.chdir(REPO); sys.path.insert(0, os.path.join(REPO, "src"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn

from dendritic_modeling.config import load_config
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)

CONFIG = "configs/examples/sglimit5_h100_cifar10_ei_1layer_bf33_10000e1250i.yaml"
dev = "cuda"
assert torch.cuda.is_available()

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=50)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--amp", action="store_true", default=True)
p.add_argument("--no-amp", dest="amp", action="store_false")
args = p.parse_args()
print(f"GPU: {torch.cuda.get_device_name(0)}  steps={args.steps} batch={args.batch} "
      f"amp(bf16)={args.amp}\n")


def set_gather(model, which):
    """which in {'fused','triton'} -> flip the flags on every indexed synapse."""
    n = 0
    for m in model.modules():
        if isinstance(m, IndexedSparseLinear):
            m.use_fused_gather = (which == "fused")
            m.use_triton_gather = (which == "triton")
            n += 1
    return n


# fixed init state, shared by both runs
cfg = load_config(CONFIG)
model, _ = initialize_model(cfg.model); model = model.to(dev)
init_state = copy.deepcopy(model.state_dict())

# fixed batch, shared by both runs
torch.manual_seed(0)
x = torch.randn(args.batch, 3072, device=dev)
y = torch.randint(0, 10, (args.batch,), device=dev)
ce = nn.CrossEntropyLoss()


def train_run(which):
    model.load_state_dict(init_state)
    n = set_gather(model, which)
    torch.manual_seed(1234)                         # identical stochasticity across runs
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    losses = []
    for _ in range(args.steps):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
            loss = ce(model(x), y)
        loss.backward(); opt.step()
        losses.append(loss.item())
    final = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return losses, final, n


l_fused, p_fused, n = train_run("fused")
l_tri, p_tri, _ = train_run("triton")
print(f"toggled {n} indexed synapses per run\n")

# loss trajectory divergence
dloss = max(abs(a - b) for a, b in zip(l_fused, l_tri))
print(f"loss[0]  fused={l_fused[0]:.6f}  triton={l_tri[0]:.6f}")
print(f"loss[-1] fused={l_fused[-1]:.6f}  triton={l_tri[-1]:.6f}")
print(f"max |loss_fused - loss_triton| over {args.steps} steps: {dloss:.3e}\n")

# final-parameter divergence
pmax, pname = 0.0, ""
for k in p_fused:
    if not torch.is_floating_point(p_fused[k]):
        continue
    e = (p_fused[k].float() - p_tri[k].float()).abs().max().item()
    if e > pmax:
        pmax, pname = e, k
print(f"max final-param |fused - triton|: {pmax:.3e}  (@ {pname})")

ok = dloss < 1e-2 and pmax < 1e-2
print(f"\nVERDICT: {'EQUIVALENT' if ok else 'DIVERGED'} -- "
      f"{'trajectories track, safe to ship the flag' if ok else 'investigate before shipping'}")
print("read: expect tiny divergence (fp32-accum / atomic-reorder), NOT a different")
print("      trajectory. If loss tracks and final params match to ~1e-3, the Triton path is")
print("      training-equivalent to the fused path -> commit and (optionally) flip it on.")
