"""Final gate for the CUDA gather path: TRAINING-equivalence vs the fused path.

The fp64-unbiasedness sweep (verify_cuda_gather.py) already passed at every synapse
(worst ratio 1.25). This is the complementary trajectory test: from an identical initial
state, run K optimizer steps twice -- once on the fused path, once on the CUDA path (wpo
forward + v2 backward) -- under bf16 autocast (the real training regime) and compare:
  * the per-step loss trajectory   (max abs difference across all K steps)
  * the final parameters           (max abs difference across every tensor)

Methodology mirrors train_equivalence_triton.py. Both the fused backward (index_add_) and
the CUDA backward (global/shared atomicAdd) are NONDETERMINISTIC -- accumulation order varies
run-to-run -- so the meaningful question is NOT "does CUDA match fused exactly" (fused does not
match ITSELF), but "does the fused->CUDA gap stay within the fused->fused nondeterminism floor".

Run:  source setup_cuda_env.sh   (CUDA path lazy-compiles on first use, ~90s)
      python train_equivalence_cuda.py --steps 50
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
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.cuda_gather import (
    cuda_gather_available,
)

CONFIG = "configs/examples/sglimit5_h100_cifar10_ei_1layer_bf33_10000e1250i.yaml"
dev = "cuda"
assert torch.cuda.is_available()
assert cuda_gather_available(), "CUDA gather did not compile -- source setup_cuda_env.sh first"

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
    """which in {'fused','cuda'} -> flip the flags on every indexed synapse."""
    n = 0
    for m in model.modules():
        if isinstance(m, IndexedSparseLinear):
            m.use_fused_gather = (which == "fused")
            m.use_triton_gather = False
            m.use_cuda_gather = (which == "cuda")
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


l_fusedA, p_fusedA, n = train_run("fused")
l_fusedB, p_fusedB, _ = train_run("fused")     # control: baseline's own nondeterminism
l_cuda, p_cuda, _ = train_run("cuda")
print(f"toggled {n} indexed synapses per run\n")


def max_loss_gap(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def max_param_gap(pa, pb):
    m, name = 0.0, ""
    for k in pa:
        if not torch.is_floating_point(pa[k]) or pa[k].numel() == 0:
            continue                               # skip empty buffers / int tensors
        e = (pa[k].float() - pb[k].float()).abs().max().item()
        if e > m:
            m, name = e, k
    return m, name


floor_loss = max_loss_gap(l_fusedA, l_fusedB)      # fused vs fused (nondeterminism floor)
test_loss = max_loss_gap(l_fusedA, l_cuda)         # fused vs cuda
floor_p, floor_pn = max_param_gap(p_fusedA, p_fusedB)
test_p, test_pn = max_param_gap(p_fusedA, p_cuda)

print(f"loss[0]  fusedA={l_fusedA[0]:.6f}  fusedB={l_fusedB[0]:.6f}  cuda={l_cuda[0]:.6f}")
print(f"loss[-1] fusedA={l_fusedA[-1]:.6f}  fusedB={l_fusedB[-1]:.6f}  cuda={l_cuda[-1]:.6f}\n")
print(f"{'':22} {'loss traj gap':>14}   {'final param gap':>16}")
print(f"{'fused vs fused (floor)':22} {floor_loss:>14.3e}   {floor_p:>16.3e}  (@ {floor_pn})")
print(f"{'fused vs cuda  (test)':22} {test_loss:>14.3e}   {test_p:>16.3e}  (@ {test_pn})")

# cuda is equivalent if its gap to fused is within the fused-vs-fused noise band
# (allow 3x headroom + a small absolute floor for the tiny-signal random-data regime)
loss_ok = test_loss <= max(3 * floor_loss, 1e-3)
param_ok = test_p <= max(3 * floor_p, 1e-4)
ok = loss_ok and param_ok
print(f"\nVERDICT: {'EQUIVALENT' if ok else 'DIVERGED'} -- cuda gap "
      f"{'within' if ok else 'EXCEEDS'} the fused-vs-fused nondeterminism floor "
      f"(loss {'ok' if loss_ok else 'FAIL'}, param {'ok' if param_ok else 'FAIL'})")
print("read: if the two rows are the same ORDER OF MAGNITUDE, the CUDA path (wpo fwd + v2 bwd)")
print("      is no less reproducible than running the fused path twice -> training-equivalent.")
print("      If test >> floor, the kernel adds real error -> investigate before flipping on.")
