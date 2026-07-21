"""What dtype do the indexed synapses actually receive under real autocast?

The baseline fused_sparse_gather crashes on a bf16 activation + fp32 weight
(index_add_ dtype mismatch). Since it trains fine under AMP, the synapse must be
getting fp32 x under this model's autocast. This confirms that directly by hooking
every IndexedSparseLinear's forward input dtype during an autocast(bf16) forward.

Run:  python check_synapse_input_dtype.py
"""
import os, sys
REPO = "/n/home12/rwhalen/DendriNet"
os.chdir(REPO); sys.path.insert(0, os.path.join(REPO, "src"))

import torch
from dendritic_modeling.config import load_config
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)

CONFIG = "configs/examples/sglimit5_h100_cifar10_ei_1layer_bf33_10000e1250i.yaml"
dev = "cuda"
assert torch.cuda.is_available()

cfg = load_config(CONFIG)
model, _ = initialize_model(cfg.model); model = model.to(dev)

seen = {}
def hook(mod, inp):
    x = inp[0]
    seen.setdefault(str(x.dtype), 0)
    seen[str(x.dtype)] += 1

n = 0
for m in model.modules():
    if isinstance(m, IndexedSparseLinear):
        m.register_forward_pre_hook(hook); n += 1
print(f"hooked {n} IndexedSparseLinear synapses")

x = torch.randn(128, 3072, device=dev)
with torch.autocast("cuda", dtype=torch.bfloat16):
    model(x)

print(f"\ninput dtypes seen at indexed synapses under autocast(bf16): {seen}")
print("read: if this is all float32, the synapse never sees bf16 under AMP -> the fp32")
print("      correctness proof IS the AMP-safety proof. If any bf16 appears, the current")
print("      fused baseline is already unsafe under AMP and the Triton kernel is the fix.")
