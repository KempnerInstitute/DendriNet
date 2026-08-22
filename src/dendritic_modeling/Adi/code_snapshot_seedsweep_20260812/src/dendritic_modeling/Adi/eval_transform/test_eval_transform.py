#!/usr/bin/env python
"""Test whether the eval preprocessing explains Houman's ~2% discrepancy.

This repo's loader uses Resize((224,224)) -- a direct squash that distorts aspect
ratio (standard_datasets.py:765). torchvision's official transform for the pretrained
weights this project freezes is Resize(256) -> CenterCrop(224). Since the encoder is
frozen and was trained under the official transform, the squash feeds it out-of-
distribution images at eval.

Re-evaluates EXISTING trained checkpoints under both transforms. No retraining.
Read-only w.r.t. every shared file.
"""
from __future__ import annotations
import sys, os, json, argparse
sys.path.insert(0, "/n/home13/adithyavm/dendritic-modeling/src")
import torch, torch._dynamo
torch._dynamo.config.disable = True
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dendritic_modeling.config import load_config
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

VAL = "/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k/val"
NORM = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

TFORMS = {
    "repo  Resize((224,224))      ": transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), NORM]),
    "tv    Resize(256)+CenterCrop ": transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), NORM]),
}

@torch.no_grad()
def top1(model, tf, device, bs=256, workers=16):
    ds = datasets.ImageFolder(VAL, transform=tf)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    c = n = 0
    for x, y in dl:
        p = model(x.to(device, non_blocking=True)).argmax(1).cpu()
        c += (p == y).sum().item(); n += y.numel()
    return c / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model, _ = initialize_model(load_config(a.config).model)
    ck = torch.load(os.path.join(a.run_dir, "final_model.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck))
    model.eval().to(dev)

    rec = None
    p = os.path.join(a.run_dir, "performance", "final.json")
    if os.path.exists(p):
        rec = json.load(open(p))["accuracy"]["test"]

    print(f"\n=== {a.label} ===", flush=True)
    print(f"  recorded test acc in final.json : {rec*100:.2f}%" if rec else "  (no final.json)", flush=True)
    out = {}
    for name, tf in TFORMS.items():
        acc = top1(model, tf, dev)
        out[name.strip()] = acc
        print(f"  {name} -> {acc*100:.2f}%", flush=True)
    k = list(out); d = (out[k[1]] - out[k[0]]) * 100
    print(f"  DELTA (official - repo squash) = {d:+.2f} points", flush=True)

if __name__ == "__main__":
    main()
