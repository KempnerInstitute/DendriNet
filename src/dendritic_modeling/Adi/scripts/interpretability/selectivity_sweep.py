"""
selectivity_sweep.py
====================
Lightweight selectivity analysis: run a single val-set forward pass,
compute per-class mean activation and lifetime sparseness selectivity index
for every neuron. No ablation — designed to run across ALL sweep models.

Usage
-----
python selectivity_sweep.py \
    --config  /path/to/config.yaml \
    --run_dir /path/to/outputs/alexnet_k1_ne256_noproj_ddp_... \
    --out_dir /path/to/outputs/interpretability_results/selectivity_sweep \
    --tag     fc7_2x2_ne256 \
    --device  cuda

Outputs (saved to <out_dir>/<tag>/)
-------------------------------------
  selectivity_index.npy      (N_E,)       lifetime sparseness per neuron
  per_class_mean_act.npy     (N_E, 1000)  mean activation per neuron per class
  summary.json                            N_E, mean/median/max SI, val accuracy
"""

from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, "/n/home13/adithyavm/dendritic-modeling/src")
from dendritic_modeling.config import load_config
from dendritic_modeling.datasets.standard_datasets import load_imagenet_as_datasets
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

IMAGENET_PATH = "/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k"


def load_model(config_path, run_dir, device):
    config   = load_config(config_path)
    model, _ = initialize_model(config.model)
    ckpt     = torch.load(os.path.join(run_dir, "final_model.pt"),
                          map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval().to(device)
    return model


def get_val_loader(batch_size, num_workers):
    _, _, val_ds = load_imagenet_as_datasets(
        train_valid_split=0.8, flatten=False, normalize=True,
        task_data_path=IMAGENET_PATH,
    )
    return DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def collect_activations(model, val_loader, device):
    acts_list, labels_list, correct_list = [], [], []

    def _hook(module, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        acts_list.append(o.detach().cpu().float())

    handle = model.core_network.register_forward_hook(_hook)
    t0 = time.time()

    with torch.no_grad():
        for i, (imgs, targets) in enumerate(val_loader):
            imgs = imgs.to(device)
            logits = model(imgs)
            preds  = logits.argmax(1).cpu()
            correct_list.append((preds == targets).float())
            labels_list.append(targets)
            if (i + 1) % 50 == 0:
                print(f"  {(i+1)*val_loader.batch_size} images  "
                      f"({time.time()-t0:.0f}s)", flush=True)

    handle.remove()
    activations = torch.cat(acts_list,   dim=0).numpy()   # (N, N_E)
    labels      = torch.cat(labels_list, dim=0).numpy()   # (N,)
    correct     = torch.cat(correct_list, dim=0).numpy()  # (N,)
    return activations, labels, correct


def compute_selectivity(activations, labels, n_classes=1000):
    n_e = activations.shape[1]
    per_class_mean = np.zeros((n_e, n_classes), dtype=np.float32)
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            per_class_mean[:, c] = activations[mask].mean(axis=0)

    r     = np.maximum(per_class_mean, 0)
    n     = n_classes
    num   = (r.sum(axis=1) / n) ** 2
    denom = (r ** 2).sum(axis=1) / n
    si    = np.where(denom > 1e-10, (1 - num / denom) / (1 - 1 / n), 0.0)
    return per_class_mean, si.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      required=True)
    p.add_argument("--run_dir",     required=True)
    p.add_argument("--out_dir",     required=True)
    p.add_argument("--tag",         required=True)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size",  type=int, default=256)
    p.add_argument("--num_workers", type=int, default=16)
    args = p.parse_args()

    save_dir = Path(args.out_dir) / args.tag
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\nSelectivity sweep: {args.tag}\n{'='*60}\n", flush=True)

    print("Loading model ...", flush=True)
    model = load_model(args.config, args.run_dir, args.device)

    print("Collecting activations ...", flush=True)
    val_loader = get_val_loader(args.batch_size, args.num_workers)
    activations, labels, correct = collect_activations(model, val_loader, args.device)

    val_acc = correct.mean()
    n_e     = activations.shape[1]
    print(f"  val top-1 acc = {val_acc*100:.2f}%  N_E = {n_e}", flush=True)

    print("Computing selectivity ...", flush=True)
    per_class_mean, si = compute_selectivity(activations, labels)
    print(f"  SI: mean={si.mean():.4f}  median={np.median(si):.4f}  max={si.max():.4f}",
          flush=True)

    np.save(save_dir / "selectivity_index.npy",  si)
    np.save(save_dir / "per_class_mean_act.npy", per_class_mean)

    summary = {
        "tag":       args.tag,
        "config":    args.config,
        "run_dir":   args.run_dir,
        "n_e":       int(n_e),
        "val_acc":   float(val_acc),
        "si_mean":   float(si.mean()),
        "si_median": float(np.median(si)),
        "si_max":    float(si.max()),
        "si_std":    float(si.std()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Saved to {save_dir}", flush=True)


if __name__ == "__main__":
    main()
