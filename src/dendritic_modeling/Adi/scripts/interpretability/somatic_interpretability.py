"""
somatic_interpretability.py
============================
Three analyses of somatic neuron specialisation in a trained EINet model.
All analyses are run across ALL N_E neurons (not a subset).

Usage
-----
python somatic_interpretability.py \\
    --config   /path/to/configs/vision/alexnet_k1_ne4096_noproj_ddp.yaml \\
    --run_dir  /path/to/outputs/alexnet_k1_ne4096_noproj_ddp_YYYYMMDD_HHMMSS \\
    --out_dir  /path/to/outputs/interpretability_results \\
    --tag      fc7_2x2_ne4096 \\
    --device   cuda

Outputs (saved to <out_dir>/<tag>/)
-------------------------------------
  fc8_weight_projection.npy    (N_E, 1000)  fc8 weight column per neuron [Analysis 1]
  val_activations.npy          (N, N_E)     raw soma activations over val set [Analysis 2]
  val_labels.npy               (N,)         ground-truth ImageNet class indices
  per_class_mean_act.npy       (N_E, 1000)  mean activation per neuron per class
  selectivity_index.npy        (N_E,)       lifetime sparseness score per neuron
  top_classes_by_weight.npy    (N_E, K)     top-K class indices from fc8 weights
  top_classes_by_activation.npy(N_E, K)     top-K class indices from mean activations
  baseline_class_acc.npy       (1000,)      per-class accuracy, no ablation [Analysis 3]
  ablation_acc_delta.npy       (N_E, 1000)  accuracy change (ablated - baseline) per neuron
  README.txt                               metadata: model paths, accuracy, timestamp
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
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
TOP_K = 10   # top-K classes to record per neuron


# ─── model loading ────────────────────────────────────────────────────────────

def load_model(config_path: str, run_dir: str, device: str):
    config   = load_config(config_path)
    model, _ = initialize_model(config.model)

    ckpt_path = os.path.join(run_dir, "final_model.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)
    return model


def get_fc8(model) -> nn.Linear:
    """Return the fc8 linear layer from the decoder backbone segment."""
    for m in model.decoder_network.modules():
        if isinstance(m, nn.Linear):
            return m
    raise RuntimeError("Could not find nn.Linear in decoder_network")


# ─── data loading ─────────────────────────────────────────────────────────────

def get_val_loader(batch_size: int = 256, num_workers: int = 8):
    _, _, val_ds = load_imagenet_as_datasets(
        train_valid_split=0.8, flatten=False, normalize=True,
        task_data_path=IMAGENET_PATH,
    )
    return DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


# ─── Analysis 1: fc8 weight projection ────────────────────────────────────────

def analysis_fc8_weight_projection(fc8: nn.Linear, n_e: int) -> np.ndarray:
    """
    For each somatic neuron i, extract fc8.weight[:, i] — the 1000-class vote
    vector. Positive entries = classes that neuron excites; negative = suppressed.

    Returns: (N_E, 1000) float32 array
    """
    print("Analysis 1: fc8 weight projection ...", flush=True)
    W = fc8.weight.detach().cpu().numpy()   # (1000, N_E_padded)
    proj = W[:, :n_e].T                     # (N_E, 1000)
    print(f"  fc8.weight slice: ({n_e}, 1000)  "
          f"min={proj.min():.4f}  max={proj.max():.4f}", flush=True)
    return proj.astype(np.float32)


# ─── Analysis 2: activation selectivity over val set ─────────────────────────

def collect_activations(model, val_loader, device: str):
    """
    Run a full forward pass over the val set with a hook on model.core_network.
    Returns activations (N, N_E) and labels (N,).
    """
    print("Analysis 2: collecting soma activations over val set ...", flush=True)

    acts_list, labels_list = [], []

    def _hook(module, inp, out):
        # out may be a tuple (exc, inh) for some EINet configs; take exc
        if isinstance(out, tuple):
            out = out[0]
        acts_list.append(out.detach().cpu().float())

    handle = model.core_network.register_forward_hook(_hook)

    with torch.no_grad():
        for batch_idx, (imgs, targets) in enumerate(val_loader):
            imgs = imgs.to(device)
            model(imgs)
            labels_list.append(targets.cpu())
            if (batch_idx + 1) % 50 == 0:
                n_done = (batch_idx + 1) * val_loader.batch_size
                print(f"  {n_done} images ...", flush=True)

    handle.remove()

    activations = torch.cat(acts_list, dim=0).numpy()   # (N, N_E)
    labels      = torch.cat(labels_list, dim=0).numpy() # (N,)
    print(f"  activations shape: {activations.shape}  "
          f"labels shape: {labels.shape}", flush=True)
    return activations, labels


def compute_selectivity(activations: np.ndarray, labels: np.ndarray, n_classes: int = 1000):
    """
    Per-class mean activation: (N_E, n_classes).
    Lifetime sparseness (Vinje & Gallant 2000) per neuron: (N_E,).
      S = (1 - (sum_k(r_k/n))^2 / sum_k(r_k^2/n)) / (1 - 1/n)
    where r_k = mean activation for class k, n = n_classes.
    Higher S = more selective.
    """
    n_e = activations.shape[1]
    per_class_mean = np.zeros((n_e, n_classes), dtype=np.float32)
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            per_class_mean[:, c] = activations[mask].mean(axis=0)

    # Lifetime sparseness
    r = per_class_mean                          # (N_E, n_classes)
    r_pos = np.maximum(r, 0)                   # rectify (only positive activations count)
    n = n_classes
    num   = (r_pos.sum(axis=1) / n) ** 2       # (N_E,)
    denom = (r_pos ** 2).sum(axis=1) / n       # (N_E,)
    si    = np.where(denom > 1e-10, (1 - num / denom) / (1 - 1 / n), 0.0)

    return per_class_mean, si.astype(np.float32)


# ─── Analysis 3: offline ablation ─────────────────────────────────────────────

def analysis_ablation(
    activations: np.ndarray,
    labels:      np.ndarray,
    fc8:         nn.Linear,
    n_e:         int,
    n_classes:   int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate zeroing each somatic neuron offline using the saved activation matrix.

    Ablated logits for neuron i:
        L_i = (A with col i = 0) @ W[:, :N_E].T + b
            = L_base - outer(A[:, i], W[:, i])

    Returns:
        baseline_class_acc : (1000,)    per-class accuracy without ablation
        acc_delta          : (N_E, 1000) accuracy change (ablated - baseline)
    """
    print("Analysis 3: offline ablation ...", flush=True)
    t0 = time.time()

    W = fc8.weight.detach().cpu().numpy().T  # (N_E_padded, 1000)
    b = fc8.bias.detach().cpu().numpy()      # (1000,)

    W_eff = W[:n_e]                          # (N_E, 1000)  only active columns

    # Base logits and predictions
    L_base  = activations @ W_eff + b[None, :]   # (N, 1000)
    preds_base = L_base.argmax(axis=1)            # (N,)

    # Per-class total image counts (precomputed once)
    class_total = np.bincount(labels, minlength=n_classes).astype(np.float32)

    def _class_acc(preds: np.ndarray) -> np.ndarray:
        correct = (preds == labels)
        class_correct = np.zeros(n_classes, dtype=np.float32)
        np.add.at(class_correct, labels, correct.astype(np.float32))
        return np.where(class_total > 0, class_correct / class_total, 0.0)

    baseline_class_acc = _class_acc(preds_base)

    acc_delta = np.zeros((n_e, n_classes), dtype=np.float32)

    for i in range(n_e):
        # Ablate neuron i: subtract its contribution from base logits
        delta  = np.outer(activations[:, i], W_eff[i])   # (N, 1000)
        L_abl  = L_base - delta
        preds_abl = L_abl.argmax(axis=1)
        acc_delta[i] = _class_acc(preds_abl) - baseline_class_acc

        if (i + 1) % 256 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (i + 1) * (n_e - i - 1)
            print(f"  neuron {i+1}/{n_e}  elapsed {elapsed:.0f}s  ETA {eta:.0f}s",
                  flush=True)

    print(f"  ablation done in {time.time()-t0:.0f}s", flush=True)
    return baseline_class_acc, acc_delta


# ─── top-K summaries ──────────────────────────────────────────────────────────

def top_k_classes(matrix: np.ndarray, k: int = TOP_K) -> np.ndarray:
    """matrix: (N_E, 1000) → indices of top-K entries per row: (N_E, K)."""
    return np.argsort(matrix, axis=1)[:, -k:][:, ::-1].astype(np.int16)


# ─── save + README ────────────────────────────────────────────────────────────

def save_results(out_dir: Path, tag: str, config_path: str, run_dir: str, **arrays):
    save_dir = out_dir / tag
    save_dir.mkdir(parents=True, exist_ok=True)

    for name, arr in arrays.items():
        np.save(save_dir / f"{name}.npy", arr)
        print(f"  saved {name}.npy  shape={arr.shape}  dtype={arr.dtype}", flush=True)

    readme = save_dir / "README.txt"
    with open(readme, "w") as f:
        f.write(f"Somatic interpretability — {tag}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"config  : {config_path}\n")
        f.write(f"run_dir : {run_dir}\n")
        f.write(f"created : {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Files\n-----\n")
        f.write("  fc8_weight_projection.npy     (N_E, 1000)  Analysis 1\n")
        f.write("  val_activations.npy           (N, N_E)     Analysis 2 — raw\n")
        f.write("  val_labels.npy                (N,)         ground-truth labels\n")
        f.write("  per_class_mean_act.npy        (N_E, 1000)  mean act per class\n")
        f.write("  selectivity_index.npy         (N_E,)       lifetime sparseness\n")
        f.write("  top_classes_by_weight.npy     (N_E, K)     top-K by fc8 weight\n")
        f.write("  top_classes_by_activation.npy (N_E, K)     top-K by mean act\n")
        f.write("  baseline_class_acc.npy        (1000,)      Analysis 3 — baseline\n")
        f.write("  ablation_acc_delta.npy        (N_E, 1000)  acc drop per neuron\n")
        f.write(f"\nTOP_K = {TOP_K}\n")
    print(f"  saved README.txt", flush=True)


# ─── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  required=True, help="Path to YAML config used for training")
    p.add_argument("--run_dir", required=True, help="Training output directory")
    p.add_argument("--out_dir", required=True, help="Root directory for interpretability results")
    p.add_argument("--tag",     required=True, help="Subdirectory label, e.g. fc7_2x2_ne4096")
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size",   type=int, default=256)
    p.add_argument("--num_workers",  type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    print(f"\n{'='*60}", flush=True)
    print(f"Somatic interpretability: {args.tag}", flush=True)
    print(f"device: {args.device}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # ── load model ────────────────────────────────────────────────────────────
    print("Loading model ...", flush=True)
    model = load_model(args.config, args.run_dir, args.device)
    fc8   = get_fc8(model)
    n_e   = fc8.weight.shape[1]   # effective dimension (N_E, since no zero-pad at ne4096)
    # For ne4096 with no zero-pad, fc8.weight is (1000, 4096) and N_E = 4096
    # For other sizes the fc8 may be padded; we trust fc8.weight.shape[1] here.
    print(f"  N_E = {n_e}  fc8: ({fc8.weight.shape[0]}, {fc8.weight.shape[1]})",
          flush=True)

    # ── Analysis 1 ────────────────────────────────────────────────────────────
    fc8_proj = analysis_fc8_weight_projection(fc8, n_e)
    top_by_weight = top_k_classes(fc8_proj)

    # ── Analysis 2 ────────────────────────────────────────────────────────────
    val_loader  = get_val_loader(args.batch_size, args.num_workers)
    activations, labels = collect_activations(model, val_loader, args.device)

    per_class_mean, sel_idx = compute_selectivity(activations, labels)
    top_by_act = top_k_classes(per_class_mean)

    print(f"  selectivity — mean={sel_idx.mean():.4f}  "
          f"max={sel_idx.max():.4f}  min={sel_idx.min():.4f}", flush=True)

    # ── Analysis 3 ────────────────────────────────────────────────────────────
    baseline_acc, acc_delta = analysis_ablation(activations, labels, fc8, n_e)
    overall_baseline = baseline_acc.mean()
    most_critical    = np.abs(acc_delta).sum(axis=1).argmax()
    print(f"  baseline top-1: {overall_baseline*100:.2f}%", flush=True)
    print(f"  most critical neuron: {most_critical}  "
          f"total |Δacc| = {np.abs(acc_delta[most_critical]).sum():.4f}", flush=True)

    # ── save ──────────────────────────────────────────────────────────────────
    print("\nSaving results ...", flush=True)
    save_results(
        out_dir, args.tag, args.config, args.run_dir,
        fc8_weight_projection     = fc8_proj,
        val_activations           = activations,
        val_labels                = labels,
        per_class_mean_act        = per_class_mean,
        selectivity_index         = sel_idx,
        top_classes_by_weight     = top_by_weight,
        top_classes_by_activation = top_by_act,
        baseline_class_acc        = baseline_acc,
        ablation_acc_delta        = acc_delta,
    )

    # ── summary.json for sweep-level aggregation ──────────────────────────────
    summary = {
        "tag":       args.tag,
        "config":    args.config,
        "run_dir":   args.run_dir,
        "n_e":       int(n_e),
        "val_acc":   float(overall_baseline),
        "si_mean":   float(sel_idx.mean()),
        "si_median": float(float(np.median(sel_idx))),
        "si_max":    float(sel_idx.max()),
        "si_std":    float(sel_idx.std()),
        "ablation_impact_median": float(np.median(np.abs(acc_delta).sum(axis=1))),
        "ablation_impact_max":    float(np.abs(acc_delta).sum(axis=1).max()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / args.tag / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved summary.json", flush=True)

    print(f"\nDone. Results in: {out_dir / args.tag}", flush=True)


if __name__ == "__main__":
    main()
