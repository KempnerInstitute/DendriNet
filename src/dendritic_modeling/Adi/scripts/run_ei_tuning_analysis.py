"""
E-I Branch Tuning Analysis
==========================
Computes excitatory (E) and inhibitory (I) synaptic currents at dendritic leaf
branches for the 2x2 and 3x3x3 DendriNet noproj sweeps across neuron counts
256, 512, 1024, 2048, 4096.

Does NOT include point-neuron or proj variants.

For each model, runs inference on a FIXED set of 100 ImageNet validation images
per class (seeded at construction so every model sees identical inputs).
Saves per-model figures and raw activations to:

    /n/holylfs06/LABS/kempner_project_b/Lab/dendritic/adivm/
        dendritic-modeling/tuning_analysis/
            {ne}_{morph}_dendrinet_noproj/
                ei_scatter.png
                vout_per_class.png
                selectivity_index.png
                activations.npz
            selectivity_sweep_summary.png

Usage:
    python run_ei_tuning_analysis.py [--n_classes 20] [--n_samples 100] [--device cuda]
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../"))
# absolute fallback
sys.path.insert(0, "/n/home13/adithyavm/dendritic-modeling/src")

from dendritic_modeling.config import load_config
from dendritic_modeling.datasets.standard_datasets import load_imagenet_as_datasets
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

# ── fixed paths ───────────────────────────────────────────────────────────────
OUTPUTS_DIRS = [
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/adivm/dendritic-modeling/outputs",
    "/n/home13/adithyavm/dendritic-modeling/outputs",
]
IMAGENET_PATH = "/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k"
CONFIGS_DIR   = "/n/home13/adithyavm/dendritic-modeling/configs/vision"
OUTPUT_BASE   = "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/adivm/dendritic-modeling/tuning_analysis"

N_E_VALUES  = [256, 512, 1024, 2048, 4096]
MORPHOLOGIES = ["2x2", "3x3x3"]

GLOB_PATTERN = {
    "2x2":   "alexnet_k1_ne{ne}_noproj_ddp_*",
    "3x3x3": "alexnet_k1_ne{ne}_3x3x3_noproj_ddp_*",
}
CONFIG_PATH = {
    "2x2":   "{configs}/alexnet_k1_ne{ne}_noproj_ddp.yaml",
    "3x3x3": "{configs}/alexnet_k1_ne{ne}_3x3x3_noproj_ddp.yaml",
}
SEED = 42


# ── utilities ─────────────────────────────────────────────────────────────────

def folder_name(ne, morph):
    return f"{ne}_{morph}_dendrinet_noproj"


def find_run_dir(ne, morph):
    pat = GLOB_PATTERN[morph].format(ne=ne)
    for base in OUTPUTS_DIRS:
        found = sorted(glob.glob(f"{base}/{pat}"))
        if found:
            return found[-1]
    return None


def load_model(ne, morph, device):
    run_dir = find_run_dir(ne, morph)
    if run_dir is None:
        raise FileNotFoundError(f"No run dir for ne={ne} morph={morph}")
    cfg_path = CONFIG_PATH[morph].format(configs=CONFIGS_DIR, ne=ne)
    config = load_config(cfg_path)
    model, _ = initialize_model(config.model)
    ckpt = torch.load(f"{run_dir}/final_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    model.to(device)
    return model


def collect_activations(model, loader, device):
    """
    Returns dict: layer_name → {'exc': ndarray [N,B], 'inh': ndarray [N,B]}
    Layers are ordered leaf → soma (sorted by name).
    """
    store = {}

    def make_hook(name):
        def hook(module, inp, _out):
            x_exc, x_inh, _x_branch = inp
            entry = store.setdefault(name, {"exc": [], "inh": []})
            if module.branch_excitation is not None:
                entry["exc"].append(module.branch_excitation(x_exc).detach().cpu())
            if module.branch_inhibition is not None and x_inh is not None:
                entry["inh"].append(module.branch_inhibition(x_inh).detach().cpu())
        return hook

    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, DendriticBranchLayer) and "excitatory_cells" in name:
            handles.append(mod.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for batch_x, _ in loader:
            model(batch_x.to(device))

    for h in handles:
        h.remove()

    return {
        lname: {k: torch.cat(v, dim=0).numpy() for k, v in tensors.items() if v}
        for lname, tensors in store.items()
    }


def selectivity_index(exc, inh, labels, classes):
    """η² per branch: between-class Vout variance / total Vout variance."""
    if inh is None:
        inh = np.zeros_like(exc)
    vout = exc / (1.0 + exc + inh + 1e-8)
    grand_mean = vout.mean(axis=0)
    ss_between = sum(
        (labels == c).sum() * (vout[labels == c].mean(axis=0) - grand_mean) ** 2
        for c in classes
    )
    ss_total = ((vout - grand_mean) ** 2).sum(axis=0)
    return ss_between / (ss_total + 1e-10)


# ── per-model plots ────────────────────────────────────────────────────────────

def plot_ei_scatter(exc_all, inh_all, all_labels, selected_classes, ne, morph, out_dir):
    """
    10×10 grid (100 panels): one panel per neuron, first leaf branch of each.
    Neurons sampled evenly across the full excitatory population.
    Each panel: E vs I coloured by class, 100 draws per class.
    """
    colors = cm.tab20(np.linspace(0, 1, len(selected_classes)))
    branches_per_neuron = 4 if morph == "2x2" else 27
    n_neurons_to_show = min(100, ne)
    neuron_indices = np.linspace(0, ne - 1, n_neurons_to_show, dtype=int)
    branch_indices = neuron_indices * branches_per_neuron   # first leaf branch per neuron

    ncols = 10
    nrows = int(np.ceil(n_neurons_to_show / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2), squeeze=False)

    for panel_i, (branch_b, neuron_n) in enumerate(zip(branch_indices, neuron_indices)):
        row, col = divmod(panel_i, ncols)
        ax = axes[row, col]
        for c_i, c in enumerate(selected_classes):
            mask = all_labels == c
            e = exc_all[mask, branch_b]
            i = inh_all[mask, branch_b] if inh_all is not None else np.zeros_like(e)
            ax.scatter(i, e, color=colors[c_i], alpha=0.4, s=3)
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], "k--", lw=0.5, alpha=0.3)
        ax.set_title(f"n{neuron_n}", fontsize=6)
        ax.tick_params(labelsize=5)

    # hide unused panels
    for panel_i in range(n_neurons_to_show, nrows * ncols):
        row, col = divmod(panel_i, ncols)
        axes[row, col].set_visible(False)

    # shared legend outside the grid
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[c_i],
                           markersize=5, label=f"cls {c}")
               for c_i, c in enumerate(selected_classes)]
    fig.legend(handles=handles, loc="lower center", ncol=len(selected_classes),
               fontsize=6, framealpha=0.7, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        f"E vs I | {morph} / N_E={ne} | {n_neurons_to_show} neurons (evenly sampled)\n"
        f"{len(selected_classes)} classes × {(all_labels == selected_classes[0]).sum()} draws  |  each panel = one neuron's leaf branch",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ei_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_vout_per_class(exc_all, inh_all, all_labels, selected_classes, ne, morph, out_dir):
    """Vout distribution per class for leaf branch 0."""
    branch_b = 0
    e = exc_all[:, branch_b]
    i = inh_all[:, branch_b] if inh_all is not None else np.zeros_like(e)
    vout = e / (1.0 + e + i + 1e-8)
    colors = cm.tab20(np.linspace(0, 1, len(selected_classes)))

    fig, ax = plt.subplots(figsize=(12, 4))
    for c_i, c in enumerate(selected_classes):
        mask = all_labels == c
        ax.scatter([c_i] * mask.sum(), vout[mask], color=colors[c_i], alpha=0.4, s=6)
        ax.errorbar(c_i, vout[mask].mean(), yerr=vout[mask].std(),
                    color=colors[c_i], capsize=3, lw=1.5)

    ax.set_xticks(range(len(selected_classes)))
    ax.set_xticklabels([f"c{c}" for c in selected_classes], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Vout = E/(1+E+I)", fontsize=11)
    ax.set_title(f"Branch output per class | {morph} / N_E={ne} | leaf branch 0", fontsize=11)
    ax.axhline(0.5, color="k", lw=0.7, ls="--", alpha=0.4, label="E=I (Vout=0.5)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "vout_per_class.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_selectivity(exc_all, inh_all, all_labels, selected_classes, ne, morph, out_dir):
    """Ranked per-branch selectivity index (η²)."""
    si = selectivity_index(exc_all, inh_all, all_labels, selected_classes)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(np.arange(len(si)), np.sort(si)[::-1], width=1.0, color="steelblue", alpha=0.8)
    ax.axhline(1 / len(selected_classes), color="r", ls="--", lw=1,
               label=f"chance (1/{len(selected_classes)})")
    ax.set_xlabel("Branch index (sorted by selectivity)", fontsize=10)
    ax.set_ylabel("Selectivity index (η²)", fontsize=10)
    ax.set_title(
        f"Class selectivity of leaf branches | {morph} / N_E={ne}\n"
        f"η² = between-class Vout variance / total Vout variance",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "selectivity_index.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return float(np.median(si)), float(np.max(si))


# ── summary plot ──────────────────────────────────────────────────────────────

def plot_sweep_summary(sweep_results, n_classes, n_samples):
    """Median η² across the full N_E × morphology sweep."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for morph, marker, color in [("2x2", "s", "darkorange"), ("3x3x3", "o", "steelblue")]:
        xs = sorted(ne for (m, ne) in sweep_results if m == morph)
        ys = [sweep_results[(morph, ne)]["median_si"] for ne in xs]
        ax.plot(xs, ys, marker=marker, color=color, label=morph)

    ax.axhline(1 / n_classes, color="r", ls="--", lw=0.8, label=f"chance (1/{n_classes})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(N_E_VALUES)
    ax.set_xticklabels(N_E_VALUES)
    ax.set_xlabel("N_E", fontsize=11)
    ax.set_ylabel("Median leaf-branch selectivity (η²)", fontsize=11)
    ax.set_title(
        f"Class selectivity across sweep | DendriNet noproj\n"
        f"({n_classes} classes × {n_samples} draws, fixed seed={SEED})",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_BASE, "selectivity_sweep_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved summary → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_classes", type=int, default=20)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--device",    type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"device: {args.device}  |  n_classes: {args.n_classes}  |  n_samples: {args.n_samples}")
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # ── load ImageNet val once ────────────────────────────────────────────────
    print("loading ImageNet val set...")
    _, _, val_ds = load_imagenet_as_datasets(
        train_valid_split=0.8, flatten=False, normalize=True, task_data_path=IMAGENET_PATH
    )

    # build class → indices map
    targets = np.array(
        val_ds.targets if hasattr(val_ds, "targets")
        else [val_ds.dataset.targets[i] for i in val_ds.indices]
    )
    class_to_idxs = {}
    for i, t in enumerate(targets):
        class_to_idxs.setdefault(int(t), []).append(i)

    selected_classes = sorted(class_to_idxs.keys())[: args.n_classes]

    # ── fixed image selection (same across ALL models) ────────────────────────
    rng = np.random.default_rng(SEED)
    fixed_idxs  = []
    all_labels  = []
    for c in selected_classes:
        pool = np.array(class_to_idxs[c])
        chosen = rng.choice(pool, size=min(args.n_samples, len(pool)), replace=False)
        chosen = np.sort(chosen)          # deterministic order
        fixed_idxs.extend(chosen.tolist())
        all_labels.extend([c] * len(chosen))

    all_labels = np.array(all_labels)
    subset     = Subset(val_ds, fixed_idxs)
    loader     = DataLoader(subset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    print(f"fixed image set: {len(subset)} samples "
          f"({args.n_samples} × {args.n_classes} classes, seed={SEED})")

    # ── per-model loop ────────────────────────────────────────────────────────
    sweep_results = {}

    for morph in MORPHOLOGIES:
        for ne in N_E_VALUES:
            tag     = folder_name(ne, morph)
            out_dir = os.path.join(OUTPUT_BASE, tag)
            os.makedirs(out_dir, exist_ok=True)
            print(f"\n{'='*60}")
            print(f"  {tag}")
            print(f"{'='*60}")

            try:
                model = load_model(ne, morph, args.device)
            except FileNotFoundError as exc:
                print(f"  SKIP: {exc}")
                continue

            acts = collect_activations(model, loader, args.device)
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()

            leaf_layer = sorted(acts.keys())[0]
            exc_all = acts[leaf_layer].get("exc")
            inh_all = acts[leaf_layer].get("inh")

            if exc_all is None:
                print("  WARNING: no excitatory activations captured")
                continue

            # save raw activations
            np.savez_compressed(
                os.path.join(out_dir, "activations.npz"),
                exc=exc_all,
                inh=inh_all if inh_all is not None else np.zeros_like(exc_all),
                labels=all_labels,
                classes=np.array(selected_classes),
                seed=SEED,
            )

            plot_ei_scatter(exc_all, inh_all, all_labels, selected_classes, ne, morph, out_dir)
            print(f"  saved ei_scatter.png")

            plot_vout_per_class(exc_all, inh_all, all_labels, selected_classes, ne, morph, out_dir)
            print(f"  saved vout_per_class.png")

            med_si, max_si = plot_selectivity(
                exc_all, inh_all, all_labels, selected_classes, ne, morph, out_dir
            )
            print(f"  saved selectivity_index.png  (median η²={med_si:.4f}, max={max_si:.4f})")

            sweep_results[(morph, ne)] = {"median_si": med_si, "max_si": max_si}

    # ── sweep summary ─────────────────────────────────────────────────────────
    if len(sweep_results) > 1:
        plot_sweep_summary(sweep_results, args.n_classes, args.n_samples)

    print("\nDone.")


if __name__ == "__main__":
    main()
