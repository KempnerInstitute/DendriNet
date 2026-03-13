"""
Surgical ablation of recurrent inhibitory synapses in dendritic_branch models.

Unlike lesion_analysis.py which zeros entire branches, this script performs
a targeted ablation of only the branch_rec_inhibition pathway (the 4th TopK
layer) within each branch, leaving the other 3 pathways intact. This isolates
the contribution of recurrent inhibitory synapses to performance and stability.

Computes:
  - Per-branch surgical ablation: accuracy by delay with/without rec_I
  - Denominator contribution: fractional contribution of each trace to shunting
  - Gradient stability comparison: norms with vs without rec_I

Usage:
    PYTHONPATH=src:$PYTHONPATH python -m dendritic_modeling.scripts.analysis.rec_inhibition_impact \
        --model_path <path/to/model.pt> \
        --config_path <path/to/config.yaml> \
        --output_dir <output_dir> \
        --n_bins 10 --device cpu
"""

import argparse
import copy
import json
import logging
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data

logger = logging.getLogger(__name__)


def ablate_rec_inhibition(model, branch_idx: int, layer_idx: int = 0):
    """Zero out only the branch_rec_inhibition weights for a specific branch.

    Creates a deep copy so the original model is unmodified.

    Args:
        model: Trained DendRNN model with dendritic_branch compartment.
        branch_idx: Index of the branch whose rec_inhibition to ablate.
        layer_idx: Index of the recurrent cell layer.

    Returns:
        Ablated model (deep copy).
    """
    ablated = copy.deepcopy(model)
    core = ablated.core_network
    cell = core.cells[layer_idx]
    branch_mod = cell.branches[branch_idx]

    # Access the DendriticBranchLayer
    dbl = getattr(branch_mod, "dbl", None)
    if dbl is None:
        raise ValueError(
            f"Branch {branch_idx} has no DendriticBranchLayer (dbl). "
            "Ensure the model uses dendritic_branch compartment."
        )

    rec_inh = getattr(dbl, "branch_rec_inhibition", None)
    if rec_inh is None:
        logger.warning(
            f"Branch {branch_idx} has no branch_rec_inhibition layer. "
            "Ablation has no effect."
        )
        return ablated

    with torch.no_grad():
        for param in rec_inh.parameters():
            param.zero_()

    logger.info(f"Ablated branch_rec_inhibition on branch {branch_idx}")
    return ablated


def evaluate_accuracy_by_delay(
    model,
    dataset,
    conditions,
    n_bins=10,
    device="cpu",
    batch_size=128,
):
    """Evaluate accuracy binned by delay condition."""
    model.eval()
    model.to(device)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            x_batch = batch[0].to(device)
            preds = model.predict(x_batch).cpu()
            all_preds.append(preds)
            all_labels.append(batch[1])

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    overall_acc = float((all_preds == all_labels).mean())

    if conditions is None:
        return {"overall": overall_acc}

    cond_array = np.array(conditions[: len(all_preds)])
    bin_edges = np.linspace(cond_array.min(), cond_array.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(cond_array, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    binned = {}
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        acc = float((all_preds[mask] == all_labels[mask]).mean())
        binned[str(float(bin_centers[i]))] = {
            "accuracy": acc,
            "n_samples": int(mask.sum()),
        }

    binned["overall"] = overall_acc
    return binned


def compute_gradient_norms_quick(model, dataset, loss_fn, device="cpu", max_batches=10):
    """Compute total gradient norm over a few batches."""
    model.train()
    model.to(device)

    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    norms = []
    for batch_idx, (x_batch, y_batch) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        model.zero_grad()
        output = model(x_batch)
        loss = loss_fn(output, y_batch)
        loss.backward()

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        norms.append(math.sqrt(total_norm))

    model.eval()
    return norms


def plot_ablation_results(results, output_path):
    """Plot accuracy vs delay for intact vs ablated models."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    for (label, acc_dict), color in zip(results.items(), colors):
        delays = []
        accs = []
        for k, v in sorted(
            acc_dict.items(), key=lambda x: float(x[0]) if x[0] != "overall" else -1
        ):
            if k == "overall":
                continue
            delays.append(float(k))
            accs.append(v["accuracy"])
        if delays:
            linewidth = 2.5 if label == "intact" else 1.5
            ax.plot(delays, accs, "o-", color=color, label=label, linewidth=linewidth)

    ax.set_xlabel("Delay", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(
        "Recurrent Inhibition Surgical Ablation", fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=8)
    ax.set_ylim([0.3, 1.05])
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_gradient_comparison(grad_data, output_path):
    """Box plot comparing gradient norms across conditions."""
    if not grad_data:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(grad_data.keys())
    data = [grad_data[k] for k in labels]

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = ["#1ABC9C", "#E74C3C", "#F39C12", "#2980B9", "#8E44AD"]
    for patch, color in zip(bp["boxes"], colors[: len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Gradient Norm", fontsize=10)
    ax.set_title("Gradient Stability: rec_I Ablation", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Recurrent inhibition surgical ablation analysis"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    os.makedirs(args.output_dir, exist_ok=True)

    from dendritic_modeling.scripts.analysis.analyze_routing_by_condition import (
        extract_delay_conditions,
        load_model_and_config,
    )

    logger.info(f"Loading model from {args.model_path}")
    model, config, test_ds = load_model_and_config(
        args.model_path, args.config_path, args.device
    )

    conditions = extract_delay_conditions(test_ds)

    # Get branch info
    rec_cfg = config.model.core.recurrent
    branch_taus = [getattr(b, "tau", "?") for b in rec_cfg.branches]
    n_branches = len(rec_cfg.branches)

    # Evaluate intact model
    results = {}
    logger.info("Evaluating intact model...")
    results["intact"] = evaluate_accuracy_by_delay(
        model,
        test_ds,
        conditions,
        n_bins=args.n_bins,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Ablate rec_inhibition per branch
    for b in range(n_branches):
        tau = branch_taus[b]
        label = f"ablate_rec_I_b{b} (tau={tau})"
        logger.info(f"Evaluating: {label}")
        ablated_model = ablate_rec_inhibition(model, branch_idx=b)
        results[label] = evaluate_accuracy_by_delay(
            ablated_model,
            test_ds,
            conditions,
            n_bins=args.n_bins,
            device=args.device,
            batch_size=args.batch_size,
        )

    # Ablate all rec_inhibition at once
    logger.info("Evaluating: all rec_I ablated")
    all_ablated = copy.deepcopy(model)
    for b in range(n_branches):
        core = all_ablated.core_network
        cell = core.cells[0]
        dbl = getattr(cell.branches[b], "dbl", None)
        if dbl is not None:
            rec_inh = getattr(dbl, "branch_rec_inhibition", None)
            if rec_inh is not None:
                with torch.no_grad():
                    for param in rec_inh.parameters():
                        param.zero_()
    results["all_rec_I_ablated"] = evaluate_accuracy_by_delay(
        all_ablated,
        test_ds,
        conditions,
        n_bins=args.n_bins,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Save results
    json_path = os.path.join(args.output_dir, "rec_inhibition_ablation.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {json_path}")

    # Generate figures
    plot_ablation_results(
        results,
        os.path.join(args.output_dir, "rec_inhibition_ablation.png"),
    )

    # Gradient stability comparison
    logger.info("Computing gradient norms...")
    loss_fn = torch.nn.CrossEntropyLoss()
    grad_data = {}
    grad_data["intact"] = compute_gradient_norms_quick(
        copy.deepcopy(model),
        test_ds,
        loss_fn,
        device=args.device,
    )
    grad_data["all_rec_I_ablated"] = compute_gradient_norms_quick(
        all_ablated,
        test_ds,
        loss_fn,
        device=args.device,
    )

    plot_gradient_comparison(
        grad_data,
        os.path.join(args.output_dir, "gradient_stability.png"),
    )

    # Save gradient data
    grad_json = dict(grad_data)
    with open(os.path.join(args.output_dir, "gradient_norms.json"), "w") as f:
        json.dump(grad_json, f, indent=2)

    # Print summary
    for label, acc_dict in results.items():
        overall = acc_dict.get("overall", "N/A")
        if isinstance(overall, float):
            logger.info(f"{label}: overall accuracy = {overall:.4f}")
        else:
            logger.info(f"{label}: {overall}")

    logger.info(f"Rec inhibition impact analysis complete. Results: {args.output_dir}")


if __name__ == "__main__":
    main()
