"""
Level lesion analysis for hierarchical dendritic EI models.

Silences one dendritic level at a time and measures performance degradation
conditioned on a task variable (e.g. delay or frequency). This
tests whether level routing is causally related to task performance -- lesioning
a distal level should selectively degrade long-delay performance.

Usage:
    PYTHONPATH=src:$PYTHONPATH python -m dendritic_modeling.scripts.analysis.lesion_analysis \
        --model_path <path/to/model.pt> \
        --config_path <path/to/config.yaml> \
        --output_dir <output_dir> \
        --dataset variable_delay_dms
"""

import argparse
import contextlib
import json
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data

from dendritic_modeling.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def lesion_level(model, depth: int, layer_idx: int = 0):
    """Silence one excitatory dendritic level while the context is open.

    ``depth`` is the soma-relative depth (0 = soma). The level is silenced
    through the population's native level mask, so its compartments are
    removed from the computation whatever the weight transform; the model is
    unchanged afterwards.
    """
    from dendritic_modeling.analysis.utils.dendritic_depth import (
        soma_relative_dendritic_depth,
    )
    from dendritic_modeling.networks.architectures.recurrent import EINetwork

    core = model.core_network
    if not isinstance(core, EINetwork):
        raise TypeError(f"Expected EINetwork, got {type(core)}")
    ei_layer = core.layers[layer_idx]
    depths = [
        soma_relative_dendritic_depth(branch)
        for branch in ei_layer.e_population.branch_layers
    ]
    if depth not in depths:
        raise ValueError(f"layer {layer_idx} has no dendritic level at depth {depth}")
    mask = torch.zeros(len(depths))
    mask[depths.index(depth)] = 1.0
    ei_layer.set_population_level_silencing("excitatory", mask)
    logger.info(f"Silenced depth {depth} in layer {layer_idx}")
    try:
        yield model
    finally:
        ei_layer.set_population_level_silencing("excitatory", None)


def level_depths_and_taus(model, layer_idx: int = 0) -> list[tuple[int, float]]:
    """Return (soma-relative depth, time constant) of each excitatory level."""
    from dendritic_modeling.analysis.utils.dendritic_depth import (
        soma_relative_dendritic_depth,
    )

    population = model.core_network.layers[layer_idx].e_population
    taus = population.current_taus.detach().reshape(-1).tolist()
    return [
        (soma_relative_dendritic_depth(branch), float(tau))
        for branch, tau in zip(population.branch_layers, taus)
    ]


def evaluate_by_condition(
    model,
    dataset,
    conditions,
    n_bins: int = 10,
    device: str = "cpu",
    batch_size: int = 128,
):
    """Evaluate accuracy binned by condition.

    Returns:
        dict mapping bin_center -> accuracy
    """
    model.eval()
    model.to(device)

    # Get all predictions
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

    # Overall accuracy
    overall_acc = float((all_preds == all_labels).mean())

    if conditions is None:
        return {"overall": overall_acc}

    # Bin by condition
    cond_array = np.array(conditions[: len(all_preds)])
    bin_edges = np.linspace(cond_array.min(), cond_array.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(cond_array, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    binned_acc = {}
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if mask.sum() == 0:
            continue
        bin_preds = all_preds[mask]
        bin_labels = all_labels[mask]
        acc = float((bin_preds == bin_labels).mean())
        binned_acc[float(bin_centers[bin_idx])] = {
            "accuracy": acc,
            "n_samples": int(mask.sum()),
        }

    binned_acc["overall"] = overall_acc
    return binned_acc


def plot_lesion_results(results, output_path: str, branch_labels=None):
    """Plot accuracy vs condition for intact and lesioned models."""
    _fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    # Markers and line styles repeat the series identity for colour-blind
    # readers.
    markers = ["o", "s", "^", "v", "D", "P", "X", "*"]
    linestyles = ["-", "--", "-.", ":"]

    for index, ((label, acc_dict), color) in enumerate(zip(results.items(), colors)):
        bins = sorted(float(k) for k in acc_dict if k != "overall")
        accuracies = [acc_dict[k]["accuracy"] for k in bins]
        if bins:
            ax.plot(
                bins,
                accuracies,
                marker=markers[index % len(markers)],
                linestyle=linestyles[index % len(linestyles)],
                color=color,
                label=label,
                linewidth=2,
            )

    ax.set_xlabel("Delay D", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Level Lesion Analysis", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim([0.3, 1.05])
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved lesion plot to {output_path}")


def plot_lesion_bar_chart(results, output_path: str):
    """Bar chart: accuracy drop per branch at short vs long delays."""
    if "intact" not in results:
        return

    intact = results["intact"]
    conditions = sorted(k for k in intact if k != "overall")
    if len(conditions) < 4:
        return

    # Split into short and long delays
    mid = len(conditions) // 2
    short_delays = conditions[:mid]
    long_delays = conditions[mid:]

    lesion_labels = [k for k in results if k != "intact"]
    if not lesion_labels:
        return

    def avg_acc(acc_dict, delay_subset):
        accs = [acc_dict[d]["accuracy"] for d in delay_subset if d in acc_dict]
        return np.mean(accs) if accs else float("nan")

    _fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(lesion_labels))
    width = 0.35

    intact_short = avg_acc(intact, short_delays)
    intact_long = avg_acc(intact, long_delays)

    short_drops = []
    long_drops = []
    for label in lesion_labels:
        lesioned = results[label]
        short_drops.append(intact_short - avg_acc(lesioned, short_delays))
        long_drops.append(intact_long - avg_acc(lesioned, long_delays))

    ax.bar(x - width / 2, short_drops, width, label="Short delays", color="#0072B2")
    ax.bar(
        x + width / 2,
        long_drops,
        width,
        label="Long delays",
        color="#D55E00",
        hatch="//",
    )

    ax.set_xlabel("Lesioned Level", fontsize=12)
    ax.set_ylabel("Accuracy Drop (intact - lesioned)", fontsize=12)
    ax.set_title("Selective Accuracy Degradation by Level Lesion", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(lesion_labels, fontsize=10)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved lesion bar chart to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Branch lesion analysis")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--dataset",
        default="variable_delay_dms",
        help="Not read: the dataset is the one of --config_path",
    )
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from dendritic_modeling.scripts.analysis.analyze_routing_by_condition import (
        extract_delay_conditions,
        load_model_and_config,
    )

    logger.info(f"Loading model from {args.model_path}")
    model, _config, test_ds = load_model_and_config(
        args.model_path, args.config_path, args.device, output_dir=args.output_dir
    )

    # Extract conditions
    conditions = extract_delay_conditions(test_ds)

    # Evaluate intact model
    results = {}
    logger.info("Evaluating intact model...")
    results["intact"] = evaluate_by_condition(
        model,
        test_ds,
        conditions,
        n_bins=args.n_bins,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Silence each excitatory dendritic level in turn, labelled by its
    # soma-relative depth and the model's time constant.
    for depth, tau in level_depths_and_taus(model):
        label = f"Lesion depth {depth} (tau={tau:.0f})"
        logger.info(f"Evaluating: {label}")
        with lesion_level(model, depth=depth):
            results[label] = evaluate_by_condition(
                model,
                test_ds,
                conditions,
                n_bins=args.n_bins,
                device=args.device,
                batch_size=args.batch_size,
            )

    # Save results
    serializable = {}
    for label, acc_dict in results.items():
        serializable[label] = {str(k): v for k, v in acc_dict.items()}

    with open(os.path.join(args.output_dir, "lesion_results.json"), "w") as f:
        json.dump(serializable, f, indent=2)

    # Generate figures
    plot_lesion_results(
        results,
        os.path.join(args.output_dir, "lesion_accuracy_vs_delay.png"),
    )
    plot_lesion_bar_chart(
        results,
        os.path.join(args.output_dir, "lesion_bar_chart.png"),
    )

    # Print summary
    for label, acc_dict in results.items():
        overall = acc_dict.get("overall", "N/A")
        logger.info(
            f"{label}: overall accuracy = {overall:.4f}"
            if isinstance(overall, float)
            else f"{label}: {overall}"
        )

    logger.info(f"Lesion analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    setup_logging()
    main()
