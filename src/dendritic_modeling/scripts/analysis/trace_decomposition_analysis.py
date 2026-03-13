"""
Trace decomposition analysis for dendritic_branch recurrent models.

Loads a trained model using the dendritic_branch compartment, runs forward
passes with store_routing=True, and extracts per-branch trace values
(trace_E, trace_E_rec, trace_I, trace_I_rec) over time. Computes:
  - Trace temporal profiles conditioned on delay
  - Effective gain G_b(t) = numerator / denominator per branch
  - E/I balance ratio per branch
  - FF vs REC dominance index per branch

Usage:
    PYTHONPATH=src:$PYTHONPATH python -m dendritic_modeling.scripts.analysis.trace_decomposition_analysis \
        --model_path <path/to/model.pt> \
        --config_path <path/to/config.yaml> \
        --output_dir <output_dir> \
        --n_samples 2000 --n_bins 10 --device cpu
"""

import argparse
import json
import logging
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data

logger = logging.getLogger(__name__)


def collect_traces(
    model,
    dataset,
    device: str = "cpu",
    batch_size: int = 64,
    n_samples: int | None = None,
):
    """Run forward passes and collect per-branch trace values over time.

    Requires that the model uses dendritic_branch compartment and has
    store_routing=True, which populates the info dict with trace tensors.

    Returns:
        traces_list: list of dicts per sample, each with:
            "branch_0": {"trace_E": [T, n_exc], "trace_E_rec": ..., etc.}
        targets: list of target values
    """
    from dendritic_modeling.networks.architectures.recurrent import DendriticRNNNetwork

    core = model.core_network
    if not isinstance(core, DendriticRNNNetwork):
        raise TypeError(f"Expected DendriticRNNNetwork, got {type(core)}")

    original_store_routing = core.store_routing
    core.store_routing = True

    if n_samples is not None and n_samples < len(dataset):
        indices = list(range(n_samples))
        subset = torch.utils.data.Subset(dataset, indices)
    else:
        subset = dataset

    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)

    all_traces = []
    all_targets = []

    try:
        with torch.no_grad():
            for batch in loader:
                x_batch = batch[0].to(device)
                y_batch = batch[1]
                _ = model(x_batch)

                seq_len = len(core._routing_info)
                if seq_len == 0:
                    continue

                batch_size_actual = x_batch.shape[0]

                for sample_idx in range(batch_size_actual):
                    sample_traces = {}
                    for t in range(seq_len):
                        info_t = core._routing_info[t]
                        # Look for traces in layer_0
                        layer_info = info_t.get("layer_0", {})
                        traces_t = layer_info.get("traces", {})

                        for branch_key, trace_dict in traces_t.items():
                            if branch_key not in sample_traces:
                                sample_traces[branch_key] = defaultdict(list)
                            for trace_name, tensor in trace_dict.items():
                                sample_traces[branch_key][trace_name].append(
                                    tensor[sample_idx].cpu()
                                )

                    # Stack timesteps: [T, n_exc] per trace
                    stacked = {}
                    for branch_key, trace_lists in sample_traces.items():
                        stacked[branch_key] = {
                            name: torch.stack(tensors, dim=0)
                            for name, tensors in trace_lists.items()
                        }

                    all_traces.append(stacked)
                    all_targets.append(y_batch[sample_idx].item())

    finally:
        core.store_routing = original_store_routing

    return all_traces, all_targets


def compute_trace_metrics(traces_list, conditions, n_bins=10):
    """Compute trace decomposition metrics binned by condition.

    Returns dict with per-bin, per-branch metrics:
        - trace_magnitudes: {branch: {trace_name: [T]}}
        - gain_profile: {branch: [T]}
        - ei_balance: {branch: [T]}
        - ff_rec_dominance: {branch: [T]}
    """
    if not traces_list:
        return {}

    cond_array = np.array(conditions[: len(traces_list)])
    bin_edges = np.linspace(cond_array.min(), cond_array.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(cond_array, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    results = {}
    eps = 1e-8

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if not np.any(mask):
            continue

        bin_center = float(bin_centers[bin_idx])
        sample_indices = np.where(mask)[0]

        # Collect per-branch metrics
        branch_metrics = {}
        branch_keys = sorted(traces_list[sample_indices[0]].keys())

        for branch_key in branch_keys:
            trace_E_list = []
            trace_E_rec_list = []
            trace_I_list = []
            trace_I_rec_list = []
            V_list = []

            for si in sample_indices:
                branch_data = traces_list[si].get(branch_key)
                if branch_data is None:
                    continue
                # Average over neurons: [T, n_exc] -> [T]
                trace_E_list.append(branch_data["trace_E"].abs().mean(dim=-1).numpy())
                trace_E_rec_list.append(
                    branch_data["trace_E_rec"].abs().mean(dim=-1).numpy()
                )
                trace_I_list.append(branch_data["trace_I"].abs().mean(dim=-1).numpy())
                trace_I_rec_list.append(
                    branch_data["trace_I_rec"].abs().mean(dim=-1).numpy()
                )
                V_list.append(branch_data["V"].abs().mean(dim=-1).numpy())

            if not trace_E_list:
                continue

            # Average over samples: [T]
            min_T = min(len(x) for x in trace_E_list)
            trace_E_avg = np.mean([x[:min_T] for x in trace_E_list], axis=0)
            trace_E_rec_avg = np.mean([x[:min_T] for x in trace_E_rec_list], axis=0)
            trace_I_avg = np.mean([x[:min_T] for x in trace_I_list], axis=0)
            trace_I_rec_avg = np.mean([x[:min_T] for x in trace_I_rec_list], axis=0)

            # Gain: numerator / denominator (using absolute values for magnitude)
            numerator = trace_E_avg + trace_E_rec_avg
            denominator = (
                1
                + np.maximum(trace_E_avg, 0)
                + np.maximum(trace_E_rec_avg, 0)
                + np.maximum(trace_I_avg, 0)
                + np.maximum(trace_I_rec_avg, 0)
                + eps
            )
            gain = numerator / denominator

            # E/I balance
            exc_total = trace_E_avg + trace_E_rec_avg
            inh_total = trace_I_avg + trace_I_rec_avg + eps
            ei_balance = exc_total / inh_total

            # FF vs REC dominance
            ff_rec_dom = trace_E_rec_avg / (trace_E_avg + trace_E_rec_avg + eps)

            branch_metrics[branch_key] = {
                "trace_magnitudes": {
                    "trace_E": trace_E_avg.tolist(),
                    "trace_E_rec": trace_E_rec_avg.tolist(),
                    "trace_I": trace_I_avg.tolist(),
                    "trace_I_rec": trace_I_rec_avg.tolist(),
                },
                "gain_profile": gain.tolist(),
                "ei_balance": ei_balance.tolist(),
                "ff_rec_dominance": ff_rec_dom.tolist(),
                "n_samples": len(sample_indices),
            }

        results[str(bin_center)] = branch_metrics

    return results


def plot_trace_heatmap(results, output_path, branch_taus=None):
    """Plot per-branch, per-trace-type heatmaps conditioned on delay."""
    if not results:
        logger.warning("No results to plot for trace heatmap")
        return

    # Get branch keys from first bin
    first_bin = next(iter(results.values()))
    branch_keys = sorted(first_bin.keys())
    trace_names = ["trace_E", "trace_E_rec", "trace_I", "trace_I_rec"]

    n_branches = len(branch_keys)
    n_traces = len(trace_names)

    fig, axes = plt.subplots(
        n_branches, n_traces, figsize=(3.5 * n_traces, 2.5 * n_branches)
    )
    if n_branches == 1:
        axes = axes[np.newaxis, :]

    delay_bins = sorted(results.keys(), key=float)

    for row, branch_key in enumerate(branch_keys):
        for col, trace_name in enumerate(trace_names):
            ax = axes[row, col]

            # Build heatmap: [n_delay_bins, T]
            rows_data = []
            for delay_bin in delay_bins:
                branch_data = results[delay_bin].get(branch_key, {})
                magnitudes = branch_data.get("trace_magnitudes", {})
                trace_vals = magnitudes.get(trace_name, [])
                if trace_vals:
                    rows_data.append(trace_vals)

            if not rows_data:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            min_T = min(len(r) for r in rows_data)
            heatmap = np.array([r[:min_T] for r in rows_data])

            im = ax.imshow(
                heatmap,
                aspect="auto",
                origin="lower",
                cmap="viridis",
                interpolation="nearest",
            )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            tau_label = f"$\\tau$={branch_taus[row]}" if branch_taus else branch_key
            if col == 0:
                ax.set_ylabel(f"{tau_label}\nDelay bin", fontsize=9)
            if row == n_branches - 1:
                ax.set_xlabel("Timestep", fontsize=9)
            if row == 0:
                ax.set_title(trace_name, fontsize=10, fontweight="bold")

    fig.suptitle(
        "Trace Decomposition by Branch and Delay",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_gain_profile(results, output_path, branch_taus=None):
    """Plot effective gain G_b(t) per branch for select delay conditions."""
    if not results:
        return

    first_bin = next(iter(results.values()))
    branch_keys = sorted(first_bin.keys())
    delay_bins = sorted(results.keys(), key=float)

    # Pick 3 representative delays: short, medium, long
    n_bins = len(delay_bins)
    selected = [delay_bins[0], delay_bins[n_bins // 2], delay_bins[-1]]

    colors = ["#E74C3C", "#F39C12", "#2980B9"]

    fig, axes = plt.subplots(1, len(branch_keys), figsize=(4 * len(branch_keys), 3.5))
    if len(branch_keys) == 1:
        axes = [axes]

    for ax, branch_key in zip(axes, branch_keys):
        for delay_bin, color in zip(selected, colors):
            branch_data = results.get(delay_bin, {}).get(branch_key, {})
            gain = branch_data.get("gain_profile", [])
            if gain:
                ax.plot(
                    gain,
                    color=color,
                    linewidth=1.5,
                    label=f"D={float(delay_bin):.0f}",
                )

        tau_label = (
            f"$\\tau$={branch_taus[branch_keys.index(branch_key)]}"
            if branch_taus
            else branch_key
        )
        ax.set_title(tau_label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Timestep", fontsize=9)
        ax.set_ylabel("Effective Gain", fontsize=9)
        ax.legend(fontsize=8)

    fig.suptitle("Effective Gain G_b(t)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_ei_balance(results, output_path, branch_taus=None):
    """Plot E/I balance and FF/REC dominance per branch."""
    if not results:
        return

    first_bin = next(iter(results.values()))
    branch_keys = sorted(first_bin.keys())
    delay_bins = sorted(results.keys(), key=float)

    n_bins = len(delay_bins)
    selected = [delay_bins[0], delay_bins[n_bins // 2], delay_bins[-1]]
    colors = ["#E74C3C", "#F39C12", "#2980B9"]

    fig, axes = plt.subplots(2, len(branch_keys), figsize=(4 * len(branch_keys), 6))
    if len(branch_keys) == 1:
        axes = axes[:, np.newaxis]

    for col, branch_key in enumerate(branch_keys):
        tau_label = f"$\\tau$={branch_taus[col]}" if branch_taus else branch_key

        # Row 0: E/I balance
        ax = axes[0, col]
        for delay_bin, color in zip(selected, colors):
            branch_data = results.get(delay_bin, {}).get(branch_key, {})
            ei = branch_data.get("ei_balance", [])
            if ei:
                ax.plot(
                    ei, color=color, linewidth=1.5, label=f"D={float(delay_bin):.0f}"
                )
        ax.set_title(tau_label, fontsize=10, fontweight="bold")
        ax.set_ylabel("E/I Ratio", fontsize=9)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.legend(fontsize=7)

        # Row 1: FF/REC dominance
        ax = axes[1, col]
        for delay_bin, color in zip(selected, colors):
            branch_data = results.get(delay_bin, {}).get(branch_key, {})
            dom = branch_data.get("ff_rec_dominance", [])
            if dom:
                ax.plot(
                    dom, color=color, linewidth=1.5, label=f"D={float(delay_bin):.0f}"
                )
        ax.set_xlabel("Timestep", fontsize=9)
        ax.set_ylabel("REC Dominance", fontsize=9)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylim([-0.05, 1.05])
        ax.legend(fontsize=7)

    fig.suptitle(
        "E/I Balance and FF/REC Dominance",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Trace decomposition analysis for dendritic_branch models"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
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

    # Extract branch taus from config
    branch_taus = None
    if hasattr(config.model.core, "recurrent"):
        rec_cfg = config.model.core.recurrent
        if hasattr(rec_cfg, "branches"):
            branch_taus = [getattr(b, "tau", "?") for b in rec_cfg.branches]

    logger.info("Collecting trace data...")
    traces_list, _targets = collect_traces(
        model,
        test_ds,
        device=args.device,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
    )

    if not traces_list:
        logger.error(
            "No trace data collected. Ensure model uses dendritic_branch "
            "compartment and store_routing is supported."
        )
        return

    logger.info(f"Collected traces for {len(traces_list)} samples")

    # Extract delay conditions
    conditions = extract_delay_conditions(test_ds)
    if conditions is None:
        logger.warning("No delay conditions found; using sample index")
        conditions = list(range(len(traces_list)))

    # Compute metrics
    results = compute_trace_metrics(traces_list, conditions, n_bins=args.n_bins)

    # Save JSON
    json_path = os.path.join(args.output_dir, "trace_decomposition.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {json_path}")

    # Generate plots
    plot_trace_heatmap(
        results,
        os.path.join(args.output_dir, "trace_heatmap.png"),
        branch_taus=branch_taus,
    )
    plot_gain_profile(
        results,
        os.path.join(args.output_dir, "gain_profile.png"),
        branch_taus=branch_taus,
    )
    plot_ei_balance(
        results,
        os.path.join(args.output_dir, "ei_balance.png"),
        branch_taus=branch_taus,
    )

    logger.info(f"Trace decomposition analysis complete. Results: {args.output_dir}")


if __name__ == "__main__":
    main()
