"""
Analyze per-level routing conditioned on a task variable.

Loads a trained recurrent EINetwork model, runs forward passes with
store_routing=True, bins per-level contribution (beta_l) and FF/REC ratio
(gamma_l) by a specified condition (e.g. delay duration, stimulus class,
frequency band), and produces:
  - Population-averaged beta_l per level per condition bin
  - Temporal heatmaps: beta_l(t | condition)
  - Spearman correlation between distal-level utilization and condition
  - FF/REC ratio per level per condition
  - Saves results as JSON + generates matplotlib figures

Usage:
    PYTHONPATH=src:$PYTHONPATH python -m dendritic_modeling.scripts.analysis.analyze_routing_by_condition \
        --model_path <path/to/model.pt> \
        --config_path <path/to/config.yaml> \
        --output_dir <output_dir> \
        --condition_key delay \
        --n_samples 2000
"""

import argparse
import json
import logging
import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from scipy import stats

logger = logging.getLogger(__name__)


def load_model_and_config(model_path: str, config_path: str, device: str = "cpu"):
    """Load a trained model and its config."""
    from dendritic_modeling.config import load_config
    from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

    config = load_config(config_path)
    model_config = config.model

    # Determine input_dim from dataset
    from dendritic_modeling.datasets.sequence_datasets import get_sequence_datasets

    dataset_name = config.data.dataset_name
    ds_params = {}
    if (
        hasattr(config.data, "dataset_params")
        and config.data.dataset_params is not None
    ):
        ds_params_section = getattr(config.data.dataset_params, dataset_name, {})
        if hasattr(ds_params_section, "__dict__"):
            ds_params = {
                k: v
                for k, v in ds_params_section.__dict__.items()
                if not k.startswith("_")
            }

    train_ds, _, test_ds = get_sequence_datasets(
        dataset_name=dataset_name,
        train_valid_split=config.experiment.train_valid_split,
        **ds_params,
    )

    sample = test_ds[0][0] if hasattr(test_ds, "__getitem__") else train_ds[0][0]
    input_dim = sample.shape[-1]

    # Set input_dim on encoder config before model construction
    if (
        hasattr(model_config.encoder, "params")
        and model_config.encoder.params is not None
    ):
        model_config.encoder.params.input_dim = input_dim
        model_config.encoder.params.input_shape = list(sample.shape)

    model, _ = initialize_model(model_config)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, config, test_ds


def collect_routing_info(
    model,
    dataset,
    device: str = "cpu",
    batch_size: int = 64,
    n_samples: Optional[int] = None,
):
    """Run forward passes with store_routing=True and collect per-level routing.

    Returns:
        all_routing: list of dicts, one per sample. Each dict has:
            - "layer_0": {
                "level_contributions": Tensor [seq_len, n_levels],
                "ff_rec_ratios": Tensor [seq_len, n_levels],
              }
        all_targets: list of target values
    """
    from dendritic_modeling.networks.architectures.recurrent import EINetwork

    core = model.core_network
    if not isinstance(core, EINetwork):
        raise TypeError(f"Expected EINetwork, got {type(core)}")

    # Enable routing storage
    original_store_routing = core.store_routing
    core.store_routing = True

    if n_samples is not None and n_samples < len(dataset):
        indices = list(range(n_samples))
        subset = torch.utils.data.Subset(dataset, indices)
    else:
        subset = dataset

    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)

    all_routing = []
    all_targets = []

    try:
        with torch.no_grad():
            for batch in loader:
                x_batch = batch[0].to(device)
                y_batch = batch[1]
                _ = model(x_batch)

                # _routing_info is a list of dicts (one per timestep)
                # Each: {"layer_0": {"excitatory": {"level_contributions": [B, L],
                #                                    "ff_rec_ratios": [B, L]}, ...}}
                batch_size_actual = x_batch.shape[0]
                seq_len = len(core._routing_info)

                if seq_len == 0:
                    continue

                # Reorganize: per-sample, per-layer -> [seq_len, n_levels]
                for sample_idx in range(batch_size_actual):
                    sample_routing = {}
                    for layer_key in core._routing_info[0]:
                        level_contribs = []
                        ff_rec_rats = []
                        for t in range(seq_len):
                            if layer_key in core._routing_info[t]:
                                info = core._routing_info[t][layer_key]
                                exc_info = info.get("excitatory", info)
                                lc = exc_info["level_contributions"]
                                fr = exc_info["ff_rec_ratios"]
                                level_contribs.append(lc[sample_idx].cpu())
                                ff_rec_rats.append(fr[sample_idx].cpu())
                        if level_contribs:
                            # Normalize level_contributions to sum to 1
                            lc_stacked = torch.stack(level_contribs, dim=0)  # [T, L]
                            lc_sum = lc_stacked.sum(dim=-1, keepdim=True).clamp(
                                min=1e-8
                            )
                            lc_normalized = lc_stacked / lc_sum
                            sample_routing[layer_key] = {
                                "level_contributions": lc_normalized,
                                "ff_rec_ratios": torch.stack(ff_rec_rats, dim=0),
                            }
                    all_routing.append(sample_routing)

                all_targets.extend(y_batch.cpu().tolist())

    finally:
        core.store_routing = original_store_routing

    return all_routing, all_targets


def extract_delay_conditions(dataset):
    """Extract delay values from VariableDelayDMSDataset."""
    if hasattr(dataset, "dataset"):
        # Subset wrapper
        base = dataset.dataset
    else:
        base = dataset

    if hasattr(base, "delays"):
        return base.delays.tolist()

    # Fallback: try to infer from dataset attributes
    logger.warning("Dataset has no 'delays' attribute; using target as condition")
    return None


def bin_routing_by_condition(
    all_routing,
    conditions,
    n_bins: int = 10,
    layer_key: str = "layer_0",
):
    """Bin per-level routing by condition (e.g., delay).

    Returns:
        binned: dict mapping bin_center -> {
            "mean_beta": [n_levels] population-averaged level contribution,
            "temporal_beta": [seq_len, n_levels] time-resolved beta_l,
            "mean_gamma": [n_levels] population-averaged FF/REC ratio,
            "temporal_gamma": [seq_len, n_levels] time-resolved gamma_l,
            "n_samples": int,
        }
    """
    if conditions is None:
        return {}

    cond_array = np.array(conditions[: len(all_routing)])
    bin_edges = np.linspace(cond_array.min(), cond_array.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(cond_array, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    binned = {}
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if mask.sum() == 0:
            continue

        sample_indices = np.where(mask)[0]
        bin_center = float(bin_centers[bin_idx])

        temporal_betas = []
        temporal_gammas = []
        for idx in sample_indices:
            if layer_key in all_routing[idx]:
                routing = all_routing[idx][layer_key]
                beta = routing["level_contributions"].numpy()  # [T, L]
                gamma = routing["ff_rec_ratios"].numpy()  # [T, L]
                temporal_betas.append(beta)
                temporal_gammas.append(gamma)

        if not temporal_betas:
            continue

        # Pad to max seq_len (variable delay -> variable total length)
        n_levels = temporal_betas[0].shape[1]
        max_len = max(a.shape[0] for a in temporal_betas)
        padded_beta = np.zeros((len(temporal_betas), max_len, n_levels))
        padded_gamma = np.zeros((len(temporal_gammas), max_len, n_levels))
        for i, (b, g) in enumerate(zip(temporal_betas, temporal_gammas)):
            padded_beta[i, : b.shape[0], :] = b
            padded_gamma[i, : g.shape[0], :] = g

        temporal_mean_beta = padded_beta.mean(axis=0)  # [max_len, L]
        temporal_mean_gamma = padded_gamma.mean(axis=0)
        mean_beta = temporal_mean_beta.mean(axis=0)  # [L]
        mean_gamma = temporal_mean_gamma.mean(axis=0)

        binned[bin_center] = {
            "mean_beta": mean_beta.tolist(),
            "temporal_beta": temporal_mean_beta.tolist(),
            "mean_gamma": mean_gamma.tolist(),
            "temporal_gamma": temporal_mean_gamma.tolist(),
            "n_samples": int(mask.sum()),
        }

    return binned


def compute_routing_statistics(binned_results, distal_level_idx: int = 0):
    """Compute Spearman correlation between beta_distal and condition."""
    delays = []
    beta_distal_vals = []
    for condition_val, data in sorted(binned_results.items()):
        delays.append(condition_val)
        beta_distal_vals.append(data["mean_beta"][distal_level_idx])

    if len(delays) < 3:
        return {"spearman_rho": None, "spearman_p": None, "n_bins": len(delays)}

    rho, p = stats.spearmanr(delays, beta_distal_vals)
    return {
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "n_bins": len(delays),
        "delays": delays,
        "beta_distal": beta_distal_vals,
    }


def plot_routing_heatmap(binned_results, output_path: str, level_labels=None):
    """Generate routing heatmap: beta_l(t | condition)."""
    conditions = sorted(binned_results.keys())
    if not conditions:
        logger.warning("No binned results to plot")
        return

    n_levels = len(binned_results[conditions[0]]["mean_beta"])
    if level_labels is None:
        level_labels = [f"Level {lvl}" for lvl in range(n_levels)]

    fig, axes = plt.subplots(1, n_levels, figsize=(4 * n_levels, 6), sharey=True)
    if n_levels == 1:
        axes = [axes]

    for lvl, (ax, label) in enumerate(zip(axes, level_labels)):
        max_len = max(len(binned_results[c]["temporal_beta"]) for c in conditions)
        heatmap = np.full((len(conditions), max_len), np.nan)
        for i, c in enumerate(conditions):
            temporal = binned_results[c]["temporal_beta"]
            for t in range(len(temporal)):
                heatmap[i, t] = temporal[t][lvl]

        im = ax.imshow(
            heatmap,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=max(0.5, np.nanmax(heatmap)),
        )
        ax.set_xlabel("Timestep")
        if lvl == 0:
            ax.set_ylabel("Delay condition")
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels([f"{c:.0f}" for c in conditions], fontsize=8)
        ax.set_title(label, fontsize=11)

    fig.colorbar(im, ax=axes, label=r"$\bar{\beta}_l(t \mid D)$", shrink=0.8)
    fig.suptitle("Level Contribution vs Delay", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved routing heatmap to {output_path}")


def plot_beta_vs_condition(
    binned_results, stats_dict, output_path: str, level_labels=None
):
    """Plot population-averaged beta per level vs condition."""
    conditions = sorted(binned_results.keys())
    if not conditions:
        return

    n_levels = len(binned_results[conditions[0]]["mean_beta"])
    if level_labels is None:
        level_labels = [f"Level {lvl}" for lvl in range(n_levels)]

    _fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.coolwarm(np.linspace(0, 1, n_levels))

    for lvl in range(n_levels):
        betas = [binned_results[c]["mean_beta"][lvl] for c in conditions]
        ax.plot(
            conditions,
            betas,
            "o-",
            color=colors[lvl],
            label=level_labels[lvl],
            linewidth=2,
        )

    ax.set_xlabel("Delay D", fontsize=12)
    ax.set_ylabel(r"$\bar{\beta}_l(D)$", fontsize=12)
    ax.legend(fontsize=10)

    if stats_dict.get("spearman_rho") is not None:
        rho = stats_dict["spearman_rho"]
        p = stats_dict["spearman_p"]
        ax.set_title(
            rf"Level contribution vs delay ($\rho_{{distal}}$={rho:.3f}, p={p:.1e})",
            fontsize=11,
        )
    else:
        ax.set_title("Level contribution vs delay", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved beta-vs-condition plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze routing by condition")
    parser.add_argument(
        "--model_path", required=True, help="Path to saved model state dict"
    )
    parser.add_argument(
        "--config_path", required=True, help="Path to experiment config YAML"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for results"
    )
    parser.add_argument("--dataset", default="variable_delay_dms", help="Dataset name")
    parser.add_argument(
        "--condition_key", default="delay", help="Condition key (delay or target)"
    )
    parser.add_argument(
        "--n_samples", type=int, default=2000, help="Number of samples to analyze"
    )
    parser.add_argument(
        "--n_bins", type=int, default=10, help="Number of condition bins"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for forward passes"
    )
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--layer_key", default="layer_0", help="Layer key to analyze")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Loading model from {args.model_path}")
    model, config, test_ds = load_model_and_config(
        args.model_path, args.config_path, args.device
    )

    # Extract conditions
    if args.condition_key == "delay":
        conditions = extract_delay_conditions(test_ds)
    else:
        conditions = None

    logger.info(f"Collecting routing info from {args.n_samples} samples...")
    all_routing, all_targets = collect_routing_info(
        model,
        test_ds,
        device=args.device,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
    )
    logger.info(f"Collected routing for {len(all_routing)} samples")

    # Use conditions or targets as binning variable
    bin_conditions = conditions if conditions is not None else all_targets

    # Bin routing by condition
    binned = bin_routing_by_condition(
        all_routing,
        bin_conditions,
        n_bins=args.n_bins,
        layer_key=args.layer_key,
    )

    # Level labels from tau values
    core_cfg = config.model.core
    level_labels = None
    if hasattr(core_cfg, "unified_ei") and core_cfg.unified_ei is not None:
        layers_cfg = core_cfg.unified_ei.layers
        if layers_cfg:
            exc_cfg = (
                layers_cfg[0].excitatory
                if hasattr(layers_cfg[0], "excitatory")
                else None
            )
            if exc_cfg is not None:
                tau_base = getattr(exc_cfg, "tau_base", 5.0)
                tau_ratio = getattr(exc_cfg, "tau_ratio", 5.0)
                bf = getattr(exc_cfg, "branch_factors", [3, 3])
                n_levels = len(bf) + 1
                level_labels = []
                for lvl in range(n_levels):
                    tau_lvl = tau_base * tau_ratio ** (n_levels - 1 - lvl)
                    level_labels.append(rf"L{lvl} ($\tau$={tau_lvl:.0f})")

    # Compute statistics
    routing_stats = compute_routing_statistics(binned, distal_level_idx=0)
    logger.info(f"Routing statistics: {routing_stats}")

    # Save results
    results = {
        "routing_stats": routing_stats,
        "binned_results": {str(k): v for k, v in binned.items()},
        "n_samples": len(all_routing),
        "condition_key": args.condition_key,
        "layer_key": args.layer_key,
    }
    with open(os.path.join(args.output_dir, "routing_by_condition.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Generate figures
    plot_routing_heatmap(
        binned,
        os.path.join(args.output_dir, "routing_heatmap.png"),
        level_labels=level_labels,
    )
    plot_beta_vs_condition(
        binned,
        routing_stats,
        os.path.join(args.output_dir, "beta_vs_condition.png"),
        level_labels=level_labels,
    )

    logger.info(f"Analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
