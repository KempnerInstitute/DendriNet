"""
Evaluate trained recurrent models on out-of-distribution delay ranges.

Generates a VariableDelayDMS dataset with a delay range outside the training
distribution and measures accuracy as a function of delay, producing binned
accuracy curves and summary statistics.

Usage:
    PYTHONPATH=src:$PYTHONPATH python -m dendritic_modeling.scripts.analysis.evaluate_ood_delay \
        --model_path <path/to/final_model.pt> \
        --config_path <path/to/config.yaml> \
        --output_dir <output_dir> \
        --min_delay 100 --max_delay 500 --n_samples 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data

from dendritic_modeling.config.conversion import dataset_specific_params
from dendritic_modeling.config.seeds import derive_seed
from dendritic_modeling.datasets.sequence_datasets import VariableDelayDMSDataset
from dendritic_modeling.scripts.analysis.analyze_routing_by_condition import (
    load_model_and_config,
)
from dendritic_modeling.utils import resolve_experiment_seeds
from dendritic_modeling.utils.logging_config import setup_logging
from dendritic_modeling.utils.reproducibility import isolated_random_seed

logger = logging.getLogger(__name__)


def _accuracy_by_delay_bins(preds, labels, delays, n_bins: int = 10):
    delays = np.asarray(delays)
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    edges = np.linspace(delays.min(), delays.max(), n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    bin_ids = np.digitize(delays, edges) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    out = {}
    for i in range(n_bins):
        mask = bin_ids == i
        if not np.any(mask):
            continue
        acc = float((preds[mask] == labels[mask]).mean())
        out[str(float(centers[i]))] = {
            "accuracy": acc,
            "n_samples": int(mask.sum()),
        }
    return out


def _plot_delay_accuracy(binned_acc: dict, output_path: str) -> None:
    xs = []
    ys = []
    for key in sorted(binned_acc, key=lambda k: float(k)):
        xs.append(float(key))
        ys.append(binned_acc[key]["accuracy"])

    if not xs:
        return

    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, ys, "o-", linewidth=2)
    plt.ylim([0.0, 1.05])
    plt.xlabel("Delay")
    plt.ylabel("Accuracy")
    plt.title("OOD Delay Accuracy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="OOD delay evaluation")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_delay", type=int, default=100)
    parser.add_argument("--max_delay", type=int, default=500)
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed of the OOD trials (default: derived from the run's evaluation seed)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, config, _ = load_model_and_config(
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device,
        output_dir=args.output_dir,
    )
    model.eval()

    if config.data.dataset_name != "variable_delay_dms":
        raise ValueError(
            "OOD delay evaluation needs a model trained on variable_delay_dms, "
            f"not {config.data.dataset_name!r}"
        )
    # Trial shape as in training (the dataset defaults where the run left
    # them unset); only the delay range and trial count change.
    configured = dataset_specific_params(config.data)
    trial_shape = {
        key: int(configured[key])
        for key in ("stimulus_dim", "stimulus_duration", "probe_duration")
        if key in configured
    }
    seed = (
        int(args.seed)
        if args.seed is not None
        else derive_seed(
            resolve_experiment_seeds(config.experiment).evaluation_seed,
            "evaluate_ood_delay",
        )
    )
    with isolated_random_seed(seed):
        dataset = VariableDelayDMSDataset(
            n_samples=args.n_samples,
            min_delay=int(args.min_delay),
            max_delay=int(args.max_delay),
            **trial_shape,
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            x_batch = batch[0].to(args.device)
            y_batch = batch[1]
            preds = model.predict(x_batch)
            all_preds.append(preds.cpu())
            all_labels.append(y_batch.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    delays = dataset.delays.numpy()

    overall_acc = float((preds == labels).mean())
    by_bin = _accuracy_by_delay_bins(preds, labels, delays, n_bins=args.n_bins)

    out = {
        "ood_min_delay": args.min_delay,
        "ood_max_delay": args.max_delay,
        "n_samples": args.n_samples,
        "seed": seed,
        "trial_shape": trial_shape,
        "overall_accuracy": overall_acc,
        "accuracy_by_delay_bin": by_bin,
    }

    with open(os.path.join(args.output_dir, "ood_delay_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    _plot_delay_accuracy(
        binned_acc=by_bin,
        output_path=os.path.join(args.output_dir, "ood_delay_accuracy.png"),
    )

    logger.info(
        "OOD delay evaluation complete: acc=%.4f over [%d, %d]",
        overall_acc,
        args.min_delay,
        args.max_delay,
    )


if __name__ == "__main__":
    setup_logging()
    main()
