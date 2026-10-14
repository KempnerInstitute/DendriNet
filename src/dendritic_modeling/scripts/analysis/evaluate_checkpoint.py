"""Evaluate a saved checkpoint on the held-out splits of its training config.

Recovery path for runs whose training completed but whose in-process final
evaluation never ran (node hang, timeout at the eval phase): rebuilds the
exact model and datasets through the package setup path, loads the
checkpoint strictly, and writes valid/test top-1 and NLL to JSON.

Usage:
    python -m dendritic_modeling.scripts.analysis.evaluate_checkpoint \
        --config <run config yaml> --checkpoint <state dict .pt> \
        --output <results json> [--splits valid test] [--batch-size 128]
"""

import argparse
import json
import os
import tempfile

import torch

from dendritic_modeling.config.config import load_config
from dendritic_modeling.networks.checkpoints import load_initial_model_checkpoint
from dendritic_modeling.scripts.script_utils.setup_utils import setup_environment
from dendritic_modeling.utils.logging_config import setup_logging


def _evaluate_split(model, dataset, device, batch_size, num_workers):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device == "cuda",
    )
    correct = total = 0
    nll_sum = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            log_probs = torch.log_softmax(logits, dim=-1)
            nll_sum += float(
                torch.nn.functional.nll_loss(log_probs, yb, reduction="sum")
            )
            correct += int((logits.argmax(-1) == yb).sum())
            total += yb.numel()
    return {
        "top1_percent": round(100.0 * correct / total, 4),
        "nll": round(nll_sum / total, 6),
        "n": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", nargs="+", default=["valid", "test"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(args.config)
    # Route setup artifacts (dirs, logs) away from the original run.
    config.outputs.results_dir = tempfile.mkdtemp(prefix="ckpt_eval_")

    _, _, valid_dataset, test_dataset, model, _ = setup_environment(
        config.model,
        config.training,
        config.data,
        config.outputs,
        config.experiment,
    )
    record = load_initial_model_checkpoint(model, args.checkpoint, strict=True)
    model = model.to(device).eval()

    datasets = {"valid": valid_dataset, "test": test_dataset}
    results = {
        "config": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(args.checkpoint),
        "sha256": record["sha256"],
    }
    for split in args.splits:
        results[split] = _evaluate_split(
            model, datasets[split], device, args.batch_size, args.num_workers
        )
        print(split, results[split], flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(results, handle, indent=1)
    print("wrote", args.output)


if __name__ == "__main__":
    setup_logging()
    main()
