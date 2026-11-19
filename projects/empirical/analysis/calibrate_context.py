"""Calibrate gates on training presentations, then use native initial_checkpoint.

The task-specific sampling is paper-owned. Gate calibration, model construction
and checkpoint loading remain native DendriNet operations. No validation or test
examples enter calibration and no library function is patched.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml

from dendritic_modeling.config import load_config
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize import (
    calibrate_reactivation_from_data,
)
from dendritic_modeling.scripts.script_utils.setup_utils import setup_environment


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    config = load_config(str(args.config))
    config.outputs.results_dir = str(args.output / "setup")
    config.outputs.run_name = "calibration"
    _, training, valid, test, model, _ = setup_environment(
        config.model, config.training, config.data, config.outputs, config.experiment
    )
    del valid, test
    model.to("cuda").eval()
    stream = training.stream.to("cuda")
    seed = int(config.experiment.seed)
    index_rng = torch.Generator(device="cuda").manual_seed(seed + 7300)
    noise_rng = torch.Generator(device="cuda").manual_seed(seed + 7301)
    common = config.training.main.common
    batches = []
    for _ in range(common.reactivation_initialization_num_batches):
        x, _ = stream.sample(
            common.reactivation_initialization_batch_size,
            index_generator=index_rng,
            noise_generator=noise_rng,
        )
        batches.append(x)
    hashes = [
        hashlib.sha256(x.cpu().contiguous().numpy().tobytes()).hexdigest()
        for x in batches
    ]
    previous = None
    changes = []
    diagnostics = None
    converged = False
    for _iteration in range(common.reactivation_initialization_max_iterations or 50):
        diagnostics = calibrate_reactivation_from_data(
            model,
            batches,
            k=None,
            device=torch.device("cuda"),
            mode=None,
            quantile_aggregation="global",
        )
        if not diagnostics:
            raise ValueError("No eligible gates in contextual calibration")
        current = {
            key: (float(value["m"]), float(value["b"]))
            for key, value in diagnostics.items()
        }
        if previous is not None and current.keys() == previous.keys():
            change = max(
                max(abs(a - b) for a, b in zip(current[k], previous[k]))
                for k in current
            )
            changes.append(change)
            if change < 1e-3:
                converged = True
                break
        previous = current
    receipt = {
        "config_sha256": sha(args.config),
        "split": "optimization_training_only",
        "sample_seed": seed + 7300,
        "noise_seed": seed + 7301,
        "batches_sha256": hashes,
        "iterations": _iteration + 1,
        "changes": changes,
        "converged": converged,
        "diagnostics": diagnostics,
    }
    (args.output / "calibration.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )
    if not converged:
        raise ValueError(
            "Contextual calibration did not converge; inspect before training"
        )
    if not all(bool(torch.isfinite(t).all()) for t in model.state_dict().values()):
        raise ValueError("Nonfinite calibrated model")
    checkpoint = args.output / "initial_state.pt"
    torch.save(model.state_dict(), checkpoint)
    payload = yaml.safe_load(args.config.read_text())
    payload["model"].update(
        initial_checkpoint=str(checkpoint.resolve()),
        initial_checkpoint_sha256=sha(checkpoint),
        initial_checkpoint_strict=True,
    )
    (args.output / "effective_config.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False)
    )


if __name__ == "__main__":
    main()
