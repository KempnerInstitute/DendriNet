"""Generate a one-epoch sweep with the installed CLI and run its job script."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from dendritic_modeling.examples import example_text
from dendritic_modeling.scripts.sweeps.slurm_profile import CHECKOUT_ROOT


def check(root: Path, *, array_task: bool = False) -> None:
    if CHECKOUT_ROOT is not None:
        raise RuntimeError("Run this check with the installed wheel outside a checkout")
    base = yaml.safe_load(example_text("synthetic_population"))
    base["training"]["main"]["common"]["epochs"] = 1
    sweep = {
        "base_config": base,
        "sweep_config": {
            "model.core.population_network.layers.0.population_defaults.ff_excitatory_synapses": (
                [7, 8] if array_task else [8]
            )
        },
        "experiment_settings": {"seeds_per_condition": 1},
    }
    spec = root / "sweep.yaml"
    spec.write_text(yaml.safe_dump(sweep))
    command = [
        str(Path(sys.executable).with_name("dendritic-sweep")),
        "--config",
        str(spec),
        "--generate-only",
        "--output-dir",
        str(root / "generated"),
    ]
    subprocess.run(command, cwd=root, check=True)
    jobs = list((root / "generated").glob("*/jobs/*.sh"))
    (job,) = jobs
    working_dir = job.parent.parent
    header = job.read_text().splitlines()
    assert f"#SBATCH --chdir={working_dir}" in header
    env = os.environ.copy()
    if array_task:
        assert job.name == "run_array_sweep.sh"
        env.update(SLURM_ARRAY_JOB_ID=env["SLURM_JOB_ID"], SLURM_ARRAY_TASK_ID="0")
    subprocess.run(["bash", str(job)], cwd=working_dir, env=env, check=True)
    config_result = working_dir / "results" / "config_0"
    summary = json.loads((config_result / "training_summary.json").read_text())
    assert (
        len(summary["train_losses"]) == 1
        and (config_result / "performance" / "final.json").is_file()
    )
    print(f"Installed sweep completed one epoch using {job.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--array-task",
        action="store_true",
        help="Run a real srun step inside an existing Slurm allocation",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
        check(Path(temp), array_task=args.array_task)
