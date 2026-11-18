#!/usr/bin/env python3
"""Write (and optionally submit) a Slurm script for one training run.

The job runs the entry point that the config's ``distributed.mode`` selects:
one process on one GPU for ``none``, and one process per GPU through
``torchrun`` on one node for ``ddp`` and ``fsdp``. For ``fsdp`` the GPU count
defaults to the smallest power of two whose sharded memory estimate fits
``--gpu-memory-gb``; ``ddp`` replicates the model, so its count is given.
Account and partition are always explicit. The header, environment and
launcher come from the ``SlurmProfile`` the sweep tools use. Multi-node FSDP
runs use ``scripts/train_fsdp.sh``.

Usage:
    python -m dendritic_modeling.scripts.tools.generate_slurm_job \\
        configs/config_exp_fsdp.yaml --account ACCOUNT --partition PARTITION
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dendritic_modeling.config import load_config
from dendritic_modeling.datasets.dimensions import dataset_input_shape
from dendritic_modeling.scripts.sweeps.slurm_profile import (
    LIBRARY_SOURCE_ROOT,
    SlurmProfile,
    job_working_directory,
)
from dendritic_modeling.utils.gpu_estimation import estimate_config
from dendritic_modeling.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

REPO_ROOT = LIBRARY_SOURCE_ROOT

TRAINING_SCRIPTS = {
    "none": "src/dendritic_modeling/scripts/training/train_experiments.py",
    "ddp": "src/dendritic_modeling/scripts/training/train_experiments.py",
    "fsdp": "src/dendritic_modeling/scripts/training/train_experiments_fsdp.py",
}


def _job_input_shape(dataset_name: str | None) -> tuple[int, ...]:
    """Return the flat input shape used by SLURM resource estimation."""
    return dataset_input_shape(
        dataset_name,
        flatten=True,
        default_input_dim=784,
    )


@dataclass(frozen=True)
class JobPlan:
    """What one training job runs and on how many GPUs."""

    mode: str
    gpus_per_node: int
    training_script: str
    # Estimated memory per GPU (GB) at gpus_per_node, when estimated.
    memory_per_gpu_gb: Optional[float] = None


def plan_job(
    config_path: str,
    *,
    gpus_per_node: Optional[int] = None,
    gpu_memory_gb: float = 80.0,
    max_gpus_per_node: int = 4,
) -> JobPlan:
    """Choose the entry point and GPU count of a run from its config."""
    config = load_config(config_path)
    mode = str(config.distributed.mode).lower()
    script = TRAINING_SCRIPTS[mode]
    if mode == "none":
        if gpus_per_node not in (None, 1):
            raise ValueError(
                "distributed.mode 'none' runs one process on one GPU; set "
                "distributed.mode to 'ddp' or 'fsdp' to use more"
            )
        return JobPlan(mode, 1, script)
    if mode == "ddp":
        if gpus_per_node is None or gpus_per_node < 2:
            raise ValueError(
                "distributed.mode 'ddp' replicates the model on every GPU; give "
                "--gpus-per-node (at least 2)"
            )
        return JobPlan(mode, int(gpus_per_node), script)
    if gpus_per_node is not None:
        return JobPlan(mode, int(gpus_per_node), script)
    estimate = estimate_config(config_path, report=False)
    count = 1
    while count <= max_gpus_per_node:
        memory = estimate.fsdp_memory_per_gpu(count)
        if memory <= gpu_memory_gb:
            return JobPlan(mode, count, script, memory)
        count *= 2
    raise ValueError(
        f"the FSDP memory estimate exceeds {gpu_memory_gb} GB per GPU on "
        f"{max_gpus_per_node} GPUs of one node; use scripts/train_fsdp.sh for a "
        "multi-node run or give --gpus-per-node"
    )


def render_job_script(
    config_path: str, plan: JobPlan, profile: SlurmProfile, log_dir: Path
) -> str:
    """The job script: header, environment and the training command."""
    if profile.gpus_per_node != plan.gpus_per_node:
        raise ValueError("the Slurm profile and the job plan request different GPUs")
    working_dir = job_working_directory(Path.cwd())
    config_arg = os.path.relpath(os.path.abspath(config_path), start=working_dir)
    header = profile.header(
        repo_root=working_dir,
        output=str(log_dir / f"{profile.job_name}_%j.out"),
        error=str(log_dir / f"{profile.job_name}_%j.err"),
    )
    return (
        header
        + "set -eo pipefail\n\n"
        + profile.environment()
        + f"{profile.training_command(plan.training_script)} {shlex.quote(config_arg)}\n"
    )


def _profile(args, plan: JobPlan, job_name: str) -> SlurmProfile:
    gpus = plan.gpus_per_node
    return SlurmProfile.from_config(
        {
            "account": args.account,
            "partition": args.partition,
            "job_name": job_name,
            "qos": args.qos,
            "time": args.time,
            "constraint": args.constraint,
            "gpus_per_node": gpus,
            "cpus_per_task": args.cpus_per_task or 8 * gpus,
            "mem": args.mem or f"{64 * gpus}G",
            "modules_to_load": args.module,
            "conda_env_path": args.conda_env_path,
            "launcher": "python" if plan.mode == "none" else "torchrun",
            "training_script": plan.training_script,
        }
    )


def build_parser(
    description: str = __doc__.split("\n\n")[0],
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("config", help="Experiment configuration file")
    parser.add_argument("--account", required=True, help="Slurm account")
    parser.add_argument("--partition", required=True, help="Slurm partition")
    parser.add_argument("--time", default="02:00:00", help="Wall time (HH:MM:SS)")
    parser.add_argument("--qos")
    parser.add_argument("--constraint")
    parser.add_argument("--gpus-per-node", type=int)
    parser.add_argument(
        "--gpu-memory-gb",
        type=float,
        default=80.0,
        help="Memory of one GPU, for the FSDP GPU count",
    )
    parser.add_argument("--max-gpus-per-node", type=int, default=4)
    parser.add_argument("--cpus-per-task", type=int, help="Default: 8 per GPU")
    parser.add_argument("--mem", help="Default: 64G per GPU")
    parser.add_argument("--module", action="append", default=[], help="Module to load")
    parser.add_argument(
        "--conda-env-path", help="Environment whose bin/python runs the job"
    )
    parser.add_argument("--job-name", help="Default: outputs.run_name")
    parser.add_argument(
        "--output-dir", help="Default: <outputs.results_dir>/slurm_jobs"
    )
    parser.add_argument("--submit", action="store_true", help="Submit with sbatch")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the script without writing it"
    )
    return parser


def run(args) -> Optional[str]:
    """Write (and with ``--submit`` submit) the job; returns the job id."""
    plan = plan_job(
        args.config,
        gpus_per_node=args.gpus_per_node,
        gpu_memory_gb=args.gpu_memory_gb,
        max_gpus_per_node=args.max_gpus_per_node,
    )
    config = load_config(args.config)
    job_name = args.job_name or config.outputs.run_name
    output_dir = Path(
        args.output_dir or Path(config.outputs.results_dir) / "slurm_jobs"
    )
    profile = _profile(args, plan, job_name)
    script = render_job_script(args.config, plan, profile, output_dir / "logs")
    if args.dry_run:
        print(script)
        return None
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    script_path = output_dir / f"{job_name}.sbatch"
    script_path.write_text(script)
    logger.info(
        "Wrote %s: %s on %d GPU(s)%s",
        script_path,
        plan.mode,
        plan.gpus_per_node,
        (
            f", estimated {plan.memory_per_gpu_gb:.1f} GB per GPU"
            if plan.memory_per_gpu_gb is not None
            else ""
        ),
    )
    if not args.submit:
        return None
    submitted = subprocess.run(
        ["sbatch", "--parsable", str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    job_id = submitted.stdout.strip().split(";")[0]
    logger.info("Submitted job %s", job_id)
    return job_id


def main(argv: Optional[Sequence[str]] = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
