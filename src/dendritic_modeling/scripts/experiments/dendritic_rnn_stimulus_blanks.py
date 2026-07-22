"""Submit optimized four-checkpoint stimulus-blank jobs by trained seed."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dendritic_modeling.scripts.analysis.evaluate_dendritic_rnn_counterfactual_bank import (
    REPO_ROOT,
    _resolve_repo_path,
    _sha256,
    evaluation_runtime_source_sha256,
)
from dendritic_modeling.scripts.analysis.evaluate_dendritic_rnn_stimulus_blanks import (
    load_study_config,
)

EVALUATOR_MODULE = (
    "dendritic_modeling.scripts.analysis.evaluate_dendritic_rnn_stimulus_blanks"
)


def submit_jobs(
    *, study_config_path: Path, replicates: list[int], dry_run: bool
) -> dict[str, Any]:
    config, study_config_path = load_study_config(study_config_path)
    study = config["study"]
    resources = config["resources"]
    runtime = config["runtime"]
    output_root = _resolve_repo_path(str(study["output_root"])).resolve()
    slurm_dir = output_root / "_slurm"
    launch_dir = output_root / "_launch"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    launch_dir.mkdir(parents=True, exist_ok=True)
    evaluator = (
        REPO_ROOT
        / "src/dendritic_modeling/scripts/analysis/evaluate_dendritic_rnn_stimulus_blanks.py"
    )
    hashes = {
        "evaluator": _sha256(evaluator),
        "runtime": evaluation_runtime_source_sha256(),
        "config": _sha256(study_config_path),
    }
    python = Path(str(runtime["python"])).absolute()
    pythonpath = _resolve_repo_path(str(runtime["pythonpath"])).resolve()
    jobs = []
    for replicate in replicates:
        if replicate not in range(10):
            raise ValueError(f"Invalid replicate: {replicate}")
        if (output_root / f"r{replicate:03d}" / "COMPLETE.json").exists():
            jobs.append({"replicate": replicate, "status": "already_complete"})
            continue
        payload = [
            "/usr/bin/env",
            f"PYTHONPATH={pythonpath}",
            "PYTHONDONTWRITEBYTECODE=1",
            "OMP_NUM_THREADS=1",
            str(python),
            "-m",
            EVALUATOR_MODULE,
            "--study-config",
            str(study_config_path),
            "--replicate",
            str(replicate),
            "--expected-evaluator-sha256",
            hashes["evaluator"],
            "--expected-runtime-source-sha256",
            hashes["runtime"],
            "--expected-config-sha256",
            hashes["config"],
        ]
        name = f"dendrnn-blank-r{replicate:03d}"
        command = [
            "sbatch",
            "--parsable",
            f"--job-name={name}",
            f"--partition={resources['partition']}",
            f"--account={resources['account']}",
            f"--gres=gpu:{resources['gpus']}",
            f"--cpus-per-task={resources['cpus_per_task']}",
            f"--mem={resources['memory']}",
            f"--time={resources['walltime']}",
            f"--output={slurm_dir / (name + '_%j.out')}",
            f"--error={slurm_dir / (name + '_%j.err')}",
            "--wrap",
            shlex.join(payload),
        ]
        job_id = None
        if not dry_run:
            completed = subprocess.run(
                command, check=True, capture_output=True, text=True
            )
            job_id = completed.stdout.strip().split(";")[0]
        jobs.append(
            {
                "replicate": replicate,
                "status": "dry_run" if dry_run else "submitted",
                "job_id": job_id,
                "command": command,
            }
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    receipt = {
        "schema_version": "dendritic-rnn-stimulus-blank-submission-receipt-v1",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dry_run": dry_run,
        "study_config": {"path": str(study_config_path), "sha256": hashes["config"]},
        "evaluator": {"path": str(evaluator), "sha256": hashes["evaluator"]},
        "runtime_source_sha256": hashes["runtime"],
        "resource_contract": dict(resources),
        "jobs": jobs,
    }
    receipt_path = launch_dir / f"submission_{stamp}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    receipt["receipt_path"] = str(receipt_path)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-config", type=Path, required=True)
    parser.add_argument("--replicates", nargs="*", type=int, default=list(range(10)))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            submit_jobs(
                study_config_path=args.study_config,
                replicates=args.replicates,
                dry_run=args.dry_run,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
