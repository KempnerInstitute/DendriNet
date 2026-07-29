"""Submit optimized per-seed counterfactual psychometric evaluation jobs."""

from __future__ import annotations

import argparse
import hashlib
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
    load_study_config,
)

RECEIPT_SCHEMA = "dendritic-rnn-counterfactual-submission-receipt-v1"
EVALUATOR_MODULE = (
    "dendritic_modeling.scripts.analysis.evaluate_dendritic_rnn_counterfactual_bank"
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _submit(command: list[str], *, dry_run: bool) -> str | None:
    if dry_run:
        return None
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip().split(";")[0]


def submit_jobs(
    *,
    study_config_path: Path,
    replicates: list[int],
    dry_run: bool,
) -> dict[str, Any]:
    config, study_config_path = load_study_config(study_config_path)
    study = config["study"]
    resources = config["resources"]
    runtime = config["runtime"]
    output_root = _resolve_repo_path(str(study["output_root"])).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    slurm_dir = output_root / "_slurm"
    launch_dir = output_root / "_launch"
    slurm_dir.mkdir(exist_ok=True)
    launch_dir.mkdir(exist_ok=True)
    evaluator_path = (
        REPO_ROOT
        / "src/dendritic_modeling/scripts/analysis/evaluate_dendritic_rnn_counterfactual_bank.py"
    )
    evaluator_sha256 = _sha256(evaluator_path)
    config_sha256 = _sha256(study_config_path)
    runtime_sha256 = evaluation_runtime_source_sha256()
    environment = Path(str(runtime["python"])).absolute()
    if not environment.is_file():
        raise RuntimeError(f"Pinned Python environment is missing: {environment}")

    jobs = []
    for replicate in replicates:
        if replicate not in range(int(study["replicates"])):
            raise ValueError(f"Invalid replicate: {replicate}")
        if (output_root / f"r{replicate:03d}" / "COMPLETE.json").exists():
            jobs.append({"replicate": replicate, "status": "already_complete"})
            continue
        pythonpath = _resolve_repo_path(str(runtime["pythonpath"])).resolve()
        payload = [
            "/usr/bin/env",
            f"PYTHONPATH={pythonpath}",
            "PYTHONDONTWRITEBYTECODE=1",
            "OMP_NUM_THREADS=1",
            str(environment),
            "-m",
            EVALUATOR_MODULE,
            "--study-config",
            str(study_config_path),
            "--replicate",
            str(replicate),
            "--expected-evaluator-sha256",
            evaluator_sha256,
            "--expected-runtime-source-sha256",
            runtime_sha256,
            "--expected-config-sha256",
            config_sha256,
        ]
        job_name = f"dendrnn-cf-r{replicate:03d}"
        command = [
            "sbatch",
            "--parsable",
            f"--job-name={job_name}",
            f"--partition={resources['partition']}",
            f"--account={resources['account']}",
            f"--gres=gpu:{int(resources['gpus'])}",
            f"--cpus-per-task={int(resources['cpus_per_task'])}",
            f"--mem={resources['memory']}",
            f"--time={resources['walltime']}",
            f"--output={slurm_dir / (job_name + '_%j.out')}",
            f"--error={slurm_dir / (job_name + '_%j.err')}",
            "--wrap",
            shlex.join(payload),
        ]
        job_id = _submit(command, dry_run=dry_run)
        jobs.append(
            {
                "replicate": replicate,
                "status": "dry_run" if dry_run else "submitted",
                "job_id": job_id,
                "command": command,
            }
        )
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dry_run": dry_run,
        "study_config": {
            "path": str(study_config_path),
            "sha256": config_sha256,
        },
        "evaluator": {"path": str(evaluator_path), "sha256": evaluator_sha256},
        "runtime_source_sha256": runtime_sha256,
        "resource_contract": dict(resources),
        "jobs": jobs,
    }
    receipt_path = launch_dir / f"submission_{_utc_stamp()}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    receipt["receipt_path"] = str(receipt_path)
    receipt["receipt_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-config", type=Path, required=True)
    parser.add_argument("--replicates", nargs="*", type=int, default=list(range(10)))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    receipt = submit_jobs(
        study_config_path=args.study_config,
        replicates=args.replicates,
        dry_run=args.dry_run,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
