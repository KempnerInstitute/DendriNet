"""Run one array-indexed frozen-checkpoint analysis from a JSONL inventory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dendritic_modeling.scripts.analysis.run_checkpoint_analysis import (
    run_checkpoint_analysis,
)


def run_declared_job(*, jobs_path: Path, job_index: int) -> dict[str, Any]:
    """Load one immutable job declaration and execute the standard analyzer."""
    jobs = [
        json.loads(line)
        for line in jobs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if job_index < 0 or job_index >= len(jobs):
        raise IndexError(f"job index {job_index} is outside 0..{len(jobs) - 1}")
    job = jobs[job_index]
    return run_checkpoint_analysis(
        config_path=job["config"],
        checkpoint_path=job["checkpoint"],
        analysis_config_path=job["analysis_config"],
        output_dir=job["output_dir"],
        filename=str(job.get("filename", "final")),
        device=job.get("device"),
        strict=True,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--job-index", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    index = args.job_index
    if index is None:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None:
            raise RuntimeError("set --job-index or SLURM_ARRAY_TASK_ID")
        index = int(raw)
    return run_declared_job(jobs_path=args.jobs, job_index=index)


if __name__ == "__main__":  # pragma: no cover
    main()
