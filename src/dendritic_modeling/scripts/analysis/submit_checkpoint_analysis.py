"""Prepare and submit YAML-declared frozen-checkpoint analysis arrays.

The manifest is figure-agnostic: every scientific job is explicit, all inputs
are hash-bound, complete inventories are immutable, and retries receive new
execution ledgers. This command schedules only the standard package checkpoint
analyzer; it does not train models or compute paper-local quantities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from dendritic_modeling.scripts.sweeps.control_plane import (
    discover_source_identity,
    validate_scheduler_profile,
)

SCHEMA_VERSION = "checkpoint_analysis_manifest_v1"
SUBMISSION_SCHEMA_VERSION = "checkpoint_analysis_submission_v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sha256(path: Path, *, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _git_head(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tracked_worktree_dirty(repository: Path) -> bool:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return bool(status)


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint-analysis manifest: {path}")
    return payload


def _completion_artifacts(manifest: Mapping[str, Any]) -> list[str]:
    artifacts = manifest.get("completion_artifacts", [])
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("completion_artifacts must be a non-empty list")
    normalized = [str(value) for value in artifacts]
    if any(
        Path(value).is_absolute() or ".." in Path(value).parts for value in normalized
    ):
        raise ValueError("completion_artifacts must be relative output paths")
    return normalized


def _build_jobs(
    manifest: Mapping[str, Any], *, manifest_path: Path
) -> list[dict[str, Any]]:
    declarations = manifest.get("jobs")
    jobs_file_value = manifest.get("jobs_file")
    if declarations is not None and jobs_file_value is not None:
        raise ValueError("declare jobs or jobs_file, not both")
    if jobs_file_value is not None:
        jobs_file = _resolve_path(str(jobs_file_value), base=manifest_path.parent)
        if not jobs_file.is_file():
            raise FileNotFoundError(jobs_file)
        jobs_payload = yaml.safe_load(jobs_file.read_text(encoding="utf-8"))
        if (
            not isinstance(jobs_payload, dict)
            or jobs_payload.get("schema_version") != "checkpoint_analysis_jobs_v1"
        ):
            raise ValueError(f"unsupported checkpoint jobs file: {jobs_file}")
        declarations = jobs_payload.get("jobs")
    if not isinstance(declarations, list) or not declarations:
        raise ValueError("jobs or jobs_file must provide a non-empty list")
    base = manifest_path.parent
    output_root = _resolve_path(str(manifest["output_root"]), base=base)
    profile = _resolve_path(str(manifest["analysis_config"]), base=base)
    if not profile.is_file():
        raise FileNotFoundError(profile)
    declared_profile_hash = manifest.get("analysis_config_sha256")
    profile_hash = _sha256(profile)
    if declared_profile_hash is not None and str(declared_profile_hash) != profile_hash:
        raise ValueError("analysis config hash differs from manifest")
    artifacts = _completion_artifacts(manifest)

    jobs: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    outputs: set[str] = set()
    for index, declaration in enumerate(declarations):
        if not isinstance(declaration, dict):
            raise TypeError(f"jobs[{index}] must be a mapping")
        missing = {"id", "config", "checkpoint"} - set(declaration)
        if missing:
            raise ValueError(f"jobs[{index}] lacks required fields: {sorted(missing)}")
        identifier = str(declaration["id"])
        if not identifier or identifier in identifiers:
            raise ValueError(f"duplicate or blank job id: {identifier!r}")
        identifiers.add(identifier)
        config = _resolve_path(str(declaration["config"]), base=base)
        checkpoint = _resolve_path(str(declaration["checkpoint"]), base=base)
        for path in (config, checkpoint):
            if not path.is_file():
                raise FileNotFoundError(path)
        output_value = declaration.get("output_dir", output_root / identifier)
        output = _resolve_path(str(output_value), base=base)
        if str(output) in outputs:
            raise ValueError(f"duplicate output directory: {output}")
        outputs.add(str(output))
        record = {
            key: value
            for key, value in declaration.items()
            if key not in {"config", "checkpoint", "output_dir"}
        }
        record.update(
            {
                "config": str(config),
                "config_sha256": _sha256(config),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "analysis_config": str(profile),
                "analysis_config_sha256": profile_hash,
                "output_dir": str(output),
                "completion_artifacts": artifacts,
            }
        )
        for field in ("config_sha256", "checkpoint_sha256"):
            declared = declaration.get(field)
            if declared is not None and str(declared) != record[field]:
                raise ValueError(f"{field} differs for job {identifier}")
        jobs.append(record)
    return jobs


def _serialized_jobs(jobs: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(job, sort_keys=True) + "\n" for job in jobs)


def _write_immutable(path: Path, payload: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"immutable inventory changed: {path}")
        return
    path.write_text(payload, encoding="utf-8")


def _next_ledger_paths(directory: Path, label: str) -> tuple[Path, Path, str]:
    candidate = label
    retry = 0
    while (directory / f"jobs_{candidate}.jsonl").exists() or (
        directory / f"submission_{candidate}.json"
    ).exists():
        retry += 1
        candidate = f"{label}_retry_{retry:03d}"
    return (
        directory / f"jobs_{candidate}.jsonl",
        directory / f"submission_{candidate}.json",
        candidate,
    )


def _pending_jobs(jobs: list[dict[str, Any]], *, rerun: bool) -> list[dict[str, Any]]:
    if rerun:
        return jobs
    pending = []
    for job in jobs:
        output = Path(str(job["output_dir"]))
        complete = (output / "checkpoint_analysis_provenance.json").is_file() and all(
            (output / artifact).is_file() for artifact in job["completion_artifacts"]
        )
        if not complete:
            pending.append(job)
    return pending


def prepare_submission(
    manifest_path: Path,
    *,
    pilot: bool = False,
    rerun: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path, Path]:
    """Validate inputs and write immutable complete/pending job inventories."""
    manifest_path = manifest_path.resolve()
    manifest = _load_manifest(manifest_path)
    repository = _resolve_path(
        str(manifest.get("repository", _repo_root())), base=manifest_path.parent
    )
    head = _git_head(repository)
    required_commit = manifest.get("required_commit")
    if required_commit is not None and not head.startswith(str(required_commit)):
        raise RuntimeError(f"repository HEAD {head} differs from {required_commit}")
    require_clean = bool(manifest.get("require_clean_worktree", True))
    dirty = _tracked_worktree_dirty(repository)
    if require_clean and dirty:
        raise RuntimeError("checkpoint analysis requires a clean tracked worktree")

    all_jobs = _build_jobs(manifest, manifest_path=manifest_path)
    if pilot:
        pilot_ids = manifest.get("pilot_job_ids")
        if not isinstance(pilot_ids, list) or not pilot_ids:
            raise ValueError("pilot_job_ids must be declared for --pilot")
        selected = {str(value) for value in pilot_ids}
        jobs = [job for job in all_jobs if str(job["id"]) in selected]
        if {str(job["id"]) for job in jobs} != selected:
            raise ValueError("pilot_job_ids include unknown jobs")
    else:
        jobs = all_jobs
    jobs = _pending_jobs(jobs, rerun=rerun)

    output_root = _resolve_path(str(manifest["output_root"]), base=manifest_path.parent)
    ledger_dir = output_root / "jobs"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    (ledger_dir / "stdout").mkdir(exist_ok=True)
    (ledger_dir / "stderr").mkdir(exist_ok=True)
    all_path = ledger_dir / "jobs_all.jsonl"
    _write_immutable(all_path, _serialized_jobs(all_jobs))
    label = "pilot" if pilot else "full"
    jobs_path, submission_path, execution_label = _next_ledger_paths(ledger_dir, label)
    jobs_path.write_text(_serialized_jobs(jobs), encoding="utf-8")
    ledger = {
        "schema_version": SUBMISSION_SCHEMA_VERSION,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "repository": str(repository),
        "repository_head": head,
        "tracked_worktree_dirty": dirty,
        "required_clean_worktree": require_clean,
        "pilot": pilot,
        "rerun": rerun,
        "execution_label": execution_label,
        "all_job_count": len(all_jobs),
        "all_jobs_path": str(all_path),
        "all_jobs_sha256": _sha256(all_path),
        "pending_job_count": len(jobs),
        "jobs_path": str(jobs_path),
        "jobs_sha256": _sha256(jobs_path),
        "source_identity": discover_source_identity(
            repository, source_files=(Path(__file__),)
        ),
    }
    if manifest.get("jobs_file") is not None:
        jobs_source = _resolve_path(
            str(manifest["jobs_file"]), base=manifest_path.parent
        )
        ledger["jobs_source_path"] = str(jobs_source)
        ledger["jobs_source_sha256"] = _sha256(jobs_source)
    submission_path.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest, jobs, jobs_path, submission_path


def _scheduler(
    manifest: Mapping[str, Any],
    *,
    account: str | None,
    partition: str | None,
) -> dict[str, Any]:
    scheduler = dict(manifest.get("scheduler", {}))
    if account is not None:
        scheduler["account"] = account
    if partition is not None:
        scheduler["partition"] = partition
    profile = validate_scheduler_profile(
        scheduler.get("account"), scheduler.get("partition")
    )
    scheduler.update(profile.as_dict())
    return scheduler


def submit_jobs(
    *,
    manifest: Mapping[str, Any],
    jobs: list[dict[str, Any]],
    jobs_path: Path,
    account: str | None = None,
    partition: str | None = None,
    max_concurrent: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Submit one Slurm array and return its job ID plus resolved scheduler."""
    scheduler = _scheduler(manifest, account=account, partition=partition)
    if not jobs:
        return "no-pending-jobs", scheduler
    repository = _resolve_path(
        str(manifest.get("repository", _repo_root())), base=Path.cwd()
    )
    concurrency = int(max_concurrent or scheduler.get("max_concurrent_jobs", len(jobs)))
    source_root = shlex.quote(str(repository / "src"))
    wrapped = f"export PYTHONPATH={source_root}:${{PYTHONPATH:-}}; " + shlex.join(
        [
            sys.executable,
            "-u",
            "-m",
            "dendritic_modeling.scripts.analysis.run_checkpoint_analysis_job",
            "--jobs",
            str(jobs_path),
        ]
    )
    ledger_dir = jobs_path.parent
    command = [
        "sbatch",
        "--parsable",
        f"--job-name={str(manifest.get('name', 'checkpoint-analysis'))[:64]}",
        f"--account={scheduler['account']}",
        f"--partition={scheduler['partition']}",
        f"--time={scheduler.get('time', '06:00:00')}",
        f"--cpus-per-task={int(scheduler.get('cpus_per_task', 8))}",
        f"--mem={scheduler.get('memory', '32G')}",
        f"--array=0-{len(jobs) - 1}%{concurrency}",
        f"--output={ledger_dir / 'stdout' / '%A_%a.out'}",
        f"--error={ledger_dir / 'stderr' / '%A_%a.err'}",
        f"--chdir={repository}",
    ]
    if int(scheduler.get("gpus", 0)):
        command.append(f"--gres=gpu:{int(scheduler['gpus'])}")
    command.extend(["--wrap", wrapped])
    job_id = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    ).stdout.strip()
    return job_id, scheduler


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--account")
    parser.add_argument("--partition")
    parser.add_argument("--max-concurrent", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest, jobs, jobs_path, submission_path = prepare_submission(
        args.manifest, pilot=args.pilot, rerun=args.rerun
    )
    if args.generate_only:
        print(f"Prepared {len(jobs)} jobs in {jobs_path}")
        print(f"Execution ledger: {submission_path}")
        return 0
    job_id, scheduler = submit_jobs(
        manifest=manifest,
        jobs=jobs,
        jobs_path=jobs_path,
        account=args.account,
        partition=args.partition,
        max_concurrent=args.max_concurrent,
    )
    ledger = json.loads(submission_path.read_text(encoding="utf-8"))
    ledger["scheduler_job_id"] = job_id
    ledger["execution_scheduler"] = scheduler
    submission_path.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Submitted {job_id}")
    print(f"Execution ledger: {submission_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
