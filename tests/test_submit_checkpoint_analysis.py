"""Tests for generic checkpoint-analysis manifest preparation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from dendritic_modeling.scripts.analysis.run_checkpoint_analysis_job import (
    run_declared_job,
)
from dendritic_modeling.scripts.analysis.submit_checkpoint_analysis import (
    prepare_submission,
)


def _git_repository(path: Path) -> str:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    tracked = path / "tracked.py"
    tracked.write_text("VALUE = 1\n")
    subprocess.run(["git", "add", "tracked.py"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=path, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_manifest(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repository"
    commit = _git_repository(repository)
    profile = tmp_path / "analysis.yaml"
    profile.write_text("analysis: {}\n")
    jobs = []
    for index in range(2):
        config = tmp_path / f"config_{index}.yaml"
        checkpoint = tmp_path / f"checkpoint_{index}.pt"
        config.write_text(f"seed: {index}\n")
        checkpoint.write_bytes(f"checkpoint-{index}".encode())
        jobs.append(
            {
                "id": f"condition_{index}",
                "config": config.name,
                "checkpoint": checkpoint.name,
                "mechanism": "shunting" if index == 0 else "additive",
            }
        )
    manifest = {
        "schema_version": "checkpoint_analysis_manifest_v1",
        "name": "test-analysis",
        "repository": str(repository),
        "required_commit": commit,
        "require_clean_worktree": True,
        "analysis_config": profile.name,
        "output_root": "outputs",
        "completion_artifacts": ["information_analysis/final.json"],
        "pilot_job_ids": ["condition_1"],
        "jobs": jobs,
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return path, repository


def test_prepare_submission_hash_binds_inventory_and_selects_pilot(
    tmp_path: Path,
) -> None:
    manifest_path, _repository = _write_manifest(tmp_path)

    _manifest, jobs, jobs_path, ledger_path = prepare_submission(
        manifest_path, pilot=True
    )

    assert [job["id"] for job in jobs] == ["condition_1"]
    assert jobs[0]["config_sha256"]
    assert jobs[0]["checkpoint_sha256"]
    assert jobs_path.name == "jobs_pilot.jsonl"
    ledger = json.loads(ledger_path.read_text())
    assert ledger["all_job_count"] == 2
    assert ledger["pending_job_count"] == 1
    assert (tmp_path / "outputs/jobs/jobs_all.jsonl").is_file()


def test_prepare_submission_accepts_reusable_jobs_file(tmp_path: Path) -> None:
    manifest_path, _repository = _write_manifest(tmp_path)
    payload = yaml.safe_load(manifest_path.read_text())
    jobs = payload.pop("jobs")
    jobs_path = tmp_path / "cohort.yaml"
    jobs_path.write_text(
        yaml.safe_dump(
            {"schema_version": "checkpoint_analysis_jobs_v1", "jobs": jobs},
            sort_keys=False,
        )
    )
    payload["jobs_file"] = jobs_path.name
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False))

    _manifest, selected, _jobs_path, _ledger = prepare_submission(
        manifest_path, pilot=True
    )
    assert [job["id"] for job in selected] == ["condition_1"]


def test_prepare_submission_rejects_dirty_source_and_changed_inventory(
    tmp_path: Path,
) -> None:
    manifest_path, repository = _write_manifest(tmp_path)
    prepare_submission(manifest_path)

    (repository / "tracked.py").write_text("VALUE = 2\n")
    with pytest.raises(RuntimeError, match="clean tracked worktree"):
        prepare_submission(manifest_path)

    subprocess.run(["git", "checkout", "--", "tracked.py"], cwd=repository, check=True)
    payload = yaml.safe_load(manifest_path.read_text())
    payload["jobs"][0]["mechanism"] = "changed"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(RuntimeError, match="immutable inventory changed"):
        prepare_submission(manifest_path)


def test_run_declared_job_dispatches_selected_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs_path = tmp_path / "jobs.jsonl"
    jobs_path.write_text(
        json.dumps(
            {
                "config": "config.yaml",
                "checkpoint": "checkpoint.pt",
                "analysis_config": "analysis.yaml",
                "output_dir": "output",
                "filename": "locked",
            }
        )
        + "\n"
    )
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(
        "dendritic_modeling.scripts.analysis.run_checkpoint_analysis_job."
        "run_checkpoint_analysis",
        fake_run,
    )
    assert run_declared_job(jobs_path=jobs_path, job_index=0) == {"ok": True}
    assert captured["filename"] == "locked"
    assert captured["strict"] is True
