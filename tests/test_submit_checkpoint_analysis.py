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
    submit_jobs,
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
    assert jobs[0]["repository"] == str((tmp_path / "repository").resolve())
    assert jobs[0]["repository_head"] == ledger["repository_head"]
    assert jobs[0]["require_clean_worktree"] is True


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


def test_run_declared_job_rejects_source_commit_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    actual_head = _git_repository(repository)
    jobs_path = tmp_path / "jobs.jsonl"
    jobs_path.write_text(
        json.dumps(
            {
                "config": "config.yaml",
                "checkpoint": "checkpoint.pt",
                "analysis_config": "analysis.yaml",
                "output_dir": "output",
                "repository": str(repository),
                "repository_head": "0" * len(actual_head),
                "require_clean_worktree": True,
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        "dendritic_modeling.scripts.analysis.run_checkpoint_analysis_job."
        "run_checkpoint_analysis",
        lambda **_kwargs: pytest.fail("analysis ran after source drift"),
    )
    with pytest.raises(RuntimeError, match="source drifted"):
        run_declared_job(jobs_path=jobs_path, job_index=0)


def test_submission_runs_the_checkout_it_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, repository = _write_manifest(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["repository"] = "repository"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    manifest, jobs, jobs_path, _ledger = prepare_submission(manifest_path)
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="123\n")

    monkeypatch.chdir(tmp_path / "outputs")
    monkeypatch.setattr(subprocess, "run", fake_run)
    job_id, _scheduler = submit_jobs(
        manifest=manifest,
        jobs=jobs,
        jobs_path=jobs_path,
        account="lab_account",
        partition="gpu_partition",
    )
    assert job_id == "123"
    assert f"--chdir={repository.resolve()}" in commands[0]
    assert str(repository.resolve() / "src") in commands[0][-1]


def test_run_declared_job_rejects_a_package_imported_elsewhere(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    head = _git_repository(repository)
    jobs_path = tmp_path / "jobs.jsonl"
    jobs_path.write_text(
        json.dumps(
            {
                "config": "config.yaml",
                "checkpoint": "checkpoint.pt",
                "analysis_config": "analysis.yaml",
                "output_dir": "output",
                "repository": str(repository),
                "repository_head": head,
                "require_clean_worktree": True,
            }
        )
        + "\n"
    )
    with pytest.raises(RuntimeError, match="not from the declared repository"):
        run_declared_job(jobs_path=jobs_path, job_index=0)


def test_an_unschedulable_submission_writes_no_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scheduler is checked before prepare_submission writes the ledger."""
    from dendritic_modeling.scripts.analysis import submit_checkpoint_analysis

    manifest_path, _repository = _write_manifest(tmp_path)
    monkeypatch.setenv("DENDRITIC_SCHEDULER_PROFILES", "none")

    def no_sbatch(command, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError(f"sbatch must not run: {command}")

    # No account or partition: submission is refused before any file is written.
    with pytest.raises(ValueError, match="account"):
        submit_checkpoint_analysis.main(["--manifest", str(manifest_path)])
    assert not (tmp_path / "outputs").exists()

    # --max-concurrent 0 is refused by the parser (exit status 2).
    with pytest.raises(SystemExit) as error:
        submit_checkpoint_analysis.main(
            [
                "--manifest",
                str(manifest_path),
                "--account",
                "lab",
                "--partition",
                "gpu",
                "--max-concurrent",
                "0",
            ]
        )
    assert error.value.code == 2
    assert not (tmp_path / "outputs").exists()

    # A manifest limit of 0 is refused before the ledger is written as well.
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["scheduler"] = {"max_concurrent_jobs": 0}
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    monkeypatch.setattr(subprocess, "run", no_sbatch)
    with pytest.raises(ValueError, match="max_concurrent_jobs must be a positive"):
        submit_checkpoint_analysis.main(
            ["--manifest", str(manifest_path), "--account", "lab", "--partition", "gpu"]
        )
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize(
    "max_concurrent,manifest_limit,expected",
    [(None, None, 2), (1, None, 1), (None, 1, 1), (2, 1, 2), (None, "1", 1)],
)
def test_the_concurrency_limit_of_the_array(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_concurrent,
    manifest_limit,
    expected,
) -> None:
    manifest_path, _repository = _write_manifest(tmp_path)
    manifest, jobs, jobs_path, _ledger = prepare_submission(manifest_path)
    if manifest_limit is not None:
        manifest["scheduler"] = {"max_concurrent_jobs": manifest_limit}
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="7\n")

    monkeypatch.setenv("DENDRITIC_SCHEDULER_PROFILES", "none")
    monkeypatch.setattr(subprocess, "run", fake_run)
    submit_jobs(
        manifest=manifest,
        jobs=jobs,
        jobs_path=jobs_path,
        account="lab",
        partition="gpu",
        max_concurrent=max_concurrent,
    )
    assert f"--array=0-1%{expected}" in commands[0]
    with pytest.raises(ValueError, match="--max-concurrent must be a positive"):
        submit_jobs(
            manifest=manifest,
            jobs=jobs,
            jobs_path=jobs_path,
            account="lab",
            partition="gpu",
            max_concurrent=0,
        )
