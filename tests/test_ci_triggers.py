"""Workflow triggers, permissions and outputs for the institutional repository."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
BRANCH_EVENTS = ("push", "pull_request", "pull_request_target")


def _events(workflow: dict) -> dict:
    """The workflow's events, with the shorthand ``on`` forms expanded."""
    # PyYAML reads the bare key ``on`` as the boolean True (YAML 1.1).
    events = workflow.get("on", workflow.get(True))
    if isinstance(events, str):
        return {events: None}
    if isinstance(events, list):
        return dict.fromkeys(events)
    return events or {}


@pytest.mark.requires_git_checkout
def test_branch_events_run_for_develop_only():
    checked = 0
    for path in sorted(WORKFLOWS.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        assert "refs/heads/main" not in text, path.name
        for event, filters in _events(yaml.safe_load(text)).items():
            if event not in BRANCH_EVENTS:
                continue
            filters = filters or {}
            keys = set(filters)
            if (
                event == "push"
                and keys & {"tags", "tags-ignore"}
                and not keys & {"branches", "branches-ignore"}
            ):
                continue  # A push filtered only by tags runs for no branch.
            # An unfiltered event, branches-ignore or a pattern could match main.
            assert filters.get("branches") == ["develop"], (path.name, event)
            checked += 1
    assert checked


@pytest.mark.requires_git_checkout
def test_ci_delivers_the_measured_coverage_badge_without_repository_writes(tmp_path):
    workflow = yaml.safe_load((WORKFLOWS / "ci-simplified.yml").read_text())
    for job in workflow["jobs"].values():
        permissions = job.get("permissions", workflow["permissions"])
        assert permissions["contents"] == "read"
        assert permissions.get("pull-requests", "none") in {"read", "none"}

    steps = workflow["jobs"]["test"]["steps"]
    assert not any("create-pull-request@" in step.get("uses", "") for step in steps)
    generation = next(
        step["run"]
        for step in steps
        if "generate_coverage_badge.py" in step.get("run", "")
    )
    command = shlex.split(generation.replace("\\\n", " "))
    report = tmp_path / "coverage.xml"
    report.write_text('<coverage line-rate="0.8126"/>', encoding="utf-8")
    subprocess.run(
        [sys.executable, WORKFLOWS.parents[1] / command[1], *command[2:]],
        cwd=tmp_path,
        check=True,
    )
    artifact = next(
        step["with"]
        for step in steps
        if step.get("uses", "").startswith("actions/upload-artifact@")
        and step["with"]["name"].startswith("coverage-")
    )
    assert set(artifact["path"].splitlines()) == {"coverage.xml", "coverage.svg"}
    assert 'aria-label="coverage: 81.3%"' in (tmp_path / "coverage.svg").read_text()
