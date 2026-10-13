"""Every documentation build of the CI workflows treats Sphinx warnings as errors."""

from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _docs_build_steps() -> list[tuple[str, str, str]]:
    steps = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job in (workflow.get("jobs") or {}).values():
            for step in job.get("steps") or []:
                run = step.get("run") or ""
                if "sphinx" in run or "make html" in run:
                    steps.append((path.name, str(step.get("name")), run))
    return steps


def test_pull_request_and_deployment_docs_builds_fail_on_warnings():
    steps = _docs_build_steps()
    assert {workflow for workflow, _, _ in steps} == {
        "ci-simplified.yml",
        "deploy-docs.yml",
    }
    for workflow, name, run in steps:
        assert "make html" not in run, (workflow, name)
        words = run.split()
        assert "-W" in words and "--keep-going" in words, (workflow, name)


def test_pull_request_checks_verify_the_generated_references():
    workflow = yaml.safe_load(
        (WORKFLOWS / "ci-simplified.yml").read_text(encoding="utf-8")
    )
    docs_jobs = [
        [step.get("run") or "" for step in job.get("steps") or []]
        for job in (workflow.get("jobs") or {}).values()
        if any("sphinx" in (step.get("run") or "") for step in job.get("steps") or [])
    ]
    assert len(docs_jobs) == 1
    runs = docs_jobs[0]
    # The job installs the package, which both generators import.
    install = next(i for i, run in enumerate(runs) if "pip install -e" in run)
    for script in ("generate_cli_reference.py", "generate_options_reference.py"):
        checks = [
            i for i, run in enumerate(runs) if f"docs/scripts/{script} --check" in run
        ]
        assert checks and checks[0] > install, script
