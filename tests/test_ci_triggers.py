"""CI and the documentation deployment run for develop, not the retired main branch."""

from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def test_workflows_run_for_develop_and_not_for_main():
    checked = 0
    for path in sorted(WORKFLOWS.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        assert "refs/heads/main" not in text, path.name
        workflow = yaml.safe_load(text)
        # PyYAML reads the bare key ``on`` as the boolean True (YAML 1.1).
        triggers = workflow.get("on", workflow.get(True)) or {}
        for event in ("push", "pull_request"):
            branches = (triggers.get(event) or {}).get("branches")
            if branches is not None:
                assert "develop" in branches, (path.name, event)
                assert "main" not in branches, (path.name, event)
                checked += 1
    assert checked
