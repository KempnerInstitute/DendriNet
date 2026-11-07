"""CI and the documentation deployment run for develop, not the retired main branch."""

from __future__ import annotations

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
