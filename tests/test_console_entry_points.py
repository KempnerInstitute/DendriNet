"""Installed commands return an exit status, never a result object."""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
ENTRY_POINTS = re.findall(
    r'^(dendritic-[\w-]+) = "([\w.]+):(\w+)"$',
    PYPROJECT.read_text(),
    flags=re.MULTILINE,
)


def test_the_console_scripts_are_listed():
    assert len(ENTRY_POINTS) >= 20


@pytest.mark.parametrize(("command", "module", "function"), ENTRY_POINTS)
def test_console_entry_points_return_an_exit_status(command, module, function):
    """``sys.exit(<dict>)`` exits 1 and prints the dict, failing set -e chains."""
    target = getattr(importlib.import_module(module), function)
    annotation = inspect.signature(target).return_annotation
    assert annotation in {
        int,
        None,
        "int",
        "None",
        inspect.Signature.empty,
    }, f"{command} -> {module}:{function} returns {annotation}"
