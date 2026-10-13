"""Every option of every installed ``dendritic-*`` command has help text."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Runs in a separate interpreter: the entry points configure logging.
_COLLECT = r"""
import argparse, contextlib, importlib, io, json, sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

scripts = tomllib.loads(open(sys.argv[1], "rb").read().decode())["project"]["scripts"]


class Stop(Exception):
    pass


found = {}


def capture(self, *args, **kwargs):
    found.setdefault("parser", self)
    raise Stop


missing = {}
for name, target in sorted(scripts.items()):
    module_name, function_name = target.split(":")
    entry = getattr(importlib.import_module(module_name), function_name)
    found.clear()
    saved = (argparse.ArgumentParser.parse_args, argparse.ArgumentParser.parse_known_args)
    argparse.ArgumentParser.parse_args = capture
    argparse.ArgumentParser.parse_known_args = capture
    sys.argv = [name, "--help"]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                entry()
            except (Stop, SystemExit):
                pass
    finally:
        argparse.ArgumentParser.parse_args, argparse.ArgumentParser.parse_known_args = saved
    parser = found.get("parser")
    if parser is None:
        missing[name] = ["<no argument parser>"]
        continue
    undocumented = [
        "/".join(action.option_strings) or action.dest
        for action in parser._actions
        if not isinstance(action, argparse._HelpAction) and not action.help
    ]
    if undocumented:
        missing[name] = undocumented
print(json.dumps(missing))
"""


def test_every_option_of_the_installed_commands_has_help_text():
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            filter(None, (str(ROOT / "src"), os.environ.get("PYTHONPATH")))
        ),
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="",
    )
    result = subprocess.run(
        [sys.executable, "-c", _COLLECT, str(ROOT / "pyproject.toml")],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    missing = json.loads(result.stdout.strip().splitlines()[-1])
    assert missing == {}
