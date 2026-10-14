"""The package metadata matches the code: dependencies, packages and extras."""

from __future__ import annotations

import ast
import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = pytest.importorskip("tomli")

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "dendritic_modeling"
IMPORT_NAMES = {"PyYAML": "yaml", "scikit-learn": "sklearn"}
# Top-level modules a distribution provides under another name.
PROVIDED_BY = {"mpl_toolkits": "matplotlib"}


def _project() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


def _import_name(requirement: str) -> str:
    distribution = re.split(r"[<>=!~;\[ ]", requirement, maxsplit=1)[0]
    return IMPORT_NAMES.get(distribution, distribution.replace("-", "_"))


def _type_checking(node: ast.If) -> bool:
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _module_level_imports(body, *, guarded: bool, found: dict) -> None:
    for node in body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                found[alias.name.split(".")[0]].append(guarded)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found[node.module.split(".")[0]].append(guarded)
        elif isinstance(node, ast.Try):
            _module_level_imports(node.body, guarded=True, found=found)
        elif isinstance(node, ast.If) and not _type_checking(node):
            _module_level_imports(node.body, guarded=guarded, found=found)
            _module_level_imports(node.orelse, guarded=guarded, found=found)


def _scan() -> tuple[dict[str, set[str]], set[str]]:
    """Unguarded module-level imports per top-level name, and every imported name."""
    unguarded: dict[str, set[str]] = defaultdict(set)
    everything: set[str] = set()
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        found: dict[str, list[bool]] = defaultdict(list)
        _module_level_imports(tree.body, guarded=False, found=found)
        for name, guards in found.items():
            if not all(guards):
                unguarded[name].add(str(path.relative_to(PACKAGE)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                everything.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                everything.add(node.module.split(".")[0])
    return unguarded, everything


def _third_party(names) -> set[str]:
    return {
        PROVIDED_BY.get(name, name)
        for name in names
        if name not in sys.stdlib_module_names
        and name not in {"__future__", "dendritic_modeling"}
    }


def test_unguarded_imports_are_declared_dependencies():
    declared = {_import_name(item) for item in _project()["dependencies"]}
    unguarded, _ = _scan()
    undeclared = {
        name: sorted(files)
        for name, files in unguarded.items()
        if _third_party([name]) and _third_party([name]).pop() not in declared
    }
    assert undeclared == {}


def test_every_hard_dependency_is_imported():
    _, everything = _scan()
    unused = [
        item
        for item in _project()["dependencies"]
        if _import_name(item) not in everything
    ]
    assert unused == []


def test_optional_imports_are_offered_as_extras():
    project = _project()
    offered = {
        _import_name(item)
        for group in project["optional-dependencies"].values()
        for item in group
    } | {_import_name(item) for item in project["dependencies"]}
    _, everything = _scan()
    assert sorted(_third_party(everything) - offered) == []


def test_every_code_directory_is_a_regular_package():
    missing = [
        str(directory.relative_to(PACKAGE))
        for directory in [PACKAGE, *PACKAGE.rglob("*")]
        if directory.is_dir()
        and directory.name != "__pycache__"
        and any(directory.glob("*.py"))
        and not (directory / "__init__.py").is_file()
    ]
    assert missing == []
