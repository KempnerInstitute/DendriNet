"""YAML examples in the documentation load through the strict config loaders.

Every ``.. code-block:: yaml`` block in docs/*.rst and fenced ``yaml`` block in
the Markdown guides is checked: experiment configs load with
``config_from_mapping`` (unknown keys raise), and excerpts of one section are
wrapped in their parent section first. Excerpts that cannot be checked on
their own are listed in ``UNCHECKED`` with the reason.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from dendritic_modeling.config import config_from_mapping
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    PopulationConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOTS = frozenset(
    {"model", "training", "data", "experiment", "outputs", "analysis"}
    | {"distributed", "metadata"}
)
SWEEP_ROOTS = frozenset(
    {"sweep_config", "filter_config", "slurm_config", "experiment_settings"}
)

# (document, root keys) -> why the excerpt is not checked alone.
UNCHECKED: dict[tuple[str, frozenset[str]], str] = {}


def _rst_blocks(path: Path):
    lines = path.read_text().splitlines()
    index = 0
    while index < len(lines):
        match = re.match(r"^(\s*)\.\. code-block:: yaml\s*$", lines[index])
        if not match:
            index += 1
            continue
        directive_indent = len(match.group(1))
        body, block_indent = [], None
        end = index + 1
        while end < len(lines):
            line = lines[end]
            if line.strip():
                current = len(line) - len(line.lstrip())
                if current <= directive_indent:
                    break
                block_indent = current if block_indent is None else block_indent
                body.append(line[block_indent:])
            else:
                body.append("")
            end += 1
        yield index + 1, "\n".join(body)
        index = end


def _markdown_blocks(path: Path):
    text = path.read_text()
    for match in re.finditer(r"```ya?ml\n(.*?)```", text, re.S):
        yield text[: match.start()].count("\n") + 1, match.group(1)


def _blocks():
    found = []
    for path in sorted((REPO_ROOT / "docs").rglob("*.rst")):
        found += [(path, line, body) for line, body in _rst_blocks(path)]
    markdown = [REPO_ROOT / "README.md", REPO_ROOT / "INSTALL.md"]
    for path in markdown + sorted((REPO_ROOT / "docs").rglob("*.md")):
        if path.exists():
            found += [(path, line, body) for line, body in _markdown_blocks(path)]
    return found


BLOCKS = _blocks()


def _check(document: str, data) -> None:
    assert isinstance(data, dict), "a YAML example must be a mapping"
    roots = frozenset(data)
    if (document, roots) in UNCHECKED:
        pytest.skip(UNCHECKED[(document, roots)])
    if roots <= EXPERIMENT_ROOTS:
        config_from_mapping(data)
    elif roots == {"population_network"}:
        config_from_mapping({"model": {"core": {"type": "population_network", **data}}})
    elif roots == {"layers"}:
        config_from_mapping(
            {
                "model": {
                    "core": {"type": "population_network", "population_network": data}
                }
            }
        )
    elif roots == {"populations"}:
        names = [population["name"] for population in data["populations"]]
        layer = {
            "name": "layer0",
            "readout_population": names[0],
            "populations": data["populations"],
            "connections": [{"source": "input", "target": name} for name in names],
        }
        config_from_mapping(
            {
                "model": {
                    "core": {
                        "type": "population_network",
                        "population_network": {"layers": [layer]},
                    }
                }
            }
        )
    elif roots == {"population_defaults"}:
        PopulationConfig(**data["population_defaults"])
    elif roots <= {"architecture", "connectivity", "sparsity", "morphology"}:
        config_from_mapping({"model": {"core": {"type": "dendritic_shunting", **data}}})
    elif roots <= SWEEP_ROOTS:
        pytest.skip("sweep axes name paths of a base config the excerpt omits")
    else:
        raise AssertionError(f"no check for an example rooted at {sorted(roots)}")


@pytest.mark.parametrize(
    "path, line, body",
    BLOCKS,
    ids=[f"{path.relative_to(REPO_ROOT)}:{line}" for path, line, _ in BLOCKS],
)
def test_documented_yaml_loads(path, line, body):
    _check(path.relative_to(REPO_ROOT).as_posix(), yaml.safe_load(body))


def test_the_documentation_has_yaml_examples():
    assert len(BLOCKS) > 50


def _native_examples():
    from dendritic_modeling.config.biological import BIOLOGICAL_DEFAULT_CORE_TYPES

    for path, line, body in BLOCKS:
        data = yaml.safe_load(body)
        if not isinstance(data, dict):
            continue
        kind = data.get("model", {}).get("core", {}).get("type", "").lower()
        if kind in BIOLOGICAL_DEFAULT_CORE_TYPES:
            yield path, line, data


@pytest.mark.parametrize(
    "path, line, data",
    list(_native_examples()),
    ids=lambda value: str(value) if isinstance(value, (Path, int)) else None,
)
def test_native_documented_models_construct(path, line, data):
    """Loading alone cannot verify the biological contract of built modules."""
    import torch

    from dendritic_modeling.config.conversion import to_plain_dict
    from dendritic_modeling.networks.architectures.factory import get_architecture

    torch.manual_seed(1)
    config = config_from_mapping(data)
    input_dim = data["model"].get("encoder", {}).get("params", {}).get("input_dim", 784)
    get_architecture(
        config.model.core.type, to_plain_dict(config.model.core), input_dim=input_dim
    )


def test_every_unchecked_entry_names_a_documented_example():
    documented = {
        (path.relative_to(REPO_ROOT).as_posix(), frozenset(data))
        for path, _line, body in BLOCKS
        if isinstance(data := yaml.safe_load(body), dict)
    }
    assert set(UNCHECKED) <= documented
