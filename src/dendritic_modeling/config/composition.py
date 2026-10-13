"""Composition of YAML files through relative ``extends`` chains and overrides."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_composed_yaml(path: str | Path) -> tuple[DictConfig, list[Path]]:
    """Load a YAML file and the chain of files it ``extends``.

    A child mapping is deep-merged over its parent and lists replace parent
    lists. A relative ``extends`` path is resolved against the directory of
    the file that names it. Returns the composed mapping and every source
    path, root parent first, for provenance records.
    """

    def _load(file_path: Path, stack: tuple[Path, ...]):
        resolved = file_path.expanduser().resolve()
        if resolved in stack:
            cycle = " -> ".join(str(item) for item in (*stack, resolved))
            raise ValueError(f"Config extends cycle: {cycle}")
        raw = OmegaConf.load(resolved)
        if not isinstance(raw, DictConfig):
            raise ValueError(f"Config file must contain a mapping: {resolved}")
        parent_reference = raw.get("extends")
        if "extends" in raw:
            del raw["extends"]
        if parent_reference is None:
            return raw, [resolved]
        if not isinstance(parent_reference, str) or not parent_reference.strip():
            raise TypeError("Config extends must be one non-empty path")
        parent_path = Path(parent_reference)
        if not parent_path.is_absolute():
            parent_path = resolved.parent / parent_path
        parent, sources = _load(parent_path, (*stack, resolved))
        return OmegaConf.merge(parent, raw), [*sources, resolved]

    return _load(Path(path), ())


def apply_overrides(
    data: Mapping[str, Any] | DictConfig, overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a copy of ``data`` with dotted-path ``overrides`` set.

    ``{"training.main.common.epochs": 3}`` sets that nested value, creating
    missing (or null) intermediate sections. A path through a value that is
    not a mapping raises ``ValueError``.
    """
    if OmegaConf.is_config(data):
        result = OmegaConf.to_container(data, resolve=True)
    else:
        result = OmegaConf.to_container(OmegaConf.create(dict(data)), resolve=True)
    for path, value in overrides.items():
        keys = str(path).split(".")
        if not all(keys):
            raise ValueError(f"Invalid override path {path!r}")
        node = result
        for depth, key in enumerate(keys[:-1]):
            child = node.get(key)
            if child is None:
                child = node[key] = {}
            if not isinstance(child, dict):
                where = ".".join(keys[: depth + 1])
                raise ValueError(
                    f"Cannot apply override {path}: {where} is not a mapping"
                )
            node = child
        node[keys[-1]] = value
    return result


__all__ = ["apply_overrides", "load_composed_yaml"]
