"""Version rules for module state that older checkpoints lack or record differently.

A module whose saved state gains a key raises its ``_version`` (PyTorch writes
it to the state dict's ``_metadata``) and lists the key in
``_STATE_KEYS_ADDED`` under that version. Loading then follows one rule:

- A file of the current version that lacks such a key fails strict loading
  like any other missing key.
- A file that predates the key (a lower metadata version, or no metadata)
  keeps the module's constructed value. This *legacy state* is warned (the
  default warning filter shows each message once per location) and recorded
  on the module, and the library's checkpoint loaders return it in their load
  records (``legacy_state``).

:func:`compare_state_dict_keys` applies the same declarations when a state
dict is compared with a model, separating keys a newer layout added from keys
that are really missing.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import torch.nn as nn

LEGACY_STATE_ATTRIBUTE = "_legacy_state_records"


def file_predates(local_metadata: Mapping[str, Any], version: int) -> bool:
    """Whether a loaded module state was written before ``version``.

    State without metadata (a dict rebuilt without ``_metadata``) counts as
    older, since its version cannot be known.
    """
    written = local_metadata.get("version")
    return written is None or int(written) < int(version)


def record_legacy_state(module: nn.Module, key: str, note: str) -> None:
    """Record that ``module`` loaded ``key`` under a legacy rule, and warn."""
    module.__dict__.setdefault(LEGACY_STATE_ATTRIBUTE, []).append(
        {"key": key, "note": note}
    )
    short = key.rsplit(".", 1)[-1]
    warnings.warn(
        f"{type(module).__name__} checkpoint state {short!r}: {note}", stacklevel=3
    )


def accept_legacy_missing_key(
    module: nn.Module,
    name: str,
    *,
    prefix: str,
    local_metadata: Mapping[str, Any],
    missing_keys: list[str],
    version: int,
    note: str,
) -> bool:
    """Accept ``prefix + name`` as missing when the file predates ``version``.

    Returns whether the key was accepted (the module keeps its constructed
    value). For a file of ``version`` or newer the key stays missing.
    """
    key = prefix + name
    if key not in missing_keys or not file_predates(local_metadata, version):
        return False
    missing_keys.remove(key)
    record_legacy_state(module, key, note)
    return True


def clear_legacy_state(model: nn.Module) -> None:
    """Forget legacy-state records from earlier loads."""
    for module in model.modules():
        module.__dict__.pop(LEGACY_STATE_ATTRIBUTE, None)


def collect_legacy_state(model: nn.Module) -> list[dict[str, str]]:
    """Remove and return the legacy-state records of the last load."""
    records: list[dict[str, str]] = []
    for module in model.modules():
        records.extend(module.__dict__.pop(LEGACY_STATE_ATTRIBUTE, []))
    return sorted(records, key=lambda record: record["key"])


def state_keys_added(module: nn.Module) -> dict[int, tuple[str, ...]]:
    """Keys each version of ``module``'s class (and its bases) added."""
    added: dict[int, set[str]] = {}
    for cls in type(module).__mro__:
        for version, names in vars(cls).get("_STATE_KEYS_ADDED", {}).items():
            added.setdefault(int(version), set()).update(names)
    return {version: tuple(sorted(names)) for version, names in added.items()}


def compare_state_dict_keys(
    model: nn.Module, state_dict: Mapping[str, Any]
) -> dict[str, list[str]]:
    """Compare a state dict's keys with ``model``'s under the version rules.

    Returns ``missing`` (keys the model needs that the file should have had),
    ``version_added`` (model keys a newer module version added after the
    file's version, per the file's ``_metadata``) and ``unexpected`` (file keys
    the model does not have).
    """
    model_keys = set(model.state_dict())
    file_keys = set(state_dict)
    metadata = getattr(state_dict, "_metadata", None) or {}
    version_added: set[str] = set()
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        written = metadata.get(module_name, {}).get("version")
        for version, names in state_keys_added(module).items():
            if written is None or int(written) < version:
                version_added.update(prefix + name for name in names)
    absent = model_keys - file_keys
    return {
        "missing": sorted(absent - version_added),
        "version_added": sorted(absent & version_added),
        "unexpected": sorted(file_keys - model_keys),
    }


__all__ = [
    "LEGACY_STATE_ATTRIBUTE",
    "accept_legacy_missing_key",
    "clear_legacy_state",
    "collect_legacy_state",
    "compare_state_dict_keys",
    "file_predates",
    "record_legacy_state",
    "state_keys_added",
]
