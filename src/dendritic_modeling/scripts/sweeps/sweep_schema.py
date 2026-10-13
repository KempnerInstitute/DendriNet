"""Sections and keys of a sweep file, checked when the sweep is loaded.

The sections the sweep tools read are checked key by key, so a misspelled
key raises instead of silently falling back to a default. Top-level keys the
tools do not read are reported: a near-miss of a section name raises, other
keys (study annotations) only warn. Keys starting with ``x-`` hold YAML
anchors, and ``metadata`` holds free-form annotations; both pass silently.
"""

from __future__ import annotations

import dataclasses
import difflib
import logging
from collections.abc import Iterable, Mapping
from typing import Any

from dendritic_modeling.scripts.sweeps.slurm_profile import SlurmProfile

logger = logging.getLogger(__name__)

SWEEP_SECTIONS = frozenset(
    {
        "base_config",
        "sweep_config",
        "sweep_variants",
        "experiment_settings",
        "filter_config",
        "slurm_config",
        "sweep_contract",
        "execution_results_dir",
        "expected_config_count",
        "output_dir",
        "outputs",
        "metadata",
    }
)
SLURM_CONFIG_KEYS = frozenset(
    {field.name for field in dataclasses.fields(SlurmProfile)} - {"job_name"}
) | {"run_name", "job_name"}
EXPERIMENT_SETTINGS_KEYS = frozenset(
    {"seeds_per_condition", "base_seed", "seed_paths", "use_array_jobs"}
)
FILTER_CONFIG_KEYS = frozenset({"mode", "branch_sweep_mode", "value", "tolerance"})
OUTPUTS_KEYS = frozenset({"dir", "run_name"})


def _check_keys(section: Any, allowed: Iterable[str], path: str) -> None:
    if not isinstance(section, Mapping):
        return
    allowed = sorted(allowed)
    unknown = sorted(str(key) for key in section if str(key) not in allowed)
    if not unknown:
        return
    details = []
    for key in unknown:
        hint = difflib.get_close_matches(key, allowed, n=1)
        details.append(f"{key!r}" + (f" (did you mean {hint[0]!r}?)" if hint else ""))
    raise ValueError(
        f"{path} has keys the sweep tools do not read: {', '.join(details)}. "
        f"Valid keys: {', '.join(allowed)}"
    )


def check_sweep_sections(config: Mapping[str, Any]) -> None:
    """Raise for misspelled sweep sections or keys; warn for ignored annotations."""
    _check_keys(config.get("slurm_config"), SLURM_CONFIG_KEYS, "slurm_config")
    _check_keys(
        config.get("experiment_settings"),
        EXPERIMENT_SETTINGS_KEYS,
        "experiment_settings",
    )
    _check_keys(config.get("filter_config"), FILTER_CONFIG_KEYS, "filter_config")
    _check_keys(config.get("outputs"), OUTPUTS_KEYS, "outputs")
    ignored = []
    for key in config:
        name = str(key)
        if name in SWEEP_SECTIONS or name.startswith("x-"):
            continue
        hint = difflib.get_close_matches(name, sorted(SWEEP_SECTIONS), n=1, cutoff=0.85)
        if hint:
            raise ValueError(
                f"Sweep section {name!r} is not one the sweep tools read; "
                f"did you mean {hint[0]!r}?"
            )
        ignored.append(name)
    if ignored:
        logger.warning(
            "Ignoring top-level sweep keys the sweep tools do not read: %s "
            "(put annotations under metadata:)",
            ", ".join(sorted(ignored)),
        )


__all__ = [
    "EXPERIMENT_SETTINGS_KEYS",
    "FILTER_CONFIG_KEYS",
    "SLURM_CONFIG_KEYS",
    "SWEEP_SECTIONS",
    "check_sweep_sections",
]
