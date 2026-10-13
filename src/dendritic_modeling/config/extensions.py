"""Configuration sections that plugins provide.

Some configuration sections belong to components that plugins supply (see
:mod:`dendritic_modeling.plugins`). A plugin registers a builder for each such
section; the loader passes the section's mapping to it and stores the typed
result. Sections may appear at these places:

* ``model.vision_replacement`` (also written ``model.pretrained_replacement``),
  ``model.transformer_replacement``, ``training.transformer_replacement`` and
  ``training.vision_replacement``: fixed slots of the model and training
  configuration;
* ``training.strategies.<name>``: settings of a trainer strategy the plugin
  registers, read from ``training.main.strategies.<name>`` (or
  ``training.encoder.strategies.<name>``) and stored in
  ``strategies.extensions[<name>]``;
* ``analysis.<name>``: settings of an analyzer the plugin registers, stored in
  ``analysis.extensions[<name>]``.

Without a registered builder, a slot holds :class:`DisabledExtensionConfig`:
an empty or explicitly disabled section loads (for example from a run made with
the plugin), enabling it raises, and a named section is an unknown key.

A plugin can also register configuration migrations
(:func:`register_config_migration`): functions that rewrite keys historical
configurations wrote for the plugin's components into their current form. The
loader runs them on every configuration before checking it, and
``dendritic-upgrade-config`` writes their result.
"""

from __future__ import annotations

import copy
import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from omegaconf import OmegaConf

from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.config.conversion import to_plain_dict

logger = logging.getLogger(__name__)

SectionBuilder = Callable[[Any], Any]
#: Rewrites a configuration mapping in place; returns one note per change.
ConfigMigration = Callable[[dict[str, Any]], "list[str] | None"]

#: Fixed slots a plugin may fill, with the configuration that holds them.
EXTENSION_SLOTS = frozenset(
    {
        "model.vision_replacement",
        "model.transformer_replacement",
        "training.transformer_replacement",
        "training.vision_replacement",
    }
)
_SLOT_ALIASES = {"model.pretrained_replacement": "model.vision_replacement"}
_NAMED_SECTION = re.compile(r"^(training\.strategies|analysis)\.([a-z][a-z0-9_]*)$")

_BUILDERS: dict[str, SectionBuilder] = {}
_MIGRATIONS: dict[str, ConfigMigration] = {}


@dataclass
class DisabledExtensionConfig(BaseConfig):
    """An empty or disabled section of a component no installed plugin provides."""

    enabled: bool = False

    def __post_init__(self):
        if self.enabled is not False:
            raise ValueError(
                "This section requires a DendriNet plugin that is not installed"
            )


def disabled_extension(data: Any, section: str) -> DisabledExtensionConfig:
    """Accept an empty or explicitly disabled section; reject enabling it."""
    values = to_plain_dict(data)
    if values and values.get("enabled") is not False:
        raise ValueError(
            f"{section} requires a DendriNet plugin that provides it; none of the "
            "installed plugins does"
        )
    if set(values) - {"enabled"}:
        logger.info("Ignoring the settings of the disabled section %s", section)
    return DisabledExtensionConfig()


def canonical_section_path(path: str) -> str:
    """Validate an extension section path and return its canonical spelling."""
    path = str(path).strip()
    path = _SLOT_ALIASES.get(path, path)
    if path in EXTENSION_SLOTS or _NAMED_SECTION.match(path):
        return path
    raise ValueError(
        f"{path!r} is not an extension section; use one of "
        f"{sorted(EXTENSION_SLOTS)}, 'training.strategies.<name>' or "
        "'analysis.<name>'"
    )


def register_config_section(
    path: str, builder: SectionBuilder, *, allow_override: bool = False
) -> None:
    """Register the builder of an extension section.

    ``builder`` receives the section's mapping (possibly empty) and returns the
    typed section; it raises for invalid content. Replacing a registered
    builder requires ``allow_override=True``.
    """
    key = canonical_section_path(path)
    if not callable(builder):
        raise TypeError("builder must be callable")
    if key in _BUILDERS and not allow_override:
        raise ValueError(
            f"Configuration section {key!r} is already registered; pass "
            "allow_override=True to replace it"
        )
    _BUILDERS[key] = builder


def unregister_config_section(path: str) -> None:
    """Remove a registered section builder if present."""
    _BUILDERS.pop(canonical_section_path(path), None)


def config_section_builder(path: str) -> SectionBuilder | None:
    """The registered builder of an extension section, or None."""
    return _BUILDERS.get(canonical_section_path(path))


def registered_config_sections() -> list[str]:
    """Extension sections that have a registered builder."""
    return sorted(_BUILDERS)


def registered_named_sections(prefix: str) -> list[str]:
    """Names registered under ``training.strategies`` or ``analysis``."""
    head = f"{prefix}."
    return sorted(path[len(head) :] for path in _BUILDERS if path.startswith(head))


def extension_default(
    path: str, fallback: Callable[[], Any] = DisabledExtensionConfig
) -> Callable[[], Any]:
    """Default factory of the slot ``path``: the defaults of its registered section.

    Configurations constructed in code (``Config()``) then hold the same typed
    section as loaded ones; without a registered builder, ``fallback()``.
    """
    key = canonical_section_path(path)

    def factory() -> Any:
        from dendritic_modeling.plugins import load_plugins

        load_plugins()
        builder = _BUILDERS.get(key)
        return fallback() if builder is None else builder({})

    return factory


def build_extension_section(
    path: str,
    data: Any,
    *,
    default: Callable[[Any], Any] | None = None,
) -> Any:
    """Build an extension section with its registered builder.

    Without a registered builder, ``default`` builds it, or the section loads
    only when empty or disabled (:func:`disabled_extension`).
    """
    builder = config_section_builder(path)
    if builder is not None:
        return builder(data)
    if default is not None:
        return default(data)
    return disabled_extension(data, path)


def register_config_migration(
    name: str, migrate: ConfigMigration, *, allow_override: bool = False
) -> None:
    """Register a migration of historical configuration keys.

    ``migrate(data)`` receives a configuration mapping (the layout of a config
    file, as plain dicts and lists) and rewrites, in place, keys that earlier
    versions of the plugin's components read into their current form. It
    changes only keys it recognises, leaves current configurations unchanged,
    and returns one note per change, which the loader logs as a warning.
    """
    key = str(name).strip()
    if not key:
        raise ValueError("configuration migration name must be non-empty")
    if not callable(migrate):
        raise TypeError("migrate must be callable")
    if key in _MIGRATIONS and not allow_override:
        raise ValueError(
            f"Configuration migration {key!r} is already registered; pass "
            "allow_override=True to replace it"
        )
    _MIGRATIONS[key] = migrate


def unregister_config_migration(name: str) -> None:
    """Remove a registered configuration migration if present."""
    _MIGRATIONS.pop(str(name).strip(), None)


def registered_config_migrations() -> list[str]:
    """Names of the registered configuration migrations, in the order they run."""
    return sorted(_MIGRATIONS)


def _plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return copy.deepcopy(value)


def apply_config_migrations(
    data: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Run the registered migrations, in name order, on a copy of ``data``.

    Returns the migrated mapping and the notes of the changes, each prefixed
    with the migration's name. Without registered migrations ``data`` is
    returned as a dict, unchanged.
    """
    if not _MIGRATIONS:
        return dict(data), []
    migrated = _plain(data)
    notes: list[str] = []
    for name in sorted(_MIGRATIONS):
        for note in _MIGRATIONS[name](migrated) or []:
            notes.append(f"{name}: {note}")
    return migrated, notes


__all__ = [
    "EXTENSION_SLOTS",
    "ConfigMigration",
    "DisabledExtensionConfig",
    "SectionBuilder",
    "apply_config_migrations",
    "build_extension_section",
    "canonical_section_path",
    "config_section_builder",
    "disabled_extension",
    "extension_default",
    "register_config_migration",
    "register_config_section",
    "registered_config_migrations",
    "registered_config_sections",
    "registered_named_sections",
    "unregister_config_migration",
    "unregister_config_section",
]
