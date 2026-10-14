"""Plugins: installed packages that extend DendriNet through its registries.

A plugin is a distribution that declares an entry point in the
``dendritic_modeling.plugins`` group, for example in its ``pyproject.toml``::

    [project.entry-points."dendritic_modeling.plugins"]
    my_plugin = "my_package.plugin:register"

The entry point names a callable without arguments that registers the
plugin's components through the library's registries: configuration sections,
model builders, architectures, sparse synapse layers, activations, additive
integration rules, datasets, trainer strategies, losses, optimizers and
analyzers (see the "Extending DendriNet" guide).

DendriNet loads every installed plugin once, before the first configuration
is loaded, whenever a registry is asked for a name it does not know and
whenever a registry lists its names. Set ``DENDRITIC_MODELING_PLUGINS=0`` to
use built-in components only. A plugin that fails to load raises
:class:`PluginLoadError`: a run never continues silently without a plugin it
may depend on. The failure is kept for the rest of the process, because the
plugin may have registered part of its components, so every later load raises
it again; fix or uninstall the plugin and start a new process. Training runs
record the loaded plugins in ``runtime_environment.json``.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, NoReturn

ENTRY_POINT_GROUP = "dendritic_modeling.plugins"
PLUGINS_ENVIRONMENT_VARIABLE = "DENDRITIC_MODELING_PLUGINS"
_DISABLED_VALUES = frozenset({"0", "false", "no", "off"})

_LOCK = threading.RLock()
_LOADED: dict[str, PluginRecord] = {}
# Plugins that failed to import or register in this process, with the reason.
_FAILED: dict[str, str] = {}
_STATE = {"discovered": False}


class PluginLoadError(RuntimeError):
    """An installed plugin could not be imported or failed to register."""


@dataclass(frozen=True)
class PluginRecord:
    """One loaded plugin, as recorded in a run's provenance."""

    name: str
    target: str
    distribution: str | None = None
    version: str | None = None
    module_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def plugins_enabled() -> bool:
    """Whether installed plugins load automatically (``DENDRITIC_MODELING_PLUGINS``)."""
    value = os.environ.get(PLUGINS_ENVIRONMENT_VARIABLE, "1")
    return value.strip().lower() not in _DISABLED_VALUES


def _module_path(target: str) -> str | None:
    import sys

    module = sys.modules.get(target.split(":", 1)[0])
    location = getattr(module, "__file__", None)
    return None if location is None else str(Path(location).resolve())


def _fail(name: str, message: str, error: BaseException) -> NoReturn:
    _FAILED[name] = message
    raise PluginLoadError(message) from error


def _raise_recorded_failures() -> None:
    if _FAILED:
        raise PluginLoadError(
            "A DendriNet plugin failed to load earlier in this process and may "
            "have registered part of its components: "
            + "; ".join(_FAILED.values())
            + ". Fix or uninstall it and start a new process, or start with "
            f"{PLUGINS_ENVIRONMENT_VARIABLE}=0 to use built-in components only."
        )


def _activate(
    name: str,
    register: Callable[[], Any],
    *,
    target: str,
    distribution: str | None,
    version: str | None,
) -> PluginRecord:
    try:
        register()
    except Exception as error:
        _fail(
            name,
            f"DendriNet plugin {name!r} ({target}) failed to register: {error}",
            error,
        )
    record = PluginRecord(
        name=name,
        target=target,
        distribution=distribution,
        version=version,
        module_path=_module_path(target),
    )
    _LOADED[name] = record
    return record


def load_plugins() -> tuple[PluginRecord, ...]:
    """Load every installed plugin once and return the loaded plugins.

    Later calls return the same plugins without loading anything again, or
    raise :class:`PluginLoadError` again when a plugin failed to load.
    """
    with _LOCK:
        _raise_recorded_failures()
        if _STATE["discovered"]:
            return loaded_plugins()
        # Set first: a plugin's registration may resolve names, which calls
        # this function again.
        _STATE["discovered"] = True
        if not plugins_enabled():
            return loaded_plugins()
        for entry in sorted(entry_points(group=ENTRY_POINT_GROUP), key=_entry_key):
            if entry.name in _LOADED:
                continue
            distribution = getattr(entry, "dist", None)
            try:
                register = entry.load()
            except Exception as error:
                _fail(
                    entry.name,
                    f"DendriNet plugin {entry.name!r} ({entry.value}) could not be "
                    f"imported: {error}",
                    error,
                )
            _activate(
                entry.name,
                register,
                target=entry.value,
                distribution=None if distribution is None else distribution.name,
                version=None if distribution is None else distribution.version,
            )
        return loaded_plugins()


def _entry_key(entry: Any) -> tuple[str, str]:
    return (entry.name, entry.value)


def activate_plugin(name: str, register: Callable[[], Any]) -> PluginRecord:
    """Register a plugin that is not installed as a distribution.

    For notebooks and tests: ``register`` runs now and the plugin is recorded
    under ``name`` like an installed one. Activating the same name again
    raises, and a failed activation is kept like a failed installed plugin.
    """
    with _LOCK:
        _raise_recorded_failures()
        if name in _LOADED:
            raise ValueError(f"DendriNet plugin {name!r} is already loaded")
        target = f"{getattr(register, '__module__', '?')}:{getattr(register, '__qualname__', '?')}"
        return _activate(name, register, target=target, distribution=None, version=None)


def loaded_plugins() -> tuple[PluginRecord, ...]:
    """The plugins loaded in this process, in loading order."""
    return tuple(_LOADED.values())


__all__ = [
    "ENTRY_POINT_GROUP",
    "PLUGINS_ENVIRONMENT_VARIABLE",
    "PluginLoadError",
    "PluginRecord",
    "activate_plugin",
    "load_plugins",
    "loaded_plugins",
    "plugins_enabled",
]
