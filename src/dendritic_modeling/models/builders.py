"""Model builders that plugins provide.

:func:`~dendritic_modeling.scripts.script_utils.setup_utils.initialize_model`
builds the encoder, core and decoder named in ``model`` and wraps them for the
task. A plugin can take over building the three components for configurations
it recognises, for example a pretrained backbone whose layers are replaced by
dendritic cores. It registers a model builder:

* ``applies(model_config)`` returns whether the builder handles the
  configuration, usually because a plugin section of it is enabled;
* ``build(model_config)`` returns ``(encoder, core, decoder, encoder_network)``,
  where ``encoder_network`` is the module whose parameters the encoder optimizer
  group trains (often ``encoder`` itself).

``initialize_model`` then wraps the components exactly as for built-in models.
At most one registered builder may apply to a configuration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch import nn

from dendritic_modeling.plugins import load_plugins

ModelComponents = tuple[nn.Module, nn.Module, nn.Module, nn.Module]


@dataclass(frozen=True)
class ModelBuilder:
    """One registered model builder."""

    name: str
    applies: Callable[[Any], bool]
    build: Callable[[Any], ModelComponents]


_BUILDERS: dict[str, ModelBuilder] = {}


def register_model_builder(
    name: str,
    *,
    applies: Callable[[Any], bool],
    build: Callable[[Any], ModelComponents],
    allow_override: bool = False,
) -> None:
    """Register a builder of the encoder, core and decoder of some models."""
    key = str(name).strip().lower()
    if not key:
        raise ValueError("model builder name must be non-empty")
    if not callable(applies) or not callable(build):
        raise TypeError("applies and build must be callable")
    if key in _BUILDERS and not allow_override:
        raise ValueError(
            f"Model builder {key!r} is already registered; pass allow_override=True "
            "to replace it"
        )
    _BUILDERS[key] = ModelBuilder(name=key, applies=applies, build=build)


def unregister_model_builder(name: str) -> None:
    """Remove a registered model builder if present."""
    _BUILDERS.pop(str(name).strip().lower(), None)


def registered_model_builders() -> list[str]:
    """Names of the registered model builders."""
    return sorted(_BUILDERS)


def resolve_model_builder(model_config: Any) -> ModelBuilder | None:
    """The registered builder that applies to ``model_config``, or None."""
    load_plugins()
    applicable = [
        builder for builder in _BUILDERS.values() if builder.applies(model_config)
    ]
    if len(applicable) > 1:
        names = ", ".join(sorted(builder.name for builder in applicable))
        raise ValueError(f"Several model builders apply to this configuration: {names}")
    return applicable[0] if applicable else None


__all__ = [
    "ModelBuilder",
    "ModelComponents",
    "register_model_builder",
    "registered_model_builders",
    "resolve_model_builder",
    "unregister_model_builder",
]
