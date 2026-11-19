"""Describe DendriNet's configuration for the Studio's forms.

Every section, option, type and default is read from the library's typed
configuration classes, and every list of allowed values from the library's own
registries or constants. The Studio does not keep a second copy of either: an
option the library adds appears here without a Studio change.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import importlib
import json
import math
import typing
from collections.abc import Callable
from typing import Any

from dendritic_modeling.config import Config

_SCALARS = {bool: "bool", int: "int", float: "float", str: "str"}

# Payload sections are plain mappings in ``Config`` and typed by their builders.
PAYLOAD_CLASSES: dict[str, Callable[[], type]] = {}


def _population_classes() -> dict[str, type]:
    from dendritic_modeling.networks.architectures.recurrent.ei_config import (
        PopulationConfig,
    )
    from dendritic_modeling.networks.architectures.recurrent.population_configs import (
        PopulationLayerConfig,
        PopulationNetworkConfig,
        PopulationProjectionConfig,
    )

    return {
        "population_network": PopulationNetworkConfig,
        "population_layer": PopulationLayerConfig,
        "population_connection": PopulationProjectionConfig,
        "population": PopulationConfig,
    }


def _recurrent_payload_classes() -> dict[str, type]:
    """Typed payloads of the recurrent baseline families."""
    found: dict[str, type] = {}
    loaders = {
        "baseline_rnn": (
            "dendritic_modeling.networks.architectures.recurrent.baseline_rnn",
            "BaselineRNNConfig",
        ),
        "heterogeneous_leak_ctrnn": (
            "dendritic_modeling.networks.architectures.recurrent."
            "heterogeneous_leak_ctrnn",
            "HeterogeneousLeakCTRNNConfig",
        ),
        "legendre_memory": (
            "dendritic_modeling.networks.architectures.recurrent.legendre_memory",
            "FixedLegendreMemoryConfig",
        ),
    }
    for key, (module, name) in loaders.items():
        try:
            found[key] = getattr(importlib.import_module(module), name)
        except Exception:
            continue
    return found


def _json_safe(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return _json_safe(value.value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _describe_type(annotation: Any) -> dict[str, Any]:
    """Reduce a type annotation to what a form needs."""
    origin = typing.get_origin(annotation)
    args = [a for a in typing.get_args(annotation) if a is not type(None)]
    optional = type(None) in typing.get_args(annotation)
    if origin is typing.Literal:
        return {
            "kind": "choice",
            "choices": [_json_safe(a) for a in typing.get_args(annotation)],
        }
    if origin in (typing.Union, getattr(__import__("types"), "UnionType", ())):
        if len(args) == 1:
            described = _describe_type(args[0])
            described["optional"] = True
            return described
        kinds = {_describe_type(a)["kind"] for a in args}
        return {"kind": "any", "optional": optional, "variants": sorted(kinds)}
    if annotation in _SCALARS:
        return {"kind": _SCALARS[annotation], "optional": optional}
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return {"kind": "choice", "choices": [m.value for m in annotation]}
    if origin in (list, tuple) or annotation in (list, tuple):
        item = _describe_type(args[0]) if args else {"kind": "any"}
        return {"kind": "list", "item": item.get("kind", "any"), "optional": optional}
    if origin is dict or annotation is dict:
        return {"kind": "mapping", "optional": optional}
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return {"kind": "section", "class": annotation.__name__, "optional": optional}
    return {"kind": "any", "optional": optional}


def _default(field: dataclasses.Field) -> Any:
    if field.default is not dataclasses.MISSING:
        return _json_safe(field.default)
    if field.default_factory is not dataclasses.MISSING:
        try:
            return _json_safe(field.default_factory())
        except Exception:
            return None
    return None


def describe_class(cls: type, path: str) -> dict[str, Any]:
    """Fields of one typed section, in declaration order."""
    hints = typing.get_type_hints(cls)
    fields = []
    for field in dataclasses.fields(cls):
        if field.name.startswith("_") or not field.init:
            continue
        info = _describe_type(hints.get(field.name, field.type))
        entry = {
            "name": field.name,
            "path": f"{path}.{field.name}" if path else field.name,
            "default": _default(field),
            **info,
        }
        choices = choices_for(entry["path"], field.name)
        if choices:
            entry.setdefault("choices", choices)
            if entry["kind"] == "str":
                entry["kind"] = "choice"
        fields.append(entry)
    doc = (cls.__doc__ or "").strip().split("\n\n")[0]
    return {
        "class": cls.__name__,
        "path": path,
        "doc": " ".join(doc.split()),
        "fields": fields,
    }


def _walk_config() -> dict[str, Any]:
    sections: dict[str, Any] = {}
    queue: list[tuple[str, type]] = [("", Config)]
    while queue:
        path, cls = queue.pop(0)
        if path in sections:
            continue
        described = describe_class(cls, path)
        sections[path] = described
        hints = typing.get_type_hints(cls)
        for field in dataclasses.fields(cls):
            annotation = hints.get(field.name, field.type)
            for candidate in (annotation, *typing.get_args(annotation)):
                if isinstance(candidate, type) and dataclasses.is_dataclass(candidate):
                    child = f"{path}.{field.name}" if path else field.name
                    queue.append((child, candidate))
    return sections


# ---------------------------------------------------------------------------
# Allowed values, each read from the library. A provider that cannot be
# imported (an optional dependency, a renamed constant) returns nothing; the
# form then falls back to free text and the library validates the value.


def _safe(provider: Callable[[], Any]) -> Callable[[], list]:
    @functools.wraps(provider)
    def wrapped() -> list:
        try:
            return sorted({str(v) for v in provider()})
        except Exception:
            return []

    return wrapped


@_safe
def _core_types():
    from dendritic_modeling.config.core_sections import BUILTIN_CORE_TYPES

    return BUILTIN_CORE_TYPES


@_safe
def _datasets():
    from dendritic_modeling.datasets.standard_datasets import _known_dataset_names

    return _known_dataset_names()


@_safe
def _strategies():
    from dendritic_modeling.training.factory import (
        SPECIAL_TRAINER_STRATEGIES,
        TRAINER_STRATEGY_REGISTRY,
    )

    return [*TRAINER_STRATEGY_REGISTRY, *SPECIAL_TRAINER_STRATEGIES]


@_safe
def _losses():
    from dendritic_modeling.training.loss.functions import LOSS_FUNCTION_REGISTRY

    return LOSS_FUNCTION_REGISTRY


@_safe
def _optimizers():
    from dendritic_modeling.training.optimizers.factory import OPTIMIZER_REGISTRY

    return OPTIMIZER_REGISTRY


@_safe
def _lr_schedules():
    from dendritic_modeling.training.strategies.standard import LR_SCHEDULES

    return LR_SCHEDULES


@_safe
def _sparse_layers():
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.factory import (
        SPARSE_LAYER_REGISTRY,
    )

    return SPARSE_LAYER_REGISTRY


@_safe
def _weight_transforms():
    from dendritic_modeling.networks.utils.weight_transforms import WeightTransformType

    return typing.get_args(WeightTransformType)


@_safe
def _reactivation_policies():
    from dendritic_modeling.config.reactivation import REACTIVATION_INIT_POLICIES

    return REACTIVATION_INIT_POLICIES


@_safe
def _topk_init_methods():
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
        TOPK_INIT_METHODS,
    )

    return TOPK_INIT_METHODS


@_safe
def _additive_modes():
    from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_dynamics import (
        ADDITIVE_MODES,
    )

    return ADDITIVE_MODES


@_safe
def _reactivation_update_modes():
    from dendritic_modeling.config.reactivation import REACTIVATION_UPDATE_MODES

    return REACTIVATION_UPDATE_MODES


@_safe
def _baseline_cells():
    from dendritic_modeling.networks.architectures.factory_recurrent import (
        _BASELINE_RNN_TYPES,
    )

    return _BASELINE_RNN_TYPES.values()


# Option name (or full path) -> provider. Names apply wherever the option
# appears (``population.topk_type`` and ``population_defaults.topk_type``).
CHOICE_PROVIDERS: dict[str, Callable[[], list]] = {
    "model.core.type": _core_types,
    "data.dataset_name": _datasets,
    "training.main.strategy": _strategies,
    "training.main.common.loss_function": _losses,
    "training.main.optimizer.name": _optimizers,
    "training.main.common.lr_schedule": _lr_schedules,
    "training.main.common.reactivation_update_mode": _reactivation_update_modes,
    "model.core.sparsity.type": _sparse_layers,
    "model.core.reactivation.init_policy": _reactivation_policies,
    "topk_type": _sparse_layers,
    "weight_transform": _weight_transforms,
    "reactivation_init_policy": _reactivation_policies,
    "topk_init_method": _topk_init_methods,
    "additive_mode": _additive_modes,
    "baseline_rnn.cell_type": _baseline_cells,
}


def choices_for(path: str, name: str) -> list:
    """Allowed values of one option: by its full path, else by its name."""
    provider = CHOICE_PROVIDERS.get(path) or CHOICE_PROVIDERS.get(name)
    return provider() if provider is not None else []


def loss_tasks() -> dict[str, list[str]]:
    """Tasks each loss declares; a loss without a declaration fits any task."""
    try:
        from dendritic_modeling.training.loss.functions import LOSS_FUNCTION_REGISTRY
    except Exception:
        return {}
    tasks = {}
    for name, builder in LOSS_FUNCTION_REGISTRY.items():
        try:
            declared = getattr(builder(), "supported_tasks", None)
        except Exception:
            declared = None
        tasks[name] = sorted(declared) if declared else []
    return tasks


def dataset_groups() -> dict[str, list[str]]:
    """Dataset names grouped by how the library produces them."""
    groups: dict[str, list[str]] = {}
    try:
        from dendritic_modeling.datasets import standard_datasets as sd
        from dendritic_modeling.datasets.registry import get_registered_dataset_names
        from dendritic_modeling.datasets.sequence.builders.registry import (
            SEQUENCE_DATASET_BUILDERS,
        )

        groups = {
            "Image (downloads once)": sorted(sd._STANDARD_VISION_DATASET_LOADERS),
            "Image variants": sorted(sd._SPECIAL_DATASET_LOADERS),
            "Synthetic (no download)": sorted(sd._THEORETICAL_SYNTHETIC_DATASET_NAMES),
            "Sequences and tasks over time": sorted(SEQUENCE_DATASET_BUILDERS),
            "Registered": sorted(get_registered_dataset_names()),
        }
    except Exception:
        names = _datasets()
        groups = {"Datasets": names} if names else {}
    return {k: v for k, v in groups.items() if v}


@functools.lru_cache(maxsize=1)
def describe_config_schema() -> dict[str, Any]:
    """The complete description the Studio's forms are built from."""
    sections = _walk_config()
    payloads = {}
    try:
        for key, cls in _population_classes().items():
            payloads[key] = describe_class(cls, key)
    except Exception:
        payloads = {}
    for key, cls in _recurrent_payload_classes().items():
        payloads[key] = describe_class(cls, key)
    try:
        from dendritic_modeling.config.core_sections import (
            BUILTIN_CORE_TYPES,
            core_family,
        )

        families = {name: core_family(name) for name in sorted(BUILTIN_CORE_TYPES)}
    except Exception:
        families = {}
    schema = {
        "sections": sections,
        "payloads": payloads,
        "core_families": families,
        "loss_tasks": loss_tasks(),
        "dataset_groups": dataset_groups(),
        "tasks": ["classification", "regression"],
    }
    json.dumps(schema, allow_nan=False)
    return schema
