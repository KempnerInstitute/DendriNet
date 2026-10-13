"""Orchestration layer that coordinates analysis pipelines across trained models.

Names are imported on first access, so tools that only read status records
(``analysis_status``) do not import the analysis manager and its models.
"""

from importlib import import_module
from typing import Any

_LAZY_IMPORTS = {
    "AnalysisIncompleteError": ".analysis_status",
    "AnalysisManager": ".analysis_manager",
}

__all__ = ["AnalysisIncompleteError", "AnalysisManager"]


def __getattr__(name: str) -> Any:
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
