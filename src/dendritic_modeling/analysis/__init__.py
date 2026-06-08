"""Public analysis API for reusable analyzers and helper metrics.

The package exposes the same top-level names as before, but imports them
lazy-on-access. This keeps utility-only imports such as
``dendritic_modeling.analysis.utils.runtime`` from importing analysis core
modules that themselves depend on training evaluation helpers.
"""

from importlib import import_module
from typing import Any

_LAZY_IMPORTS = {
    "AblationAnalyzer": ("dendritic_modeling.analysis.core", "AblationAnalyzer"),
    "AnalyzerSpec": ("dendritic_modeling.analysis.registry", "AnalyzerSpec"),
    "AnalysisManager": (
        "dendritic_modeling.analysis.managers.analysis_manager",
        "AnalysisManager",
    ),
    "BranchActivationAnalyzer": (
        "dendritic_modeling.analysis.core",
        "BranchActivationAnalyzer",
    ),
    "CorrelationAnalyzer": (
        "dendritic_modeling.analysis.tools",
        "CorrelationAnalyzer",
    ),
    "DendriticTimetracesAnalyzer": (
        "dendritic_modeling.analysis.tools",
        "DendriticTimetracesAnalyzer",
    ),
    "InformationAnalyzer": ("dendritic_modeling.analysis.core", "InformationAnalyzer"),
    "InhibitorySpecializationAnalyzer": (
        "dendritic_modeling.analysis.tools",
        "InhibitorySpecializationAnalyzer",
    ),
    "JacobianSpectrumAnalyzer": (
        "dendritic_modeling.analysis.tools",
        "JacobianSpectrumAnalyzer",
    ),
    "LocalRuleComponentAnalyzer": (
        "dendritic_modeling.analysis.tools",
        "LocalRuleComponentAnalyzer",
    ),
    "NoisePerturbationAnalyzer": (
        "dendritic_modeling.analysis.metrics",
        "NoisePerturbationAnalyzer",
    ),
    "PerformanceAnalyzer": (
        "dendritic_modeling.analysis.metrics",
        "PerformanceAnalyzer",
    ),
    "ReactivationDynamicsAnalyzer": (
        "dendritic_modeling.analysis.tools",
        "ReactivationDynamicsAnalyzer",
    ),
    "SingleLayerContributionAnalyzer": (
        "dendritic_modeling.analysis.metrics",
        "SingleLayerContributionAnalyzer",
    ),
    "SynapticActivationAnalyzer": (
        "dendritic_modeling.analysis.tools",
        "SynapticActivationAnalyzer",
    ),
    "SynapticTurnoverAnalyzer": (
        "dendritic_modeling.analysis.tools",
        "SynapticTurnoverAnalyzer",
    ),
    "WeightAnalyzer": ("dendritic_modeling.analysis.tools", "WeightAnalyzer"),
    "get_analyzer_spec": (
        "dendritic_modeling.analysis.registry",
        "get_analyzer_spec",
    ),
    "get_registered_analyzer_names": (
        "dendritic_modeling.analysis.registry",
        "get_registered_analyzer_names",
    ),
    "get_registered_analyzers": (
        "dendritic_modeling.analysis.registry",
        "get_registered_analyzers",
    ),
    "register_analyzer": (
        "dendritic_modeling.analysis.registry",
        "register_analyzer",
    ),
    "unregister_analyzer": (
        "dendritic_modeling.analysis.registry",
        "unregister_analyzer",
    ),
    "direction_selectivity_index": (
        "dendritic_modeling.analysis.metrics",
        "direction_selectivity_index",
    ),
    "length_selectivity_index": (
        "dendritic_modeling.analysis.metrics",
        "length_selectivity_index",
    ),
    "orientation_selectivity_index": (
        "dendritic_modeling.analysis.metrics",
        "orientation_selectivity_index",
    ),
    "size_tuning_index": (
        "dendritic_modeling.analysis.metrics",
        "size_tuning_index",
    ),
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
