"""Analyzer metadata registry.

``AnalysisManager`` still owns analyzer execution so current analysis behavior
is unchanged.  This registry records analyzer metadata in one place and creates
a safe extension surface for future manager refactors.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Literal

AnalyzerConstructor = Literal["params", "model_params", "none"]


@dataclass(frozen=True)
class AnalyzerSpec:
    """Metadata describing one analysis component."""

    name: str
    config_field: str
    class_path: str
    attribute_name: str
    save_subdir: str
    constructor_style: AnalyzerConstructor = "params"
    supports_training: bool = True
    description: str = ""

    def load_class(self) -> type:
        """Resolve the analyzer class declared by this spec."""
        module_name, _, class_name = self.class_path.rpartition(".")
        if not module_name or not class_name:
            raise ValueError(f"Invalid analyzer class path: {self.class_path!r}")
        module = importlib.import_module(module_name)
        analyzer_cls = getattr(module, class_name)
        if not isinstance(analyzer_cls, type):
            raise TypeError(
                f"Analyzer path does not resolve to a class: {self.class_path}"
            )
        return analyzer_cls


_ANALYZER_REGISTRY: dict[str, AnalyzerSpec] = {}


def register_analyzer(
    name: str,
    *,
    config_field: str,
    class_path: str,
    attribute_name: str,
    save_subdir: str,
    constructor_style: AnalyzerConstructor = "params",
    supports_training: bool = True,
    description: str = "",
    allow_override: bool = False,
) -> None:
    """Register analyzer metadata without changing execution behavior."""
    key = str(name).lower()
    if not key:
        raise ValueError("analyzer name must be non-empty")
    if key in _ANALYZER_REGISTRY and not allow_override:
        raise ValueError(f"Analyzer '{key}' is already registered")
    if constructor_style not in ("params", "model_params", "none"):
        raise ValueError(f"Unsupported analyzer constructor style: {constructor_style}")
    _ANALYZER_REGISTRY[key] = AnalyzerSpec(
        name=key,
        config_field=config_field,
        class_path=class_path,
        attribute_name=attribute_name,
        save_subdir=save_subdir,
        constructor_style=constructor_style,
        supports_training=supports_training,
        description=description,
    )


def unregister_analyzer(name: str) -> None:
    """Remove analyzer metadata if present."""
    _ANALYZER_REGISTRY.pop(str(name).lower(), None)


def get_analyzer_spec(name: str) -> AnalyzerSpec | None:
    """Return metadata for one analyzer."""
    return _ANALYZER_REGISTRY.get(str(name).lower())


def get_registered_analyzer_names() -> list[str]:
    """Return registered analyzer names."""
    return sorted(_ANALYZER_REGISTRY)


def get_registered_analyzers() -> dict[str, AnalyzerSpec]:
    """Return a copy of all analyzer specs."""
    return dict(_ANALYZER_REGISTRY)


def load_analyzer_class(name: str) -> type:
    """Resolve the registered analyzer class for one analyzer name."""
    spec = get_analyzer_spec(name)
    if spec is None:
        raise KeyError(f"Analyzer {name!r} is not registered")
    return spec.load_class()


def _register_builtin_analyzers() -> None:
    constructor_styles: dict[str, AnalyzerConstructor] = {
        "synapse_turnover": "model_params",
        "routing": "none",
    }
    builtins = [
        (
            "performance",
            "performance_analysis",
            "dendritic_modeling.analysis.metrics.performance.PerformanceAnalyzer",
            "performance_analyzer",
            "performance",
        ),
        (
            "information",
            "information_analysis",
            "dendritic_modeling.analysis.core.information.InformationAnalyzer",
            "information_analyzer",
            "information_analysis",
        ),
        (
            "synapse_turnover",
            "synapse_turnover_analysis",
            "dendritic_modeling.analysis.tools.synaptic_turnover.SynapticTurnoverAnalyzer",
            "syn_turnover_analyzer",
            "synapse_turnover_analysis",
        ),
        (
            "noise_perturbation",
            "noise_perturbation_analysis",
            "dendritic_modeling.analysis.metrics.noise_perturbation.NoisePerturbationAnalyzer",
            "noise_perturbation_analyzer",
            "noise_perturbation_analysis",
        ),
        (
            "single_layer_contribution",
            "single_layer_contribution",
            "dendritic_modeling.analysis.metrics.layer_contribution.SingleLayerContributionAnalyzer",
            "single_layer_analyzer",
            "single_layer_contribution",
        ),
        (
            "ablation",
            "ablation_analysis",
            "dendritic_modeling.analysis.core.ablation.AblationAnalyzer",
            "ablation_analyzer",
            "ablation_analysis",
        ),
        (
            "weight",
            "weight_analysis",
            "dendritic_modeling.analysis.tools.weight.WeightAnalyzer",
            "weight_analyzer",
            "weight_analysis",
        ),
        (
            "compartment_statistics",
            "compartment_statistics_analysis",
            "dendritic_modeling.analysis.tools.compartment_statistics.CompartmentStatisticsAnalyzer",
            "compartment_statistics_analyzer",
            "compartment_statistics_analysis",
        ),
        (
            "compartment_snr",
            "compartment_snr_analysis",
            "dendritic_modeling.analysis.tools.compartment_snr.CompartmentSNRAnalyzer",
            "compartment_snr_analyzer",
            "compartment_snr_analysis",
        ),
        (
            "multiplicative_gain",
            "multiplicative_gain_analysis",
            "dendritic_modeling.analysis.tools.multiplicative_gain.MultiplicativeGainAnalyzer",
            "multiplicative_gain_analyzer",
            "multiplicative_gain_analysis",
        ),
        (
            "receptive_field",
            "receptive_field_analysis",
            "dendritic_modeling.analysis.tools.receptive_fields.ReceptiveFieldAnalyzer",
            "receptive_field_analyzer",
            "receptive_field_analysis",
        ),
        (
            "reactivation_dynamics",
            "reactivation_dynamics_analysis",
            "dendritic_modeling.analysis.tools.reactivation_dynamics.ReactivationDynamicsAnalyzer",
            "reactivation_dynamics_analyzer",
            "reactivation_dynamics",
        ),
        (
            "synaptic_activation",
            "synaptic_activation_analysis",
            "dendritic_modeling.analysis.tools.synaptic_activation.SynapticActivationAnalyzer",
            "synaptic_activation_analyzer",
            "synaptic_activation_analysis",
        ),
        (
            "branch_activation",
            "branch_activation_analysis",
            "dendritic_modeling.analysis.core.branch_activation.BranchActivationAnalyzer",
            "branch_activation_analyzer",
            "branch_activation_analysis",
        ),
        (
            "local_rule_components",
            "local_rule_component_analysis",
            "dendritic_modeling.analysis.tools.local_rule_components.LocalRuleComponentAnalyzer",
            "local_rule_component_analyzer",
            "local_rule_components",
        ),
        (
            "correlation",
            "correlation_analysis",
            "dendritic_modeling.analysis.tools.correlation_analysis.CorrelationAnalyzer",
            "correlation_analyzer",
            "correlation_analysis",
        ),
        (
            "routing",
            "routing_analysis",
            "dendritic_modeling.analysis.tools.routing_analysis.RoutingAnalyzer",
            "routing_analyzer",
            "routing_analysis",
        ),
        (
            "inhibitory_specialization",
            "inhibitory_specialization_analysis",
            "dendritic_modeling.analysis.tools.inhibitory_specialization.InhibitorySpecializationAnalyzer",
            "inhibitory_specialization_analyzer",
            "inhibitory_specialization",
        ),
        (
            "jacobian_spectrum",
            "jacobian_spectrum_analysis",
            "dendritic_modeling.analysis.tools.jacobian_spectrum.JacobianSpectrumAnalyzer",
            "jacobian_spectrum_analyzer",
            "jacobian_spectrum",
        ),
        (
            "dendritic_timetraces",
            "dendritic_timetraces_analysis",
            "dendritic_modeling.analysis.tools.dendritic_timetraces.DendriticTimetracesAnalyzer",
            "dendritic_timetraces_analyzer",
            "dendritic_timetraces",
        ),
        (
            "spike_train",
            "spike_train_analysis",
            "dendritic_modeling.analysis.tools.spike_trains.SpikeTrainAnalyzer",
            "spike_train_analyzer",
            "spike_train_analysis",
        ),
        (
            "adversarial_robustness",
            "adversarial_robustness_analysis",
            "dendritic_modeling.analysis.metrics.adversarial_robustness.AdversarialRobustnessAnalyzer",
            "adversarial_robustness_analyzer",
            "adversarial_robustness_analysis",
        ),
    ]
    for name, config_field, class_path, attribute_name, save_subdir in builtins:
        register_analyzer(
            name,
            config_field=config_field,
            class_path=class_path,
            attribute_name=attribute_name,
            save_subdir=save_subdir,
            constructor_style=constructor_styles.get(name, "params"),
        )


_register_builtin_analyzers()


__all__ = [
    "AnalyzerConstructor",
    "AnalyzerSpec",
    "get_analyzer_spec",
    "get_registered_analyzer_names",
    "get_registered_analyzers",
    "load_analyzer_class",
    "register_analyzer",
    "unregister_analyzer",
]
