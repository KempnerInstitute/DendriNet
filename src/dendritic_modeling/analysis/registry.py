"""Analyzer registry.

Each analyzer is described by an :class:`AnalyzerSpec`. ``AnalysisManager``
builds the analyzers whose configuration section is enabled and runs them in
its fixed order. Plugins add analyzers with :func:`register_analyzer`; an
analyzer registered with a ``phase`` runs in that phase of the order, and its
configuration section is the plugin section ``analysis.<config_field>``
(registered with :func:`dendritic_modeling.config.extensions.register_config_section`).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Literal

from dendritic_modeling.plugins import load_plugins

AnalyzerConstructor = Literal["params", "model_params", "none"]
#: Where a plugin analyzer runs in the analysis order: with the test-dataset
#: analyses before or after the weight and reactivation analyses, or with the
#: model-only analyses.
AnalyzerPhase = Literal["pre_weight", "post_reactivation", "model_only"]
ANALYZER_PHASES = ("pre_weight", "post_reactivation", "model_only")


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
    # Analyzers that compare snapshots taken during training.
    requires_training: bool = False
    description: str = ""
    # Plugin analyzers: where they run, whether the runner passes ``training``
    # and whether a missing analyzer instance is an error.
    phase: AnalyzerPhase | None = None
    pass_training: bool = False
    require_analyzer: bool = False

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


_BUILTIN_ANALYZERS: tuple[AnalyzerSpec, ...] = (
    AnalyzerSpec(
        name="performance",
        config_field="performance",
        class_path="dendritic_modeling.analysis.metrics.performance.PerformanceAnalyzer",
        attribute_name="performance_analyzer",
        save_subdir="performance",
    ),
    AnalyzerSpec(
        name="information",
        config_field="information",
        class_path="dendritic_modeling.analysis.core.information.InformationAnalyzer",
        attribute_name="information_analyzer",
        save_subdir="information_analysis",
    ),
    AnalyzerSpec(
        name="representation_accessibility",
        config_field="representation_accessibility",
        class_path=(
            "dendritic_modeling.analysis.tools.representation_accessibility."
            "RepresentationAccessibilityAnalyzer"
        ),
        attribute_name="representation_accessibility_analyzer",
        save_subdir="representation_accessibility",
        supports_training=False,
        description=(
            "Validation-fit linear/nonlinear probes and matched-support class "
            "information for stacked soma populations"
        ),
    ),
    AnalyzerSpec(
        name="source_tuning_support",
        config_field="source_tuning_support",
        class_path=(
            "dendritic_modeling.analysis.tools.source_tuning."
            "SourceTuningSupportAnalyzer"
        ),
        attribute_name="source_tuning_support_analyzer",
        save_subdir="source_tuning_support_analysis",
        supports_training=False,
        description=(
            "Split-safe source-coordinate tuning aligned with exact sparse "
            "contacts and effective conductances"
        ),
    ),
    AnalyzerSpec(
        name="input_region_intervention",
        config_field="input_region_intervention",
        class_path=(
            "dendritic_modeling.analysis.tools.input_region_intervention."
            "InputRegionInterventionAnalyzer"
        ),
        attribute_name="input_region_intervention_analyzer",
        save_subdir="input_region_intervention",
        supports_training=False,
        description=(
            "Paired image-region interventions with layerwise E/I response propagation"
        ),
    ),
    AnalyzerSpec(
        name="synapse_turnover",
        config_field="synapse_turnover",
        class_path="dendritic_modeling.analysis.tools.synaptic_turnover.SynapticTurnoverAnalyzer",
        attribute_name="syn_turnover_analyzer",
        save_subdir="synapse_turnover_analysis",
        constructor_style="model_params",
        requires_training=True,
    ),
    AnalyzerSpec(
        name="synapse_pruning",
        config_field="synapse_pruning",
        class_path="dendritic_modeling.analysis.tools.synaptic_pruning.SynapticPruningAnalyzer",
        attribute_name="syn_pruning_analyzer",
        save_subdir="synapse_pruning_analysis",
        supports_training=False,
        description="Realized-contact census and the contacts each pruning step removed",
    ),
    AnalyzerSpec(
        name="noise_perturbation",
        config_field="noise_perturbation",
        class_path="dendritic_modeling.analysis.metrics.noise_perturbation.NoisePerturbationAnalyzer",
        attribute_name="noise_perturbation_analyzer",
        save_subdir="noise_perturbation_analysis",
    ),
    AnalyzerSpec(
        name="single_layer_contribution",
        config_field="single_layer_contribution",
        class_path="dendritic_modeling.analysis.metrics.layer_contribution.SingleLayerContributionAnalyzer",
        attribute_name="single_layer_analyzer",
        save_subdir="single_layer_contribution",
    ),
    AnalyzerSpec(
        name="ablation",
        config_field="ablation",
        class_path="dendritic_modeling.analysis.core.ablation.AblationAnalyzer",
        attribute_name="ablation_analyzer",
        save_subdir="ablation_analysis",
    ),
    AnalyzerSpec(
        name="path_matched_intervention",
        config_field="path_matched_intervention",
        class_path=(
            "dendritic_modeling.analysis.tools.path_matched_intervention."
            "PathMatchedInterventionAnalyzer"
        ),
        attribute_name="path_matched_intervention_analyzer",
        save_subdir="path_matched_intervention",
        supports_training=False,
        description=(
            "Fixed-count, nested soma-to-distal coordinate interventions across "
            "dendritic depth"
        ),
    ),
    AnalyzerSpec(
        name="branch_local_sensitivity",
        config_field="branch_local_sensitivity",
        class_path=(
            "dendritic_modeling.analysis.tools.branch_local_sensitivity."
            "BranchLocalSensitivityAnalyzer"
        ),
        attribute_name="branch_local_sensitivity_analyzer",
        save_subdir="branch_local_sensitivity",
        supports_training=False,
        description=(
            "Observed local E/I/inherited-current branch sensitivities including "
            "the fitted reactivation derivative"
        ),
    ),
    AnalyzerSpec(
        name="node_perturbation_credit",
        config_field="node_perturbation_credit",
        class_path=(
            "dendritic_modeling.analysis.tools.node_perturbation_credit."
            "NodePerturbationCreditAnalyzer"
        ),
        attribute_name="node_perturbation_credit_analyzer",
        save_subdir="node_perturbation_credit",
        supports_training=False,
        description=(
            "Compartment errors estimated from loss changes under antithetic "
            "perturbations, compared with the exact errors"
        ),
    ),
    AnalyzerSpec(
        name="local_inhibitory_population_intervention",
        config_field="local_inhibitory_population_intervention",
        class_path=(
            "dendritic_modeling.analysis.tools.layer_population_intervention."
            "LocalInhibitoryPopulationInterventionAnalyzer"
        ),
        attribute_name="local_inhibitory_population_intervention_analyzer",
        save_subdir="local_inhibitory_population_intervention",
        supports_training=False,
        description=(
            "Whole local inhibitory soma-population interventions across "
            "feedforward network layers"
        ),
    ),
    AnalyzerSpec(
        name="path_matched_information",
        config_field="path_matched_information",
        class_path=(
            "dendritic_modeling.analysis.tools.path_matched_information."
            "PathMatchedInformationAnalyzer"
        ),
        attribute_name="path_matched_information_analyzer",
        save_subdir="path_matched_information",
        supports_training=False,
        description=(
            "Validation-set joint class information on fixed-width, nested "
            "soma-to-distal coordinate supports"
        ),
    ),
    AnalyzerSpec(
        name="weight",
        config_field="weight",
        class_path="dendritic_modeling.analysis.tools.weight.WeightAnalyzer",
        attribute_name="weight_analyzer",
        save_subdir="weight_analysis",
    ),
    AnalyzerSpec(
        name="compartment_statistics",
        config_field="compartment_statistics",
        class_path="dendritic_modeling.analysis.tools.compartment_statistics.CompartmentStatisticsAnalyzer",
        attribute_name="compartment_statistics_analyzer",
        save_subdir="compartment_statistics_analysis",
    ),
    AnalyzerSpec(
        name="compartment_snr",
        config_field="compartment_snr",
        class_path="dendritic_modeling.analysis.tools.compartment_snr.CompartmentSNRAnalyzer",
        attribute_name="compartment_snr_analyzer",
        save_subdir="compartment_snr_analysis",
    ),
    AnalyzerSpec(
        name="multiplicative_gain",
        config_field="multiplicative_gain",
        class_path="dendritic_modeling.analysis.tools.multiplicative_gain.MultiplicativeGainAnalyzer",
        attribute_name="multiplicative_gain_analyzer",
        save_subdir="multiplicative_gain_analysis",
    ),
    AnalyzerSpec(
        name="gain_load_perturbation",
        config_field="gain_load_perturbation",
        class_path=(
            "dendritic_modeling.analysis.tools.gain_load_perturbation."
            "GainLoadPerturbationAnalyzer"
        ),
        attribute_name="gain_load_perturbation_analyzer",
        save_subdir="gain_load_perturbation",
        supports_training=False,
        description=(
            "Mean-one trial-shared lognormal gain crossed with an explicit "
            "independent input-load intervention"
        ),
    ),
    AnalyzerSpec(
        name="receptive_field",
        config_field="receptive_field",
        class_path="dendritic_modeling.analysis.tools.receptive_fields.ReceptiveFieldAnalyzer",
        attribute_name="receptive_field_analyzer",
        save_subdir="receptive_field_analysis",
    ),
    AnalyzerSpec(
        name="reactivation_dynamics",
        config_field="reactivation_dynamics",
        class_path="dendritic_modeling.analysis.tools.reactivation_dynamics.ReactivationDynamicsAnalyzer",
        attribute_name="reactivation_dynamics_analyzer",
        save_subdir="reactivation_dynamics",
    ),
    AnalyzerSpec(
        name="synaptic_activation",
        config_field="synaptic_activation",
        class_path="dendritic_modeling.analysis.tools.synaptic_activation.SynapticActivationAnalyzer",
        attribute_name="synaptic_activation_analyzer",
        save_subdir="synaptic_activation_analysis",
    ),
    AnalyzerSpec(
        name="branch_activation",
        config_field="branch_activation",
        class_path="dendritic_modeling.analysis.core.branch_activation.BranchActivationAnalyzer",
        attribute_name="branch_activation_analyzer",
        save_subdir="branch_activation_analysis",
    ),
    AnalyzerSpec(
        name="local_rule_components",
        config_field="local_rule_components",
        class_path="dendritic_modeling.analysis.tools.local_rule_components.LocalRuleComponentAnalyzer",
        attribute_name="local_rule_component_analyzer",
        save_subdir="local_rule_components",
    ),
    AnalyzerSpec(
        name="correlation",
        config_field="correlation",
        class_path="dendritic_modeling.analysis.tools.correlation_analysis.CorrelationAnalyzer",
        attribute_name="correlation_analyzer",
        save_subdir="correlation_analysis",
    ),
    AnalyzerSpec(
        name="routing",
        config_field="routing",
        class_path="dendritic_modeling.analysis.tools.routing_analysis.RoutingAnalyzer",
        attribute_name="routing_analyzer",
        save_subdir="routing_analysis",
        constructor_style="none",
    ),
    AnalyzerSpec(
        name="inhibitory_specialization",
        config_field="inhibitory_specialization",
        class_path="dendritic_modeling.analysis.tools.inhibitory_specialization.InhibitorySpecializationAnalyzer",
        attribute_name="inhibitory_specialization_analyzer",
        save_subdir="inhibitory_specialization",
    ),
    AnalyzerSpec(
        name="jacobian_spectrum",
        config_field="jacobian_spectrum",
        class_path="dendritic_modeling.analysis.tools.jacobian_spectrum.JacobianSpectrumAnalyzer",
        attribute_name="jacobian_spectrum_analyzer",
        save_subdir="jacobian_spectrum",
    ),
    AnalyzerSpec(
        name="dendritic_timetraces",
        config_field="dendritic_timetraces",
        class_path="dendritic_modeling.analysis.tools.dendritic_timetraces.DendriticTimetracesAnalyzer",
        attribute_name="dendritic_timetraces_analyzer",
        save_subdir="dendritic_timetraces",
    ),
    AnalyzerSpec(
        name="spike_train",
        config_field="spike_train",
        class_path="dendritic_modeling.analysis.tools.spike_trains.SpikeTrainAnalyzer",
        attribute_name="spike_train_analyzer",
        save_subdir="spike_train_analysis",
    ),
    AnalyzerSpec(
        name="adversarial_robustness",
        config_field="adversarial_robustness",
        class_path="dendritic_modeling.analysis.metrics.adversarial_robustness.AdversarialRobustnessAnalyzer",
        attribute_name="adversarial_robustness_analyzer",
        save_subdir="adversarial_robustness_analysis",
    ),
)


def register_analyzer(
    name: str,
    *,
    config_field: str,
    class_path: str,
    attribute_name: str,
    save_subdir: str,
    constructor_style: AnalyzerConstructor = "params",
    supports_training: bool = True,
    requires_training: bool = False,
    description: str = "",
    phase: AnalyzerPhase | None = None,
    pass_training: bool = False,
    require_analyzer: bool = False,
    allow_override: bool = False,
) -> None:
    """Register an analyzer.

    Without ``phase`` the analyzer is metadata only: ``AnalysisManager`` builds
    it when its section is enabled but never runs it. With ``phase`` it runs
    there, receiving the test dataset (``pre_weight``, ``post_reactivation``)
    or only the model (``model_only``).
    """
    key = str(name).lower()
    if not key:
        raise ValueError("analyzer name must be non-empty")
    if key in _ANALYZER_REGISTRY and not allow_override:
        raise ValueError(f"Analyzer '{key}' is already registered")
    if constructor_style not in ("params", "model_params", "none"):
        raise ValueError(f"Unsupported analyzer constructor style: {constructor_style}")
    if phase is not None and phase not in ANALYZER_PHASES:
        raise ValueError(
            f"Unsupported analyzer phase {phase!r}; use one of {ANALYZER_PHASES}"
        )
    _ANALYZER_REGISTRY[key] = AnalyzerSpec(
        name=key,
        config_field=config_field,
        class_path=class_path,
        attribute_name=attribute_name,
        save_subdir=save_subdir,
        constructor_style=constructor_style,
        supports_training=supports_training,
        requires_training=requires_training,
        description=description,
        phase=phase,
        pass_training=pass_training,
        require_analyzer=require_analyzer,
    )


def unregister_analyzer(name: str) -> None:
    """Remove analyzer metadata if present."""
    _ANALYZER_REGISTRY.pop(str(name).lower(), None)


def get_analyzer_spec(name: str) -> AnalyzerSpec | None:
    """Return metadata for one analyzer."""
    key = str(name).lower()
    if key not in _ANALYZER_REGISTRY:
        load_plugins()
    return _ANALYZER_REGISTRY.get(key)


def get_registered_analyzer_names() -> list[str]:
    """Return registered analyzer names, including those of plugins."""
    load_plugins()
    return sorted(_ANALYZER_REGISTRY)


def get_registered_analyzers() -> dict[str, AnalyzerSpec]:
    """Return a copy of all analyzer specs, including those of plugins."""
    load_plugins()
    return dict(_ANALYZER_REGISTRY)


def get_builtin_analyzer_names() -> tuple[str, ...]:
    """Return built-in analyzer names in their declared initialization order."""
    return tuple(spec.name for spec in _BUILTIN_ANALYZERS)


def get_extension_analyzer_names() -> tuple[str, ...]:
    """Return the names of registered analyzers that are not built in, sorted."""
    load_plugins()
    builtin = set(get_builtin_analyzer_names())
    return tuple(sorted(name for name in _ANALYZER_REGISTRY if name not in builtin))


def load_analyzer_class(name: str) -> type:
    """Resolve the registered analyzer class for one analyzer name."""
    spec = get_analyzer_spec(name)
    if spec is None:
        raise KeyError(f"Analyzer {name!r} is not registered")
    return spec.load_class()


def _register_builtin_analyzers() -> None:
    for spec in _BUILTIN_ANALYZERS:
        register_analyzer(
            spec.name,
            config_field=spec.config_field,
            class_path=spec.class_path,
            attribute_name=spec.attribute_name,
            save_subdir=spec.save_subdir,
            constructor_style=spec.constructor_style,
            supports_training=spec.supports_training,
            requires_training=spec.requires_training,
            description=spec.description,
        )


_register_builtin_analyzers()


__all__ = [
    "ANALYZER_PHASES",
    "AnalyzerConstructor",
    "AnalyzerPhase",
    "AnalyzerSpec",
    "get_analyzer_spec",
    "get_builtin_analyzer_names",
    "get_extension_analyzer_names",
    "get_registered_analyzer_names",
    "get_registered_analyzers",
    "load_analyzer_class",
    "register_analyzer",
    "unregister_analyzer",
]
