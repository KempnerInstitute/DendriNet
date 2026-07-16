"""Reusable analysis tools for weights, routing, dynamics, and E/I motifs."""

from .causal_intervention import SameCheckpointCausalInterventionAnalyzer
from .class_accessibility import (
    LinearClassAccessibilityAnalyzer,
    cross_validated_linear_decode,
    validation_to_test_linear_decode,
)
from .class_information import (
    MatchedClassInformationAnalyzer,
    matched_support_class_information,
)
from .correlation_analysis import CorrelationAnalyzer
from .dendritic_timetraces import DendriticTimetracesAnalyzer
from .ei_connectivity_motifs import (
    LOCAL_RECURRENT_ROUTE_KEYS,
    ROUTE_SPECS,
    TARGETING_CLASS_ORDER,
    analyze_ei_connectivity_motifs_from_state_dict,
    extract_compartment_weights_from_state_dict,
    pair_stats_by_class,
    summarize_compartment_targeting_from_compartment_weights,
)
from .inhibitory_routing_comparison import (
    InhibitoryRoutingComparison,
    fixed_budget_inhibitory_routing_comparison,
)
from .inhibitory_specialization import InhibitorySpecializationAnalyzer
from .jacobian_spectrum import JacobianSpectrumAnalyzer
from .local_rule_components import LocalRuleComponentAnalyzer
from .poisson_exposure_transfer import (
    POISSON_EXPOSURE_DRAW_SEED_RULE,
    POSTHOC_INTEGRATION_NOTE,
    PoissonExposureTransferAnalyzer,
    evaluate_poisson_exposure_accuracy,
    poisson_exposure_draw_seed,
    poisson_exposure_duration_key,
    validate_poisson_exposure_durations,
)
from .reactivation_dynamics import ReactivationDynamicsAnalyzer
from .spike_trains import SpikeTrainAnalyzer
from .state_transition_jacobian import StateTransitionJacobianAnalyzer
from .synaptic_activation import SynapticActivationAnalyzer
from .synaptic_pruning import (
    SynapticPruningAnalyzer,
    integrate_pruning_with_turnover_analysis,
)
from .synaptic_turnover import SynapticTurnoverAnalyzer
from .weight import WeightAnalyzer

__all__ = [
    "LOCAL_RECURRENT_ROUTE_KEYS",
    "POISSON_EXPOSURE_DRAW_SEED_RULE",
    "POSTHOC_INTEGRATION_NOTE",
    "ROUTE_SPECS",
    "TARGETING_CLASS_ORDER",
    "CorrelationAnalyzer",
    "DendriticTimetracesAnalyzer",
    "InhibitoryRoutingComparison",
    "InhibitorySpecializationAnalyzer",
    "JacobianSpectrumAnalyzer",
    "LinearClassAccessibilityAnalyzer",
    "LocalRuleComponentAnalyzer",
    "MatchedClassInformationAnalyzer",
    "PoissonExposureTransferAnalyzer",
    "ReactivationDynamicsAnalyzer",
    "SameCheckpointCausalInterventionAnalyzer",
    "SpikeTrainAnalyzer",
    "StateTransitionJacobianAnalyzer",
    "SynapticActivationAnalyzer",
    "SynapticPruningAnalyzer",
    "SynapticTurnoverAnalyzer",
    "WeightAnalyzer",  # Now includes all weight analysis functionality
    "analyze_ei_connectivity_motifs_from_state_dict",
    "cross_validated_linear_decode",
    "evaluate_poisson_exposure_accuracy",
    "extract_compartment_weights_from_state_dict",
    "fixed_budget_inhibitory_routing_comparison",
    "integrate_pruning_with_turnover_analysis",
    "matched_support_class_information",
    "pair_stats_by_class",
    "poisson_exposure_draw_seed",
    "poisson_exposure_duration_key",
    "summarize_compartment_targeting_from_compartment_weights",
    "validate_poisson_exposure_durations",
    "validation_to_test_linear_decode",
]
