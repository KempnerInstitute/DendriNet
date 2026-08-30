"""Reusable analysis tools for weights, routing, dynamics, and E/I motifs."""

from .compositional_ood import CompositionalOODAnalyzer
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
from .inhibitory_specialization import InhibitorySpecializationAnalyzer
from .jacobian_spectrum import JacobianSpectrumAnalyzer
from .local_rule_components import LocalRuleComponentAnalyzer
from .reactivation_dynamics import ReactivationDynamicsAnalyzer
from .spike_trains import SpikeTrainAnalyzer
from .synaptic_activation import SynapticActivationAnalyzer
from .synaptic_pruning import (
    SynapticPruningAnalyzer,
    integrate_pruning_with_turnover_analysis,
)
from .synaptic_turnover import SynapticTurnoverAnalyzer
from .weight import WeightAnalyzer

__all__ = [
    "LOCAL_RECURRENT_ROUTE_KEYS",
    "ROUTE_SPECS",
    "TARGETING_CLASS_ORDER",
    "CompositionalOODAnalyzer",
    "CorrelationAnalyzer",
    "DendriticTimetracesAnalyzer",
    "InhibitorySpecializationAnalyzer",
    "JacobianSpectrumAnalyzer",
    "LocalRuleComponentAnalyzer",
    "ReactivationDynamicsAnalyzer",
    "SpikeTrainAnalyzer",
    "SynapticActivationAnalyzer",
    "SynapticPruningAnalyzer",
    "SynapticTurnoverAnalyzer",
    "WeightAnalyzer",  # Now includes all weight analysis functionality
    "analyze_ei_connectivity_motifs_from_state_dict",
    "extract_compartment_weights_from_state_dict",
    "integrate_pruning_with_turnover_analysis",
    "pair_stats_by_class",
    "summarize_compartment_targeting_from_compartment_weights",
]
