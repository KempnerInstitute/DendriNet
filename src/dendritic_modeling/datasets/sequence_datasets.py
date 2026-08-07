"""Compatibility exports for sequence datasets.

The implementations are grouped under :mod:`dendritic_modeling.datasets.sequence`;
this module preserves the historical import path.
"""

from dendritic_modeling.datasets.sequence import (
    AddingProblemDataset,
    ContextDependentDecisionDataset,
    CopyTaskDataset,
    FactualBayesianReadySetGoDataset,
    GainModulatedContextualComparisonDataset,
    HierarchicalTemporalDataset,
    LorenzSequenceDataset,
    MultiFrequencyDataset,
    MultiSineForecastDataset,
    OculomotorDelayedResponseDataset,
    RomoDelayComparisonDataset,
    SequentialMNISTDataset,
    StringerV1Dataset,
    SwitchingContextDecisionDataset,
    SwitchingLDSDataset,
    TimescaleGeneralizationDataset,
    VariableDelayDMSDataset,
    _resolve_stringer_session_file,
    get_registered_sequence_dataset_names,
    get_sequence_dataset_builder,
    get_sequence_datasets,
    has_sequence_dataset_builder,
    register_sequence_dataset_builder,
    unregister_sequence_dataset_builder,
)

__all__ = [
    "AddingProblemDataset",
    "ContextDependentDecisionDataset",
    "CopyTaskDataset",
    "FactualBayesianReadySetGoDataset",
    "GainModulatedContextualComparisonDataset",
    "HierarchicalTemporalDataset",
    "LorenzSequenceDataset",
    "MultiFrequencyDataset",
    "MultiSineForecastDataset",
    "OculomotorDelayedResponseDataset",
    "RomoDelayComparisonDataset",
    "SequentialMNISTDataset",
    "StringerV1Dataset",
    "SwitchingContextDecisionDataset",
    "SwitchingLDSDataset",
    "TimescaleGeneralizationDataset",
    "VariableDelayDMSDataset",
    "_resolve_stringer_session_file",
    "get_registered_sequence_dataset_names",
    "get_sequence_dataset_builder",
    "get_sequence_datasets",
    "has_sequence_dataset_builder",
    "register_sequence_dataset_builder",
    "unregister_sequence_dataset_builder",
]
