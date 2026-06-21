"""Compatibility exports for sequence datasets.

The implementations are grouped under :mod:`dendritic_modeling.datasets.sequence`;
this module preserves the historical import path.
"""

from dendritic_modeling.datasets.sequence import (
    AddingProblemDataset,
    ContextDependentDecisionDataset,
    CopyTaskDataset,
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
    get_sequence_datasets,
)

__all__ = [
    "AddingProblemDataset",
    "ContextDependentDecisionDataset",
    "CopyTaskDataset",
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
    "get_sequence_datasets",
]
