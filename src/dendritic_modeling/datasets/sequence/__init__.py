"""Sequence dataset package."""

from dendritic_modeling.datasets.sequence.decision_tasks import (
    ContextDependentDecisionDataset,
    HierarchicalTemporalDataset,
    SwitchingContextDecisionDataset,
    TimescaleGeneralizationDataset,
)
from dendritic_modeling.datasets.sequence.factory import get_sequence_datasets
from dendritic_modeling.datasets.sequence.memory_tasks import (
    AddingProblemDataset,
    CopyTaskDataset,
    SequentialMNISTDataset,
    VariableDelayDMSDataset,
)
from dendritic_modeling.datasets.sequence.neuro_tasks import (
    GainModulatedContextualComparisonDataset,
    OculomotorDelayedResponseDataset,
    RomoDelayComparisonDataset,
    StringerV1Dataset,
)
from dendritic_modeling.datasets.sequence.signal_tasks import (
    LorenzSequenceDataset,
    MultiFrequencyDataset,
    MultiSineForecastDataset,
    SwitchingLDSDataset,
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
