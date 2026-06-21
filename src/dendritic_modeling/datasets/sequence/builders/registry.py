"""Registry for built-in sequence dataset builders."""

from __future__ import annotations

from dendritic_modeling.datasets.sequence.builders.common import SequenceDatasetBuilder
from dendritic_modeling.datasets.sequence.builders.decision import (
    DECISION_SEQUENCE_BUILDERS,
)
from dendritic_modeling.datasets.sequence.builders.memory import (
    MEMORY_SEQUENCE_BUILDERS,
)
from dendritic_modeling.datasets.sequence.builders.neuro import NEURO_SEQUENCE_BUILDERS
from dendritic_modeling.datasets.sequence.builders.signal import (
    SIGNAL_SEQUENCE_BUILDERS,
)

SEQUENCE_DATASET_BUILDERS: dict[str, SequenceDatasetBuilder] = {
    **DECISION_SEQUENCE_BUILDERS,
    **MEMORY_SEQUENCE_BUILDERS,
    **NEURO_SEQUENCE_BUILDERS,
    **SIGNAL_SEQUENCE_BUILDERS,
}


__all__ = ["SEQUENCE_DATASET_BUILDERS"]
