"""Registry for built-in sequence dataset builders."""

from __future__ import annotations

from collections.abc import Mapping

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


def merge_sequence_dataset_builders(
    *builder_maps: Mapping[str, SequenceDatasetBuilder],
) -> dict[str, SequenceDatasetBuilder]:
    """Merge builder maps while rejecting accidental duplicate dataset names."""
    merged: dict[str, SequenceDatasetBuilder] = {}
    for builder_map in builder_maps:
        duplicates = sorted(set(merged).intersection(builder_map))
        if duplicates:
            duplicate_names = ", ".join(duplicates)
            raise ValueError(
                f"duplicate sequence dataset builder names: {duplicate_names}"
            )
        merged.update(builder_map)
    return merged


SEQUENCE_DATASET_BUILDERS = merge_sequence_dataset_builders(
    DECISION_SEQUENCE_BUILDERS,
    MEMORY_SEQUENCE_BUILDERS,
    NEURO_SEQUENCE_BUILDERS,
    SIGNAL_SEQUENCE_BUILDERS,
)


__all__ = ["SEQUENCE_DATASET_BUILDERS", "merge_sequence_dataset_builders"]
