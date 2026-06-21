"""Builders for decision and temporal-generalization sequence datasets."""

from __future__ import annotations

from typing import Any

from dendritic_modeling.datasets.sequence.builders.common import (
    SequenceDatasetBuild,
    build_deferred_train_test_split,
    build_explicit_or_deferred_split,
    positive_input_encoding,
)
from dendritic_modeling.datasets.sequence.decision_tasks import (
    ContextDependentDecisionDataset,
    HierarchicalTemporalDataset,
    SwitchingContextDecisionDataset,
    TimescaleGeneralizationDataset,
)


def build_hierarchical_temporal(
    _dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 20000)
    common = {
        "n_contexts": kwargs.get("n_contexts", 2),
        "n_block_types": kwargs.get("n_block_types", 4),
        "n_blocks": kwargs.get("n_blocks", 8),
        "context_duration": kwargs.get("context_duration", 20),
        "block_duration": kwargs.get("block_duration", 25),
        "signal_dim": kwargs.get("signal_dim", 16),
        "noise_std": kwargs.get("noise_std", 0.5),
    }
    return build_deferred_train_test_split(
        dataset_cls=HierarchicalTemporalDataset,
        n_train=n_train,
        common_defaults=common,
        min_test_size=2000,
    )


def build_context_dependent_decision(
    dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 20000)
    common = {
        "signal_dim": kwargs.get("signal_dim", 8),
        "context_duration": kwargs.get("context_duration", 10),
        "integration_duration": kwargs.get("integration_duration", 50),
        "coherence_levels": tuple(
            kwargs.get("coherence_levels", (0.05, 0.1, 0.2, 0.4))
        ),
        "noise_std": kwargs.get("noise_std", 1.0),
        "variable_delay": kwargs.get("variable_delay", True),
        "min_integration": kwargs.get("min_integration", 20),
        "max_integration": kwargs.get("max_integration", 80),
        "delay_duration": kwargs.get("delay_duration", 0),
        "min_delay": kwargs.get("min_delay", None),
        "max_delay": kwargs.get("max_delay", None),
        "conflict_fraction": kwargs.get("conflict_fraction", 0.0),
        "distractor_burst_prob": kwargs.get("distractor_burst_prob", 0.0),
        "distractor_burst_scale": kwargs.get("distractor_burst_scale", 2.0),
        "distractor_burst_duration": kwargs.get("distractor_burst_duration", 5),
        "cue_dropout_prob": kwargs.get("cue_dropout_prob", 0.0),
        "cue_keep_duration": kwargs.get("cue_keep_duration", 1),
        "late_context_switch_prob": kwargs.get("late_context_switch_prob", 0.0),
        "context_signal_scale": kwargs.get("context_signal_scale", 1.0),
        "return_seq_lengths": kwargs.get("return_seq_lengths", False),
        "positive_input_encoding": positive_input_encoding(dataset_name, kwargs),
    }
    return build_deferred_train_test_split(
        dataset_cls=ContextDependentDecisionDataset,
        n_train=n_train,
        common_defaults=common,
        min_test_size=2000,
    )


def build_switching_context_decision(
    dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 16000)
    common = {
        "signal_dim": kwargs.get("signal_dim", 8),
        "seq_len": kwargs.get("seq_len", 360),
        "trial_duration": kwargs.get("trial_duration", 12),
        "min_context_duration": kwargs.get("min_context_duration", 90),
        "max_context_duration": kwargs.get("max_context_duration", 150),
        "coherence_levels": tuple(
            kwargs.get("coherence_levels", (0.03, 0.06, 0.12, 0.24))
        ),
        "noise_std": kwargs.get("noise_std", 1.1),
        "conflict_fraction": kwargs.get("conflict_fraction", 0.75),
        "distractor_burst_prob": kwargs.get("distractor_burst_prob", 0.35),
        "distractor_burst_scale": kwargs.get("distractor_burst_scale", 2.0),
        "distractor_burst_duration": kwargs.get("distractor_burst_duration", 3),
        "context_cue_duration": kwargs.get("context_cue_duration", 10),
        "switch_cue_duration": kwargs.get("switch_cue_duration", 4),
        "context_signal_scale": kwargs.get("context_signal_scale", 1.0),
        "switch_signal_scale": kwargs.get("switch_signal_scale", 0.35),
        "min_blocks": kwargs.get("min_blocks", 2),
        "positive_input_encoding": positive_input_encoding(dataset_name, kwargs),
    }
    return build_deferred_train_test_split(
        dataset_cls=SwitchingContextDecisionDataset,
        n_train=n_train,
        common_defaults=common,
        min_test_size=2000,
    )


def build_timescale_generalization(
    dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 20000)
    common_defaults = {
        "stimulus_dim": kwargs.get("stimulus_dim", 16),
        "stimulus_duration": kwargs.get("stimulus_duration", 10),
        "min_delay": kwargs.get("min_delay", 10),
        "max_delay": kwargs.get("max_delay", 100),
        "probe_duration": kwargs.get("probe_duration", 10),
        "positive_input_encoding": positive_input_encoding(dataset_name, kwargs),
        "noise_std": kwargs.get("noise_std", 0.3),
    }
    return build_explicit_or_deferred_split(
        dataset_cls=TimescaleGeneralizationDataset,
        kwargs=kwargs,
        common_defaults=common_defaults,
        n_train=n_train,
        min_test_size=2000,
    )


DECISION_SEQUENCE_BUILDERS = {
    "context_dependent_decision": build_context_dependent_decision,
    "context_dependent_decision_hard": build_context_dependent_decision,
    "context_dependent_decision_hard_positive": build_context_dependent_decision,
    "context_dependent_decision_positive": build_context_dependent_decision,
    "hierarchical_temporal": build_hierarchical_temporal,
    "switching_context_decision": build_switching_context_decision,
    "switching_context_decision_positive": build_switching_context_decision,
    "timescale_generalization": build_timescale_generalization,
    "timescale_generalization_positive": build_timescale_generalization,
}


__all__ = [
    "DECISION_SEQUENCE_BUILDERS",
    "build_context_dependent_decision",
    "build_hierarchical_temporal",
    "build_switching_context_decision",
    "build_timescale_generalization",
]
