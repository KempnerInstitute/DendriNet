"""Builders for Bayesian interval-reproduction datasets."""

from __future__ import annotations

from numbers import Integral
from typing import Any

from dendritic_modeling.datasets.sequence.bayesian_timing import (
    BayesianReadySetGoDataset,
    default_observed_interval_bounds,
)
from dendritic_modeling.datasets.sequence.builders.common import (
    SequenceDatasetBuild,
    require_distinct_explicit_split_seeds,
    validate_noop_standard_processing,
    validated_split_overrides,
)

_SPLIT_NAMES = ("train", "valid", "test")


def _validated_even_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("Bayesian Ready-Set-Go split sizes must be positive integers")
    if int(value) % 2:
        raise ValueError(
            "Bayesian Ready-Set-Go split sizes must be even for paired trials"
        )
    return int(value)


def _split_default_size(n_train: int) -> int:
    size = max(n_train // 5, 2000)
    return size + size % 2


def _base_parameters(dataset_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "prior_supports": tuple(
            tuple(support)
            for support in kwargs.get(
                "prior_supports",
                ((24, 28, 32, 36, 40), (40, 45, 50, 55, 60)),
            )
        ),
        "prior_probabilities": kwargs.get("prior_probabilities"),
        "weber_fraction": kwargs.get("weber_fraction", 0.12),
        "measurement_noise_floor": kwargs.get("measurement_noise_floor", 0.0),
        "likelihood_tail_sigma": kwargs.get("likelihood_tail_sigma", 4.0),
        "production_scale": kwargs.get("production_scale", 1.0),
        "pulse_duration": kwargs.get("pulse_duration", 2),
        "context_start_step": kwargs.get("context_start_step", 0),
        "context_duration": kwargs.get("context_duration", 10),
        "context_gain": kwargs.get("context_gain", 1.0),
        "target_sigma": kwargs.get("target_sigma", 2.0),
        "target_tail_sigma": kwargs.get("target_tail_sigma", 4.0),
        "post_go_duration": kwargs.get("post_go_duration", 20),
        "cue_gain_min": kwargs.get("cue_gain_min", 0.7),
        "cue_gain_max": kwargs.get("cue_gain_max", 1.3),
        "input_noise_std": kwargs.get("input_noise_std", 0.05),
        "mean_distractors_per_trial": kwargs.get("mean_distractors_per_trial", 1.0),
        "distractor_amplitude": kwargs.get("distractor_amplitude", 1.0),
        "distractor_pulse_duration": kwargs.get("distractor_pulse_duration", 2),
        "positive_input_encoding": dataset_name.endswith("_positive")
        or kwargs.get("positive_input_encoding", False),
    }


def build_bayesian_ready_set_go(
    dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    """Build seeded Bayesian RSG splits with shared timing geometry."""

    validate_noop_standard_processing(
        kwargs,
        dataset_label="Bayesian Ready-Set-Go",
    )

    n_train = _validated_even_size(kwargs.get("n_samples", 10000))
    base_seed = kwargs.get("seed", 0)
    if isinstance(base_seed, bool) or not isinstance(base_seed, Integral):
        raise TypeError("Bayesian Ready-Set-Go seed must be an integer")
    base = _base_parameters(dataset_name, kwargs)
    split_overrides = validated_split_overrides(
        kwargs,
        dataset_label="Bayesian Ready-Set-Go",
    )
    forbidden = {
        "context_duration",
        "context_start_step",
        "likelihood_tail_sigma",
        "measurement_noise_floor",
        "observed_interval_max",
        "observed_interval_min",
        "prior_probabilities",
        "prior_supports",
        "production_scale",
        "sequence_duration",
        "set_max_step",
        "set_min_step",
        "target_sigma",
        "target_tail_sigma",
        "weber_fraction",
    }
    if any(forbidden.intersection(override) for override in split_overrides.values()):
        raise ValueError(
            "Bayesian observer semantics and absolute timing must be shared across "
            "Ready-Set-Go splits; categorical prior cues cannot specify changed "
            "prior or likelihood parameters"
        )

    default_sizes = {
        "train": n_train,
        "valid": _split_default_size(n_train),
        "test": _split_default_size(n_train),
    }
    parameters_by_split: dict[str, tuple[int, dict[str, Any]]] = {}
    seed_offsets = (0, 1, 2) if split_overrides else (0, 1, 1)
    for seed_offset, split_name in zip(seed_offsets, _SPLIT_NAMES, strict=True):
        override = dict(split_overrides.get(split_name, {}))
        n_split = _validated_even_size(
            override.pop("n_samples", default_sizes[split_name])
        )
        parameters = {**base, **override}
        parameters.setdefault("seed", int(base_seed) + seed_offset)
        parameters_by_split[split_name] = (n_split, parameters)
    if split_overrides:
        require_distinct_explicit_split_seeds(
            {
                name: parameters["seed"]
                for name, (_size, parameters) in parameters_by_split.items()
            },
            dataset_label="Bayesian Ready-Set-Go",
        )

    derived_bounds = tuple(
        default_observed_interval_bounds(
            parameters["prior_supports"],
            weber_fraction=parameters["weber_fraction"],
            measurement_noise_floor=parameters["measurement_noise_floor"],
            likelihood_tail_sigma=parameters["likelihood_tail_sigma"],
            minimum_interval=parameters["pulse_duration"],
        )
        for _, parameters in parameters_by_split.values()
    )
    observed_min = kwargs.get(
        "observed_interval_min", min(bounds[0] for bounds in derived_bounds)
    )
    observed_max = kwargs.get(
        "observed_interval_max", max(bounds[1] for bounds in derived_bounds)
    )
    configured_set_min = kwargs.get("set_min_step")
    set_min_step = (
        observed_max + base["context_start_step"] + base["context_duration"] + 10
        if configured_set_min is None
        else configured_set_min
    )
    configured_set_max = kwargs.get("set_max_step")
    default_translation_span = max(40, observed_max - observed_min)
    set_max_step = (
        set_min_step + default_translation_span
        if configured_set_max is None
        else configured_set_max
    )

    required_duration = max(
        set_max_step
        + max(
            round(parameters["production_scale"] * interval)
            for support in parameters["prior_supports"]
            for interval in support
        )
        + parameters["post_go_duration"]
        + 1
        for _, parameters in parameters_by_split.values()
    )
    configured_duration = kwargs.get("sequence_duration")
    sequence_duration = (
        required_duration if configured_duration is None else configured_duration
    )
    shared_geometry = {
        "observed_interval_min": observed_min,
        "observed_interval_max": observed_max,
        "set_min_step": set_min_step,
        "set_max_step": set_max_step,
        "sequence_duration": sequence_duration,
    }

    def construct(split_name: str) -> BayesianReadySetGoDataset:
        n_split, parameters = parameters_by_split[split_name]
        return BayesianReadySetGoDataset(
            n_samples=n_split,
            **parameters,
            **shared_geometry,
        )

    if split_overrides:
        return SequenceDatasetBuild(
            split_datasets=tuple(construct(name) for name in _SPLIT_NAMES)
        )
    return SequenceDatasetBuild(
        train_full=construct("train"),
        test_ds=construct("test"),
    )


BAYESIAN_TIMING_SEQUENCE_BUILDERS = {
    "bayesian_ready_set_go": build_bayesian_ready_set_go,
    "bayesian_ready_set_go_positive": build_bayesian_ready_set_go,
}


__all__ = [
    "BAYESIAN_TIMING_SEQUENCE_BUILDERS",
    "build_bayesian_ready_set_go",
]
