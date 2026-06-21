"""Factory for sequence dataset train/valid/test triplets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import random_split

from dendritic_modeling.datasets.registry import build_registered_dataset
from dendritic_modeling.datasets.sequence.decision_tasks import (
    ContextDependentDecisionDataset,
    HierarchicalTemporalDataset,
    SwitchingContextDecisionDataset,
    TimescaleGeneralizationDataset,
)
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


@dataclass(frozen=True)
class _SequenceDatasetBuild:
    train_full: Any | None = None
    test_ds: Any | None = None
    split_datasets: tuple[Any, Any, Any] | None = None


SequenceDatasetBuilder = Callable[
    [str, str | None, dict[str, Any]], _SequenceDatasetBuild
]


def _test_size(n_train: int, minimum: int) -> int:
    return max(n_train // 5, minimum)


def _deferred_split(train_full: Any, test_ds: Any) -> _SequenceDatasetBuild:
    return _SequenceDatasetBuild(train_full=train_full, test_ds=test_ds)


def _explicit_split(
    train_ds: Any,
    valid_ds: Any,
    test_ds: Any,
) -> _SequenceDatasetBuild:
    return _SequenceDatasetBuild(split_datasets=(train_ds, valid_ds, test_ds))


def _positive_input_encoding(dataset_name: str, kwargs: dict[str, Any]) -> bool:
    return dataset_name.endswith("_positive") or kwargs.get(
        "positive_input_encoding", False
    )


def _has_explicit_splits(kwargs: dict[str, Any]) -> bool:
    return any(
        isinstance(kwargs.get(split_name), dict)
        for split_name in ("train", "valid", "test")
    )


def _build_explicit_or_deferred_split(
    *,
    dataset_cls: Callable[..., Any],
    kwargs: dict[str, Any],
    common_defaults: dict[str, Any],
    n_train: int,
    min_test_size: int,
) -> _SequenceDatasetBuild:
    if _has_explicit_splits(kwargs):

        def _build_split(split_name: str, default_n_samples: int) -> Any:
            split_kwargs = dict(kwargs.get(split_name, {}))
            split_params = {
                **common_defaults,
                **{
                    key: value
                    for key, value in split_kwargs.items()
                    if key != "n_samples"
                },
            }
            split_n = split_kwargs.get("n_samples", default_n_samples)
            return dataset_cls(n_samples=split_n, **split_params)

        return _explicit_split(
            _build_split("train", n_train),
            _build_split("valid", _test_size(n_train, min_test_size)),
            _build_split("test", _test_size(n_train, min_test_size)),
        )

    return _deferred_split(
        dataset_cls(n_samples=n_train, **common_defaults),
        dataset_cls(n_samples=_test_size(n_train, min_test_size), **common_defaults),
    )


def _build_sequential_mnist(
    _dataset_name: str,
    data_path: str | None,
    kwargs: dict[str, Any],
) -> _SequenceDatasetBuild:
    mode = kwargs.get("mode", "pixel")
    normalize = kwargs.get("normalize", True)
    return _deferred_split(
        SequentialMNISTDataset(
            mode=mode,
            train=True,
            data_path=data_path,
            normalize=normalize,
        ),
        SequentialMNISTDataset(
            mode=mode,
            train=False,
            data_path=data_path,
            normalize=normalize,
        ),
    )


def _build_copy_task(
    _dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> _SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 10000)
    common = {
        "seq_len": kwargs.get("seq_len", 10),
        "delay": kwargs.get("delay", 50),
        "n_symbols": kwargs.get("n_symbols", 8),
    }
    return _deferred_split(
        CopyTaskDataset(n_samples=n_train, **common),
        CopyTaskDataset(n_samples=_test_size(n_train, 1000), **common),
    )


def _build_adding_problem(
    _dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> _SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 10000)
    common = {"seq_len": kwargs.get("seq_len", 200)}
    return _deferred_split(
        AddingProblemDataset(n_samples=n_train, **common),
        AddingProblemDataset(n_samples=_test_size(n_train, 1000), **common),
    )


def _build_variable_delay_dms(
    _dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> _SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 10000)
    common = {
        "stimulus_dim": kwargs.get("stimulus_dim", 16),
        "stimulus_duration": kwargs.get("stimulus_duration", 10),
        "min_delay": kwargs.get("min_delay", 10),
        "max_delay": kwargs.get("max_delay", 100),
        "probe_duration": kwargs.get("probe_duration", 10),
    }
    return _deferred_split(
        VariableDelayDMSDataset(n_samples=n_train, **common),
        VariableDelayDMSDataset(n_samples=_test_size(n_train, 1000), **common),
    )


def _build_multi_frequency(
    _dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> _SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 10000)
    common = {
        "seq_len": kwargs.get("seq_len", 200),
        "n_frequencies": kwargs.get("n_frequencies", 5),
        "base_freq": kwargs.get("base_freq", 0.01),
        "noise_std": kwargs.get("noise_std", 0.1),
    }
    return _deferred_split(
        MultiFrequencyDataset(n_samples=n_train, **common),
        MultiFrequencyDataset(n_samples=_test_size(n_train, 1000), **common),
    )


def _build_hierarchical_temporal(
    _dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> _SequenceDatasetBuild:
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
    return _deferred_split(
        HierarchicalTemporalDataset(n_samples=n_train, **common),
        HierarchicalTemporalDataset(n_samples=_test_size(n_train, 2000), **common),
    )


def _build_context_dependent_decision(
    dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> _SequenceDatasetBuild:
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
        "positive_input_encoding": _positive_input_encoding(dataset_name, kwargs),
    }
    return _deferred_split(
        ContextDependentDecisionDataset(n_samples=n_train, **common),
        ContextDependentDecisionDataset(n_samples=_test_size(n_train, 2000), **common),
    )


def _build_switching_context_decision(
    dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> _SequenceDatasetBuild:
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
        "positive_input_encoding": _positive_input_encoding(dataset_name, kwargs),
    }
    return _deferred_split(
        SwitchingContextDecisionDataset(n_samples=n_train, **common),
        SwitchingContextDecisionDataset(n_samples=_test_size(n_train, 2000), **common),
    )


def _build_timescale_generalization(
    dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> _SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 20000)
    common_defaults = {
        "stimulus_dim": kwargs.get("stimulus_dim", 16),
        "stimulus_duration": kwargs.get("stimulus_duration", 10),
        "min_delay": kwargs.get("min_delay", 10),
        "max_delay": kwargs.get("max_delay", 100),
        "probe_duration": kwargs.get("probe_duration", 10),
        "positive_input_encoding": _positive_input_encoding(dataset_name, kwargs),
        "noise_std": kwargs.get("noise_std", 0.3),
    }
    return _build_explicit_or_deferred_split(
        dataset_cls=TimescaleGeneralizationDataset,
        kwargs=kwargs,
        common_defaults=common_defaults,
        n_train=n_train,
        min_test_size=2000,
    )


def _build_odr_generalization(
    dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> _SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 20000)
    common_defaults = {
        "n_targets": kwargs.get("n_targets", 8),
        "input_tuning_dim": kwargs.get("input_tuning_dim", 16),
        "fixation_duration": kwargs.get("fixation_duration", 6),
        "cue_duration": kwargs.get("cue_duration", 8),
        "min_delay": kwargs.get("min_delay", 10),
        "max_delay": kwargs.get("max_delay", 100),
        "go_duration": kwargs.get("go_duration", 8),
        "tuning_kappa": kwargs.get("tuning_kappa", 2.0),
        "cue_noise_std": kwargs.get("cue_noise_std", 0.05),
        "positive_input_encoding": _positive_input_encoding(dataset_name, kwargs),
    }
    return _build_explicit_or_deferred_split(
        dataset_cls=OculomotorDelayedResponseDataset,
        kwargs=kwargs,
        common_defaults=common_defaults,
        n_train=n_train,
        min_test_size=2000,
    )


def _build_romo_delay_comparison(
    dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> _SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 20000)
    common_defaults = {
        "frequency_values": tuple(
            kwargs.get("frequency_values", (2.0, 3.0, 4.0, 5.0, 6.0, 7.0))
        ),
        "stimulus_duration": kwargs.get("stimulus_duration", 20),
        "min_delay": kwargs.get("min_delay", 10),
        "max_delay": kwargs.get("max_delay", 100),
        "response_duration": kwargs.get("response_duration", 8),
        "noise_std": kwargs.get("noise_std", 0.08),
        "positive_input_encoding": _positive_input_encoding(dataset_name, kwargs),
        "tuning_dim": kwargs.get("tuning_dim", 8),
        "tuning_sigma": kwargs.get("tuning_sigma", None),
    }
    return _build_explicit_or_deferred_split(
        dataset_cls=RomoDelayComparisonDataset,
        kwargs=kwargs,
        common_defaults=common_defaults,
        n_train=n_train,
        min_test_size=2000,
    )


def _build_gain_modulated_contextual_comparison(
    dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> _SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 24000)
    common_defaults = {
        "n_pathways": kwargs.get("n_pathways", 4),
        "pathway_dim": kwargs.get("pathway_dim", 4),
        "cue_duration": kwargs.get("cue_duration", 8),
        "stimulus_duration": kwargs.get("stimulus_duration", 16),
        "min_delay": kwargs.get("min_delay", 20),
        "max_delay": kwargs.get("max_delay", 80),
        "response_duration": kwargs.get("response_duration", 8),
        "amplitude_values": tuple(
            kwargs.get(
                "amplitude_values",
                (0.45, 0.6, 0.75, 0.9, 1.05, 1.2),
            )
        ),
        "comparison_margin": kwargs.get("comparison_margin", 0.15),
        "global_gain_min": kwargs.get("global_gain_min", 0.85),
        "global_gain_max": kwargs.get("global_gain_max", 1.15),
        "pathway_gain_min": kwargs.get("pathway_gain_min", 0.8),
        "pathway_gain_max": kwargs.get("pathway_gain_max", 1.2),
        "gain_drift_std": kwargs.get("gain_drift_std", 0.03),
        "noise_std": kwargs.get("noise_std", 0.08),
        "cue_noise_std": kwargs.get("cue_noise_std", 0.02),
        "distractor_burst_prob": kwargs.get("distractor_burst_prob", 0.35),
        "distractor_burst_scale": kwargs.get("distractor_burst_scale", 1.5),
        "distractor_burst_duration": kwargs.get("distractor_burst_duration", 4),
        "positive_input_encoding": _positive_input_encoding(dataset_name, kwargs),
    }
    return _build_explicit_or_deferred_split(
        dataset_cls=GainModulatedContextualComparisonDataset,
        kwargs=kwargs,
        common_defaults=common_defaults,
        n_train=n_train,
        min_test_size=2000,
    )


def _build_multi_sine_forecast(
    _dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> _SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 12000)
    common = {
        "seq_len": kwargs.get("seq_len", 160),
        "rollout_horizon": kwargs.get("rollout_horizon", 160),
        "prediction_mode": kwargs.get("prediction_mode", "all"),
        "n_components": kwargs.get("n_components", 3),
        "frequency_bank": kwargs.get("frequency_bank", [1.0, 2.5, 5.5, 9.0, 14.0]),
        "amplitude_min": kwargs.get("amplitude_min", 0.3),
        "amplitude_max": kwargs.get("amplitude_max", 1.0),
        "envelope_scale": kwargs.get("envelope_scale", 0.15),
        "envelope_freq_min": kwargs.get("envelope_freq_min", 0.15),
        "envelope_freq_max": kwargs.get("envelope_freq_max", 0.6),
        "observation_noise_std": kwargs.get("observation_noise_std", 0.02),
        "seed": kwargs.get("seed", 42),
        "normalize_signal": kwargs.get("normalize_signal", True),
    }
    return _deferred_split(
        MultiSineForecastDataset(n_samples=n_train, **common),
        MultiSineForecastDataset(
            n_samples=_test_size(n_train, 1000),
            **{
                **common,
                "seed": kwargs.get("test_seed", common["seed"] + 1),
            },
        ),
    )


def _build_lorenz_sequence(
    _dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> _SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 10000)
    common = {
        "seq_len": kwargs.get("seq_len", 50),
        "dt": kwargs.get("dt", 0.01),
        "sigma": kwargs.get("sigma", 10.0),
        "rho": kwargs.get("rho", 28.0),
        "beta": kwargs.get("beta", 8.0 / 3.0),
        "rollout_horizon": kwargs.get("rollout_horizon", 50),
        "prediction_mode": kwargs.get("prediction_mode", "last"),
        "observation_mode": kwargs.get("observation_mode", "full"),
        "observation_dim": kwargs.get("observation_dim", 3),
        "observation_seed": kwargs.get("observation_seed", kwargs.get("seed", 42)),
        "observation_noise_std": kwargs.get("observation_noise_std", 0.0),
        "burn_in_steps": kwargs.get("burn_in_steps", 200),
        "seed": kwargs.get("seed", 42),
        "integration_method": kwargs.get("integration_method", "euler"),
        "normalize_states": kwargs.get("normalize_states", True),
        "normalize_observations": kwargs.get("normalize_observations", True),
        "normalization_steps": kwargs.get("normalization_steps", 10000),
        "initial_state_scale": kwargs.get("initial_state_scale", 0.1),
    }
    return _deferred_split(
        LorenzSequenceDataset(n_samples=n_train, **common),
        LorenzSequenceDataset(
            n_samples=_test_size(n_train, 1000),
            **{
                **common,
                "seed": kwargs.get("test_seed", common["seed"] + 1),
            },
        ),
    )


def _build_switching_lds(
    _dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> _SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 12000)
    common = {
        "seq_len": kwargs.get("seq_len", 96),
        "latent_dim": kwargs.get("latent_dim", 4),
        "observation_dim": kwargs.get("observation_dim", 2),
        "n_modes": kwargs.get("n_modes", 3),
        "rollout_horizon": kwargs.get("rollout_horizon", 80),
        "prediction_mode": kwargs.get("prediction_mode", "all"),
        "observation_noise_std": kwargs.get("observation_noise_std", 0.03),
        "transition_noise_std": kwargs.get("transition_noise_std", 0.02),
        "min_mode_duration": kwargs.get("min_mode_duration", 36),
        "max_mode_duration": kwargs.get("max_mode_duration", 96),
        "burn_in_steps": kwargs.get("burn_in_steps", 24),
        "seed": kwargs.get("seed", 42),
        "normalize_states": kwargs.get("normalize_states", True),
        "normalize_observations": kwargs.get("normalize_observations", True),
        "normalization_samples": kwargs.get("normalization_samples", 4096),
        "stats_seed": kwargs.get("stats_seed", 1234),
        "initial_state_scale": kwargs.get("initial_state_scale", 0.35),
    }
    return _deferred_split(
        SwitchingLDSDataset(n_samples=n_train, **common),
        SwitchingLDSDataset(
            n_samples=_test_size(n_train, 1000),
            **{
                **common,
                "seed": kwargs.get("test_seed", common["seed"] + 1),
            },
        ),
    )


def _build_stringer_v1(
    _dataset_name: str,
    data_path: str | None,
    kwargs: dict[str, Any],
) -> _SequenceDatasetBuild:
    data_dir = kwargs.get("data_dir", data_path)
    if data_dir is None:
        raise ValueError(
            "stringer_v1 requires data_dir (path to directory with "
            "gratings_drifting_GT3_2019_04_05_1.npy)"
        )
    n_neurons = kwargs.get("n_neurons", 500)
    n_classes = kwargs.get("n_classes", 8)
    normalize = kwargs.get("normalize", True)
    seed = kwargs.get("label_noise_seed", 42)
    session_idx = kwargs.get("session_idx", 20)
    running_quantile = kwargs.get("running_quantile", 0.5)
    running_state = kwargs.get("running_state", "all")
    train_running_state = kwargs.get("train_running_state", running_state)
    test_running_state = kwargs.get("test_running_state", running_state)
    gain_group_count = kwargs.get("gain_group_count", 1)
    gain_grouping = kwargs.get("gain_grouping", "preferred_class")
    poisson_sampling = kwargs.get("poisson_sampling", False)
    poisson_rate_scale = kwargs.get("poisson_rate_scale", 1.0)
    gain_seed = kwargs.get("gain_seed", seed)

    def _state_kwargs(prefix: str, state: str) -> dict[str, Any]:
        return {
            "running_state": state,
            "running_quantile": running_quantile,
            "gain_group_count": kwargs.get(
                f"{prefix}gain_group_count", gain_group_count
            ),
            "gain_grouping": kwargs.get(f"{prefix}gain_grouping", gain_grouping),
            "global_gain_min": kwargs.get(
                f"{prefix}global_gain_min", kwargs.get("global_gain_min", 1.0)
            ),
            "global_gain_max": kwargs.get(
                f"{prefix}global_gain_max", kwargs.get("global_gain_max", 1.0)
            ),
            "group_gain_min": kwargs.get(
                f"{prefix}group_gain_min", kwargs.get("group_gain_min", 1.0)
            ),
            "group_gain_max": kwargs.get(
                f"{prefix}group_gain_max", kwargs.get("group_gain_max", 1.0)
            ),
            "gain_seed": kwargs.get(f"{prefix}gain_seed", gain_seed),
            "poisson_sampling": kwargs.get(
                f"{prefix}poisson_sampling", poisson_sampling
            ),
            "poisson_rate_scale": kwargs.get(
                f"{prefix}poisson_rate_scale", poisson_rate_scale
            ),
        }

    common = {
        "data_dir": data_dir,
        "n_neurons": n_neurons,
        "n_classes": n_classes,
        "normalize": normalize,
        "seed": seed,
        "session_idx": session_idx,
    }

    use_state_specific_split = (
        train_running_state != "all" or test_running_state != "all"
    )
    if use_state_specific_split:

        def _state_set(state: str) -> set[str]:
            return {"low", "high"} if state == "all" else {state}

        if _state_set(train_running_state) & _state_set(test_running_state):
            raise ValueError(
                "Stringer V1 state-specific train/test requests must be "
                "disjoint to avoid trial overlap. Use train_running_state='low' "
                "with test_running_state='high' (or the reverse), or set both "
                "states to 'all' for the standard fixed split."
            )
        train_full = StringerV1Dataset(
            split="all",
            **common,
            **_state_kwargs("train_", train_running_state),
        )
        test_ds = StringerV1Dataset(
            split="all",
            **common,
            **_state_kwargs("test_", test_running_state),
            normalization_stats=getattr(train_full, "normalization_stats", None),
        )
        # Name-level disjointness (above) is necessary but not sufficient: the
        # low/high running-speed masks are both boundary-inclusive, so at the
        # default running_quantile=0.5 (or with ties in a zero-inflated running
        # trace) the realized "low" and "high" trial sets can still overlap.
        # Enforce disjointness on the actual selected trial indices.
        train_idx = set(np.asarray(train_full._split_indices).ravel().tolist())
        test_idx = set(np.asarray(test_ds._split_indices).ravel().tolist())
        overlap = train_idx & test_idx
        if overlap:
            raise ValueError(
                f"Stringer V1 state-specific split produced {len(overlap)} trial(s) "
                f"in both train (running_state={train_running_state!r}) and test "
                f"(running_state={test_running_state!r}). The running-speed thresholds "
                f"are inclusive, so trials tied at the quantile boundary fall into both "
                f"bands. Use a running_quantile < 0.5 so the low/high bands are strictly "
                f"separated (e.g. running_quantile=0.25 for bottom/top quartiles)."
            )
        return _deferred_split(train_full, test_ds)

    # Fixed 90/10 train/test split; train_valid_split handled by factory below.
    test_fraction = 0.1
    train_full = StringerV1Dataset(
        split="train",
        train_fraction=1.0 - test_fraction,
        **common,
        **_state_kwargs("train_", train_running_state),
    )
    test_ds = StringerV1Dataset(
        split="test",
        train_fraction=1.0 - test_fraction,
        **common,
        **_state_kwargs("test_", test_running_state),
    )
    return _deferred_split(train_full, test_ds)


_SEQUENCE_DATASET_BUILDERS: dict[str, SequenceDatasetBuilder] = {
    "adding_problem": _build_adding_problem,
    "context_dependent_decision": _build_context_dependent_decision,
    "context_dependent_decision_hard": _build_context_dependent_decision,
    "context_dependent_decision_hard_positive": _build_context_dependent_decision,
    "context_dependent_decision_positive": _build_context_dependent_decision,
    "copy_task": _build_copy_task,
    "gain_modulated_contextual_comparison": (
        _build_gain_modulated_contextual_comparison
    ),
    "gain_modulated_contextual_comparison_positive": (
        _build_gain_modulated_contextual_comparison
    ),
    "hierarchical_temporal": _build_hierarchical_temporal,
    "lorenz": _build_lorenz_sequence,
    "lorenz_sequence": _build_lorenz_sequence,
    "multi_frequency": _build_multi_frequency,
    "multi_sine_forecast": _build_multi_sine_forecast,
    "odr_generalization": _build_odr_generalization,
    "odr_generalization_positive": _build_odr_generalization,
    "romo_delay_comparison": _build_romo_delay_comparison,
    "romo_delay_comparison_positive": _build_romo_delay_comparison,
    "sequential_mnist": _build_sequential_mnist,
    "stringer_v1": _build_stringer_v1,
    "switching_context_decision": _build_switching_context_decision,
    "switching_context_decision_positive": _build_switching_context_decision,
    "switching_lds": _build_switching_lds,
    "timescale_generalization": _build_timescale_generalization,
    "timescale_generalization_positive": _build_timescale_generalization,
    "variable_delay_dms": _build_variable_delay_dms,
}


def _split_train_valid(
    train_full: Any,
    test_ds: Any,
    *,
    train_valid_split: float,
    split_seed: int,
) -> tuple[Any, Any, Any]:
    n_train = int(len(train_full) * train_valid_split)
    n_valid = len(train_full) - n_train
    if n_valid > 0:
        train_ds, valid_ds = random_split(
            train_full,
            [n_train, n_valid],
            generator=torch.Generator().manual_seed(split_seed),
        )
    else:
        train_ds = train_full
        valid_ds = test_ds
    return train_ds, valid_ds, test_ds


def get_sequence_datasets(
    dataset_name: str,
    data_path: str | None = None,
    train_valid_split: float = 0.9,
    **kwargs,
):
    """Factory function for sequence datasets.

    Args:
        dataset_name: Name of the dataset.
        data_path: Path for datasets that need file storage.
        train_valid_split: Fraction of training data to use for training.
        **kwargs: Dataset-specific parameters.

    Returns:
        Tuple of (train_ds, valid_ds, test_ds).
    """
    registered = build_registered_dataset(
        dataset_name,
        {
            "dataset": dataset_name,
            "data_path": data_path,
            "train_valid_split": train_valid_split,
            "parameters": dict(kwargs),
        },
    )
    if registered is not None:
        return registered

    builder = _SEQUENCE_DATASET_BUILDERS.get(dataset_name)
    if builder is None:
        raise ValueError(f"Unknown sequence dataset: {dataset_name}")

    build = builder(dataset_name, data_path, kwargs)
    if build.split_datasets is not None:
        return build.split_datasets
    if build.train_full is None or build.test_ds is None:
        raise RuntimeError(
            f"Sequence dataset builder returned no datasets: {dataset_name}"
        )

    split_seed = int(kwargs.get("split_seed", kwargs.get("seed", 42)) or 0)
    return _split_train_valid(
        build.train_full,
        build.test_ds,
        train_valid_split=train_valid_split,
        split_seed=split_seed,
    )


__all__ = ["get_sequence_datasets"]
