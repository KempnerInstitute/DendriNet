"""Factory for sequence dataset train/valid/test triplets."""

from __future__ import annotations

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
    split_seed = int(kwargs.get("split_seed", kwargs.get("seed", 42)) or 0)

    if dataset_name == "sequential_mnist":
        mode = kwargs.get("mode", "pixel")
        normalize = kwargs.get("normalize", True)
        train_full = SequentialMNISTDataset(
            mode=mode, train=True, data_path=data_path, normalize=normalize
        )
        test_ds = SequentialMNISTDataset(
            mode=mode, train=False, data_path=data_path, normalize=normalize
        )
    elif dataset_name == "copy_task":
        n_train = kwargs.get("n_samples", 10000)
        train_full = CopyTaskDataset(
            n_samples=n_train,
            seq_len=kwargs.get("seq_len", 10),
            delay=kwargs.get("delay", 50),
            n_symbols=kwargs.get("n_symbols", 8),
        )
        test_ds = CopyTaskDataset(
            n_samples=max(n_train // 5, 1000),
            seq_len=kwargs.get("seq_len", 10),
            delay=kwargs.get("delay", 50),
            n_symbols=kwargs.get("n_symbols", 8),
        )
    elif dataset_name == "adding_problem":
        n_train = kwargs.get("n_samples", 10000)
        train_full = AddingProblemDataset(
            n_samples=n_train,
            seq_len=kwargs.get("seq_len", 200),
        )
        test_ds = AddingProblemDataset(
            n_samples=max(n_train // 5, 1000),
            seq_len=kwargs.get("seq_len", 200),
        )
    elif dataset_name == "variable_delay_dms":
        n_train = kwargs.get("n_samples", 10000)
        train_full = VariableDelayDMSDataset(
            n_samples=n_train,
            stimulus_dim=kwargs.get("stimulus_dim", 16),
            stimulus_duration=kwargs.get("stimulus_duration", 10),
            min_delay=kwargs.get("min_delay", 10),
            max_delay=kwargs.get("max_delay", 100),
            probe_duration=kwargs.get("probe_duration", 10),
        )
        test_ds = VariableDelayDMSDataset(
            n_samples=max(n_train // 5, 1000),
            stimulus_dim=kwargs.get("stimulus_dim", 16),
            stimulus_duration=kwargs.get("stimulus_duration", 10),
            min_delay=kwargs.get("min_delay", 10),
            max_delay=kwargs.get("max_delay", 100),
            probe_duration=kwargs.get("probe_duration", 10),
        )
    elif dataset_name == "multi_frequency":
        n_train = kwargs.get("n_samples", 10000)
        train_full = MultiFrequencyDataset(
            n_samples=n_train,
            seq_len=kwargs.get("seq_len", 200),
            n_frequencies=kwargs.get("n_frequencies", 5),
            base_freq=kwargs.get("base_freq", 0.01),
            noise_std=kwargs.get("noise_std", 0.1),
        )
        test_ds = MultiFrequencyDataset(
            n_samples=max(n_train // 5, 1000),
            seq_len=kwargs.get("seq_len", 200),
            n_frequencies=kwargs.get("n_frequencies", 5),
            base_freq=kwargs.get("base_freq", 0.01),
            noise_std=kwargs.get("noise_std", 0.1),
        )
    elif dataset_name == "hierarchical_temporal":
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
        train_full = HierarchicalTemporalDataset(n_samples=n_train, **common)
        test_ds = HierarchicalTemporalDataset(
            n_samples=max(n_train // 5, 2000), **common
        )
    elif dataset_name in {
        "context_dependent_decision",
        "context_dependent_decision_hard",
        "context_dependent_decision_positive",
        "context_dependent_decision_hard_positive",
    }:
        n_train = kwargs.get("n_samples", 20000)
        positive_input_encoding = dataset_name.endswith("_positive") or kwargs.get(
            "positive_input_encoding", False
        )
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
            "positive_input_encoding": positive_input_encoding,
        }
        train_full = ContextDependentDecisionDataset(n_samples=n_train, **common)
        test_ds = ContextDependentDecisionDataset(
            n_samples=max(n_train // 5, 2000), **common
        )
    elif dataset_name in {
        "switching_context_decision",
        "switching_context_decision_positive",
    }:
        n_train = kwargs.get("n_samples", 16000)
        positive_input_encoding = dataset_name.endswith("_positive") or kwargs.get(
            "positive_input_encoding", False
        )
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
            "positive_input_encoding": positive_input_encoding,
        }
        train_full = SwitchingContextDecisionDataset(n_samples=n_train, **common)
        test_ds = SwitchingContextDecisionDataset(
            n_samples=max(n_train // 5, 2000), **common
        )
    elif dataset_name in {
        "timescale_generalization",
        "timescale_generalization_positive",
    }:
        n_train = kwargs.get("n_samples", 20000)
        positive_input_encoding = dataset_name.endswith("_positive") or kwargs.get(
            "positive_input_encoding", False
        )
        common_defaults = {
            "stimulus_dim": kwargs.get("stimulus_dim", 16),
            "stimulus_duration": kwargs.get("stimulus_duration", 10),
            "min_delay": kwargs.get("min_delay", 10),
            "max_delay": kwargs.get("max_delay", 100),
            "probe_duration": kwargs.get("probe_duration", 10),
            "positive_input_encoding": positive_input_encoding,
            "noise_std": kwargs.get("noise_std", 0.3),
        }
        explicit_splits = any(
            isinstance(kwargs.get(split_name), dict)
            for split_name in ("train", "valid", "test")
        )
        if explicit_splits:

            def _build_split(split_name: str, default_n_samples: int):
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
                return TimescaleGeneralizationDataset(n_samples=split_n, **split_params)

            train_ds = _build_split("train", n_train)
            valid_ds = _build_split("valid", max(n_train // 5, 2000))
            test_ds = _build_split("test", max(n_train // 5, 2000))
            return train_ds, valid_ds, test_ds

        train_full = TimescaleGeneralizationDataset(
            n_samples=n_train, **common_defaults
        )
        test_ds = TimescaleGeneralizationDataset(
            n_samples=max(n_train // 5, 2000), **common_defaults
        )
    elif dataset_name in {
        "odr_generalization",
        "odr_generalization_positive",
    }:
        n_train = kwargs.get("n_samples", 20000)
        positive_input_encoding = dataset_name.endswith("_positive") or kwargs.get(
            "positive_input_encoding", False
        )
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
            "positive_input_encoding": positive_input_encoding,
        }
        explicit_splits = any(
            isinstance(kwargs.get(split_name), dict)
            for split_name in ("train", "valid", "test")
        )
        if explicit_splits:

            def _build_split(split_name: str, default_n_samples: int):
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
                return OculomotorDelayedResponseDataset(
                    n_samples=split_n, **split_params
                )

            train_ds = _build_split("train", n_train)
            valid_ds = _build_split("valid", max(n_train // 5, 2000))
            test_ds = _build_split("test", max(n_train // 5, 2000))
            return train_ds, valid_ds, test_ds

        train_full = OculomotorDelayedResponseDataset(
            n_samples=n_train, **common_defaults
        )
        test_ds = OculomotorDelayedResponseDataset(
            n_samples=max(n_train // 5, 2000), **common_defaults
        )
    elif dataset_name in {
        "romo_delay_comparison",
        "romo_delay_comparison_positive",
    }:
        n_train = kwargs.get("n_samples", 20000)
        positive_input_encoding = dataset_name.endswith("_positive") or kwargs.get(
            "positive_input_encoding", False
        )
        common_defaults = {
            "frequency_values": tuple(
                kwargs.get("frequency_values", (2.0, 3.0, 4.0, 5.0, 6.0, 7.0))
            ),
            "stimulus_duration": kwargs.get("stimulus_duration", 20),
            "min_delay": kwargs.get("min_delay", 10),
            "max_delay": kwargs.get("max_delay", 100),
            "response_duration": kwargs.get("response_duration", 8),
            "noise_std": kwargs.get("noise_std", 0.08),
            "positive_input_encoding": positive_input_encoding,
            "tuning_dim": kwargs.get("tuning_dim", 8),
            "tuning_sigma": kwargs.get("tuning_sigma", None),
        }
        explicit_splits = any(
            isinstance(kwargs.get(split_name), dict)
            for split_name in ("train", "valid", "test")
        )
        if explicit_splits:

            def _build_split(split_name: str, default_n_samples: int):
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
                return RomoDelayComparisonDataset(n_samples=split_n, **split_params)

            train_ds = _build_split("train", n_train)
            valid_ds = _build_split("valid", max(n_train // 5, 2000))
            test_ds = _build_split("test", max(n_train // 5, 2000))
            return train_ds, valid_ds, test_ds

        train_full = RomoDelayComparisonDataset(n_samples=n_train, **common_defaults)
        test_ds = RomoDelayComparisonDataset(
            n_samples=max(n_train // 5, 2000), **common_defaults
        )
    elif dataset_name in {
        "gain_modulated_contextual_comparison",
        "gain_modulated_contextual_comparison_positive",
    }:
        n_train = kwargs.get("n_samples", 24000)
        positive_input_encoding = dataset_name.endswith("_positive") or kwargs.get(
            "positive_input_encoding", False
        )
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
            "positive_input_encoding": positive_input_encoding,
        }
        explicit_splits = any(
            isinstance(kwargs.get(split_name), dict)
            for split_name in ("train", "valid", "test")
        )
        if explicit_splits:

            def _build_split(split_name: str, default_n_samples: int):
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
                return GainModulatedContextualComparisonDataset(
                    n_samples=split_n, **split_params
                )

            train_ds = _build_split("train", n_train)
            valid_ds = _build_split("valid", max(n_train // 5, 2000))
            test_ds = _build_split("test", max(n_train // 5, 2000))
            return train_ds, valid_ds, test_ds

        train_full = GainModulatedContextualComparisonDataset(
            n_samples=n_train, **common_defaults
        )
        test_ds = GainModulatedContextualComparisonDataset(
            n_samples=max(n_train // 5, 2000), **common_defaults
        )
    elif dataset_name == "multi_sine_forecast":
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
        train_full = MultiSineForecastDataset(
            n_samples=n_train,
            **common,
        )
        test_ds = MultiSineForecastDataset(
            n_samples=max(n_train // 5, 1000),
            **{
                **common,
                "seed": kwargs.get("test_seed", common["seed"] + 1),
            },
        )
    elif dataset_name in {"lorenz", "lorenz_sequence"}:
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
        train_full = LorenzSequenceDataset(
            n_samples=n_train,
            **common,
        )
        test_ds = LorenzSequenceDataset(
            n_samples=max(n_train // 5, 1000),
            **{
                **common,
                "seed": kwargs.get("test_seed", common["seed"] + 1),
            },
        )
    elif dataset_name == "switching_lds":
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
        train_full = SwitchingLDSDataset(
            n_samples=n_train,
            **common,
        )
        test_ds = SwitchingLDSDataset(
            n_samples=max(n_train // 5, 1000),
            **{
                **common,
                "seed": kwargs.get("test_seed", common["seed"] + 1),
            },
        )
    elif dataset_name == "stringer_v1":
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

        def _state_kwargs(prefix: str, state: str) -> dict:
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
        else:
            # Fixed 90/10 train/test split; train_valid_split handled by factory below
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
    else:
        raise ValueError(f"Unknown sequence dataset: {dataset_name}")

    # Split training into train/valid
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
        valid_ds = test_ds  # use test set as validation if no split

    return train_ds, valid_ds, test_ds


__all__ = ["get_sequence_datasets"]
