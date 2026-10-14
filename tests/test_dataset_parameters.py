"""Dataset parameters from the training pipeline take effect or raise."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Subset, TensorDataset

from dendritic_modeling.config.config import DataProcessingConfig, ExperimentConfig
from dendritic_modeling.datasets import get_unified_datasets
from dendritic_modeling.datasets.standard_datasets import (
    _maybe_apply_label_noise_to_triplet,
)


def _task(dataset, *, train_valid_split=0.8, **parameters):
    return SimpleNamespace(
        dataset=dataset,
        data_path=None,
        train_valid_split=train_valid_split,
        parameters={"split_seed": 0, "label_noise_seed": 0, "seed": 0, **parameters},
    )


def test_label_noise_on_a_dataset_that_does_not_apply_it_raises():
    with pytest.raises(ValueError, match=r"label_noise_rate=0\.2 would be ignored"):
        get_unified_datasets(_task("linear_separable", label_noise_rate=0.2))


def test_label_noise_needs_an_in_memory_training_split():
    data = TensorDataset(torch.zeros(4, 2), torch.tensor([0, 1, 0, 1]))
    with pytest.raises(TypeError, match="TensorDataset"):
        _maybe_apply_label_noise_to_triplet(
            Subset(data, [0, 1]),
            data,
            data,
            label_noise_rate=0.5,
            label_noise_seed=0,
        )


@pytest.mark.parametrize("rate", [-0.1, 1.5, True])
def test_label_noise_rate_must_be_a_fraction(rate):
    with pytest.raises(ValueError, match="label_noise_rate must lie in"):
        DataProcessingConfig(label_noise_rate=rate)


@pytest.mark.parametrize("split", [-0.1, 1.2])
def test_train_valid_split_must_be_a_fraction(split):
    with pytest.raises(ValueError, match=r"train_valid_split .* must lie in"):
        ExperimentConfig(train_valid_split=split)


def test_zero_train_valid_split_raises_for_datasets_the_pipeline_splits():
    with pytest.raises(ValueError, match="no training samples"):
        get_unified_datasets(_task("linear_separable", train_valid_split=0.0))


def test_sequence_builder_rejects_a_key_it_does_not_read():
    from dendritic_modeling.datasets.sequence_datasets import get_sequence_datasets

    with pytest.raises(ValueError, match=r"seq_lenn \(did you mean seq_len\?\)"):
        get_sequence_datasets("adding_problem", n_samples=8, seq_lenn=5)


def test_pipeline_keys_are_not_dataset_parameters():
    from dendritic_modeling.datasets.sequence_datasets import get_sequence_datasets

    train, _, _ = get_sequence_datasets(
        "adding_problem",
        n_samples=8,
        seq_len=5,
        flatten=False,
        normalize=False,
        label_noise_rate=0.0,
        label_noise_seed=0,
        split_seed=0,
        seed=0,
    )
    assert len(train) > 0


def test_builders_registered_at_run_time_check_their_own_parameters():
    from dendritic_modeling.datasets.sequence.builders.common import deferred_split
    from dendritic_modeling.datasets.sequence_datasets import (
        get_sequence_datasets,
        register_sequence_dataset_builder,
        unregister_sequence_dataset_builder,
    )

    data = TensorDataset(torch.zeros(4, 2), torch.tensor([0, 1, 0, 1]))
    register_sequence_dataset_builder(
        "_parameters_test_task", lambda *_: deferred_split(data, data)
    )
    try:
        get_sequence_datasets("_parameters_test_task", custom_option=1)
    finally:
        unregister_sequence_dataset_builder("_parameters_test_task")


@pytest.mark.parametrize(
    "dataset, parameters, unread",
    [
        ("mnist", {"shuffle_iterations": 2}, "shuffle_iterations"),
        ("linear_separable", {"n_sampls": 10}, "n_sampls"),
        ("poisson_generator", {"base_datset": "mnist"}, "base_datset"),
    ],
)
def test_unified_loaders_reject_keys_they_do_not_read(dataset, parameters, unread):
    with pytest.raises(ValueError, match=rf"dataset_params\.{dataset} sets {unread}"):
        get_unified_datasets(_task(dataset, **parameters))


def test_positive_dataset_name_and_signed_encoding_flag_conflict():
    from dendritic_modeling.datasets.sequence.builders.common import (
        positive_input_encoding,
    )

    assert positive_input_encoding("romo_delay_comparison_positive", {})
    assert positive_input_encoding(
        "romo_delay_comparison", {"positive_input_encoding": 1}
    )
    assert not positive_input_encoding("romo_delay_comparison", {})
    with pytest.raises(ValueError, match="contradicts"):
        positive_input_encoding(
            "romo_delay_comparison_positive", {"positive_input_encoding": False}
        )


def test_stringer_split_seed_follows_the_dataset_seed(monkeypatch, tmp_path):
    from dendritic_modeling.datasets.sequence.builders import neuro

    seeds = []

    class Recorder:
        _GRATING_FILE = neuro.StringerV1Dataset._GRATING_FILE

        def __init__(self, **kwargs):
            seeds.append(kwargs["seed"])

    monkeypatch.setattr(neuro, "StringerV1Dataset", Recorder)
    neuro.build_stringer_v1(
        "stringer_v1", str(tmp_path), {"seed": 7, "label_noise_seed": 3}
    )
    neuro.build_stringer_v1("stringer_v1", str(tmp_path), {"label_noise_seed": 3})
    assert seeds == [7, 7, 3, 3]


def test_contextual_remapping_requires_the_presentation_stream_trainer():
    from dendritic_modeling.datasets.registry import (
        reject_unsupported_training_strategy,
    )

    reject_unsupported_training_strategy("contextual_remapping", "presentation_stream")
    reject_unsupported_training_strategy("mnist", "standard")
    with pytest.raises(ValueError, match="strategy: presentation_stream"):
        reject_unsupported_training_strategy("contextual_remapping", "standard")


def test_unset_validation_mode_of_a_gain_dataset_warns(monkeypatch, caplog):
    import logging

    from dendritic_modeling.datasets import synthetic_datasets

    built = []
    spec = synthetic_datasets._SyntheticDatasetSpec(
        factory=lambda **kwargs: built.append(kwargs) or (None, None, None),
        defaults={"valid_split_mode": "ood"},
    )
    monkeypatch.setitem(synthetic_datasets._SYNTHETIC_DATASET_SPECS, "_gain", spec)
    with caplog.at_level(logging.WARNING):
        synthetic_datasets.get_synthetic_datasets({"type": "_gain"})
    assert "shifted test distribution" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        synthetic_datasets.get_synthetic_datasets(
            {"type": "_gain", "valid_split_mode": "ood"}
        )
    assert "shifted test distribution" not in caplog.text
    assert built == [{"valid_split_mode": "ood"}] * 2


def _copy_task(**kwargs):
    from dendritic_modeling.datasets.sequence_datasets import get_sequence_datasets

    train, _, test = get_sequence_datasets(
        "copy_task", n_samples=16, seq_len=4, split_seed=0, **kwargs
    )
    return train.dataset[:][0], test[:][0]


def test_seeded_sequence_data_match_the_pipeline_and_restore_the_rng():
    from dendritic_modeling.utils.reproducibility import set_seed

    set_seed(11)
    pipeline_train, pipeline_test = _copy_task()  # global stream seeded with 11

    torch.manual_seed(999)
    state = torch.random.get_rng_state()
    train, test = _copy_task(seed=11)
    assert torch.equal(train, pipeline_train)
    assert torch.equal(test, pipeline_test)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert not torch.equal(_copy_task(seed=12)[0], train)


def test_synthetic_generators_leave_the_global_rng_unchanged():
    from dendritic_modeling.datasets.synthetic_datasets import get_synthetic_datasets

    torch.manual_seed(5)
    state = torch.random.get_rng_state()
    get_synthetic_datasets({"type": "linear_separable", "n_samples": 20, "seed": 3})
    assert torch.equal(torch.random.get_rng_state(), state)


def test_validation_loss_without_samples_is_nan():
    import math

    from dendritic_modeling.training.strategies.standard import _ValidationLossStats

    stats = _ValidationLossStats(0.0, 0, 0.0, 0)
    assert math.isnan(stats.batch_mean) and math.isnan(stats.sample_mean)


@pytest.mark.parametrize(
    ("dataset", "parameters", "option"),
    [
        ("linear_separable", {"n_samples": 40, "normalize": True}, "normalize"),
        ("adding_problem", {"n_samples": 8, "seq_len": 5, "flatten": True}, "flatten"),
        (
            "adding_problem",
            {"n_samples": 8, "seq_len": 5, "normalize": True},
            "normalize",
        ),
        ("contextual_remapping", {"normalize": True}, "normalize"),
    ],
)
def test_processing_a_loader_does_not_apply_raises(dataset, parameters, option):
    from types import SimpleNamespace

    from dendritic_modeling.datasets import get_unified_datasets

    task = SimpleNamespace(
        dataset=dataset,
        data_path=None,
        train_valid_split=0.8,
        parameters={"label_noise_rate": 0.0, **parameters},
    )
    with pytest.raises(ValueError, match=f"{option}: true would be ignored"):
        get_unified_datasets(task_cfg=task)


def test_processing_that_asks_for_no_effect_or_is_already_met_passes():
    from types import SimpleNamespace

    from dendritic_modeling.datasets import get_unified_datasets

    for dataset, parameters in (
        ("linear_separable", {"n_samples": 40, "flatten": True, "normalize": False}),
        ("adding_problem", {"n_samples": 8, "seq_len": 5, "flatten": False}),
    ):
        task = SimpleNamespace(
            dataset=dataset,
            data_path=None,
            train_valid_split=0.8,
            parameters={"label_noise_rate": 0.0, **parameters},
        )
        train, _, _ = get_unified_datasets(task_cfg=task)
        assert len(train) > 0
