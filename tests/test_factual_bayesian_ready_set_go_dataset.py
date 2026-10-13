"""Tests for factual-outcome Bayesian Ready--Set--Go data."""

from __future__ import annotations

import hashlib

import pytest
import torch
from torch.utils.data import DataLoader

from dendritic_modeling.datasets.sequence import (
    BayesianReadySetGoDataset,
    FactualBayesianReadySetGoDataset,
    finite_gaussian_motor_kernel,
    get_registered_sequence_dataset_names,
    get_sequence_datasets,
)
from dendritic_modeling.datasets.unified_options import _SEQUENCE_DATASET_NAMES
from dendritic_modeling.training.loss.functions import get_loss_function


def _small_dataset(**overrides) -> FactualBayesianReadySetGoDataset:
    parameters = {
        "n_samples": 60,
        "prior_supports": ((8, 10, 12), (14, 16, 18)),
        "prior_probabilities": ((0.2, 0.5, 0.3), (0.25, 0.5, 0.25)),
        "weber_fraction": 0.2,
        "observed_interval_min": 4,
        "observed_interval_max": 22,
        "production_scale": 1.0,
        "motor_sigma": 1.0,
        "motor_tail_sigma": 2.0,
        "set_step_values": (28, 30, 32),
        "pulse_duration": 1,
        "context_start_step": 0,
        "context_duration": 2,
        "post_go_duration": 3,
        "sequence_duration": 56,
        "cue_gain_min": 0.8,
        "cue_gain_max": 1.2,
        "input_noise_std": 0.0,
        "mean_distractors_per_trial": 0.0,
        "positive_input_encoding": True,
        "seed": 117,
    }
    return FactualBayesianReadySetGoDataset(**{**parameters, **overrides})


def test_finite_gaussian_motor_kernel_is_normalized_and_auditable():
    offsets, probabilities = finite_gaussian_motor_kernel(
        sigma=1.25,
        tail_sigma=3.0,
    )
    assert offsets.dtype == torch.long
    assert probabilities.dtype == torch.float64
    assert torch.equal(offsets, torch.arange(-4, 5))
    assert probabilities.sum().item() == pytest.approx(1.0)
    assert torch.equal(probabilities, probabilities.flip(0))
    assert int(torch.argmax(probabilities)) == 4

    exact_offsets, exact_probabilities = finite_gaussian_motor_kernel(
        sigma=0.0,
        tail_sigma=4.0,
    )
    assert torch.equal(exact_offsets, torch.tensor([0]))
    assert torch.equal(exact_probabilities, torch.ones(1, dtype=torch.float64))
    with pytest.raises(ValueError, match="non-negative"):
        finite_gaussian_motor_kernel(sigma=-0.1)
    with pytest.raises(ValueError, match="positive"):
        finite_gaussian_motor_kernel(sigma=1.0, tail_sigma=0.0)


def test_factual_rsg_has_only_factual_hard_outcomes():
    dataset = _small_dataset(n_samples=61)

    assert dataset.lineage_name == "factual_bayesian_ready_set_go_v1"
    assert dataset.is_factual_only
    assert not dataset.is_counterfactual_paired
    assert dataset.target_kind == "hard_integer_go_step"
    assert dataset.input_dim == 5
    assert dataset.targets.dtype == torch.long
    assert dataset.targets.shape == (61,)
    assert torch.equal(dataset.targets, dataset.go_steps)
    assert torch.equal(
        dataset.ready_steps, dataset.set_steps - dataset.observed_intervals
    )
    assert torch.equal(
        dataset.go_steps, dataset.set_steps + dataset.production_intervals
    )
    assert torch.equal(
        dataset.set_steps, dataset.set_step_values[dataset.set_step_indices]
    )
    assert torch.equal(
        dataset.motor_offsets,
        dataset.motor_offset_values[dataset.motor_offset_indices],
    )
    assert torch.equal(
        dataset.production_intervals,
        dataset.production_centers[dataset.latent_interval_indices]
        + dataset.motor_offsets,
    )
    assert torch.all(dataset.targets >= 0)
    assert torch.all(dataset.targets < dataset.sequence_duration)
    assert (
        int(torch.bincount(dataset.context_indices).max())
        - int(torch.bincount(dataset.context_indices).min())
        <= 1
    )

    for sample_index in range(len(dataset)):
        context = int(dataset.context_indices[sample_index])
        latent = int(dataset.latent_intervals[sample_index])
        assert latent in dataset.prior_supports[context]
        input_sample, target = dataset[sample_index]
        assert input_sample.shape == (dataset.sequence_duration, dataset.input_dim)
        assert target.ndim == 0
        assert target.dtype == torch.long
        context_slice = input_sample[
            dataset.context_start_step : dataset.context_start_step
            + dataset.context_duration,
            2:4,
        ]
        assert torch.all(context_slice[:, context] == dataset.context_gain)
        assert torch.all(context_slice[:, 1 - context] == 0.0)

    for forbidden_attribute in (
        "counterfactual_indices",
        "factual_mask",
        "pair_ids",
        "posterior_mean_intervals",
        "posterior_probabilities",
    ):
        assert not hasattr(dataset, forbidden_attribute)


def test_factual_rsg_declared_distributions_match_generated_trials():
    dataset = _small_dataset(n_samples=6000, seed=391)

    assert torch.allclose(
        dataset.measurement_likelihood.sum(dim=1),
        torch.ones(len(dataset.latent_interval_values), dtype=torch.float64),
    )
    assert torch.allclose(
        dataset.motor_kernel.sum(dim=1),
        torch.ones(len(dataset.latent_interval_values), dtype=torch.float64),
    )
    assert dataset.motor_kernel_probabilities.sum().item() == pytest.approx(1.0)

    for context in range(2):
        context_rows = dataset.context_indices == context
        support_counts = torch.bincount(
            dataset.latent_support_indices[context_rows],
            minlength=len(dataset.prior_supports[context]),
        ).to(torch.float64)
        empirical = support_counts / support_counts.sum()
        declared = torch.tensor(
            dataset.prior_support_probabilities[context], dtype=torch.float64
        )
        assert torch.all(torch.abs(empirical - declared) < 1.0 / support_counts.sum())

    for latent_index in range(len(dataset.latent_interval_values)):
        latent_rows = dataset.latent_interval_indices == latent_index
        assert int(latent_rows.sum()) > 500
        observed_counts = torch.bincount(
            dataset.observed_interval_indices[latent_rows],
            minlength=len(dataset.observed_interval_values),
        ).to(torch.float64)
        observed_empirical = observed_counts / observed_counts.sum()
        assert (
            torch.abs(
                observed_empirical - dataset.measurement_likelihood[latent_index]
            ).sum()
            < 0.16
        )

        production_counts = torch.bincount(
            dataset.production_interval_indices[latent_rows],
            minlength=len(dataset.production_interval_values),
        ).to(torch.float64)
        production_empirical = production_counts / production_counts.sum()
        assert (
            torch.abs(production_empirical - dataset.motor_kernel[latent_index]).sum()
            < 0.12
        )

    distances = (
        dataset.production_interval_values[None, :]
        - dataset.production_centers[:, None]
    ).abs()
    assert torch.all(dataset.motor_kernel[distances > dataset.motor_radius] == 0.0)


def test_factual_rsg_set_draw_is_independent_of_latent_prior():
    first = _small_dataset(n_samples=600, seed=719)
    shifted_prior = _small_dataset(
        n_samples=600,
        seed=719,
        prior_probabilities=((0.8, 0.1, 0.1), (0.1, 0.1, 0.8)),
    )

    assert torch.equal(first.context_indices, shifted_prior.context_indices)
    assert torch.equal(first.set_steps, shifted_prior.set_steps)
    assert not torch.equal(first.latent_intervals, shifted_prior.latent_intervals)
    assert first.set_sampling == "balanced_explicit_grid"

    set_counts = torch.bincount(
        first.set_step_indices, minlength=len(first.set_step_values)
    ).to(torch.float64)
    assert int(set_counts.max() - set_counts.min()) <= 1


def test_factual_rsg_motor_seed_is_a_separate_auditable_stream():
    first = _small_dataset(n_samples=300, seed=719, motor_seed=1201)
    changed_motor = _small_dataset(n_samples=300, seed=719, motor_seed=1202)
    derived_first = _small_dataset(n_samples=300, seed=719)
    derived_second = _small_dataset(n_samples=300, seed=719)

    for field in (
        "context_indices",
        "latent_intervals",
        "observed_intervals",
        "ready_steps",
        "set_steps",
        "cue_gains",
        "distractor_mask",
        "inputs",
    ):
        assert torch.equal(getattr(first, field), getattr(changed_motor, field))
    assert not torch.equal(first.motor_offsets, changed_motor.motor_offsets)
    assert not torch.equal(first.targets, changed_motor.targets)
    assert (first.motor_seed, changed_motor.motor_seed) == (1201, 1202)
    assert not first.motor_seed_was_derived
    assert derived_first.motor_seed_was_derived
    assert derived_first.motor_seed != derived_first.seed
    assert torch.equal(derived_first.motor_offsets, derived_second.motor_offsets)

    exact_motor = _small_dataset(n_samples=30, motor_sigma=0.0)
    assert torch.equal(exact_motor.motor_offset_values, torch.tensor([0]))
    assert torch.equal(
        exact_motor.motor_kernel_probabilities,
        torch.ones(1, dtype=torch.float64),
    )
    assert torch.all(exact_motor.motor_offsets == 0)
    assert torch.equal(
        exact_motor.production_intervals,
        exact_motor.production_centers[exact_motor.latent_interval_indices],
    )


def test_factual_rsg_is_locally_deterministic_and_supports_nuisances():
    parameters = {
        "n_samples": 90,
        "seed": 881,
        "input_noise_std": 0.04,
        "mean_distractors_per_trial": 2.0,
        "distractor_amplitude": 0.7,
    }
    first = _small_dataset(**parameters)
    torch.manual_seed(999999)
    second = _small_dataset(**parameters)
    different = _small_dataset(**{**parameters, "seed": 882})

    for field in (
        "inputs",
        "targets",
        "context_indices",
        "latent_intervals",
        "observed_intervals",
        "set_steps",
        "production_intervals",
        "cue_gains",
        "distractor_mask",
    ):
        assert torch.equal(getattr(first, field), getattr(second, field))
    assert not torch.equal(first.inputs, different.inputs)
    assert torch.all(first.inputs >= 0.0)
    assert torch.any(first.distractor_mask)
    assert float(first.cue_gains.min()) >= first.cue_gain_min
    assert float(first.cue_gains.max()) <= first.cue_gain_max
    assert torch.unique(first.cue_gains).numel() > 1

    clean = _small_dataset(
        input_noise_std=0.0,
        mean_distractors_per_trial=0.0,
        cue_gain_min=1.0,
        cue_gain_max=1.0,
    )
    assert not torch.any(clean.distractor_mask)
    assert torch.all(clean.inputs[..., 4] == 0.0)
    assert torch.all(clean.cue_gains == 1.0)


def test_factual_rsg_zero_nested_default_preserves_frozen_tensor_bytes():
    dataset = _small_dataset(
        n_samples=73,
        input_noise_std=0.04,
        mean_distractors_per_trial=2.0,
        distractor_amplitude=0.7,
    )
    expected = {
        "inputs": "397a666b2a719aa063535db32ca4b7f34883e7cd5424045dc6cec6e09f582199",
        "targets": "b899ce92db4f174934b109e0a513e89c13e2aa2e6cf8d0b530efbcd0dd1199c1",
        "observed_intervals": "3609379a900082aec009ce93989e9eb8cd468f679091cbdffda2acbc15e4c756",
        "cue_gains": "25a739b38c794cfbc5e5b46f04b6ccb79def37115399e170c88689eb06026f50",
        "distractor_mask": "215e4d9d3f19fb3c2574f975d7352207677d4fe7a420ea7c112e4ab764046803",
    }

    assert dataset.nested_distractor_pairs_per_trial == 0
    assert dataset.nested_distractor_ready_steps.shape == (73, 0)
    for name, expected_sha256 in expected.items():
        tensor = getattr(dataset, name).detach().cpu().contiguous()
        observed_sha256 = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        assert observed_sha256 == expected_sha256


def test_factual_rsg_nested_pairs_are_independent_and_strictly_inside_outer_pair():
    common = {
        "n_samples": 120,
        "observed_interval_min": 12,
        "observed_interval_max": 22,
        "mean_distractors_per_trial": 0.0,
        "input_noise_std": 0.0,
        "cue_gain_min": 1.0,
        "cue_gain_max": 1.0,
    }
    clean = _small_dataset(**common)
    nested = _small_dataset(
        **common,
        nested_distractor_pairs_per_trial=2,
        nested_distractor_interval_values=(2, 3),
        nested_distractor_clearance=1,
    )
    changed_nested_stream = _small_dataset(
        **common,
        nested_distractor_pairs_per_trial=2,
        nested_distractor_interval_values=(2, 3),
        nested_distractor_clearance=1,
        nested_distractor_seed=nested.nested_distractor_seed + 1,
    )

    for name in (
        "targets",
        "context_indices",
        "latent_intervals",
        "observed_intervals",
        "ready_steps",
        "set_steps",
        "production_intervals",
        "cue_gains",
        "distractor_mask",
    ):
        assert torch.equal(getattr(clean, name), getattr(nested, name))
        assert torch.equal(getattr(nested, name), getattr(changed_nested_stream, name))
    assert not torch.equal(clean.inputs, nested.inputs)
    assert not torch.equal(
        nested.nested_distractor_ready_steps,
        changed_nested_stream.nested_distractor_ready_steps,
    )

    ready = nested.nested_distractor_ready_steps
    set_steps = nested.nested_distractor_set_steps
    intervals = nested.nested_distractor_intervals
    assert torch.equal(set_steps - ready, intervals)
    assert torch.all(ready[:, 0] >= nested.ready_steps + 2)
    assert torch.all(set_steps[:, -1] + nested.pulse_duration <= nested.set_steps - 1)
    assert torch.all(ready[:, 1:] >= set_steps[:, :-1] + nested.pulse_duration)
    assert set(intervals.unique().tolist()) == {2, 3}
    for sample_index in range(len(nested)):
        for pair_index in range(nested.nested_distractor_pairs_per_trial):
            nested_ready = int(ready[sample_index, pair_index])
            nested_set = int(set_steps[sample_index, pair_index])
            assert torch.all(
                nested.inputs[
                    sample_index,
                    nested_ready : nested_ready + nested.pulse_duration,
                    0,
                ]
                == 1.0
            )
            assert torch.all(
                nested.inputs[
                    sample_index,
                    nested_set : nested_set + nested.pulse_duration,
                    1,
                ]
                == 1.0
            )


def test_factual_rsg_hard_targets_work_with_temporal_event_ce():
    dataset = _small_dataset(n_samples=12)
    inputs, targets = next(iter(DataLoader(dataset, batch_size=6)))
    logits = torch.zeros(inputs.shape[0], dataset.sequence_duration, 1)
    loss = get_loss_function("temporal_event_ce").from_predictions(logits, targets)

    assert torch.isfinite(loss)
    assert float(loss) == pytest.approx(
        torch.log(torch.tensor(float(dataset.sequence_duration))).item()
    )


def test_factual_rsg_builder_creates_isolated_deterministic_splits():
    kwargs = {
        "n_samples": 12,
        "train": {"n_samples": 12},
        "valid": {"n_samples": 8},
        "test": {"n_samples": 10},
        "prior_supports": ((8, 10, 12), (14, 16, 18)),
        "prior_probabilities": ((0.2, 0.5, 0.3), (0.25, 0.5, 0.25)),
        "weber_fraction": 0.2,
        "observed_interval_min": 4,
        "observed_interval_max": 22,
        "motor_sigma": 1.0,
        "motor_tail_sigma": 2.0,
        "set_step_values": (28, 30, 32),
        "pulse_duration": 1,
        "context_duration": 2,
        "post_go_duration": 3,
        "sequence_duration": 56,
        "input_noise_std": 0.0,
        "mean_distractors_per_trial": 0.0,
        "seed": 101,
        "motor_seed": 701,
    }
    first = get_sequence_datasets("factual_bayesian_ready_set_go_positive", **kwargs)
    second = get_sequence_datasets("factual_bayesian_ready_set_go_positive", **kwargs)

    assert tuple(len(split) for split in first) == (12, 8, 10)
    assert tuple(split.seed for split in first) == (101, 102, 103)
    assert tuple(split.motor_seed for split in first) == (701, 702, 703)
    assert len({id(split) for split in first}) == 3
    assert all(isinstance(split, FactualBayesianReadySetGoDataset) for split in first)
    assert all(split.positive_input_encoding for split in first)
    for first_split, second_split in zip(first, second, strict=True):
        assert torch.equal(first_split.inputs, second_split.inputs)
        assert torch.equal(first_split.targets, second_split.targets)

    nested = get_sequence_datasets(
        "factual_bayesian_ready_set_go_positive",
        **{
            **kwargs,
            "observed_interval_min": 12,
            "nested_distractor_pairs_per_trial": 2,
            "nested_distractor_interval_values": (2, 3),
            "nested_distractor_clearance": 1,
            "nested_distractor_seed": 1701,
        },
    )
    assert tuple(split.nested_distractor_seed for split in nested) == (
        1701,
        1702,
        1703,
    )
    assert all(split.nested_distractor_pairs_per_trial == 2 for split in nested)

    with pytest.raises(ValueError, match="seeds must be distinct"):
        get_sequence_datasets(
            "factual_bayesian_ready_set_go_positive",
            **{
                **kwargs,
                "train": {"n_samples": 12, "seed": 7},
                "valid": {"n_samples": 8, "seed": 7},
                "test": {"n_samples": 10, "seed": 9},
            },
        )


def test_factual_rsg_names_are_in_both_public_registries():
    names = {
        "factual_bayesian_ready_set_go",
        "factual_bayesian_ready_set_go_positive",
    }
    assert names <= set(_SEQUENCE_DATASET_NAMES)
    assert names <= set(get_registered_sequence_dataset_names())


def test_existing_paired_bayesian_rsg_lineage_is_unchanged():
    existing = BayesianReadySetGoDataset(
        n_samples=10,
        prior_supports=((8, 10, 12), (12, 16, 20)),
        prior_probabilities=((0.2, 0.5, 0.3), (0.25, 0.5, 0.25)),
        weber_fraction=0.2,
        observed_interval_min=4,
        observed_interval_max=24,
        set_min_step=30,
        set_max_step=34,
        pulse_duration=1,
        context_duration=2,
        target_sigma=1.0,
        target_tail_sigma=4.0,
        post_go_duration=4,
        sequence_duration=59,
        input_noise_std=0.0,
        mean_distractors_per_trial=0.0,
        seed=71,
    )

    assert existing.is_counterfactual_paired
    assert existing.n_counterfactual_pairs == 5
    assert existing.targets.dtype == torch.float32
    assert existing.targets.shape == (10, 59)
    assert hasattr(existing, "posterior_probabilities")
    assert hasattr(existing, "counterfactual_indices")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n_samples": 0}, "positive"),
        ({"motor_sigma": -1.0}, "non-negative"),
        ({"motor_tail_sigma": 0.0}, "positive"),
        ({"motor_seed": 117}, "differ"),
        ({"motor_sigma": 10.0}, "overlap"),
        ({"sequence_duration": 50}, "too short"),
        ({"set_min_step": 28}, "mutually exclusive"),
        ({"nested_distractor_pairs_per_trial": -1}, "non-negative"),
        (
            {
                "nested_distractor_pairs_per_trial": 2,
                "nested_distractor_interval_values": (3, 4),
                "nested_distractor_clearance": 1,
            },
            "cannot contain",
        ),
    ],
)
def test_factual_rsg_rejects_invalid_kernel_or_geometry(overrides, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _small_dataset(**overrides)
