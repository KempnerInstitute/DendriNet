"""Analytic and geometry tests for Bayesian Ready--Set--Go."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Subset

from dendritic_modeling.datasets.sequence import (
    BayesianReadySetGoDataset,
    discrete_bayesian_interval_posterior,
    discrete_weber_likelihood,
    get_sequence_datasets,
)


def _small_dataset(**overrides) -> BayesianReadySetGoDataset:
    parameters = {
        "n_samples": 30,
        "prior_supports": ((8, 10, 12), (12, 16, 20)),
        "prior_probabilities": ((0.2, 0.5, 0.3), (0.25, 0.5, 0.25)),
        "weber_fraction": 0.2,
        "observed_interval_min": 4,
        "observed_interval_max": 24,
        "set_min_step": 30,
        "set_max_step": 34,
        "pulse_duration": 1,
        "context_start_step": 0,
        "context_duration": 2,
        "target_sigma": 1.0,
        "target_tail_sigma": 4.0,
        "post_go_duration": 4,
        "sequence_duration": 59,
        "input_noise_std": 0.05,
        "mean_distractors_per_trial": 1.5,
        "distractor_amplitude": 0.5,
        "positive_input_encoding": True,
        "seed": 117,
    }
    return BayesianReadySetGoDataset(**{**parameters, **overrides})


def test_discrete_weber_likelihood_and_posterior_limiting_cases():
    latent = torch.tensor([8, 12, 16])
    observed = torch.arange(6, 19)
    likelihood = discrete_weber_likelihood(
        latent,
        observed,
        weber_fraction=0.2,
    )
    assert likelihood.dtype == torch.float64
    assert torch.allclose(likelihood.sum(dim=1), torch.ones(3, dtype=torch.float64))
    posterior = discrete_bayesian_interval_posterior(
        (0.2, 0.3, 0.5), likelihood, observed_index=6
    )
    expected = torch.tensor((0.2, 0.3, 0.5), dtype=torch.float64) * likelihood[:, 6]
    expected /= expected.sum()
    assert torch.allclose(posterior, expected)
    assert posterior.sum().item() == pytest.approx(1.0)

    identity = discrete_weber_likelihood(
        latent,
        latent,
        weber_fraction=0.0,
        measurement_noise_floor=0.0,
    )
    assert torch.equal(identity, torch.eye(3, dtype=torch.float64))
    exact = discrete_bayesian_interval_posterior(
        (0.2, 0.3, 0.5), identity, observed_index=1
    )
    assert torch.equal(exact, torch.tensor((0.0, 1.0, 0.0)))

    broad = discrete_weber_likelihood(
        latent,
        observed,
        weber_fraction=1e6,
    )
    broad_posterior = discrete_bayesian_interval_posterior(
        (0.2, 0.3, 0.5), broad, observed_index=6
    )
    assert torch.allclose(
        broad_posterior,
        torch.tensor((0.2, 0.3, 0.5), dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )


def test_bayesian_rsg_counterfactuals_share_every_noncontext_input():
    dataset = _small_dataset()
    assert dataset.is_counterfactual_paired
    assert dataset.input_dim == 5
    assert torch.all(dataset.inputs >= 0.0)
    assert torch.allclose(
        dataset.targets.sum(dim=1), torch.ones(dataset.n_samples), atol=1e-6
    )
    assert torch.allclose(
        dataset.posterior_probabilities.sum(dim=1),
        torch.ones(dataset.n_samples, dtype=torch.float64),
    )
    assert torch.equal(
        dataset.set_steps - dataset.ready_steps, dataset.observed_intervals
    )
    assert torch.any(dataset.latent_intervals != dataset.observed_intervals)
    assert torch.all(
        dataset.factual_mask ^ dataset.factual_mask[dataset.counterfactual_indices]
    )

    allowed_difference = torch.zeros(
        dataset.sequence_duration, dataset.input_dim, dtype=torch.bool
    )
    allowed_difference[
        dataset.context_start_step : dataset.context_start_step
        + dataset.context_duration,
        2:4,
    ] = True
    for sample_index in range(dataset.n_samples):
        partner_index = int(dataset.counterfactual_indices[sample_index])
        if sample_index > partner_index:
            continue
        pair = torch.tensor((sample_index, partner_index))
        assert set(dataset.context_indices[pair].tolist()) == {0, 1}
        for field in (
            "source_prior_indices",
            "source_support_indices",
            "latent_intervals",
            "observed_interval_indices",
            "observed_intervals",
            "ready_steps",
            "set_steps",
            "cue_gains",
            "distractor_mask",
        ):
            values = getattr(dataset, field)
            assert torch.equal(values[sample_index], values[partner_index])

        difference = dataset.inputs[sample_index] != dataset.inputs[partner_index]
        assert not torch.any(difference & ~allowed_difference)
        assert torch.any(difference & allowed_difference)
        assert not torch.equal(
            dataset.posterior_probabilities[sample_index],
            dataset.posterior_probabilities[partner_index],
        )
        assert not torch.equal(
            dataset.targets[sample_index], dataset.targets[partner_index]
        )

        # There is no Go channel or event: both event channels are exactly the
        # shared noisy Ready--Set measurement, never the posterior response.
        for member in pair.tolist():
            map_go = int(dataset.posterior_map_go_steps[member])
            assert torch.equal(
                dataset.inputs[member, map_go, :2],
                dataset.inputs[partner_index, map_go, :2],
            )


def test_bayesian_rsg_stores_exact_posterior_and_temporal_mixture():
    dataset = _small_dataset(input_noise_std=0.0, mean_distractors_per_trial=0.0)
    sample_index = 3
    context = int(dataset.context_indices[sample_index])
    observed_index = int(dataset.observed_interval_indices[sample_index])
    expected_posterior = discrete_bayesian_interval_posterior(
        dataset.prior_probabilities[context],
        dataset.measurement_likelihood,
        observed_index,
    )
    assert torch.equal(
        dataset.posterior_probabilities[sample_index], expected_posterior
    )

    intervals = dataset.latent_interval_values.to(torch.float64)
    expected_mean = torch.dot(expected_posterior, intervals)
    expected_variance = torch.dot(
        expected_posterior, (intervals - expected_mean).square()
    )
    assert dataset.posterior_mean_intervals[sample_index] == expected_mean
    assert dataset.posterior_interval_variances[sample_index] == expected_variance

    target = torch.zeros(dataset.sequence_duration, dtype=torch.float64)
    offsets = torch.arange(
        -dataset.target_radius, dataset.target_radius + 1, dtype=torch.float64
    )
    component = torch.exp(-0.5 * (offsets / dataset.target_sigma).square())
    component /= component.sum()
    set_step = int(dataset.set_steps[sample_index])
    for interval, mass in zip(
        dataset.latent_interval_values.tolist(), expected_posterior, strict=True
    ):
        center = set_step + round(dataset.production_scale * interval)
        target[center - dataset.target_radius : center + dataset.target_radius + 1] += (
            mass * component
        )
        assert torch.equal(
            dataset.inputs[sample_index, center], torch.zeros(dataset.input_dim)
        )
    assert torch.allclose(
        dataset.targets[sample_index].to(torch.float64),
        target,
        atol=2e-8,
        rtol=1e-6,
    )


def test_bayesian_rsg_is_locally_seeded_reproducible_and_balanced():
    first = _small_dataset(n_samples=42, seed=901)
    torch.manual_seed(999999)
    second = _small_dataset(n_samples=42, seed=901)
    different = _small_dataset(n_samples=42, seed=902)
    for field in (
        "inputs",
        "targets",
        "pair_ids",
        "counterfactual_indices",
        "context_indices",
        "source_prior_indices",
        "latent_intervals",
        "observed_intervals",
        "posterior_probabilities",
    ):
        assert torch.equal(getattr(first, field), getattr(second, field))
    assert not torch.equal(first.inputs, different.inputs)

    assert torch.equal(
        torch.bincount(first.context_indices, minlength=2),
        torch.tensor((21, 21)),
    )
    # Pair-level source priors differ by at most one.
    representatives = torch.nonzero(
        torch.arange(first.n_samples) < first.counterfactual_indices,
        as_tuple=False,
    ).flatten()
    source_counts = torch.bincount(
        first.source_prior_indices[representatives], minlength=2
    )
    assert int(source_counts.max() - source_counts.min()) <= 1
    assert torch.equal(
        torch.bincount(first.pair_ids, minlength=first.n_counterfactual_pairs),
        torch.full((first.n_counterfactual_pairs,), 2),
    )
    set_counts = torch.bincount(
        first.set_steps[representatives] - first.set_min_step,
        minlength=first.set_max_step - first.set_min_step + 1,
    )
    assert int(set_counts.max() - set_counts.min()) <= 1

    # Largest-remainder stratification keeps source support counts within one
    # trial of their declared expectation.
    for source_prior in range(2):
        source_rows = representatives[
            first.source_prior_indices[representatives] == source_prior
        ]
        observed_counts = torch.bincount(
            first.source_support_indices[source_rows],
            minlength=len(first.prior_supports[source_prior]),
        ).to(torch.float64)
        probabilities = torch.tensor(
            first.prior_support_probabilities[source_prior], dtype=torch.float64
        )
        expected_counts = probabilities * source_rows.numel()
        assert torch.all(torch.abs(observed_counts - expected_counts) < 1.0)


def test_bayesian_rsg_registry_and_deferred_split_keep_pairs_together():
    train, valid, test = get_sequence_datasets(
        "bayesian_ready_set_go_positive",
        n_samples=20,
        train_valid_split=0.8,
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
        split_seed=81,
    )
    assert isinstance(train, Subset)
    assert isinstance(valid, Subset)
    assert len(train) == 16
    assert len(valid) == 4
    assert len(test) == 2000
    assert train.dataset is valid.dataset
    full_dataset = train.dataset
    assert (full_dataset.seed, test.seed) == (71, 72)
    for indices in (set(train.indices), set(valid.indices)):
        for sample_index in indices:
            assert int(full_dataset.counterfactual_indices[sample_index]) in indices


def test_bayesian_rsg_builder_rejects_unencoded_prior_shift():
    with pytest.raises(ValueError, match="categorical prior cues"):
        get_sequence_datasets(
            "bayesian_ready_set_go_positive",
            n_samples=20,
            train={"n_samples": 20},
            test={
                "n_samples": 10,
                "prior_supports": ((10, 12, 14), (18, 20, 22)),
            },
        )


def test_bayesian_rsg_builder_rejects_unimplemented_standard_processing():
    with pytest.raises(ValueError, match="normalize=false"):
        get_sequence_datasets(
            "bayesian_ready_set_go_positive",
            n_samples=20,
            normalize=True,
        )


def test_bayesian_rsg_explicit_splits_require_distinct_seeds():
    with pytest.raises(ValueError, match="seeds must be distinct"):
        get_sequence_datasets(
            "bayesian_ready_set_go_positive",
            n_samples=20,
            train={"n_samples": 20, "seed": 71},
            valid={"n_samples": 10, "seed": 71},
            test={"n_samples": 10, "seed": 73},
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n_samples": 3}, "even"),
        ({"prior_probabilities": ((0.5, 0.5), (0.25, 0.5, 0.25))}, "match"),
        ({"observed_interval_min": 1, "pulse_duration": 2}, "overlap"),
        ({"context_duration": 8}, "earliest"),
        ({"sequence_duration": 50}, "too short"),
    ],
)
def test_bayesian_rsg_rejects_invalid_probability_or_geometry(overrides, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _small_dataset(**overrides)
