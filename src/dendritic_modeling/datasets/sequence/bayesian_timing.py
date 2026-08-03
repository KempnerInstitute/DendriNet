"""Bayesian interval-reproduction sequence tasks.

This module implements a finite, fully specified observer model for a
two-prior Ready--Set--Go task.  Its default supports are the 480--800 ms and
800--1200 ms priors of Sohn et al. (2019), expressed in 20 ms steps.  The
latent interval and its noisy integer
measurement are generated from the same discrete Weber likelihood used to
construct the training target.  This keeps the target an auditable posterior
predictive distribution rather than a hand-selected event label.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral, Real

import torch
from torch.utils.data import Dataset


def _positive_integer_tuple(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    if len(values) == 0:
        raise ValueError(f"{name} must not be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in values
    ):
        raise TypeError(f"{name} must contain integers")
    result = tuple(int(value) for value in values)
    if any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive integers")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _finite_nonnegative(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _validated_prior_probabilities(
    prior_supports: tuple[tuple[int, ...], tuple[int, ...]],
    prior_probabilities: Sequence[Sequence[float]] | None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if prior_probabilities is None:
        first = tuple(1.0 / len(prior_supports[0]) for _ in prior_supports[0])
        second = tuple(1.0 / len(prior_supports[1]) for _ in prior_supports[1])
        return first, second
    if len(prior_probabilities) != 2:
        raise ValueError("prior_probabilities must describe exactly two priors")

    validated: list[tuple[float, ...]] = []
    for prior_index, (support, probabilities) in enumerate(
        zip(prior_supports, prior_probabilities, strict=True)
    ):
        if len(probabilities) != len(support):
            raise ValueError(
                "each prior probability vector must match its support; "
                f"prior {prior_index} has {len(support)} support values and "
                f"{len(probabilities)} probabilities"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, Real)
            for value in probabilities
        ):
            raise TypeError("prior probabilities must be real numbers")
        values = tuple(float(value) for value in probabilities)
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("prior probabilities must be finite and strictly positive")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("each prior probability vector must sum to one")
        validated.append(values)
    return validated[0], validated[1]


def default_observed_interval_bounds(
    prior_supports: Sequence[Sequence[int]],
    *,
    weber_fraction: float,
    measurement_noise_floor: float = 0.0,
    likelihood_tail_sigma: float = 4.0,
    minimum_interval: int = 1,
) -> tuple[int, int]:
    """Return integer measurement bounds covering each latent by a fixed tail.

    The returned bounds only define the finite sample space.  The likelihood
    is normalized on that declared space, so posterior inference remains exact
    even when a deliberately narrower range is configured.
    """

    supports = tuple(
        _positive_integer_tuple(support, name="prior support")
        for support in prior_supports
    )
    if len(supports) != 2:
        raise ValueError("prior_supports must describe exactly two priors")
    weber = _finite_nonnegative(weber_fraction, name="weber_fraction")
    floor = _finite_nonnegative(measurement_noise_floor, name="measurement_noise_floor")
    if (
        isinstance(likelihood_tail_sigma, bool)
        or not isinstance(likelihood_tail_sigma, Real)
        or not math.isfinite(float(likelihood_tail_sigma))
        or float(likelihood_tail_sigma) <= 0.0
    ):
        raise ValueError("likelihood_tail_sigma must be finite and positive")
    if isinstance(minimum_interval, bool) or not isinstance(minimum_interval, Integral):
        raise TypeError("minimum_interval must be an integer")
    if minimum_interval <= 0:
        raise ValueError("minimum_interval must be positive")

    tail = float(likelihood_tail_sigma)
    latent_values = tuple(value for support in supports for value in support)
    lower = min(
        latent - tail * math.hypot(weber * latent, floor) for latent in latent_values
    )
    upper = max(
        latent + tail * math.hypot(weber * latent, floor) for latent in latent_values
    )
    return max(int(minimum_interval), math.floor(lower)), math.ceil(upper)


def discrete_weber_likelihood(
    latent_intervals: Sequence[int] | torch.Tensor,
    observed_intervals: Sequence[int] | torch.Tensor,
    *,
    weber_fraction: float,
    measurement_noise_floor: float = 0.0,
) -> torch.Tensor:
    """Construct ``P(observed | latent)`` on a declared integer sample space.

    For nonzero noise the unnormalized mass is a Gaussian kernel with
    ``sigma(latent)^2 = (weber_fraction * latent)^2 + noise_floor^2``.
    Each latent row is normalized over the supplied observation values.  Zero
    total noise is the exact identity channel and therefore requires every
    latent value to occur in the observation support.
    """

    latent = torch.as_tensor(latent_intervals, dtype=torch.float64)
    observed = torch.as_tensor(observed_intervals, dtype=torch.float64)
    if latent.ndim != 1 or latent.numel() == 0:
        raise ValueError("latent_intervals must be a non-empty one-dimensional array")
    if observed.ndim != 1 or observed.numel() == 0:
        raise ValueError("observed_intervals must be a non-empty one-dimensional array")
    for values, name in (
        (latent, "latent_intervals"),
        (observed, "observed_intervals"),
    ):
        if not torch.isfinite(values).all() or torch.any(values <= 0.0):
            raise ValueError(f"{name} must contain finite positive values")
        if not torch.equal(values, values.round()):
            raise TypeError(f"{name} must contain integers")
        if torch.unique(values).numel() != values.numel():
            raise ValueError(f"{name} must not contain duplicates")

    weber = _finite_nonnegative(weber_fraction, name="weber_fraction")
    floor = _finite_nonnegative(measurement_noise_floor, name="measurement_noise_floor")
    scales = torch.hypot(weber * latent, torch.full_like(latent, floor))
    zero_scale = scales == 0.0
    if torch.any(zero_scale):
        if not torch.all(zero_scale):
            raise ValueError(
                "likelihood scales must be either all zero or all positive"
            )
        matches = latent[:, None] == observed[None, :]
        if not torch.all(matches.any(dim=1)):
            raise ValueError(
                "zero-noise likelihood requires every latent interval in the "
                "observation support"
            )
        return matches.to(torch.float64)

    log_mass = -0.5 * ((observed[None, :] - latent[:, None]) / scales[:, None]).square()
    return torch.softmax(log_mass, dim=1)


def discrete_bayesian_interval_posterior(
    prior_probabilities: Sequence[float] | torch.Tensor,
    likelihood: torch.Tensor,
    observed_index: int,
) -> torch.Tensor:
    """Return an exact posterior over a finite latent interval support."""

    prior = torch.as_tensor(prior_probabilities, dtype=torch.float64)
    likelihood = torch.as_tensor(likelihood, dtype=torch.float64)
    if prior.ndim != 1 or prior.numel() == 0:
        raise ValueError("prior_probabilities must be a non-empty vector")
    if likelihood.ndim != 2 or likelihood.shape[0] != prior.numel():
        raise ValueError("likelihood rows must match prior_probabilities")
    if isinstance(observed_index, bool) or not isinstance(observed_index, Integral):
        raise TypeError("observed_index must be an integer")
    if not 0 <= int(observed_index) < likelihood.shape[1]:
        raise IndexError("observed_index is outside the likelihood support")
    if not torch.isfinite(prior).all() or torch.any(prior < 0.0):
        raise ValueError("prior_probabilities must be finite and non-negative")
    if not torch.isfinite(likelihood).all() or torch.any(likelihood < 0.0):
        raise ValueError("likelihood must be finite and non-negative")
    if not torch.isclose(prior.sum(), torch.tensor(1.0, dtype=torch.float64)):
        raise ValueError("prior_probabilities must sum to one")
    if not torch.allclose(
        likelihood.sum(dim=1),
        torch.ones(likelihood.shape[0], dtype=torch.float64),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("each likelihood row must sum to one")

    joint = prior * likelihood[:, int(observed_index)]
    evidence = joint.sum()
    if not torch.isfinite(evidence) or evidence <= 0.0:
        raise ValueError("the observation has zero probability under the prior")
    return joint / evidence


def _largest_remainder_counts(
    n_items: int,
    probabilities: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    expected = probabilities * n_items
    counts = torch.floor(expected).to(torch.long)
    remainder = n_items - int(counts.sum())
    if remainder:
        # Seeded tie-breaking avoids a permanent preference for low indices.
        tie_order = torch.randperm(probabilities.numel(), generator=generator)
        ranked = tie_order[
            torch.argsort(expected[tie_order] - counts[tie_order], descending=True)
        ]
        counts[ranked[:remainder]] += 1
    return counts


def _stratified_categories(
    n_items: int,
    probabilities: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    if n_items == 0:
        return torch.empty(0, dtype=torch.long)
    counts = _largest_remainder_counts(n_items, probabilities, generator=generator)
    categories = torch.repeat_interleave(torch.arange(probabilities.numel()), counts)
    return categories[torch.randperm(n_items, generator=generator)]


class BayesianReadySetGoDataset(Dataset):
    """Counterfactually paired Bayesian interval reproduction.

    A pair shares a source prior, latent interval, noisy Ready--Set
    measurement, absolute cue times, gain, distractors, and additive noise.
    Its two members differ only in an early one-hot short/long prior cue and in
    the resulting posterior-predictive Go target.  Inputs contain five
    channels: Ready, Set, short-prior context, long-prior context, and a
    task-irrelevant distractor channel.  There is no Go input.

    The finite observer model is fully represented by ``prior_probabilities``
    on ``latent_interval_values`` and by ``measurement_likelihood`` on
    ``observed_interval_values``.  Targets mix normalized discrete temporal
    kernels according to the exact posterior, preserving posterior uncertainty
    instead of collapsing it to a point estimate.
    """

    def __init__(
        self,
        n_samples: int = 10000,
        prior_supports: tuple[tuple[int, ...], tuple[int, ...]] = (
            (24, 28, 32, 36, 40),
            (40, 45, 50, 55, 60),
        ),
        prior_probabilities: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
        weber_fraction: float = 0.12,
        measurement_noise_floor: float = 0.0,
        observed_interval_min: int | None = None,
        observed_interval_max: int | None = None,
        likelihood_tail_sigma: float = 4.0,
        production_scale: float = 1.0,
        set_min_step: int | None = None,
        set_max_step: int | None = None,
        pulse_duration: int = 2,
        context_start_step: int = 0,
        context_duration: int = 10,
        context_gain: float = 1.0,
        target_sigma: float = 2.0,
        target_tail_sigma: float = 4.0,
        post_go_duration: int = 20,
        sequence_duration: int | None = None,
        cue_gain_min: float = 0.7,
        cue_gain_max: float = 1.3,
        input_noise_std: float = 0.05,
        mean_distractors_per_trial: float = 1.0,
        distractor_amplitude: float = 1.0,
        distractor_pulse_duration: int = 2,
        positive_input_encoding: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if (
            isinstance(n_samples, bool)
            or not isinstance(n_samples, Integral)
            or n_samples <= 0
        ):
            raise ValueError("n_samples must be a positive integer")
        if n_samples % 2:
            raise ValueError("Bayesian Ready-Set-Go requires an even n_samples")

        supports = tuple(
            _positive_integer_tuple(support, name="prior support")
            for support in prior_supports
        )
        if len(supports) != 2:
            raise ValueError("prior_supports must describe exactly two priors")
        supports = (supports[0], supports[1])
        probabilities = _validated_prior_probabilities(supports, prior_probabilities)
        union_support = tuple(sorted(set(supports[0]).union(supports[1])))

        weber = _finite_nonnegative(weber_fraction, name="weber_fraction")
        noise_floor = _finite_nonnegative(
            measurement_noise_floor, name="measurement_noise_floor"
        )
        if (
            isinstance(likelihood_tail_sigma, bool)
            or not isinstance(likelihood_tail_sigma, Real)
            or not math.isfinite(float(likelihood_tail_sigma))
            or float(likelihood_tail_sigma) <= 0.0
        ):
            raise ValueError("likelihood_tail_sigma must be finite and positive")
        derived_min, derived_max = default_observed_interval_bounds(
            supports,
            weber_fraction=weber,
            measurement_noise_floor=noise_floor,
            likelihood_tail_sigma=float(likelihood_tail_sigma),
            minimum_interval=pulse_duration,
        )
        if observed_interval_min is None:
            observed_interval_min = derived_min
        if observed_interval_max is None:
            observed_interval_max = derived_max

        integer_options = {
            "observed_interval_min": observed_interval_min,
            "observed_interval_max": observed_interval_max,
            "pulse_duration": pulse_duration,
            "context_start_step": context_start_step,
            "context_duration": context_duration,
            "post_go_duration": post_go_duration,
            "distractor_pulse_duration": distractor_pulse_duration,
            "seed": seed,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in integer_options.values()
        ):
            raise TypeError(
                "interval, step, duration, and seed options must be integers"
            )
        if observed_interval_min < pulse_duration:
            raise ValueError(
                "observed_interval_min must prevent Ready/Set pulse overlap"
            )
        if observed_interval_max < observed_interval_min:
            raise ValueError("observed interval bounds must be ordered")
        observed_values = tuple(
            range(int(observed_interval_min), int(observed_interval_max) + 1)
        )

        if not math.isfinite(production_scale) or production_scale <= 0.0:
            raise ValueError("production_scale must be finite and positive")
        if pulse_duration <= 0 or context_duration <= 0:
            raise ValueError("cue and context durations must be positive")
        if context_start_step < 0:
            raise ValueError("context_start_step must be non-negative")
        if not math.isfinite(context_gain) or context_gain <= 0.0:
            raise ValueError("context_gain must be finite and positive")
        if not math.isfinite(target_sigma) or target_sigma <= 0.0:
            raise ValueError("target_sigma must be finite and positive")
        if not math.isfinite(target_tail_sigma) or target_tail_sigma <= 0.0:
            raise ValueError("target_tail_sigma must be finite and positive")
        target_radius = math.ceil(target_sigma * target_tail_sigma)
        if post_go_duration < target_radius:
            raise ValueError(
                "post_go_duration must cover the complete temporal target kernel"
            )
        production_values = tuple(
            round(production_scale * interval) for interval in union_support
        )
        if min(production_values) - target_radius < pulse_duration:
            raise ValueError(
                "the earliest temporal target kernel overlaps the Set pulse"
            )
        if distractor_pulse_duration <= 0:
            raise ValueError("distractor_pulse_duration must be positive")

        if set_min_step is None:
            set_min_step = (
                int(observed_interval_max) + context_start_step + context_duration + 10
            )
        if set_max_step is None:
            translation_span = max(
                40, int(observed_interval_max) - int(observed_interval_min)
            )
            set_max_step = int(set_min_step) + translation_span
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in (set_min_step, set_max_step)
        ):
            raise TypeError("set_min_step and set_max_step must be integers or None")
        if set_max_step < set_min_step:
            raise ValueError("Set-time bounds must be ordered")
        earliest_ready = int(set_min_step) - int(observed_interval_max)
        if context_start_step + context_duration > earliest_ready:
            raise ValueError(
                "prior context must end no later than the earliest possible Ready pulse"
            )

        required_duration = (
            int(set_max_step) + max(production_values) + post_go_duration + 1
        )
        if sequence_duration is None:
            sequence_duration = required_duration
        if isinstance(sequence_duration, bool) or not isinstance(
            sequence_duration, Integral
        ):
            raise TypeError("sequence_duration must be an integer or None")
        if sequence_duration < required_duration:
            raise ValueError(
                "sequence_duration is too short for the complete posterior "
                f"target: need at least {required_duration}, got {sequence_duration}"
            )
        if distractor_pulse_duration > sequence_duration:
            raise ValueError("distractor pulse cannot exceed sequence duration")
        if (
            not math.isfinite(cue_gain_min)
            or not math.isfinite(cue_gain_max)
            or cue_gain_min <= 0.0
            or cue_gain_max < cue_gain_min
        ):
            raise ValueError("cue gain bounds must be positive and ordered")
        input_noise = _finite_nonnegative(input_noise_std, name="input_noise_std")
        distractor_rate = _finite_nonnegative(
            mean_distractors_per_trial, name="mean_distractors_per_trial"
        )
        distractor_gain = _finite_nonnegative(
            distractor_amplitude, name="distractor_amplitude"
        )

        self.n_samples = int(n_samples)
        self.n_counterfactual_pairs = self.n_samples // 2
        self.is_counterfactual_paired = True
        self.seed = int(seed)
        self.input_dim = 5
        self.prior_supports = supports
        self.prior_support_probabilities = probabilities
        self.latent_interval_values = torch.tensor(union_support, dtype=torch.long)
        self.observed_interval_values = torch.tensor(observed_values, dtype=torch.long)
        self.weber_fraction = weber
        self.measurement_noise_floor = noise_floor
        self.likelihood_tail_sigma = float(likelihood_tail_sigma)
        self.production_scale = float(production_scale)
        self.set_min_step = int(set_min_step)
        self.set_max_step = int(set_max_step)
        self.pulse_duration = int(pulse_duration)
        self.context_start_step = int(context_start_step)
        self.context_duration = int(context_duration)
        self.context_gain = float(context_gain)
        self.target_sigma = float(target_sigma)
        self.target_tail_sigma = float(target_tail_sigma)
        self.target_radius = int(target_radius)
        self.post_go_duration = int(post_go_duration)
        self.sequence_duration = int(sequence_duration)
        self.cue_gain_min = float(cue_gain_min)
        self.cue_gain_max = float(cue_gain_max)
        self.input_noise_std = input_noise
        self.mean_distractors_per_trial = distractor_rate
        self.distractor_amplitude = distractor_gain
        self.distractor_pulse_duration = int(distractor_pulse_duration)
        self.positive_input_encoding = bool(positive_input_encoding)

        support_lookup = {value: index for index, value in enumerate(union_support)}
        prior_matrix = torch.zeros(2, len(union_support), dtype=torch.float64)
        for prior_index, (support, prior_values) in enumerate(
            zip(supports, probabilities, strict=True)
        ):
            for interval, probability in zip(support, prior_values, strict=True):
                prior_matrix[prior_index, support_lookup[interval]] = probability
        self.prior_probabilities = prior_matrix
        self.prior_support_mask = prior_matrix > 0.0
        self.measurement_likelihood = discrete_weber_likelihood(
            self.latent_interval_values,
            self.observed_interval_values,
            weber_fraction=self.weber_fraction,
            measurement_noise_floor=self.measurement_noise_floor,
        )

        # Reject a context manipulation that would have no target consequence.
        production_groups = torch.tensor(production_values, dtype=torch.long)
        unique_productions = torch.unique(production_groups, sorted=True)
        for observed_index in range(self.observed_interval_values.numel()):
            posteriors = tuple(
                discrete_bayesian_interval_posterior(
                    self.prior_probabilities[context],
                    self.measurement_likelihood,
                    observed_index,
                )
                for context in range(2)
            )
            if torch.allclose(posteriors[0], posteriors[1], rtol=0.0, atol=1e-12):
                raise ValueError(
                    "the configured priors do not produce distinct counterfactual "
                    "posteriors throughout the observation support"
                )
            predictive_masses = tuple(
                torch.stack(
                    [
                        posterior[production_groups == production].sum()
                        for production in unique_productions
                    ]
                )
                for posterior in posteriors
            )
            if torch.allclose(
                predictive_masses[0],
                predictive_masses[1],
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "production_scale collapses the counterfactual posteriors "
                    "to the same temporal target"
                )

        self.inputs = torch.zeros(
            self.n_samples, self.sequence_duration, self.input_dim
        )
        self.targets = torch.zeros(self.n_samples, self.sequence_duration)
        self.context_indices = torch.zeros(self.n_samples, dtype=torch.long)
        self.source_prior_indices = torch.zeros(self.n_samples, dtype=torch.long)
        self.source_support_indices = torch.zeros(self.n_samples, dtype=torch.long)
        self.latent_intervals = torch.zeros(self.n_samples, dtype=torch.long)
        self.observed_interval_indices = torch.zeros(self.n_samples, dtype=torch.long)
        self.observed_intervals = torch.zeros(self.n_samples, dtype=torch.long)
        self.ready_steps = torch.zeros(self.n_samples, dtype=torch.long)
        self.set_steps = torch.zeros(self.n_samples, dtype=torch.long)
        self.cue_gains = torch.zeros(self.n_samples)
        self.posterior_probabilities = torch.zeros(
            self.n_samples, len(union_support), dtype=torch.float64
        )
        self.posterior_mean_intervals = torch.zeros(self.n_samples, dtype=torch.float64)
        self.posterior_interval_variances = torch.zeros(
            self.n_samples, dtype=torch.float64
        )
        self.posterior_mean_production_intervals = torch.zeros(
            self.n_samples, dtype=torch.float64
        )
        self.posterior_production_variances = torch.zeros(
            self.n_samples, dtype=torch.float64
        )
        self.posterior_mean_go_steps = torch.zeros(self.n_samples, dtype=torch.float64)
        self.posterior_map_intervals = torch.zeros(self.n_samples, dtype=torch.long)
        self.posterior_map_go_steps = torch.zeros(self.n_samples, dtype=torch.long)
        self.factual_mask = torch.zeros(self.n_samples, dtype=torch.bool)
        self.distractor_mask = torch.zeros(
            self.n_samples, self.sequence_duration, dtype=torch.bool
        )
        self.pair_ids = torch.full((self.n_samples,), -1, dtype=torch.long)
        self.counterfactual_indices = torch.full(
            (self.n_samples,), -1, dtype=torch.long
        )

        generator = torch.Generator().manual_seed(self.seed)
        n_pairs = self.n_counterfactual_pairs
        source_prior_schedule = torch.arange(n_pairs) % 2
        source_prior_schedule = source_prior_schedule[
            torch.randperm(n_pairs, generator=generator)
        ]
        source_support_schedule = torch.full((n_pairs,), -1, dtype=torch.long)
        latent_schedule = torch.full((n_pairs,), -1, dtype=torch.long)
        for source_prior in range(2):
            pair_indices = torch.nonzero(
                source_prior_schedule == source_prior, as_tuple=False
            ).flatten()
            source_probabilities = torch.tensor(
                probabilities[source_prior], dtype=torch.float64
            )
            categories = _stratified_categories(
                pair_indices.numel(),
                source_probabilities,
                generator=generator,
            )
            source_support_schedule[pair_indices] = categories
            source_values = torch.tensor(supports[source_prior], dtype=torch.long)
            latent_schedule[pair_indices] = source_values[categories]

        observed_schedule = torch.zeros(n_pairs, dtype=torch.long)
        for pair_id in range(n_pairs):
            latent_index = support_lookup[int(latent_schedule[pair_id])]
            observed_schedule[pair_id] = torch.multinomial(
                self.measurement_likelihood[latent_index],
                1,
                generator=generator,
            )[0]
        set_schedule = torch.arange(n_pairs) % (
            self.set_max_step - self.set_min_step + 1
        )
        set_schedule = (
            set_schedule[torch.randperm(n_pairs, generator=generator)]
            + self.set_min_step
        )
        pair_slots = torch.randperm(self.n_samples, generator=generator).reshape(
            n_pairs, 2
        )

        latent_values_float = self.latent_interval_values.to(torch.float64)
        production_values_tensor = torch.tensor(production_values, dtype=torch.float64)
        target_offsets = torch.arange(
            -self.target_radius, self.target_radius + 1, dtype=torch.float64
        )
        component_shape = torch.exp(
            -0.5 * (target_offsets / self.target_sigma).square()
        )
        component_shape /= component_shape.sum()

        for pair_id in range(n_pairs):
            source_prior = int(source_prior_schedule[pair_id])
            source_support = int(source_support_schedule[pair_id])
            latent_interval = int(latent_schedule[pair_id])
            observed_index = int(observed_schedule[pair_id])
            observed_interval = observed_values[observed_index]
            set_step = int(set_schedule[pair_id])
            ready_step = set_step - observed_interval
            cue_gain = float(
                torch.empty(1)
                .uniform_(
                    self.cue_gain_min,
                    self.cue_gain_max,
                    generator=generator,
                )
                .item()
            )

            base_input = torch.zeros(self.sequence_duration, self.input_dim)
            base_input[ready_step : ready_step + self.pulse_duration, 0] = cue_gain
            base_input[set_step : set_step + self.pulse_duration, 1] = cue_gain
            base_distractor_mask = torch.zeros(self.sequence_duration, dtype=torch.bool)
            n_distractors = int(
                torch.poisson(
                    torch.tensor(self.mean_distractors_per_trial),
                    generator=generator,
                ).item()
            )
            if n_distractors > 0 and self.distractor_amplitude > 0.0:
                latest_start = self.sequence_duration - self.distractor_pulse_duration
                for _ in range(n_distractors):
                    start = int(
                        torch.randint(
                            0,
                            latest_start + 1,
                            (1,),
                            generator=generator,
                        ).item()
                    )
                    stop = start + self.distractor_pulse_duration
                    base_input[start:stop, 4] += self.distractor_amplitude
                    base_distractor_mask[start:stop] = True

            shared_noise = None
            if self.input_noise_std > 0.0:
                shared_noise = self.input_noise_std * torch.randn(
                    self.sequence_duration,
                    self.input_dim,
                    generator=generator,
                )

            first_slot, second_slot = (int(slot) for slot in pair_slots[pair_id])
            for context_index, sample_index in enumerate((first_slot, second_slot)):
                sample_input = base_input.clone()
                sample_input[
                    self.context_start_step : self.context_start_step
                    + self.context_duration,
                    2 + context_index,
                ] = self.context_gain
                if shared_noise is not None:
                    sample_input += shared_noise
                if self.positive_input_encoding:
                    sample_input.clamp_(min=0.0)

                posterior = discrete_bayesian_interval_posterior(
                    self.prior_probabilities[context_index],
                    self.measurement_likelihood,
                    observed_index,
                )
                posterior_mean = torch.dot(posterior, latent_values_float)
                posterior_variance = torch.dot(
                    posterior, (latent_values_float - posterior_mean).square()
                )
                production_mean = torch.dot(posterior, production_values_tensor)
                production_variance = torch.dot(
                    posterior,
                    (production_values_tensor - production_mean).square(),
                )
                map_index = int(torch.argmax(posterior))

                target = torch.zeros(self.sequence_duration, dtype=torch.float64)
                for latent_index, posterior_mass in enumerate(posterior):
                    if posterior_mass == 0.0:
                        continue
                    center = set_step + production_values[latent_index]
                    start = center - self.target_radius
                    stop = center + self.target_radius + 1
                    target[start:stop] += posterior_mass * component_shape

                self.inputs[sample_index] = sample_input
                self.targets[sample_index] = target.to(torch.float32)
                self.context_indices[sample_index] = context_index
                self.source_prior_indices[sample_index] = source_prior
                self.source_support_indices[sample_index] = source_support
                self.latent_intervals[sample_index] = latent_interval
                self.observed_interval_indices[sample_index] = observed_index
                self.observed_intervals[sample_index] = observed_interval
                self.ready_steps[sample_index] = ready_step
                self.set_steps[sample_index] = set_step
                self.cue_gains[sample_index] = cue_gain
                self.posterior_probabilities[sample_index] = posterior
                self.posterior_mean_intervals[sample_index] = posterior_mean
                self.posterior_interval_variances[sample_index] = posterior_variance
                self.posterior_mean_production_intervals[sample_index] = production_mean
                self.posterior_production_variances[sample_index] = production_variance
                self.posterior_mean_go_steps[sample_index] = set_step + production_mean
                self.posterior_map_intervals[sample_index] = union_support[map_index]
                self.posterior_map_go_steps[sample_index] = (
                    set_step + production_values[map_index]
                )
                self.factual_mask[sample_index] = context_index == source_prior
                self.distractor_mask[sample_index] = base_distractor_mask
                self.pair_ids[sample_index] = pair_id

            self.counterfactual_indices[first_slot] = second_slot
            self.counterfactual_indices[second_slot] = first_slot

        if not torch.allclose(
            self.targets.sum(dim=1),
            torch.ones(self.n_samples),
            rtol=0.0,
            atol=2e-6,
        ):
            raise RuntimeError("posterior temporal targets failed normalization")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int):
        return self.inputs[index], self.targets[index]


__all__ = [
    "BayesianReadySetGoDataset",
    "default_observed_interval_bounds",
    "discrete_bayesian_interval_posterior",
    "discrete_weber_likelihood",
]
