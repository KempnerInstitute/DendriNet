import math

import pytest
import torch

from dendritic_modeling.analysis.utils.cascade_response import (
    decay_from_tau,
    equal_tau_serial_closed_form,
    first_order_impulse,
    frequency_response,
    normalized_nonnormality,
    parallel_bank_impulse,
    serial_cascade_impulse,
    serial_cascade_state_space,
    summarize_kernel,
)


def test_equal_tau_closed_form_matches_direct_serial_update():
    direct = serial_cascade_impulse([12.0, 12.0, 12.0], n_steps=300)
    closed = equal_tau_serial_closed_form(tau=12.0, depth=3, n_steps=300)
    assert torch.allclose(direct, closed, atol=1e-14, rtol=1e-12)


def test_serial_moments_follow_negative_binomial_law():
    tau = 9.0
    depth = 4
    rho = decay_from_tau(tau)
    kernel = equal_tau_serial_closed_form(
        tau=tau,
        depth=depth,
        n_steps=2000,
    )
    summary = summarize_kernel(kernel)
    expected_mean = depth * rho / (1.0 - rho)
    expected_sd = math.sqrt(depth * rho) / (1.0 - rho)
    assert summary.mass == pytest.approx(1.0, abs=1e-12)
    assert summary.mean_delay == pytest.approx(expected_mean, rel=1e-10)
    assert summary.standard_deviation == pytest.approx(expected_sd, rel=1e-10)
    assert summary.peak_index > 0


def test_positive_parallel_bank_is_monotone_but_serial_path_peaks_late():
    taus = [20.0, 20.0, 20.0]
    parallel = parallel_bank_impulse(taus, n_steps=300)
    serial = serial_cascade_impulse(taus, n_steps=300)
    assert torch.all(parallel[1:] <= parallel[:-1])
    assert summarize_kernel(parallel).peak_index == 0
    assert summarize_kernel(serial).peak_index > 0


def test_serial_state_space_matches_impulse_and_is_stable_but_nonnormal():
    taus = [5.0, 12.0, 20.0]
    realization = serial_cascade_state_space(taus)
    state = torch.zeros(len(taus), dtype=torch.float64)
    response = []
    for timestep in range(300):
        impulse = 1.0 if timestep == 0 else 0.0
        state = realization.transition @ state + realization.input_vector * impulse
        response.append(realization.readout_vector @ state)
    direct = serial_cascade_impulse(taus, n_steps=300)

    assert torch.allclose(torch.stack(response), direct, atol=1e-14, rtol=1e-12)
    eigenvalues = torch.linalg.eigvals(realization.transition).real.sort().values
    expected = torch.tensor(
        sorted(decay_from_tau(tau) for tau in taus), dtype=torch.float64
    )
    assert torch.allclose(eigenvalues, expected, atol=1e-12, rtol=1e-12)
    assert float(eigenvalues.max()) < 1.0
    assert normalized_nonnormality(realization.transition) > 0.0
    assert normalized_nonnormality(torch.eye(3, dtype=torch.float64)) == 0.0


def test_frequency_response_has_unit_dc_gain_and_matches_fft_convention():
    omega = torch.tensor([0.0, 0.1, 0.5], dtype=torch.float64)
    serial = frequency_response([5.0, 15.0], omega, topology="serial")
    parallel = frequency_response([5.0, 15.0], omega, topology="parallel")
    assert serial[0] == pytest.approx(1.0 + 0.0j)
    assert parallel[0] == pytest.approx(1.0 + 0.0j)
    assert torch.all(serial.abs()[1:] < 1.0)
    assert torch.all(parallel.abs()[1:] < 1.0)


@pytest.mark.parametrize(
    "call",
    (
        lambda: first_order_impulse(2.0, n_steps=4, dtype=torch.int64),
        lambda: serial_cascade_impulse([2.0], n_steps=4, dtype=torch.int64),
        lambda: serial_cascade_state_space([2.0], dtype=torch.int64),
        lambda: equal_tau_serial_closed_form(
            tau=2.0, depth=2, n_steps=4, dtype=torch.int64
        ),
        lambda: parallel_bank_impulse([2.0], n_steps=4, dtype=torch.int64),
    ),
)
def test_cascade_constructors_reject_non_floating_output_dtypes(call):
    with pytest.raises(TypeError, match=r"floating-point torch\.dtype"):
        call()


@pytest.mark.parametrize(
    ("frequencies", "error", "message"),
    (
        (torch.tensor([0, 1]), TypeError, "floating-point dtype"),
        (torch.tensor([0.0 + 0.0j]), TypeError, "floating-point dtype"),
        (torch.tensor([0.0, float("nan")]), ValueError, "finite"),
        (torch.tensor([0.0, float("inf")]), ValueError, "finite"),
    ),
)
def test_frequency_response_rejects_unsupported_or_nonfinite_frequencies(
    frequencies,
    error,
    message,
):
    with pytest.raises(error, match=message):
        frequency_response([2.0], frequencies, topology="serial")


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: decay_from_tau(0.0), "positive"),
        (lambda: first_order_impulse(1.0, n_steps=0), "positive integer"),
        (lambda: serial_cascade_impulse([], n_steps=4), "at least one"),
        (
            lambda: parallel_bank_impulse([1.0, 2.0], n_steps=4, weights=[1.0, -1.0]),
            "non-negative",
        ),
        (
            lambda: frequency_response([1.0], torch.tensor([0.0]), topology="other"),
            "topology",
        ),
        (lambda: normalized_nonnormality(torch.zeros(2, 3)), "square"),
    ],
)
def test_invalid_cascade_response_inputs_fail_closed(call, message):
    with pytest.raises((TypeError, ValueError), match=message):
        call()
