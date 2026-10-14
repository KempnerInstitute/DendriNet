import pytest
import torch

from dendritic_modeling.analysis.utils.shunting_invariance import (
    global_log_gain_gradient,
    global_shunting_output,
    local_log_gain_jacobian,
    local_shunting_outputs,
    shunting_mixed_ei_curvature,
)


def test_analytic_gain_derivatives_match_autograd() -> None:
    signal = torch.tensor([0.4, 1.1, 2.0], dtype=torch.float64)
    conductance = torch.tensor([0.7, 1.4, 0.9], dtype=torch.float64)
    log_gains = torch.tensor([-0.2, 0.3, 0.8], dtype=torch.float64, requires_grad=True)
    gains = log_gains.exp()

    local = local_shunting_outputs(signal, conductance, gains, baseline=0.6)
    local_rows = []
    for index in range(local.numel()):
        local_rows.append(
            torch.autograd.grad(local[index], log_gains, retain_graph=True)[0]
        )
    torch.testing.assert_close(
        torch.stack(local_rows),
        local_log_gain_jacobian(signal, conductance, gains, baseline=0.6),
    )

    global_output = global_shunting_output(signal, conductance, gains, baseline=0.6)
    actual_global = torch.autograd.grad(global_output, log_gains)[0]
    torch.testing.assert_close(
        actual_global,
        global_log_gain_gradient(signal, conductance, gains, baseline=0.6),
    )


def test_local_dividers_remove_independent_high_gain_directions() -> None:
    signal = torch.tensor([0.5, 1.5, 2.5, 0.8], dtype=torch.float64)
    conductance = torch.tensor([1.0, 0.7, 1.3, 0.9], dtype=torch.float64)
    moderate = torch.ones(4, dtype=torch.float64)
    high = torch.full((4,), 1e7, dtype=torch.float64)

    moderate_local = local_log_gain_jacobian(signal, conductance, moderate)
    high_local = local_log_gain_jacobian(signal, conductance, high)
    high_global = global_log_gain_gradient(signal, conductance, high)

    assert float(torch.linalg.matrix_norm(high_local)) < 1e-6 * float(
        torch.linalg.matrix_norm(moderate_local)
    )
    assert float(torch.linalg.vector_norm(high_global)) > 1e-2
    assert float(torch.abs(high_global.sum())) < 1e-7


def test_shunting_mixed_curvature_matches_second_autograd_derivative() -> None:
    excitation = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    inhibition = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    voltage = excitation / (1.0 + excitation + inhibition)
    derivative_n = torch.autograd.grad(voltage, excitation, create_graph=True)[0]
    actual = torch.autograd.grad(derivative_n, inhibition)[0]
    expected = shunting_mixed_ei_curvature(excitation, inhibition)
    torch.testing.assert_close(actual, expected)
    assert float(expected) != 0.0


@pytest.mark.parametrize(
    "call",
    (
        lambda: local_shunting_outputs(torch.ones(2), torch.ones(3), torch.ones(2)),
        lambda: local_shunting_outputs(
            torch.ones(2), torch.ones(2), torch.tensor([1.0, 0.0])
        ),
        lambda: global_shunting_output(
            torch.ones(2), torch.ones(2), torch.ones(2), baseline=-1.0
        ),
    ),
)
def test_invalid_shunting_invariance_inputs_fail_closed(call) -> None:
    with pytest.raises((TypeError, ValueError)):
        call()
