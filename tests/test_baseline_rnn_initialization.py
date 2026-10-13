"""Regression tests for opt-in baseline recurrent initialization."""

import math

import pytest
import torch
from torch import nn

from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.networks.architectures.recurrent import (
    BaselineRNN,
    BaselineRNNConfig,
)


def _assert_state_dicts_equal(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> None:
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0.0, atol=0.0)


def test_default_gru_initialization_exactly_matches_pytorch() -> None:
    """The new all-zero defaults must not alter PyTorch's draw or parameters."""
    torch.manual_seed(173)
    expected = nn.GRU(
        input_size=3,
        hidden_size=5,
        num_layers=2,
        batch_first=True,
        dropout=0.2,
    )

    torch.manual_seed(173)
    actual = BaselineRNN(
        BaselineRNNConfig(
            cell_type="gru",
            input_dim=3,
            hidden_dim=5,
            num_layers=2,
            dropout=0.2,
        )
    )

    _assert_state_dicts_equal(actual.rnn.state_dict(), expected.state_dict())


def test_gru_fast_slow_biases_are_exact_targets_in_every_layer() -> None:
    hidden_dim = 6
    common = {
        "cell_type": "gru",
        "input_dim": 4,
        "hidden_dim": hidden_dim,
        "num_layers": 2,
    }
    torch.manual_seed(191)
    reference = BaselineRNN(BaselineRNNConfig(**common))
    torch.manual_seed(191)
    initialized = BaselineRNN(
        BaselineRNNConfig(
            **common,
            gru_fast_update_bias=-0.25,
            gru_slow_update_bias=5.75,
            gru_slow_fraction=0.5,
        )
    )

    expected_update_bias = torch.tensor([-0.25] * 3 + [5.75] * 3)
    reference_parameters = dict(reference.rnn.named_parameters())
    for name, parameter in initialized.rnn.named_parameters():
        reference_parameter = reference_parameters[name]
        if name.startswith("bias_ih_"):
            torch.testing.assert_close(
                parameter[:hidden_dim], reference_parameter[:hidden_dim]
            )
            torch.testing.assert_close(
                parameter[hidden_dim : 2 * hidden_dim], expected_update_bias
            )
            torch.testing.assert_close(
                parameter[2 * hidden_dim :], reference_parameter[2 * hidden_dim :]
            )
        elif name.startswith("bias_hh_"):
            torch.testing.assert_close(
                parameter[:hidden_dim], reference_parameter[:hidden_dim]
            )
            torch.testing.assert_close(
                parameter[hidden_dim : 2 * hidden_dim], torch.zeros(hidden_dim)
            )
            torch.testing.assert_close(
                parameter[2 * hidden_dim :], reference_parameter[2 * hidden_dim :]
            )
        else:
            torch.testing.assert_close(parameter, reference_parameter)

    for layer in range(2):
        effective_update_bias = (
            getattr(initialized.rnn, f"bias_ih_l{layer}")[hidden_dim : 2 * hidden_dim]
            + getattr(initialized.rnn, f"bias_hh_l{layer}")[hidden_dim : 2 * hidden_dim]
        )
        torch.testing.assert_close(effective_update_bias, expected_update_bias)


def test_nonzero_fraction_activates_zero_target_initialization() -> None:
    """A nonzero fraction is itself opt-in, even when both targets are zero."""
    network = BaselineRNN(
        BaselineRNNConfig(
            cell_type="gru",
            input_dim=2,
            hidden_dim=4,
            gru_slow_fraction=0.5,
        )
    )

    torch.testing.assert_close(network.rnn.bias_ih_l0[4:8], torch.zeros(4))
    torch.testing.assert_close(network.rnn.bias_hh_l0[4:8], torch.zeros(4))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("gru_fast_update_bias", math.inf, "gru_fast_update_bias must be finite"),
        ("gru_slow_update_bias", math.nan, "gru_slow_update_bias must be finite"),
        ("gru_slow_fraction", -0.01, "inclusive range"),
        ("gru_slow_fraction", 1.01, "inclusive range"),
    ],
)
def test_gru_update_bias_options_reject_invalid_values(
    field: str, value: float, message: str
) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError, match=message):
        BaselineRNN(BaselineRNNConfig(**kwargs))


@pytest.mark.parametrize("cell_type", ["lstm", "vanilla"])
def test_non_gru_rejects_active_gru_update_bias_options(cell_type: str) -> None:
    with pytest.raises(ValueError, match="require cell_type='gru'"):
        BaselineRNN(
            BaselineRNNConfig(
                cell_type=cell_type,
                gru_slow_update_bias=4.0,
                gru_slow_fraction=0.5,
            )
        )


@pytest.mark.parametrize("cell_type", ["lstm", "vanilla"])
def test_non_gru_accepts_default_gru_update_bias_options(cell_type: str) -> None:
    BaselineRNN(BaselineRNNConfig(cell_type=cell_type))


def test_factory_propagates_gru_fast_slow_update_bias_options() -> None:
    torch.manual_seed(211)
    reference = get_architecture(
        "gru",
        {"baseline_rnn": {"hidden_dim": 4, "num_layers": 2}},
        input_dim=3,
    )
    torch.manual_seed(211)
    initialized = get_architecture(
        "gru",
        {
            "baseline_rnn": {
                "hidden_dim": 4,
                "num_layers": 2,
                "gru_fast_update_bias": 0.0,
                "gru_slow_update_bias": 5.0,
                "gru_slow_fraction": 0.5,
            }
        },
        input_dim=3,
    )

    for layer in range(2):
        reference_ih = getattr(reference.rnn, f"bias_ih_l{layer}")
        reference_hh = getattr(reference.rnn, f"bias_hh_l{layer}")
        initialized_ih = getattr(initialized.rnn, f"bias_ih_l{layer}")
        initialized_hh = getattr(initialized.rnn, f"bias_hh_l{layer}")
        torch.testing.assert_close(initialized_ih[:4], reference_ih[:4])
        torch.testing.assert_close(initialized_ih[8:], reference_ih[8:])
        torch.testing.assert_close(initialized_hh[:4], reference_hh[:4])
        torch.testing.assert_close(initialized_hh[8:], reference_hh[8:])
        effective_update_bias = initialized_ih[4:8] + initialized_hh[4:8]
        torch.testing.assert_close(
            effective_update_bias, torch.tensor([0.0, 0.0, 5.0, 5.0])
        )


def test_positive_slow_update_bias_retains_state_and_gradient() -> None:
    """A positive slow bias supplies a directly testable long carry path."""
    hidden_dim = 4
    torch.manual_seed(227)
    network = BaselineRNN(
        BaselineRNNConfig(
            cell_type="gru",
            input_dim=1,
            hidden_dim=hidden_dim,
            gru_slow_update_bias=6.0,
            gru_slow_fraction=0.5,
            output_mode="all",
        )
    )

    # Isolate the GRU carry term while preserving its exact initialized target.
    with torch.no_grad():
        for name, parameter in network.rnn.named_parameters():
            if name.startswith("weight_"):
                parameter.zero_()
            else:
                parameter[:hidden_dim].zero_()
                parameter[2 * hidden_dim :].zero_()

    sequence = torch.zeros(1, 32, 1)
    initial_state = torch.ones(1, 1, hidden_dim, requires_grad=True)
    output, _ = network(sequence, hidden=initial_state, return_hidden=True)
    output[0, -1].sum().backward()

    expected_carry = torch.sigmoid(torch.tensor([0.0, 0.0, 6.0, 6.0])).pow(32)
    torch.testing.assert_close(output[0, -1], expected_carry)
    assert initial_state.grad is not None
    torch.testing.assert_close(initial_state.grad[0, 0], expected_carry)
