"""Tests for local-learning soma-error mapping behavior."""

import torch
import torch.nn as nn

from dendritic_modeling.training.strategies.local_learning import LocalCreditAssignment


class _DummyModel(nn.Module):
    """Minimal model exposing only a decoder network."""

    def __init__(self, decoder_network: nn.Module):
        super().__init__()
        self.decoder_network = decoder_network


def _make_trainer() -> LocalCreditAssignment:
    """Build an uninitialized trainer instance for helper-method tests."""
    trainer = LocalCreditAssignment.__new__(LocalCreditAssignment)
    trainer._decoder_cache = {}
    trainer._warned_decoder_soma_fallback = False
    return trainer


def test_linear_decoder_maps_output_error_to_core_soma_error():
    """Single linear decoder should map delta_out into core-soma space."""
    torch.manual_seed(0)
    decoder = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        decoder.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 4.0]]))

    model = _DummyModel(decoder)
    trainer = _make_trainer()

    h_in = torch.randn(5, 3)
    y_hat = torch.randn(5, 2)
    delta_out = torch.randn(5, 2)
    trainer._decoder_cache["module"] = decoder
    trainer._decoder_cache["input"] = h_in

    v0_local, delta_local = trainer._resolve_local_soma_signals(model, y_hat, delta_out)

    expected_delta = delta_out @ decoder.weight.detach()
    torch.testing.assert_close(v0_local, h_in)
    torch.testing.assert_close(delta_local, expected_delta)


def test_nonlinear_decoder_maps_output_error_via_decoder_jacobian():
    """Multi-layer decoder should map error back to decoder input when cached."""
    torch.manual_seed(1)
    decoder = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    model = _DummyModel(decoder)
    trainer = _make_trainer()

    h_in = torch.randn(4, 3, requires_grad=True)
    y_hat = decoder(h_in)
    delta_out = torch.randn(4, 2)
    trainer._decoder_cache["decoder_input"] = h_in

    v0_local, delta_local = trainer._resolve_local_soma_signals(model, y_hat, delta_out)

    expected_delta = torch.autograd.grad(
        outputs=y_hat,
        inputs=h_in,
        grad_outputs=delta_out,
        retain_graph=True,
    )[0]
    torch.testing.assert_close(v0_local, h_in.detach())
    torch.testing.assert_close(delta_local, expected_delta.detach())


def test_nonlinear_decoder_without_decoder_input_falls_back_to_output_space():
    """Multi-layer decoder should keep legacy output-space behavior without cache."""
    torch.manual_seed(1)
    decoder = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    model = _DummyModel(decoder)
    trainer = _make_trainer()

    y_hat = torch.randn(4, 2, requires_grad=True)
    delta_out = torch.randn(4, 2)
    trainer._decoder_cache["module"] = decoder[-1]
    trainer._decoder_cache["input"] = torch.randn(4, 4)

    v0_local, delta_local = trainer._resolve_local_soma_signals(model, y_hat, delta_out)

    torch.testing.assert_close(v0_local, y_hat.detach())
    torch.testing.assert_close(delta_local, delta_out)


def test_linear_decoder_without_cache_falls_back_to_output_space():
    """Missing decoder cache should keep legacy output-space behavior."""
    decoder = nn.Linear(3, 2)
    model = _DummyModel(decoder)
    trainer = _make_trainer()

    y_hat = torch.randn(3, 2)
    delta_out = torch.randn(3, 2)
    v0_local, delta_local = trainer._resolve_local_soma_signals(model, y_hat, delta_out)

    torch.testing.assert_close(v0_local, y_hat)
    torch.testing.assert_close(delta_local, delta_out)
