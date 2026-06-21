"""Tests for local-learning soma-error mapping behavior."""

import logging

import torch
import torch.nn as nn

from dendritic_modeling.training.strategies.local_learning import LocalCreditAssignment
from dendritic_modeling.training.strategies.local_learning_signals import (
    _decoder_linear_layers,
    _has_single_linear_decoder,
    _linear_decoder_shapes_compatible,
    _map_decoder_input_error,
    _prepare_bce_prediction,
    _prepare_ce_target,
)


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


def test_map_decoder_input_error_returns_detached_jacobian_mapping():
    decoder_input = torch.randn(4, 3, requires_grad=True)
    weight = torch.randn(3, 2)
    y_hat = decoder_input @ weight
    delta_out = torch.randn(4, 2)

    mapped = _map_decoder_input_error(
        y_hat=y_hat,
        decoder_input=decoder_input,
        delta_out=delta_out,
    )

    assert mapped is not None
    v0_local, delta_local = mapped
    torch.testing.assert_close(v0_local, decoder_input.detach())
    torch.testing.assert_close(delta_local, delta_out @ weight.t())
    assert v0_local.requires_grad is False
    assert delta_local.requires_grad is False


def test_map_decoder_input_error_returns_none_for_unavailable_mapping():
    decoder_input = torch.randn(4, 3, requires_grad=True)
    other_input = torch.randn(4, 3, requires_grad=True)
    y_hat = other_input.square()
    delta_out = torch.randn(4, 3)

    assert (
        _map_decoder_input_error(
            y_hat=y_hat,
            decoder_input=decoder_input,
            delta_out=delta_out,
        )
        is None
    )
    assert (
        _map_decoder_input_error(
            y_hat=torch.randn(4, 3),
            decoder_input=decoder_input,
            delta_out=delta_out,
        )
        is None
    )
    assert (
        _map_decoder_input_error(
            y_hat=y_hat,
            decoder_input=None,
            delta_out=delta_out,
        )
        is None
    )


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


def test_decoder_soma_fallback_warning_logs_once(caplog):
    decoder = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    model = _DummyModel(decoder)
    trainer = _make_trainer()

    y_hat = torch.randn(4, 2)
    delta_out = torch.randn(4, 2)
    message = "Local soma-error mapping fallback: decoder is not a single linear layer."

    caplog.set_level(
        logging.INFO,
        logger="dendritic_modeling.training.strategies.local_learning_signals",
    )

    first_v0, first_delta = trainer._resolve_local_soma_signals(model, y_hat, delta_out)
    second_v0, second_delta = trainer._resolve_local_soma_signals(
        model, y_hat, delta_out
    )

    torch.testing.assert_close(first_v0, y_hat.detach())
    torch.testing.assert_close(first_delta, delta_out)
    torch.testing.assert_close(second_v0, y_hat.detach())
    torch.testing.assert_close(second_delta, delta_out)
    assert trainer._warned_decoder_soma_fallback is True
    assert caplog.text.count(message) == 1


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


def test_decoder_linear_layers_preserves_traversal_failure_fallback():
    class _BrokenDecoder(nn.Module):
        def modules(self):
            raise RuntimeError("broken traversal")

    assert _decoder_linear_layers(_BrokenDecoder()) == []
    assert not _has_single_linear_decoder(_BrokenDecoder())


def test_single_linear_decoder_detection_counts_nested_linear_layers():
    assert _has_single_linear_decoder(nn.Linear(3, 2))
    assert not _has_single_linear_decoder(
        nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    )


def test_linear_decoder_shapes_compatible_matches_mapping_requirements():
    lin = nn.Linear(3, 2)
    delta_out = torch.zeros(4, 2)
    h_in = torch.zeros(4, 3)

    assert _linear_decoder_shapes_compatible(
        delta_out=delta_out,
        h_in=h_in,
        lin_mod=lin,
    )
    assert not _linear_decoder_shapes_compatible(
        delta_out=delta_out.unsqueeze(1),
        h_in=h_in,
        lin_mod=lin,
    )
    assert not _linear_decoder_shapes_compatible(
        delta_out=torch.zeros(4, 3),
        h_in=h_in,
        lin_mod=lin,
    )
    assert not _linear_decoder_shapes_compatible(
        delta_out=delta_out,
        h_in=torch.zeros(4, 4),
        lin_mod=lin,
    )


def test_prepare_bce_prediction_preserves_logits_and_probability_paths():
    logits = torch.tensor([[-1.0, 2.0], [0.5, 0.0]])
    probabilities = torch.tensor([[0.0, 1.0], [0.25, 0.75]])

    torch.testing.assert_close(_prepare_bce_prediction(logits), torch.sigmoid(logits))
    torch.testing.assert_close(
        _prepare_bce_prediction(probabilities),
        torch.clamp(probabilities, min=1e-6, max=1.0 - 1e-6),
    )


def test_prepare_ce_target_preserves_class_index_and_soft_label_paths():
    y_hat = torch.zeros(2, 3, dtype=torch.float64)
    class_target = _prepare_ce_target(y_hat, torch.tensor([[2], [0]]))
    soft_target = torch.tensor([[0.1, 0.9, 0.0], [1.0, 0.0, 0.0]])

    torch.testing.assert_close(
        class_target,
        torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float64),
    )
    assert _prepare_ce_target(y_hat, soft_target) is soft_target


def test_error_by_mode_uses_extracted_ce_and_bce_helpers():
    y_hat = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    y_class = torch.tensor([0, 1])
    y_binary = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    ce_error = LocalCreditAssignment._error_by_mode(y_hat, y_class, "ce")
    bce_error = LocalCreditAssignment._error_by_mode(y_hat, y_binary, "bce")

    expected_ce = torch.softmax(y_hat, dim=-1) - _prepare_ce_target(y_hat, y_class)
    expected_bce = _prepare_bce_prediction(y_hat) - y_binary
    torch.testing.assert_close(ce_error, expected_ce)
    torch.testing.assert_close(bce_error, expected_bce)
