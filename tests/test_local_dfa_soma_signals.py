"""Tests for the DFA between-neuron soma-error source in local learning."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from dendritic_modeling.config.training import LocalRuleConfig
from dendritic_modeling.training.strategies.local_learning import LocalCreditAssignment


class _DummyModel(nn.Module):
    """Minimal model exposing only a decoder network."""

    def __init__(self, decoder_network: nn.Module):
        super().__init__()
        self.decoder_network = decoder_network


def _make_trainer(local_cfg=None) -> LocalCreditAssignment:
    """Build an uninitialized trainer instance for helper-method tests."""
    trainer = LocalCreditAssignment.__new__(LocalCreditAssignment)
    trainer._decoder_cache = {}
    trainer._warned_decoder_soma_fallback = False
    if local_cfg is not None:
        trainer.local_cfg = local_cfg
    return trainer


def _reference_feedback(soma_dim: int, output_dim: int, seed: int, scale: float):
    """The SomaDFA feedback construction the local-CA path must reproduce."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    feedback = torch.randn((soma_dim, output_dim), generator=generator)
    feedback.mul_(scale / float(output_dim) ** 0.5)
    return feedback


def test_decoder_source_is_default_and_preserves_exact_mapping():
    """soma_error_source='decoder' must leave the exact Jacobian path unchanged."""
    decoder = nn.Linear(3, 2, bias=False)
    model = _DummyModel(decoder)
    trainer = _make_trainer(SimpleNamespace(soma_error_source="decoder"))
    h_in = torch.randn(5, 3)
    y_hat = torch.randn(5, 2)
    delta_out = torch.randn(5, 2)
    trainer._decoder_cache["module"] = decoder
    trainer._decoder_cache["input"] = h_in

    v0_local, delta_local = trainer._resolve_local_soma_signals(model, y_hat, delta_out)

    torch.testing.assert_close(v0_local, h_in)
    torch.testing.assert_close(delta_local, delta_out @ decoder.weight.detach())


def test_missing_local_cfg_defaults_to_decoder_path():
    """Trainers without a local_cfg attribute keep the legacy behavior."""
    decoder = nn.Linear(3, 2, bias=False)
    model = _DummyModel(decoder)
    trainer = _make_trainer()
    h_in = torch.randn(4, 3)
    trainer._decoder_cache["module"] = decoder
    trainer._decoder_cache["input"] = h_in
    delta_out = torch.randn(4, 2)

    v0_local, delta_local = trainer._resolve_local_soma_signals(
        model, torch.randn(4, 2), delta_out
    )

    torch.testing.assert_close(v0_local, h_in)
    torch.testing.assert_close(delta_local, delta_out @ decoder.weight.detach())


def test_dfa_source_replaces_decoder_jacobian_with_fixed_random_map():
    """soma_error_source='dfa' must ignore decoder weights entirely."""
    decoder = nn.Linear(6, 4, bias=False)
    model = _DummyModel(decoder)
    trainer = _make_trainer(
        SimpleNamespace(
            soma_error_source="dfa",
            dfa_feedback_seed=123,
            dfa_feedback_scale=2.0,
        )
    )
    h_in = torch.randn(5, 6)
    delta_out = torch.randn(5, 4)
    trainer._decoder_cache["module"] = decoder
    trainer._decoder_cache["input"] = h_in

    v0_local, delta_local = trainer._resolve_local_soma_signals(
        model, torch.randn(5, 4), delta_out
    )

    feedback = _reference_feedback(soma_dim=6, output_dim=4, seed=123, scale=2.0)
    torch.testing.assert_close(v0_local, h_in)
    torch.testing.assert_close(delta_local, delta_out @ feedback.t())
    assert delta_local.requires_grad is False


def test_dfa_feedback_matrix_is_cached_and_deterministic():
    """Repeated resolutions must reuse one fixed matrix, not re-sample."""
    decoder = nn.Linear(6, 4, bias=False)
    model = _DummyModel(decoder)
    trainer = _make_trainer(
        SimpleNamespace(
            soma_error_source="dfa",
            dfa_feedback_seed=7,
            dfa_feedback_scale=1.0,
        )
    )
    h_in = torch.randn(3, 6)
    trainer._decoder_cache["module"] = decoder
    trainer._decoder_cache["input"] = h_in
    delta_out = torch.randn(3, 4)

    _, first = trainer._resolve_local_soma_signals(model, torch.randn(3, 4), delta_out)
    matrix_after_first = trainer._dfa_feedback_matrix.clone()
    _, second = trainer._resolve_local_soma_signals(model, torch.randn(3, 4), delta_out)

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(trainer._dfa_feedback_matrix, matrix_after_first)


def test_dfa_seed_defaults_to_trainer_seed():
    decoder = nn.Linear(6, 4, bias=False)
    model = _DummyModel(decoder)
    trainer = _make_trainer(
        SimpleNamespace(
            soma_error_source="dfa",
            dfa_feedback_seed=None,
            dfa_feedback_scale=1.0,
        )
    )
    trainer.seed = 99
    h_in = torch.randn(3, 6)
    trainer._decoder_cache["module"] = decoder
    trainer._decoder_cache["input"] = h_in
    delta_out = torch.randn(3, 4)

    _, delta_local = trainer._resolve_local_soma_signals(
        model, torch.randn(3, 4), delta_out
    )

    feedback = _reference_feedback(soma_dim=6, output_dim=4, seed=99, scale=1.0)
    torch.testing.assert_close(delta_local, delta_out @ feedback.t())


def test_dfa_source_raises_without_cached_decoder_input():
    """DFA is explicit opt-in: it must not silently fall back to output space."""
    decoder = nn.Linear(3, 2)
    model = _DummyModel(decoder)
    trainer = _make_trainer(SimpleNamespace(soma_error_source="dfa"))

    with pytest.raises(RuntimeError, match="cached two-dimensional decoder input"):
        trainer._resolve_local_soma_signals(model, torch.randn(3, 2), torch.randn(3, 2))


def test_dfa_source_raises_on_geometry_change():
    decoder = nn.Linear(6, 4, bias=False)
    model = _DummyModel(decoder)
    trainer = _make_trainer(
        SimpleNamespace(
            soma_error_source="dfa",
            dfa_feedback_seed=1,
            dfa_feedback_scale=1.0,
        )
    )
    trainer._decoder_cache["module"] = decoder
    trainer._decoder_cache["input"] = torch.randn(3, 6)
    trainer._resolve_local_soma_signals(model, torch.randn(3, 4), torch.randn(3, 4))

    trainer._decoder_cache["input"] = torch.randn(3, 5)
    with pytest.raises(RuntimeError, match="geometry changed"):
        trainer._resolve_local_soma_signals(model, torch.randn(3, 4), torch.randn(3, 4))


def test_local_rule_config_validates_soma_error_source():
    cfg = LocalRuleConfig(soma_error_source="DFA")
    assert cfg.soma_error_source == "dfa"
    assert LocalRuleConfig().soma_error_source == "decoder"

    with pytest.raises(ValueError, match="soma_error_source"):
        LocalRuleConfig(soma_error_source="random")
    with pytest.raises(ValueError, match="dfa_feedback_scale"):
        LocalRuleConfig(soma_error_source="dfa", dfa_feedback_scale=0.0)
