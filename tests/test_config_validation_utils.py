"""Tests for unified config validation helpers."""

import dendritic_modeling.utils.config_validation as config_validation


def test_run_architecture_validation_returns_false_on_failure(monkeypatch):
    monkeypatch.setattr(
        config_validation, "validate_model_architecture", lambda _cfg: False
    )

    assert config_validation._run_architecture_validation({}) is False


def test_run_architecture_validation_returns_true_on_success(monkeypatch):
    monkeypatch.setattr(
        config_validation, "validate_model_architecture", lambda _cfg: True
    )

    assert config_validation._run_architecture_validation({}) is True


def test_run_training_validation_applies_strict_policy(monkeypatch):
    monkeypatch.setattr(
        config_validation, "validate_training_stability", lambda _cfg: False
    )

    assert config_validation._run_training_validation({}, strict=True) is False
    assert config_validation._run_training_validation({}, strict=False) is True


def test_run_training_validation_returns_true_on_success(monkeypatch):
    monkeypatch.setattr(
        config_validation, "validate_training_stability", lambda _cfg: True
    )

    assert config_validation._run_training_validation({}, strict=True) is True
    assert config_validation._run_training_validation({}, strict=False) is True


def test_validate_full_config_short_circuits_when_architecture_fails(monkeypatch):
    called = {"training": False}

    monkeypatch.setattr(
        config_validation, "validate_model_architecture", lambda _cfg: False
    )

    def _training(_cfg):
        called["training"] = True
        return True

    monkeypatch.setattr(config_validation, "validate_training_stability", _training)

    assert config_validation.validate_full_config({}, strict=True) is False
    assert called["training"] is False


def test_validate_full_config_strict_mode_fails_on_training(monkeypatch):
    monkeypatch.setattr(
        config_validation, "validate_model_architecture", lambda _cfg: True
    )
    monkeypatch.setattr(
        config_validation, "validate_training_stability", lambda _cfg: False
    )

    assert config_validation.validate_full_config({}, strict=True) is False


def test_validate_full_config_non_strict_mode_allows_training_failure(monkeypatch):
    monkeypatch.setattr(
        config_validation, "validate_model_architecture", lambda _cfg: True
    )
    monkeypatch.setattr(
        config_validation, "validate_training_stability", lambda _cfg: False
    )

    assert config_validation.validate_full_config({}, strict=False) is True


def test_validate_full_config_passes_when_both_valid(monkeypatch):
    monkeypatch.setattr(
        config_validation, "validate_model_architecture", lambda _cfg: True
    )
    monkeypatch.setattr(
        config_validation, "validate_training_stability", lambda _cfg: True
    )

    assert config_validation.validate_full_config({}, strict=True) is True


def test_validate_architecture_only_delegates(monkeypatch):
    monkeypatch.setattr(
        config_validation, "validate_model_architecture", lambda _cfg: True
    )
    assert config_validation.validate_architecture_only({}) is True


def test_validate_training_only_delegates(monkeypatch):
    monkeypatch.setattr(
        config_validation, "validate_training_stability", lambda _cfg: True
    )
    assert config_validation.validate_training_only({}) is True


def test_get_validation_summary_aggregates_both_results(monkeypatch):
    monkeypatch.setattr(
        config_validation, "validate_model_architecture", lambda _cfg: True
    )
    monkeypatch.setattr(
        config_validation, "validate_training_stability", lambda _cfg: False
    )

    summary = config_validation.get_validation_summary({})

    assert summary == {
        "architecture_valid": True,
        "training_stable": False,
    }
