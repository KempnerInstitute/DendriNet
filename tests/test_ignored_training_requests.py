"""Training requests that change nothing are reported, and still change nothing."""

from __future__ import annotations

import logging

import pytest
import torch
from test_trainer_configuration import _data, _einet_model, _make_trainer, _state

from dendritic_modeling.training.strategies.local_learning import (
    inactive_auxiliary_terms,
)


@pytest.fixture
def package_logs(monkeypatch, caplog):
    """Let caplog see the package's warnings."""
    monkeypatch.setattr(logging.getLogger("dendritic_modeling"), "propagate", True)
    caplog.set_level(logging.WARNING, logger="dendritic_modeling")
    return caplog


def _warnings(caplog, text):
    return [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and text in record.getMessage()
    ]


def _train(model, **overrides):
    trainer, _ = _make_trainer("standard", model, epochs=2, **overrides)
    data = _data()
    trainer.train(model, data, data)
    return trainer


def test_recalibration_of_analytical_gates_warns_once_and_changes_no_gate(
    package_logs,
):
    # Without shuffling, the batches the recalibration draws change nothing.
    reference = _einet_model()
    _train(reference)
    assert not _warnings(package_logs, "recalibrate_reactivation_every")

    model = _einet_model()
    _train(model, recalibrate_reactivation_every=1)
    records = _warnings(package_logs, "recalibrate_reactivation_every=1")
    assert len(records) == 1
    message = records[0].getMessage()
    assert "from_init_policy" in message
    assert "still draws training batches" in message
    assert "keep the setting to reproduce an earlier run" in message
    after, expected = _state(model), _state(reference)
    assert all(torch.equal(after[name], expected[name]) for name in expected)


def test_a_no_op_recalibration_still_advances_a_shuffled_data_order(package_logs):
    # The warning's reason to keep the setting: with shuffling, removing the
    # recalibration that changes no gate changes the order of later batches.
    reference = _einet_model()
    _train(reference, shuffle=True)
    model = _einet_model()
    _train(model, shuffle=True, recalibrate_reactivation_every=1)
    assert len(_warnings(package_logs, "still draws training batches")) == 1
    after, expected = _state(model), _state(reference)
    assert not all(torch.equal(after[name], expected[name]) for name in expected)


def test_recalibration_that_calibrates_gates_does_not_warn(package_logs):
    _train(
        _einet_model(),
        recalibrate_reactivation_every=1,
        reactivation_recalibration_mode="occupancy_quantile",
    )
    assert not _warnings(package_logs, "recalibrate_reactivation_every")


def test_amp_without_cuda_warns_and_trains_in_full_precision(package_logs, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    trainer, _ = _make_trainer("standard", _einet_model(), use_amp=False)
    assert not _warnings(package_logs, "use_amp")
    trainer, _ = _make_trainer("standard", _einet_model(), use_amp=True)
    assert trainer.enable_amp is False
    records = _warnings(package_logs, "use_amp=true")
    assert len(records) == 1
    assert "full precision" in records[0].getMessage()


_INACTIVE = [
    ({"hsic": {"enabled": True, "weight": 0.0}}, "hsic (weight=0)"),
    ({"hsic": {"enabled": True, "weight": 1.0, "self_weight": 0.0}}, None),
    ({"hsic": {"enabled": True, "weight": -1.0}}, None),
    (
        {"inhibitory_homeostasis": {"enabled": True, "weight": 0.0}},
        "inhibitory_homeostasis (weight=0)",
    ),
    (
        {"voltage_homeostasis": {"enabled": True, "weight": 0.0}},
        "voltage_homeostasis (weight=0)",
    ),
    (
        {"gate_homeostasis": {"enabled": True}},
        "gate_homeostasis (center_weight=0, saturation_weight=0)",
    ),
    ({"voltage_homeostasis": {"enabled": True, "weight": 0.5}}, None),
    ({"gate_homeostasis": {"enabled": True, "saturation_weight": 0.2}}, None),
    ({"voltage_homeostasis": {"enabled": False, "weight": 0.0}}, None),
]


@pytest.mark.parametrize("local_rule_config, expected", _INACTIVE)
def test_zero_weight_auxiliary_terms_of_local_ca_warn(
    package_logs, local_rule_config, expected
):
    trainer, _ = _make_trainer(
        "local_ca", _einet_model(), local_rule_config=local_rule_config
    )
    records = _warnings(package_logs, "auxiliary terms")
    if expected is None:
        assert not records
        assert inactive_auxiliary_terms(trainer.local_cfg) == []
    else:
        assert len(records) == 1
        assert expected in records[0].getMessage()
        assert inactive_auxiliary_terms(trainer.local_cfg) == [expected]


def test_hsic_without_self_or_target_weight_is_reported(package_logs):
    trainer, _ = _make_trainer(
        "local_ca",
        _einet_model(),
        local_rule_config={
            "hsic": {
                "enabled": True,
                "weight": 1.0,
                "self_weight": 0.0,
                "target_weight": 0.0,
            }
        },
    )
    assert inactive_auxiliary_terms(trainer.local_cfg) == [
        "hsic (self_weight=0 and target_weight=0)"
    ]
    assert len(_warnings(package_logs, "auxiliary terms")) == 1
