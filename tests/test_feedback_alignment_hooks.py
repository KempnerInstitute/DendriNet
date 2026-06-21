"""Focused tests for feedback-alignment hook lifecycle helpers."""

import pytest
from torch import nn

from dendritic_modeling.training.strategies.feedback_alignment import (
    FeedbackAlignmentTrainer,
    ShuntingFeedbackAlignmentTrainer,
)


def _hook_count(*modules: nn.Module) -> int:
    return sum(len(module._forward_hooks) for module in modules)


def test_feedback_alignment_activation_hooks_removed_after_registration_failure(
    monkeypatch,
):
    layer_a = nn.Linear(2, 2)
    layer_b = nn.Linear(2, 2)
    trainer = FeedbackAlignmentTrainer.__new__(FeedbackAlignmentTrainer)
    trainer._layer_info = [
        (layer_a, "weight", 2, 2),
        (layer_b, "weight", 2, 2),
    ]

    def _failing_register(*args, **kwargs):
        raise RuntimeError("feedback hook registration failed")

    monkeypatch.setattr(layer_b, "register_forward_hook", _failing_register)

    with pytest.raises(RuntimeError, match="feedback hook registration failed"):
        trainer._attach_activation_hooks({})

    assert _hook_count(layer_a, layer_b) == 0


def test_shunting_feedback_hooks_removed_after_voltage_registration_failure(
    monkeypatch,
):
    layer = nn.Linear(2, 2)
    branch_a = nn.Linear(2, 2)
    branch_b = nn.Linear(2, 2)
    trainer = ShuntingFeedbackAlignmentTrainer.__new__(ShuntingFeedbackAlignmentTrainer)
    trainer._layer_info = [(layer, "weight", 2, 2)]
    trainer._layer_voltages = {}
    trainer._layer_g_tot = {}

    def _failing_register(*args, **kwargs):
        raise RuntimeError("shunting hook registration failed")

    monkeypatch.setattr(branch_b, "register_forward_hook", _failing_register)

    with pytest.raises(RuntimeError, match="shunting hook registration failed"):
        trainer._attach_shunting_hooks({}, [branch_a, branch_b])

    assert _hook_count(layer, branch_a, branch_b) == 0
