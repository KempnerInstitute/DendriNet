"""Focused tests for feedback-alignment hook lifecycle helpers."""

import pytest
import torch
from torch import nn

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear import (
    BlockLinear,
)
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


def test_feedback_alignment_collect_trainable_layers_preserves_linear_order():
    first = nn.Linear(3, 4)
    second = nn.Linear(4, 2)
    model = nn.Sequential(first, nn.ReLU(), second)

    layers = FeedbackAlignmentTrainer._collect_trainable_layers(model)

    assert layers == [
        (first, "weight", 4, 3),
        (second, "weight", 2, 4),
    ]


def test_feedback_alignment_blocklinear_log_weight_predicate():
    block = BlockLinear(in_features=4, out_features=2)
    linear = nn.Linear(4, 2)

    assert FeedbackAlignmentTrainer._is_blocklinear_log_weight(block, "log_weight")
    assert not FeedbackAlignmentTrainer._is_blocklinear_log_weight(block, "weight")
    assert not FeedbackAlignmentTrainer._is_blocklinear_log_weight(linear, "log_weight")


def test_feedback_alignment_assign_linear_bias_grad_accumulates_with_bias_dtype():
    layer = nn.Linear(2, 3, bias=True).double()
    layer.bias.grad = torch.ones_like(layer.bias)
    error = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 4.0]], dtype=torch.float32)

    FeedbackAlignmentTrainer._assign_linear_bias_grad(layer, error)

    expected = torch.tensor([2.5, 2.0, 8.0], dtype=torch.float64)
    torch.testing.assert_close(layer.bias.grad, expected)


def test_feedback_alignment_assign_linear_bias_grad_ignores_modules_without_bias():
    no_bias = nn.Linear(2, 3, bias=False)
    relu = nn.ReLU()
    error = torch.ones(2, 3)

    FeedbackAlignmentTrainer._assign_linear_bias_grad(no_bias, error)
    FeedbackAlignmentTrainer._assign_linear_bias_grad(relu, error)

    assert no_bias.bias is None


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
