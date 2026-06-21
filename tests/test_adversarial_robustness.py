from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from dendritic_modeling.analysis.metrics.adversarial_robustness import (
    AdversarialRobustnessAnalyzer,
)


def _linear_classifier() -> nn.Linear:
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -0.5], [-0.25, 0.75]]))
    return model


def test_fgsm_attack_uses_input_gradient_without_parameter_gradients():
    model = _linear_classifier()
    analyzer = AdversarialRobustnessAnalyzer(SimpleNamespace(batch_size=2))
    x = torch.tensor([[0.2, 0.7], [0.8, 0.1]])
    y = torch.tensor([0, 1])
    epsilon = 0.1

    x_ref = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_ref), y)
    (input_grad,) = torch.autograd.grad(loss, x_ref)
    expected = torch.clamp(x + epsilon * input_grad.sign(), 0.0, 1.0)

    attacked = analyzer.fgsm_attack(model, x, y, epsilon)

    torch.testing.assert_close(attacked, expected)
    assert all(param.grad is None for param in model.parameters())


def test_evaluate_accuracy_under_attack_restores_training_mode_and_gradients():
    model = _linear_classifier()
    model.train()
    analyzer = AdversarialRobustnessAnalyzer(SimpleNamespace(batch_size=1))
    dataset = TensorDataset(
        torch.tensor([[0.2, 0.7], [0.8, 0.1]]),
        torch.tensor([0, 1]),
    )

    accuracy = analyzer.evaluate_accuracy_under_attack(
        model,
        dataset,
        epsilon=0.1,
        device="cpu",
    )

    assert 0.0 <= accuracy <= 1.0
    assert model.training is True
    assert all(param.grad is None for param in model.parameters())
