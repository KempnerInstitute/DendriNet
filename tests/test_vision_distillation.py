from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.replacement import PositiveFeatureScale
from dendritic_modeling.training.strategies.vision_distillation import (
    AlexNetBoundaryDistillationLoss,
)


class _TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(*(nn.Identity() for _ in range(12)))
        self.avgpool = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Identity(),
            nn.Linear(48, 48),
            nn.ReLU(),
            nn.Identity(),
            nn.Linear(48, 48),
            nn.ReLU(),
            nn.Linear(48, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(torch.flatten(x, 1))


class _TinyStudentCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_blocks = nn.ModuleList(nn.Identity() for _ in range(5))
        self.spatial_output_adapters = nn.ModuleList(
            PositiveFeatureScale(3, feature_axis=1) for _ in range(5)
        )
        self.feedforward_blocks = nn.ModuleList(
            [nn.Linear(48, 48), nn.Linear(48, 48), nn.Linear(48, 3)]
        )
        self.feedforward_output_adapters = nn.ModuleList(
            [
                PositiveFeatureScale(48, feature_axis=-1),
                PositiveFeatureScale(48, feature_axis=-1),
                nn.Identity(),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block, adapter in zip(
            self.spatial_blocks,
            self.spatial_output_adapters,
            strict=True,
        ):
            x = adapter(block(x))
        x = torch.flatten(x, 1)
        for index, (block, adapter) in enumerate(
            zip(
                self.feedforward_blocks,
                self.feedforward_output_adapters,
                strict=True,
            )
        ):
            x = block(x)
            if index < 2:
                x = torch.relu(x)
            x = adapter(x)
        return x


class _TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.core_network = _TinyStudentCore()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core_network(x)


def test_alexnet_boundary_distillation_is_finite_and_differentiable(monkeypatch):
    monkeypatch.setattr(
        "dendritic_modeling.training.strategies.vision_distillation.get_model",
        lambda _backbone, weights: _TinyTeacher(),
    )
    loss_fn = AlexNetBoundaryDistillationLoss(
        {
            "boundary_weights": {
                "conv1": 1.0,
                "conv2": 1.0,
                "conv3": 1.0,
                "conv4": 1.0,
                "conv5": 1.0,
                "fc6": 1.0,
                "fc7": 1.0,
            },
            "feature_weight": 1.0,
            "logit_kl_weight": 1.0,
            "supervised_weight": 1.0,
        }
    )
    student = _TinyStudent()
    loss = loss_fn(student, torch.randn(4, 3, 4, 4), torch.tensor([0, 1, 2, 1]))

    assert torch.isfinite(loss)
    loss.backward()
    assert student.core_network.feedforward_blocks[0].weight.grad is not None
    assert student.core_network.spatial_output_adapters[0].log_scale.grad is not None
    assert all(parameter.grad is None for parameter in loss_fn.teacher.parameters())


def test_alexnet_boundary_distillation_rejects_unknown_boundary():
    with pytest.raises(ValueError, match="Unknown vision-distillation boundaries"):
        AlexNetBoundaryDistillationLoss({"boundary_weights": {"pool3": 1.0}})
