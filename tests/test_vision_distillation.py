from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.replacement import (
    PopulationToFeatureReduction,
    PositiveFeatureScale,
)
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


class _RecordingIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x.detach().clone()
        return x


class _RepeatInterleave(nn.Module):
    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat_interleave(2, dim=self.axis)


class _TinyStudentCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_blocks = nn.ModuleList(_RecordingIdentity() for _ in range(5))
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
    inputs = torch.randn(4, 3, 4, 4)
    loss = loss_fn(student, inputs, torch.tensor([0, 1, 2, 1]))

    assert torch.isfinite(loss)
    assert all(
        torch.equal(block.last_input, inputs)
        for block in student.core_network.spatial_blocks
    )
    loss.backward()
    assert student.core_network.feedforward_blocks[0].weight.grad is not None
    assert student.core_network.spatial_output_adapters[0].log_scale.grad is not None
    assert all(parameter.grad is None for parameter in loss_fn.teacher.parameters())


def test_teacher_forced_boundaries_apply_population_reducers():
    core = nn.Module()
    core.input_adapter = nn.Identity()
    core.spatial_blocks = nn.ModuleList(_RepeatInterleave(1) for _ in range(5))
    core.spatial_population_reducers = nn.ModuleList(
        PopulationToFeatureReduction(
            3,
            somas_per_feature=2,
            feature_axis=1,
            mode="fixed_mean",
        )
        for _ in range(5)
    )
    core.spatial_output_adapters = nn.ModuleList(nn.Identity() for _ in range(5))
    core.feedforward_blocks = nn.ModuleList(
        [_RepeatInterleave(-1), _RepeatInterleave(-1), nn.Identity()]
    )
    core.feedforward_population_reducers = nn.ModuleList(
        [
            PopulationToFeatureReduction(
                48,
                somas_per_feature=2,
                feature_axis=-1,
                mode="fixed_mean",
            ),
            PopulationToFeatureReduction(
                48,
                somas_per_feature=2,
                feature_axis=-1,
                mode="fixed_mean",
            ),
            nn.Identity(),
        ]
    )
    core.feedforward_output_adapters = nn.ModuleList(nn.Identity() for _ in range(3))
    teacher_inputs = {
        **{
            name: torch.randn(2, 3, 4, 4)
            for name in ("conv1", "conv2", "conv3", "conv4", "conv5")
        },
        "fc6": torch.randn(2, 48),
        "fc7": torch.randn(2, 48),
        "fc8": torch.randn(2, 3),
    }

    outputs = AlexNetBoundaryDistillationLoss._conditioned_student_outputs(
        core,
        teacher_inputs,
        teacher_forcing_ratio=1.0,
    )

    assert set(outputs) == set(teacher_inputs)
    for name in outputs:
        torch.testing.assert_close(outputs[name], teacher_inputs[name])


def test_alexnet_boundary_distillation_rejects_unknown_boundary():
    with pytest.raises(ValueError, match="Unknown vision-distillation boundaries"):
        AlexNetBoundaryDistillationLoss({"boundary_weights": {"pool3": 1.0}})


def test_teacher_forced_boundary_distillation_uses_local_block_inputs(monkeypatch):
    monkeypatch.setattr(
        "dendritic_modeling.training.strategies.vision_distillation.get_model",
        lambda _backbone, weights: _TinyTeacher(),
    )
    loss_fn = AlexNetBoundaryDistillationLoss(
        {
            "boundary_weights": dict.fromkeys(
                (
                    "conv1",
                    "conv2",
                    "conv3",
                    "conv4",
                    "conv5",
                    "fc6",
                    "fc7",
                    "fc8",
                ),
                1.0,
            ),
            "boundary_input_mode": "teacher_forced",
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


def test_alexnet_boundary_distillation_rejects_unknown_input_mode():
    with pytest.raises(ValueError, match="boundary_input_mode"):
        AlexNetBoundaryDistillationLoss({"boundary_input_mode": "unsupported"})


def test_mixed_boundary_distillation_blends_teacher_and_composed_inputs(monkeypatch):
    monkeypatch.setattr(
        "dendritic_modeling.training.strategies.vision_distillation.get_model",
        lambda _backbone, weights: _TinyTeacher(),
    )
    loss_fn = AlexNetBoundaryDistillationLoss(
        {
            "boundary_weights": {"conv2": 1.0},
            "boundary_input_mode": "mixed",
            "teacher_forcing_ratio": 0.25,
            "feature_weight": 1.0,
            "logit_kl_weight": 0.0,
            "supervised_weight": 0.0,
        }
    )
    student = _TinyStudent()
    with torch.no_grad():
        student.core_network.spatial_output_adapters[0].log_scale.fill_(
            torch.log(torch.tensor(2.0))
        )
    inputs = torch.randn(4, 3, 4, 4)
    loss = loss_fn(student, inputs, torch.tensor([0, 1, 2, 1]))

    expected_conv2_input = 0.25 * inputs + 0.75 * (2.0 * inputs)
    assert torch.allclose(
        student.core_network.spatial_blocks[1].last_input,
        expected_conv2_input,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert student.core_network.spatial_output_adapters[0].log_scale.grad is not None


@pytest.mark.parametrize("ratio", [-0.1, 0.0, 1.0, 1.1])
def test_mixed_boundary_distillation_rejects_invalid_ratio(ratio):
    with pytest.raises(ValueError, match="teacher_forcing_ratio"):
        AlexNetBoundaryDistillationLoss(
            {
                "boundary_input_mode": "mixed",
                "teacher_forcing_ratio": ratio,
            }
        )


def test_mixed_boundary_inputs_follow_composed_dtype():
    core = _TinyStudentCore()
    teacher_inputs = {
        **{
            name: torch.randn(2, 3, 4, 4, dtype=torch.float16)
            for name in ("conv1", "conv2", "conv3", "conv4", "conv5")
        },
        "fc6": torch.randn(2, 48, dtype=torch.float16),
        "fc7": torch.randn(2, 48, dtype=torch.float16),
        "fc8": torch.randn(2, 48, dtype=torch.float16),
    }
    composed_inputs = {name: value.float() for name, value in teacher_inputs.items()}

    outputs = AlexNetBoundaryDistillationLoss._conditioned_student_outputs(
        core,
        teacher_inputs,
        teacher_forcing_ratio=0.5,
        composed_inputs=composed_inputs,
    )

    assert all(value.dtype == torch.float32 for value in outputs.values())


def test_teacher_forced_single_boundary_keeps_all_student_parameters_in_graph(
    monkeypatch,
):
    monkeypatch.setattr(
        "dendritic_modeling.training.strategies.vision_distillation.get_model",
        lambda _backbone, weights: _TinyTeacher(),
    )
    loss_fn = AlexNetBoundaryDistillationLoss(
        {
            "boundary_weights": {"conv1": 1.0},
            "boundary_input_mode": "teacher_forced",
            "feature_weight": 1.0,
            "logit_kl_weight": 0.0,
            "supervised_weight": 0.0,
        }
    )
    student = _TinyStudent()
    loss = loss_fn(student, torch.randn(4, 3, 4, 4), torch.tensor([0, 1, 2, 1]))

    loss.backward()

    student_parameters = list(student.parameters())
    assert all(parameter.grad is not None for parameter in student_parameters)
    assert (
        torch.count_nonzero(student.core_network.feedforward_blocks[1].weight.grad) == 0
    )
