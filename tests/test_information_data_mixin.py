"""Focused tests for information-analysis data collection helpers."""

import logging

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.analysis.core.information_parts.information_data_mixin import (
    InformationDataMixin,
)
from dendritic_modeling.config.analysis import EvaluationRuntimeConfig


class _RecordingInformationDataHarness(InformationDataMixin):
    """Minimal harness for exercising InformationDataMixin helper boundaries."""

    def __init__(self):
        self.max_samples = None
        self.logger = logging.getLogger(__name__)
        self.data_dict = {"layer0": {}}
        self.removed_handles = []

    def attach_forward_hooks(self, model):
        return ["handle"]

    def remove_forward_hooks(self, handles):
        self.removed_handles.append(list(handles))

    def _is_discrete_class_labels(self, labels):
        return np.issubdtype(labels.dtype, np.integer)

    def _discrete_entropy_nats(self, labels):
        return 0.75


class _RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_inputs = None

    def forward(self, inputs):
        self.seen_inputs = inputs.detach().clone()
        return inputs


class _FailingModel(torch.nn.Module):
    def forward(self, inputs):
        raise RuntimeError("forward failed")


def test_information_data_collection_preserves_outputs_and_hook_lifecycle():
    harness = _RecordingInformationDataHarness()
    model = _RecordingModel()
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    y = torch.tensor([0, 1, 1, 0])
    dataset = TensorDataset(x, y)

    result = harness._collect_analysis_data(
        model=model,
        test_dataset=dataset,
        device="cpu",
        runtime=EvaluationRuntimeConfig(mode="materialize", num_workers=0),
    )

    torch.testing.assert_close(result.inputs, x)
    torch.testing.assert_close(result.labels, y)
    np.testing.assert_array_equal(result.C, y.numpy())
    assert result.entropy_C == 0.75
    torch.testing.assert_close(model.seen_inputs, x)
    assert harness.removed_handles == [["handle"]]


def test_information_forward_hook_collection_removes_hooks_after_forward_failure():
    harness = _RecordingInformationDataHarness()

    with pytest.raises(RuntimeError, match="forward failed"):
        harness._collect_forward_hook_activations(
            model=_FailingModel(),
            inputs=torch.ones(2, 3),
        )

    assert harness.removed_handles == [["handle"]]
