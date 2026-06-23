from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dendritic_modeling.analysis.tools.linearity_analysis import (
    LinearityAnalyzer,
    _count_linear_svc_margin_samples,
    compute_linear_decoder_performance,
)


def test_linearity_ensure_dataloader_wraps_dataset_without_replacing_loader():
    analyzer = LinearityAnalyzer(SimpleNamespace(n_samples=4))
    dataset = TensorDataset(torch.arange(12).reshape(6, 2).float(), torch.arange(6))
    loader = DataLoader(dataset, batch_size=2)

    wrapped = analyzer._ensure_dataloader(dataset)
    same_loader = analyzer._ensure_dataloader(loader)

    assert isinstance(wrapped, DataLoader)
    assert wrapped.batch_size == 256
    assert same_loader is loader


def test_linearity_collect_samples_preserves_existing_limit_semantics():
    analyzer = LinearityAnalyzer(SimpleNamespace(n_samples=5))
    x = torch.arange(14).reshape(7, 2).float()
    y = torch.arange(7)
    loader = DataLoader(TensorDataset(x, y), batch_size=3, shuffle=False)

    inputs, labels = analyzer._collect_samples(loader)

    torch.testing.assert_close(inputs, x[:5])
    torch.testing.assert_close(labels, y[:5])


def test_linearity_resolve_analysis_device_uses_model_device():
    analyzer = LinearityAnalyzer()
    model = nn.Linear(2, 1)

    assert analyzer._resolve_analysis_device(model, "cpu") == torch.device("cpu")


def test_linearity_per_class_linearity_uses_class_slices():
    analyzer = LinearityAnalyzer()
    inputs = torch.tensor([[1.0], [2.0], [10.0], [20.0]])
    labels = torch.tensor([0, 0, 1, 1])
    seen = []

    def fake_linearity(_model, class_inputs):
        seen.append(class_inputs.clone())
        return float(class_inputs.mean())

    analyzer._compute_linearity_index = fake_linearity

    result = analyzer._compute_per_class_linearity(nn.Identity(), inputs, labels)

    assert result == {0: 1.5, 1: 15.0}
    torch.testing.assert_close(seen[0], inputs[:2])
    torch.testing.assert_close(seen[1], inputs[2:])


def test_linear_decoder_performance_runs_with_linear_svc():
    class0 = np.array(
        [[-2.0, -1.5], [-1.8, -1.2], [-1.5, -1.4], [-1.2, -1.1], [-1.1, -1.6]]
    )
    class1 = np.array([[1.0, 1.4], [1.2, 1.1], [1.4, 1.6], [1.6, 1.3], [1.8, 1.5]])
    branch_outputs = np.vstack([class0, class1])
    labels = np.array([0] * len(class0) + [1] * len(class1))

    result = compute_linear_decoder_performance(
        branch_outputs, labels, test_fraction=0.4
    )

    assert set(result) == {
        "train_accuracy",
        "test_accuracy",
        "margin",
        "n_support_vectors",
    }
    assert result["train_accuracy"] == 1.0
    assert result["test_accuracy"] == 1.0
    assert result["margin"] >= 0.0
    assert 0 <= result["n_support_vectors"] <= 6


def test_linear_svc_margin_sample_count_binary():
    class DummyLinearSVC:
        classes_ = np.array([0, 1])

        def decision_function(self, _branch_outputs):
            return np.array([-1.2, -0.4, 0.8, 1.4])

    count = _count_linear_svc_margin_samples(
        DummyLinearSVC(),
        np.zeros((4, 2)),
        np.array([0, 0, 1, 1]),
    )

    assert count == 2
