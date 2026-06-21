import logging

import pytest
import torch
import torch.nn as nn

from dendritic_modeling.models import Classifier, RecurrentClassifier
from dendritic_modeling.training.utils.evaluation import (
    _CachedOutputModel,
    _compute_metric_score,
    _compute_pred_label_mi,
)


class _RecurrentIdentity(nn.Module):
    is_recurrent = True

    def forward(self, x, seq_lengths=None):
        return x


def test_cached_output_model_predict_matches_recurrent_classifier_argmax():
    model = RecurrentClassifier(nn.Identity(), nn.Identity(), nn.Identity())
    cached = torch.tensor(
        [
            [[0.1, 2.0, -1.0], [3.0, 0.0, 0.5]],
            [[-1.0, 0.0, 4.0], [0.2, 0.7, 0.1]],
        ]
    )

    wrapper = _CachedOutputModel(model, cached)

    assert torch.equal(wrapper.predict(), cached.argmax(dim=-1))


def test_pred_label_mi_normalizes_recurrent_sequence_shapes():
    model = RecurrentClassifier(nn.Identity(), _RecurrentIdentity(), nn.Identity())
    logits = torch.tensor(
        [
            [[4.0, 0.0], [0.0, 5.0]],
            [[3.0, 0.0], [0.0, 2.0]],
        ]
    )
    labels = torch.tensor([[0, 1], [0, 1]])

    score = _compute_pred_label_mi(model, logits, labels)

    assert score.item() == pytest.approx(1.0)


def test_compute_metric_score_logs_classifier_skip_without_stdout(caplog, capsys):
    model = Classifier(nn.Identity(), nn.Identity(), nn.Identity())
    inputs = torch.randn(2, 3)
    labels = torch.tensor([0, 1])

    def _compute_mse(*_args, **_kwargs):
        raise AssertionError("classifier skip should return before metric execution")

    with caplog.at_level(
        logging.DEBUG,
        logger="dendritic_modeling.training.utils.evaluation",
    ):
        score = _compute_metric_score(
            _compute_mse,
            model,
            inputs,
            labels,
            move_device=False,
        )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert torch.isnan(score)
    assert "Skipping _compute_mse for Classifier model" in caplog.text


def test_compute_metric_score_logs_exception_and_preserves_nan_return(caplog, capsys):
    model = nn.Identity()
    inputs = torch.randn(2, 3)
    labels = torch.randn(2, 3)

    def broken_metric(*_args, **_kwargs):
        raise RuntimeError("boom")

    with caplog.at_level(
        logging.WARNING,
        logger="dendritic_modeling.training.utils.evaluation",
    ):
        score = _compute_metric_score(
            broken_metric,
            model,
            inputs,
            labels,
            move_device=False,
        )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert torch.isnan(score)
    assert score.device == inputs.device
    assert "Error computing metric in broken_metric: boom" in caplog.text
    assert "input shapes: inputs=(2, 3), labels=(2, 3)" in caplog.text
