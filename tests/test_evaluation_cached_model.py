import logging

import pytest
import torch
import torch.nn as nn

from dendritic_modeling.models import Classifier, RecurrentClassifier, Regressor
from dendritic_modeling.training.utils.evaluation import (
    _CachedOutputModel,
    _call_metric_computation,
    _compute_metric_score,
    _compute_pred_label_mi,
    _metric_accepts_seq_lengths,
    _prepare_classification_logits_and_labels,
    _prepare_regression_predictions_and_labels,
    _sequence_valid_mask,
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


def test_sequence_valid_mask_selects_timesteps_before_sequence_length():
    seq_lengths = torch.tensor([1, 3])

    mask = _sequence_valid_mask(seq_lengths, n_timesteps=4, device=torch.device("cpu"))

    assert mask.tolist() == [
        [True, False, False, False],
        [True, True, True, False],
    ]


def test_prepare_recurrent_classification_uses_shared_sequence_mask():
    model = RecurrentClassifier(nn.Identity(), _RecurrentIdentity(), nn.Identity())
    logits = torch.tensor(
        [
            [[9.0, 0.0], [0.0, 8.0], [7.0, 0.0]],
            [[0.0, 6.0], [5.0, 0.0], [0.0, 4.0]],
        ]
    )
    labels = torch.tensor([[0, 1, 0], [1, 0, 1]])
    seq_lengths = torch.tensor([2, 1])

    prepared_logits, prepared_labels = _prepare_classification_logits_and_labels(
        model,
        logits,
        labels,
        seq_lengths=seq_lengths,
    )

    torch.testing.assert_close(
        prepared_logits,
        torch.tensor([[9.0, 0.0], [0.0, 8.0], [0.0, 6.0]]),
    )
    torch.testing.assert_close(prepared_labels, torch.tensor([0, 1, 1]))


def test_prepare_recurrent_regression_uses_shared_sequence_mask():
    model = Regressor(nn.Identity(), _RecurrentIdentity(), nn.Identity())
    predictions = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ]
    )
    labels = predictions + 0.5
    seq_lengths = torch.tensor([1, 2])

    prepared_predictions, prepared_labels = _prepare_regression_predictions_and_labels(
        model,
        predictions,
        labels,
        seq_lengths=seq_lengths,
    )

    torch.testing.assert_close(
        prepared_predictions,
        torch.tensor([[1.0, 2.0], [7.0, 8.0], [9.0, 10.0]]),
    )
    torch.testing.assert_close(prepared_labels, prepared_predictions + 0.5)


def test_metric_call_helper_passes_seq_lengths_only_when_supported():
    model = nn.Identity()
    inputs = torch.randn(2, 3)
    labels = torch.randn(2, 3)
    seq_lengths = torch.tensor([1, 2])
    calls = []

    def metric_with_lengths(model, inputs, labels, reduce_dim, seq_lengths=None):
        calls.append(("with", model, inputs, labels, reduce_dim, seq_lengths))
        return inputs.sum()

    def metric_without_lengths(model, inputs, labels, reduce_dim):
        calls.append(("without", model, inputs, labels, reduce_dim))
        return labels.sum()

    assert _metric_accepts_seq_lengths(metric_with_lengths)
    assert not _metric_accepts_seq_lengths(metric_without_lengths)

    with_result = _call_metric_computation(
        metric_with_lengths,
        model,
        inputs,
        labels,
        1,
        seq_lengths=seq_lengths,
    )
    without_result = _call_metric_computation(
        metric_without_lengths,
        model,
        inputs,
        labels,
        1,
        seq_lengths=seq_lengths,
    )

    torch.testing.assert_close(with_result, inputs.sum())
    torch.testing.assert_close(without_result, labels.sum())
    assert calls[0][0] == "with"
    assert calls[0][1] is model
    assert calls[0][2] is inputs
    assert calls[0][3] is labels
    assert calls[0][4] == 1
    assert calls[0][5] is seq_lengths
    assert calls[1][0] == "without"
    assert calls[1][1] is model
    assert calls[1][2] is inputs
    assert calls[1][3] is labels
    assert calls[1][4] == 1


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
