import pytest
import torch
import torch.nn as nn

from dendritic_modeling.models import RecurrentClassifier
from dendritic_modeling.training.utils.evaluation import (
    _CachedOutputModel,
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
