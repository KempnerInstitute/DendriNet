import pytest
import torch
import torch.nn.functional as F

from dendritic_modeling.training._transformer_replacement.common import (
    _compute_distillation_loss as _compute_transformer_distillation_loss,
)
from dendritic_modeling.training.replacement_common import (
    _compute_distillation_loss as _compute_shared_distillation_loss,
    _resolve_dtype,
)
from dendritic_modeling.training.vision_replacement import (
    _compute_distillation_loss as _compute_vision_distillation_loss,
)


def test_transformer_distillation_loss_matches_previous_formula():
    prediction = torch.tensor(
        [
            [[0.2, -0.4, 1.0], [0.5, 0.1, -0.3]],
            [[-0.7, 0.8, 0.4], [0.9, -0.2, 0.6]],
        ],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [
            [[0.1, -0.2, 0.7], [0.4, -0.1, -0.5]],
            [[-0.5, 0.9, 0.2], [0.8, -0.4, 0.3]],
        ],
        dtype=torch.float64,
    )
    cosine_weight = 0.35

    expected = F.mse_loss(prediction, target)
    expected = expected + cosine_weight * (
        1.0
        - F.cosine_similarity(
            prediction.reshape(-1, prediction.shape[-1]),
            target.reshape(-1, target.shape[-1]),
            dim=-1,
        ).mean()
    )

    actual = _compute_transformer_distillation_loss(
        prediction,
        target,
        loss_name="mse",
        cosine_weight=cosine_weight,
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_vision_distillation_loss_matches_previous_formula():
    prediction = torch.tensor(
        [
            [[0.2, -0.4, 1.0], [0.5, 0.1, -0.3]],
            [[-0.7, 0.8, 0.4], [0.9, -0.2, 0.6]],
        ],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [
            [[0.1, -0.2, 0.7], [0.4, -0.1, -0.5]],
            [[-0.5, 0.9, 0.2], [0.8, -0.4, 0.3]],
        ],
        dtype=torch.float64,
    )
    cosine_weight = 0.2

    expected = F.smooth_l1_loss(prediction, target)
    expected = expected + cosine_weight * (
        1.0
        - F.cosine_similarity(
            prediction.reshape(prediction.shape[0], -1),
            target.reshape(target.shape[0], -1),
            dim=-1,
        ).mean()
    )

    actual = _compute_vision_distillation_loss(
        prediction,
        target,
        loss_name="smooth_l1",
        cosine_weight=cosine_weight,
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_transformer_and_vision_cosine_flattening_remain_domain_specific():
    prediction = torch.tensor(
        [
            [[10.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 1.0]],
        ],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [
            [[10.0, 0.0], [1.0, 0.0]],
            [[2.0, 0.0], [1.0, 0.0]],
        ],
        dtype=torch.float64,
    )

    transformer_loss = _compute_transformer_distillation_loss(
        prediction,
        target,
        loss_name="mse",
        cosine_weight=1.0,
    )
    vision_loss = _compute_vision_distillation_loss(
        prediction,
        target,
        loss_name="mse",
        cosine_weight=1.0,
    )

    assert transformer_loss.item() != vision_loss.item()


def test_distillation_loss_error_messages_preserve_domain_context():
    prediction = torch.zeros(2, 3)
    target = torch.ones(2, 3)

    with pytest.raises(ValueError, match="transformer replacement loss"):
        _compute_transformer_distillation_loss(
            prediction,
            target,
            loss_name="mae",
        )

    with pytest.raises(ValueError, match="vision replacement loss"):
        _compute_vision_distillation_loss(
            prediction,
            target,
            loss_name="mae",
        )


def test_shared_dtype_resolution_accepts_aliases():
    expected = {
        "float": torch.float32,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }

    for alias, dtype in expected.items():
        assert _resolve_dtype(alias) is dtype
        assert _resolve_dtype(alias.upper()) is dtype


def test_shared_dtype_resolution_rejects_unknown_alias():
    with pytest.raises(ValueError, match="Unknown dtype 'float64'"):
        _resolve_dtype("float64")


def test_shared_distillation_loss_rejects_unknown_cosine_flatten_mode():
    prediction = torch.zeros(2, 3)
    target = torch.ones(2, 3)

    with pytest.raises(ValueError, match="cosine_flatten"):
        _compute_shared_distillation_loss(
            prediction,
            target,
            loss_name="mse",
            cosine_weight=1.0,
            cosine_flatten="token",
        )


def test_shared_distillation_loss_skips_cosine_validation_when_weight_is_zero():
    prediction = torch.zeros(2, 3)
    target = torch.ones(2, 3)

    actual = _compute_shared_distillation_loss(
        prediction,
        target,
        loss_name="l2",
        cosine_weight=0.0,
        cosine_flatten="token",
    )

    torch.testing.assert_close(actual, F.mse_loss(prediction, target))
