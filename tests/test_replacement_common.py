import json

import pytest
import torch
import torch.nn.functional as F

from dendritic_modeling.training._transformer_replacement.common import (
    _compute_distillation_loss as _compute_transformer_distillation_loss,
)
from dendritic_modeling.training.replacement_common import (
    _compute_distillation_loss as _compute_shared_distillation_loss,
    _infinite_loader,
    _make_cuda_grad_scaler,
    _resolve_dtype,
    _write_metrics_json,
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


def test_shared_amp_scaler_is_disabled_for_cpu_device():
    scaler = _make_cuda_grad_scaler(use_amp=True, device=torch.device("cpu"))

    assert scaler.is_enabled() is False


def test_shared_amp_scaler_respects_disabled_amp_on_cuda_device():
    scaler = _make_cuda_grad_scaler(use_amp=False, device=torch.device("cuda"))

    assert scaler.is_enabled() is False


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


def test_shared_infinite_loader_restarts_without_cycle_cache():
    dataset = torch.utils.data.TensorDataset(torch.tensor([1, 2, 3]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    iterator = _infinite_loader(loader)
    observed = [int(next(iterator)[0].item()) for _ in range(5)]

    assert observed == [1, 2, 3, 1, 2]


def test_write_metrics_json_supports_default_and_explicit_output_dir(tmp_path):
    class Result:
        save_dir = str(tmp_path / "train")

        def asdict(self):
            return {"loss": 0.25, "steps": [1, 2]}

    result = Result()

    _write_metrics_json(result)
    _write_metrics_json(
        result,
        output_dir=tmp_path / "benchmark",
        filename="benchmark_metrics.json",
    )

    assert json.loads((tmp_path / "train" / "metrics.json").read_text()) == {
        "loss": 0.25,
        "steps": [1, 2],
    }
    assert json.loads(
        (tmp_path / "benchmark" / "benchmark_metrics.json").read_text()
    ) == {
        "loss": 0.25,
        "steps": [1, 2],
    }
