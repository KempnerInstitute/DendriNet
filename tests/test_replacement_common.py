import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from dendritic_modeling.training._transformer_replacement.common import (
    _compute_distillation_loss as _compute_transformer_distillation_loss,
)
from dendritic_modeling.training.replacement_common import (
    ReplacementTrainingHistory,
    _compute_distillation_loss as _compute_shared_distillation_loss,
    _infinite_loader,
    _load_required_tensor,
    _load_tensor_payload,
    _make_cuda_grad_scaler,
    _make_replacement_loader,
    _move_replacement_batch_tensors,
    _resolve_dtype,
    _should_run_periodic_step,
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


def test_shared_replacement_batch_tensor_move_preserves_values_and_casts_dtype():
    inputs = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    target = torch.tensor([[0.25, 0.5], [0.75, 1.0]], dtype=torch.float64)

    moved_inputs, moved_target = _move_replacement_batch_tensors(
        inputs,
        target,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert moved_inputs.dtype is torch.float32
    assert moved_target.dtype is torch.float32
    torch.testing.assert_close(moved_inputs, inputs.float())
    torch.testing.assert_close(moved_target, target.float())


def test_shared_tensor_payload_loader_preserves_loaded_payload(tmp_path):
    payload = {"layer_0": torch.tensor([1.0, 2.0])}
    path = tmp_path / "payload.pt"
    torch.save(payload, path)

    loaded = _load_tensor_payload(
        str(path),
        empty_path_message="missing tensor cache",
    )

    assert loaded.keys() == payload.keys()
    torch.testing.assert_close(loaded["layer_0"], payload["layer_0"])


def test_shared_required_tensor_loader_validates_path_and_type(tmp_path):
    tensor_path = tmp_path / "tensor.pt"
    dict_path = tmp_path / "dict.pt"
    expected = torch.tensor([[1.0, 2.0]])
    torch.save(expected, tensor_path)
    torch.save({"not": "a tensor"}, dict_path)

    loaded = _load_required_tensor(
        str(tensor_path),
        empty_path_message="empty path",
        type_error_message="must be tensor",
    )

    torch.testing.assert_close(loaded, expected)
    with pytest.raises(ValueError, match="empty path"):
        _load_required_tensor(
            "",
            empty_path_message="empty path",
            type_error_message="must be tensor",
        )
    with pytest.raises(TypeError, match="must be tensor"):
        _load_required_tensor(
            str(dict_path),
            empty_path_message="empty path",
            type_error_message="must be tensor",
        )


def test_shared_replacement_loader_preserves_configured_tuning():
    dataset = torch.utils.data.TensorDataset(
        torch.arange(5).unsqueeze(1),
        torch.arange(5).unsqueeze(1),
    )
    train_cfg = SimpleNamespace(
        num_workers=1,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=3,
        in_order=False,
    )

    loader = _make_replacement_loader(
        dataset,
        batch_size=2,
        shuffle=False,
        train_cfg=train_cfg,
        device="cpu",
    )

    assert loader.num_workers == 1
    assert loader.pin_memory is False
    assert loader.persistent_workers is True
    assert loader.prefetch_factor == 3
    assert loader.in_order is False
    assert loader.drop_last is False


def test_replacement_training_history_tracks_best_validation_loss():
    history = ReplacementTrainingHistory.start(1.5)

    history.record_train_loss(1.25)
    history.record_valid_loss(5, 1.6)
    history.record_valid_loss(10, 1.2)
    history.record_valid_loss(15, 1.2)

    assert history.initial_valid_loss == 1.5
    assert history.train_losses == [1.25]
    assert history.valid_losses == [1.5, 1.6, 1.2, 1.2]
    assert history.best_valid_loss == 1.2
    assert history.best_step == 10
    assert history.final_valid_loss == 1.2


def test_replacement_training_history_result_kwargs_match_direct_fields():
    history = ReplacementTrainingHistory.start(1.5)
    history.record_train_loss(1.25)
    history.record_valid_loss(5, 1.6)
    history.record_valid_loss(10, 1.2)

    payload = history.as_result_kwargs()

    assert payload == {
        "train_losses": [1.25],
        "valid_losses": [1.5, 1.6, 1.2],
        "initial_valid_loss": 1.5,
        "final_valid_loss": 1.2,
        "best_valid_loss": 1.2,
        "best_step": 10,
    }
    assert payload["train_losses"] is history.train_losses
    assert payload["valid_losses"] is history.valid_losses


def test_should_run_periodic_step_matches_loop_conditions():
    assert _should_run_periodic_step(1, every=100, max_steps=1)
    assert _should_run_periodic_step(10, every=5, max_steps=99)
    assert _should_run_periodic_step(40, every=7, max_steps=40)
    assert not _should_run_periodic_step(11, every=5, max_steps=40)


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
