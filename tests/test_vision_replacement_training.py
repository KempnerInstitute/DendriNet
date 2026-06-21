import logging

import pytest
import torch
import torch.nn as nn

from dendritic_modeling.config import Config
from dendritic_modeling.config.training import VisionReplacementTrainingConfig
from dendritic_modeling.training.vision_replacement import (
    _infinite_loader,
    _make_vision_loader,
    _select_vision_trainable_parameters,
    run_vision_replacement_training,
)


class _TinyVisionReplacementModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_network = nn.Identity()
        self.core_network = nn.Linear(4, 3)
        self.decoder_network = nn.Linear(3, 2)

    def forward(self, x):
        x = self.encoder_network(x)
        x = self.core_network(x)
        return self.decoder_network(x)


def _small_vision_distillation_config(tmp_path):
    cfg = Config()
    cfg.experiment.seed = 123
    cfg.training.main.optimizer.name = "adamw"
    cfg.training.main.optimizer.lr = 0.05
    cfg.training.main.optimizer.weight_decay = 0.0
    cfg.training.vision_replacement = VisionReplacementTrainingConfig(
        enabled=True,
        mode="blockwise_distillation",
        teacher_source="tensor_cache",
        train_target="replacement_only",
        device="cpu",
        dtype="float32",
        seed=123,
        batch_size=16,
        max_steps=60,
        eval_every=20,
        log_every=100,
        save_dir=str(tmp_path),
        save_replacement=True,
    )
    return cfg


def test_tensor_cache_vision_replacement_distillation_reduces_loss(
    tmp_path,
    monkeypatch,
):
    torch.set_num_threads(1)
    cfg = _small_vision_distillation_config(tmp_path)

    generator = torch.Generator().manual_seed(123)
    train_x = torch.randn(96, 4, generator=generator)
    valid_x = torch.randn(48, 4, generator=generator)
    teacher = nn.Linear(4, 3, bias=True)
    with torch.no_grad():
        teacher.weight.copy_(
            torch.tensor(
                [
                    [1.0, -0.5, 0.25, 0.1],
                    [-0.2, 0.7, 0.3, -0.4],
                    [0.5, 0.2, -0.6, 0.8],
                ]
            )
        )
        teacher.bias.copy_(torch.tensor([0.1, -0.2, 0.05]))
    train_y = teacher(train_x).detach()
    valid_y = teacher(valid_x).detach()

    paths = {
        "train_inputs": tmp_path / "train_inputs.pt",
        "train_targets": tmp_path / "train_targets.pt",
        "valid_inputs": tmp_path / "valid_inputs.pt",
        "valid_targets": tmp_path / "valid_targets.pt",
    }
    torch.save(train_x, paths["train_inputs"])
    torch.save(train_y, paths["train_targets"])
    torch.save(valid_x, paths["valid_inputs"])
    torch.save(valid_y, paths["valid_targets"])
    cfg.training.vision_replacement.tensor_cache = {
        key: str(value) for key, value in paths.items()
    }

    monkeypatch.setattr(
        "dendritic_modeling.training.vision_replacement.initialize_model",
        lambda model_config: (_TinyVisionReplacementModel(), None),
    )

    result = run_vision_replacement_training(cfg)

    assert result.final_valid_loss < result.initial_valid_loss
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "replacement_core.pt").exists()


def test_image_cache_full_model_distillation_saves_student_checkpoint(
    tmp_path,
    monkeypatch,
):
    torch.set_num_threads(1)
    cfg = _small_vision_distillation_config(tmp_path)
    cfg.training.vision_replacement.teacher_source = "image_tensor_cache"
    cfg.training.vision_replacement.train_target = "full_model"
    cfg.training.vision_replacement.max_steps = 1
    cfg.training.vision_replacement.eval_every = 1

    generator = torch.Generator().manual_seed(321)
    train_x = torch.randn(16, 4, generator=generator)
    valid_x = torch.randn(8, 4, generator=generator)
    train_y = torch.randn(16, 2, generator=generator)
    valid_y = torch.randn(8, 2, generator=generator)

    paths = {
        "train_inputs": tmp_path / "train_inputs.pt",
        "train_targets": tmp_path / "train_targets.pt",
        "valid_inputs": tmp_path / "valid_inputs.pt",
        "valid_targets": tmp_path / "valid_targets.pt",
    }
    torch.save(train_x, paths["train_inputs"])
    torch.save(train_y, paths["train_targets"])
    torch.save(valid_x, paths["valid_inputs"])
    torch.save(valid_y, paths["valid_targets"])
    cfg.training.vision_replacement.tensor_cache = {
        key: str(value) for key, value in paths.items()
    }

    monkeypatch.setattr(
        "dendritic_modeling.training.vision_replacement.initialize_model",
        lambda model_config: (_TinyVisionReplacementModel(), None),
    )

    run_vision_replacement_training(cfg)

    student_payload = torch.load(
        tmp_path / "student_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert student_payload["train_target"] == "full_model"
    assert "core_network.weight" in student_payload["state_dict"]
    assert "decoder_network.weight" in student_payload["state_dict"]


def test_vision_trainable_parameter_selection_supports_full_model():
    model = _TinyVisionReplacementModel()

    replacement_params = _select_vision_trainable_parameters(
        model,
        "replacement_only",
    )
    assert {id(param) for param in replacement_params} == {
        id(param) for param in model.core_network.parameters()
    }
    assert all(param.requires_grad for param in model.core_network.parameters())
    assert all(not param.requires_grad for param in model.decoder_network.parameters())

    full_model_params = _select_vision_trainable_parameters(model, "full_model")
    assert {id(param) for param in full_model_params} == {
        id(param) for param in model.parameters()
    }
    assert all(param.requires_grad for param in model.parameters())


def test_vision_trainable_parameter_selection_restores_after_empty_replacement():
    model = _TinyVisionReplacementModel()
    model.core_network = nn.Identity()
    model.decoder_network.bias.requires_grad_(False)
    original_states = [param.requires_grad for param in model.parameters()]

    with pytest.raises(ValueError, match="No vision replacement parameters"):
        _select_vision_trainable_parameters(model, "replacement_only")

    assert [param.requires_grad for param in model.parameters()] == original_states


def test_vision_replacement_loader_uses_configured_worker_options():
    dataset = torch.utils.data.TensorDataset(
        torch.arange(8).float().unsqueeze(1), torch.arange(8).float().unsqueeze(1)
    )
    cfg = VisionReplacementTrainingConfig(
        batch_size=2,
        num_workers=1,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=3,
    )

    loader = _make_vision_loader(
        dataset,
        cfg,
        device=torch.device("cpu"),
        shuffle=False,
    )

    assert loader.num_workers == 1
    assert loader.pin_memory is False
    assert loader.persistent_workers is True
    assert loader.prefetch_factor == 3


def test_vision_replacement_loader_logs_tuning_when_requested(caplog):
    dataset = torch.utils.data.TensorDataset(
        torch.arange(8).float().unsqueeze(1), torch.arange(8).float().unsqueeze(1)
    )
    cfg = VisionReplacementTrainingConfig(batch_size=2, num_workers=0)

    with caplog.at_level(
        logging.INFO,
        logger="dendritic_modeling.training.vision_replacement",
    ):
        _make_vision_loader(
            dataset,
            cfg,
            device=torch.device("cpu"),
            shuffle=False,
            log_tuning=True,
        )

    assert "Vision replacement DataLoader tuning:" in caplog.text
    assert "dataset=torch.utils.data.dataset.TensorDataset" in caplog.text
    assert "num_workers=0" in caplog.text


def test_vision_replacement_loader_rejects_invalid_persistent_workers():
    dataset = torch.utils.data.TensorDataset(torch.randn(4, 2), torch.randn(4, 1))
    cfg = VisionReplacementTrainingConfig(
        batch_size=2,
        num_workers=0,
        persistent_workers=True,
    )

    with pytest.raises(ValueError, match="persistent_workers"):
        _make_vision_loader(
            dataset,
            cfg,
            device=torch.device("cpu"),
            shuffle=False,
        )


def test_infinite_loader_restarts_without_itertools_cycle_cache():
    dataset = torch.utils.data.TensorDataset(
        torch.arange(3).unsqueeze(1), torch.arange(3).unsqueeze(1)
    )
    cfg = VisionReplacementTrainingConfig(batch_size=2, num_workers=0)
    loader = _make_vision_loader(
        dataset,
        cfg,
        device=torch.device("cpu"),
        shuffle=False,
    )

    iterator = _infinite_loader(loader)
    batches = [next(iterator)[0].squeeze(1).tolist() for _ in range(4)]

    assert batches == [[0, 1], [2], [0, 1], [2]]
