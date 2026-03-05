import torch.nn as nn

from dendritic_modeling.config.training import ParamGroupsConfig
from dendritic_modeling.models.base import BaseModel
from dendritic_modeling.scripts.script_utils.config_utils import (
    normalize_param_groups_config,
    prepare_trainer_config,
)


class _DummyModel(BaseModel):
    def __init__(self):
        super().__init__(
            encoder_network=nn.Identity(),
            core_network=nn.Identity(),
            decoder_network=nn.Identity(),
        )

    def predict(self, x):
        return self.forward(x)


def test_normalize_param_groups_from_dict_applies_lr():
    model = _DummyModel()

    # Historical behavior with raw dict falls back to BaseModel defaults.
    groups_raw = model.get_param_groups({"lr": 0.02})
    assert groups_raw[0]["lr"] == 0.001

    normalized = normalize_param_groups_config({"lr": 0.02, "split_params": False})
    groups_normalized = model.get_param_groups(normalized)
    assert groups_normalized[0]["lr"] == 0.02


def test_normalize_param_groups_passthrough_for_typed_config():
    config = ParamGroupsConfig(lr=0.03, split_params=False, decoder_lr=0.07)
    normalized = normalize_param_groups_config(config)

    assert isinstance(normalized, ParamGroupsConfig)
    assert normalized.lr == 0.03
    assert normalized.decoder_lr == 0.07


def test_normalize_param_groups_ignores_unknown_fields():
    normalized = normalize_param_groups_config(
        {"lr": 0.04, "split_params": False, "unknown_key": 123}
    )

    assert isinstance(normalized, ParamGroupsConfig)
    assert normalized.lr == 0.04
    assert not hasattr(normalized, "unknown_key")


def test_prepare_trainer_config_includes_recurrent_strategy_block():
    main_train_config = {
        "strategy": "recurrent",
        "common": {
            "epochs": 3,
            "batch_size": 4,
            "param_groups": {"lr": 0.001, "split_params": False},
        },
        "strategies": {
            "recurrent": {
                "grad_clip": 0.5,
                "enable_amp": True,
                "lr_schedule": "cosine",
                "lr_warmup_epochs": 2,
            }
        },
    }
    experiment_config = {
        "checkpointing": {"enabled": False},
        "enable_profiling": False,
        "enable_amp": False,
        "enable_hooks": False,
    }

    trainer_cfg = prepare_trainer_config(main_train_config, experiment_config, "/tmp")
    assert "recurrent_config" in trainer_cfg
    assert trainer_cfg["recurrent_config"]["grad_clip"] == 0.5
    assert trainer_cfg["recurrent_config"]["enable_amp"] is True
    assert trainer_cfg["recurrent_config"]["lr_schedule"] == "cosine"
