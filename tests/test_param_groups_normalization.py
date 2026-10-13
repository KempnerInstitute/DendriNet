import torch.nn as nn

from dendritic_modeling.config.regularization import RegularizationConfig
from dendritic_modeling.config.training import ParamGroupsConfig
from dendritic_modeling.models.base import BaseModel
from dendritic_modeling.networks import (
    BlockLinear,
    IndexedDynamicTopKLinear,
    IndexedSparseLinear,
    ParametricTanh,
    TopKLinear,
)
from dendritic_modeling.scripts.script_utils.config_utils import (
    normalize_param_groups_config,
    prepare_trainer_config,
)
from dendritic_modeling.training.regularization import RegularizationManager


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
    assert groups_normalized[0]["name"] == "all"


def test_base_model_split_param_groups_are_named():
    model = _DummyModel()
    model.core_network = nn.Linear(2, 2)
    model.decoder_network = nn.Linear(2, 1)

    groups = model.get_param_groups(ParamGroupsConfig(lr=0.02, split_params=True))
    names = {group["name"] for group in groups}

    assert "decoder" in names
    assert "catch_all" in names
    assert all("name" in group for group in groups)


def test_base_model_split_param_groups_preserve_core_module_order():
    model = _DummyModel()
    model.core_network = nn.Sequential(
        TopKLinear(4, 3, K=2),
        BlockLinear(4, 2),
        ParametricTanh(2),
    )
    model.decoder_network = nn.Linear(2, 1)
    config = ParamGroupsConfig(
        lr=0.05,
        topk_lr=0.01,
        blocklinear_lr=0.02,
        reactivation_lr=0.03,
        decoder_lr=0.04,
        split_params=True,
    )

    groups = model.get_param_groups(config)

    assert [(group["name"], group["lr"]) for group in groups] == [
        ("topk", 0.01),
        ("blocklinear", 0.02),
        ("reactivation", 0.03),
        ("decoder", 0.04),
    ]
    param_group_by_id = {
        id(param): group["name"] for group in groups for param in group["params"]
    }
    assert param_group_by_id[id(model.core_network[0].pre_w)] == "topk"
    assert param_group_by_id[id(model.core_network[1].log_weight)] == "blocklinear"
    assert param_group_by_id[id(model.core_network[2].log_m)] == "reactivation"
    assert param_group_by_id[id(model.decoder_network.weight)] == "decoder"


def test_base_model_assigns_indexed_synapses_to_topk_group():
    model = _DummyModel()
    model.core_network = nn.Sequential(
        IndexedSparseLinear(4, 3, K=2, seed=1),
        IndexedDynamicTopKLinear(3, 2, K=1, candidate_size=2, seed=2),
    )
    model.decoder_network = nn.Linear(2, 1)

    groups = model.get_param_groups(
        ParamGroupsConfig(
            lr=0.05,
            topk_lr=0.01,
            decoder_lr=0.04,
            split_params=True,
        )
    )
    group_by_param_id = {
        id(param): group for group in groups for param in group["params"]
    }

    assert group_by_param_id[id(model.core_network[0].pre_w)]["name"] == "topk"
    assert group_by_param_id[id(model.core_network[0].pre_w)]["lr"] == 0.01
    assert group_by_param_id[id(model.core_network[1].pre_w)]["name"] == "topk"
    assert group_by_param_id[id(model.core_network[1].pre_w)]["lr"] == 0.01


def test_base_model_scales_only_decoder_input_weight():
    model = _DummyModel()
    model.decoder_network = nn.Sequential(
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )

    groups = model.get_param_groups(
        ParamGroupsConfig(
            lr=0.05,
            decoder_input_lr=1e-5,
            decoder_lr=1e-3,
            split_params=True,
        )
    )
    group_by_param_id = {
        id(param): group for group in groups for param in group["params"]
    }

    assert group_by_param_id[id(model.decoder_network[0].weight)]["name"] == (
        "decoder_input"
    )
    assert group_by_param_id[id(model.decoder_network[0].weight)]["lr"] == 1e-5
    assert group_by_param_id[id(model.decoder_network[0].bias)]["name"] == "decoder"
    assert group_by_param_id[id(model.decoder_network[2].weight)]["name"] == "decoder"
    assert group_by_param_id[id(model.decoder_network[2].bias)]["lr"] == 1e-3


def test_regularization_uses_explicit_param_group_name():
    manager = RegularizationManager(RegularizationConfig())

    assert manager._identify_group_type({"name": "mlp_inh_net"}, 3) == "mlp_inh_net"


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
                "grad_clip_value": 0.5,
                "use_amp": True,
                "lr_schedule": "cosine",
                "lr_warmup_epochs": 2,
            }
        },
    }
    experiment_config = {
        "checkpointing": {"enabled": False},
    }

    trainer_cfg = prepare_trainer_config(main_train_config, experiment_config, "/tmp")
    assert "recurrent_config" in trainer_cfg
    assert trainer_cfg["recurrent_config"]["grad_clip_value"] == 0.5
    assert trainer_cfg["recurrent_config"]["use_amp"] is True
    assert trainer_cfg["recurrent_config"]["lr_schedule"] == "cosine"


def test_prepare_trainer_config_enables_standard_and_fsdp_checkpoint_keys():
    main_train_config = {
        "strategy": "standard",
        "common": {
            "epochs": 3,
            "batch_size": 4,
            "param_groups": {"lr": 0.001},
        },
    }
    experiment_config = {
        "checkpointing": {
            "enabled": True,
            "save_every_n_epochs": 7,
            "checkpoint_dir": "/tmp/explicit-checkpoints",
        },
    }

    trainer_cfg = prepare_trainer_config(main_train_config, experiment_config, "/tmp")
    assert trainer_cfg["checkpointing"] is True
    assert trainer_cfg["checkpoint_interval"] == 7
    assert trainer_cfg["save_every_n_epochs"] == 7
    # Checkpoints are written under the run directory; checkpoint_dir is retired.
    assert "checkpoint_dir" not in trainer_cfg
