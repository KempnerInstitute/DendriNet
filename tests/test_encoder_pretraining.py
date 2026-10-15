"""Reconstruction pretraining of autoencoder encoders (training.encoder_pretraining)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.config import EncoderPretrainingConfig, config_from_mapping
from dendritic_modeling.networks.architectures.classical.autoencoder import (
    CNNAutoencoder,
    MLPAutoencoder,
    MLPVariationalAutoencoder,
)
from dendritic_modeling.scripts.script_utils import setup_utils
from dendritic_modeling.scripts.training import train_experiments
from dendritic_modeling.training.encoder_pretraining import pretrain_autoencoder


def _dataset(n: int = 64, width: int = 12, seed: int = 0) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.rand(n, width, generator=generator)
    return TensorDataset(inputs, torch.arange(n) % 3)


def _autoencoder(cls=MLPAutoencoder):
    torch.manual_seed(0)
    return cls(
        input_dim=12, encoder_hidden_dims=[8], latent_dim=4, decoder_hidden_dims=[8]
    )


@pytest.mark.parametrize(
    "field, value",
    [
        ("epochs", -1),
        ("batch_size", 0),
        ("patience", 0),
        ("learning_rate", 0.0),
        ("weight_decay", -0.1),
        ("grad_clip_value", 0.0),
        ("loss_function", "ce"),
    ],
)
def test_invalid_settings_raise(field, value):
    with pytest.raises(ValueError, match=f"training.encoder_pretraining.{field}"):
        EncoderPretrainingConfig(**{field: value})


def test_the_section_loads_and_is_off_by_default():
    assert config_from_mapping({}).training.encoder_pretraining.epochs == 0
    loaded = config_from_mapping(
        {"training": {"encoder_pretraining": {"epochs": 3, "loss_function": "BCE"}}}
    ).training.encoder_pretraining
    assert (loaded.epochs, loaded.loss_function) == (3, "bce")
    with pytest.raises(ValueError, match="encoder_pretraining"):
        config_from_mapping({"training": {"encoder_pretraining": {"epoch": 3}}})


@pytest.mark.parametrize(
    "cls, loss", [(MLPAutoencoder, "mse"), (MLPVariationalAutoencoder, "vae_elbo")]
)
def test_pretraining_lowers_the_reconstruction_loss(cls, loss, tmp_path):
    autoencoder = _autoencoder(cls)
    settings = EncoderPretrainingConfig(
        epochs=15, batch_size=16, loss_function=loss, learning_rate=0.01
    )
    summary = pretrain_autoencoder(
        autoencoder,
        _dataset(),
        _dataset(seed=1),
        settings,
        save_path=str(tmp_path),
        device="cpu",
    )
    assert len(summary["train_losses"]) == 15
    assert summary["train_losses"][-1] < 0.5 * summary["train_losses"][0]
    assert summary["best_valid_loss"] == min(summary["valid_losses"])


def test_the_elbo_needs_a_variational_autoencoder():
    with pytest.raises(ValueError, match="variational autoencoder"):
        pretrain_autoencoder(
            _autoencoder(),
            _dataset(),
            _dataset(),
            EncoderPretrainingConfig(epochs=1, loss_function="vae_elbo"),
        )
    with pytest.raises(TypeError, match="autoencoder"):
        pretrain_autoencoder(
            torch.nn.Linear(12, 12),
            _dataset(),
            _dataset(),
            EncoderPretrainingConfig(epochs=1),
        )


@pytest.mark.parametrize("cls", [MLPAutoencoder, MLPVariationalAutoencoder])
def test_a_frozen_encoder_stays_fixed_in_evaluation_mode(cls):
    autoencoder = _autoencoder(cls)
    encoder = autoencoder.encoder
    keys = list(encoder.state_dict())
    encoder.freeze()
    autoencoder.train()
    assert encoder.frozen and not encoder.training
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    assert list(encoder.state_dict()) == keys
    inputs = torch.rand(5, 12)
    assert torch.equal(encoder(inputs), encoder(inputs))


def test_a_cnn_encoder_freezes():
    torch.manual_seed(0)
    autoencoder = CNNAutoencoder(
        input_shape=(1, 8, 8),
        encoder_hidden_dims=[4],
        latent_shape=(2, 2, 2),
        decoder_hidden_dims=[4],
    )
    autoencoder.encoder.freeze()
    autoencoder.train()
    assert not autoencoder.encoder.training


def _settings(**options):
    return EncoderPretrainingConfig(
        **{"epochs": 2, "batch_size": 16, "learning_rate": 0.01, **options}
    )


def _encoder_datasets(encoder_network, **options):
    data = _dataset()
    return train_experiments._encoder_datasets(
        encoder_network=encoder_network,
        train_ds=data,
        valid_ds=data,
        test_ds=data,
        **options,
    )


def test_the_entry_point_pretrains_and_freezes_the_encoder(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    autoencoder = _autoencoder()
    before = {key: value.clone() for key, value in autoencoder.state_dict().items()}
    _encoder_datasets(
        autoencoder, pretraining=_settings(), run_save_path=str(tmp_path), seed=3
    )
    after = autoencoder.state_dict()
    assert any(not torch.equal(before[key], after[key].cpu()) for key in before)
    assert autoencoder.encoder.frozen
    summary = json.loads(
        (tmp_path / "encoder_pretraining" / "summary.json").read_text()
    )
    assert len(summary["train_losses"]) == 2


def test_an_unfrozen_encoder_keeps_training():
    autoencoder = _autoencoder()
    _encoder_datasets(autoencoder, pretraining=_settings(freeze_encoder=False))
    assert not autoencoder.encoder.frozen
    assert all(
        parameter.requires_grad for parameter in autoencoder.encoder.parameters()
    )


def test_a_checkpoint_supplies_the_trained_encoder():
    autoencoder = _autoencoder()
    before = {key: value.clone() for key, value in autoencoder.state_dict().items()}
    _encoder_datasets(
        autoencoder, pretraining=_settings(epochs=0), initial_checkpoint="run.pt"
    )
    assert all(
        torch.equal(before[key], value)
        for key, value in autoencoder.state_dict().items()
    )
    assert autoencoder.encoder.frozen


@pytest.mark.parametrize(
    "encoder, options, message",
    [
        ("plain", {"pretraining": _settings()}, "which the main phase trains"),
        ("autoencoder", {"pretraining": _settings(epochs=0)}, "must be trained"),
        (
            "autoencoder",
            {"pretraining": _settings(), "initial_checkpoint": "run.pt"},
            "already holds the trained encoder",
        ),
        (
            "autoencoder",
            {"pretraining": _settings(), "world_size": 2},
            "single-process",
        ),
        ("autoencoder", {"pretraining": _settings(), "fsdp": True}, "FSDP"),
    ],
)
def test_unsupported_combinations_raise(encoder, options, message):
    network = (
        _autoencoder() if encoder == "autoencoder" else SimpleNamespace(output_dim=12)
    )
    with pytest.raises(ValueError, match=message):
        _encoder_datasets(network, **options)


def _experiment(**pretraining):
    return {
        "experiment": {"seed": 0},
        "data": {"dataset_name": "mnist"},
        "model": {
            "task": "classification",
            "encoder": {
                "type": "mlpautoencoder",
                "params": {
                    "input_dim": 12,
                    "encoder_hidden_dims": [8],
                    "latent_dim": 4,
                    "decoder_hidden_dims": [8],
                },
            },
            "core": {
                "biological_neuron": False,
                "type": "einet",
                "architecture": {
                    "excitatory_layer_sizes": [6],
                    "excitatory_branch_factors": [2],
                },
                "connectivity": {"ee_synapses_per_branch_per_layer": [2]},
            },
            "decoder": {"type": "MLP", "params": {"hidden_dims": [], "output_dim": 3}},
        },
        "training": {
            "encoder_pretraining": {
                "epochs": 3,
                "batch_size": 16,
                "learning_rate": 0.01,
                **pretraining,
            },
            "main": {
                "strategy": "standard",
                "common": {
                    "epochs": 1,
                    "batch_size": 16,
                    "loss_function": "ce",
                    "early_stopping": False,
                    "load_best_state_dict": False,
                },
                "optimizer": {"name": "adam"},
            },
        },
    }


def test_a_pretrained_frozen_encoder_trains_the_model_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    data = _dataset(n=48)
    monkeypatch.setattr(
        setup_utils, "get_unified_datasets", lambda *, task_cfg: (data, data, data)
    )
    monkeypatch.setattr(
        setup_utils.logger_manager, "set_log_directory", lambda *_: None
    )
    config = config_from_mapping(_experiment())
    run_path, train_ds, valid_ds, _test, model, encoder_network = (
        setup_utils.setup_environment(
            model_config=config.model,
            training_config=config.training,
            data_config=config.data,
            outputs_config=SimpleNamespace(results_dir=str(tmp_path), run_name="ae"),
            experiment_config=config.experiment,
            is_main=True,
        )
    )
    assert model.encoder_network is encoder_network.encoder
    _encoder, train_ds, valid_ds, _ = train_experiments._encoder_datasets(
        encoder_network=encoder_network,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=valid_ds,
        pretraining=config.training.encoder_pretraining,
        run_save_path=run_path,
    )
    train_experiments._install_encoder_network(model, encoder_network)
    pretrained = {
        key: value.detach().cpu().clone()
        for key, value in model.encoder_network.state_dict().items()
    }
    core_before = {
        key: value.detach().cpu().clone()
        for key, value in model.core_network.state_dict().items()
    }
    setup = train_experiments._build_main_trainer(
        unwrapped_model=model,
        training_config=config.training,
        model_config=config.model,
        experiment_config=config.experiment,
        run_save_path=run_path,
        encoded_train_ds=train_ds,
        analysis_manager=None,
    )
    setup.trainer.train(model, train_ds, valid_ds)

    encoder_after = model.encoder_network.state_dict()
    assert all(
        torch.equal(value, encoder_after[key].cpu())
        for key, value in pretrained.items()
    )
    core_after = model.core_network.state_dict()
    assert any(
        value.is_floating_point() and not torch.equal(value, core_after[key].cpu())
        for key, value in core_before.items()
    )
    logits = model(torch.rand(2, 12, device=next(model.parameters()).device))
    assert logits.shape == (2, 3)


def test_builtin_types_tell_whether_they_build_an_autoencoder():
    from dendritic_modeling.networks.architectures.factory import builds_autoencoder

    for name in (
        "mlpautoencoder",
        "CNNAutoencoder",
        "mlpvariationalautoencoder",
        "cnnvariationalautoencoder",
    ):
        assert builds_autoencoder(name) is True
    for name in ("identity", "mlp", "cnn", "einet", "population_network"):
        assert builds_autoencoder(name) is False
    assert builds_autoencoder("not_a_registered_architecture") is None


def _write_experiment(tmp_path, *, encoder=None, checkpoint=None, **pretraining):
    import yaml

    experiment = _experiment(**pretraining)
    if encoder is not None:
        experiment["model"]["encoder"] = encoder
    if checkpoint is not None:
        experiment["model"]["initial_checkpoint"] = str(tmp_path / checkpoint)
    experiment["outputs"] = {"results_dir": str(tmp_path / "results")}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(experiment))
    return path


_REJECTED_REQUESTS = [
    ({"epochs": 0}, None, None, "must be trained"),
    ({}, None, "trained_run.pt", "already holds the trained encoder"),
    (
        {},
        {"type": "identity", "params": {"input_dim": 12}},
        None,
        "which the main phase trains",
    ),
]


@pytest.mark.parametrize(
    "pretraining, encoder, checkpoint, message", _REJECTED_REQUESTS
)
def test_a_rejected_encoder_request_creates_no_run_directory(
    monkeypatch, tmp_path, pretraining, encoder, checkpoint, message
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    data = _dataset(n=48)
    monkeypatch.setattr(
        setup_utils, "get_unified_datasets", lambda *, task_cfg: (data, data, data)
    )
    monkeypatch.setattr(
        setup_utils.logger_manager, "set_log_directory", lambda *_: None
    )
    config_path = _write_experiment(
        tmp_path, encoder=encoder, checkpoint=checkpoint, **pretraining
    )
    with pytest.raises(ValueError, match=message):
        train_experiments.main(str(config_path), args=None)
    results = tmp_path / "results"
    assert not results.exists() or not any(results.iterdir())


@pytest.mark.parametrize(
    "options, message",
    [
        ({"pretraining": _settings(epochs=0)}, "must be trained"),
        (
            {"pretraining": _settings(), "initial_checkpoint": "run.pt"},
            "already holds the trained encoder",
        ),
        ({"pretraining": _settings(), "world_size": 2}, "single-process"),
        ({"pretraining": _settings(), "fsdp": True}, "FSDP"),
    ],
)
def test_the_early_check_raises_what_the_built_encoder_check_raises(options, message):
    config = config_from_mapping(_experiment())
    with pytest.raises(ValueError, match=message) as early:
        train_experiments._validate_encoder_request(
            SimpleNamespace(
                encoder=config.model.encoder,
                initial_checkpoint=options.get("initial_checkpoint"),
            ),
            SimpleNamespace(encoder_pretraining=options["pretraining"]),
            world_size=options.get("world_size", 1),
            fsdp=options.get("fsdp", False),
        )
    with pytest.raises(ValueError) as built:
        _encoder_datasets(_autoencoder(), **options)
    assert str(early.value) == str(built.value)


def test_valid_encoder_requests_pass_the_early_check():
    encoder = config_from_mapping(_experiment()).model.encoder
    for pretraining, checkpoint in ((_settings(), None), (_settings(epochs=0), "r.pt")):
        train_experiments._validate_encoder_request(
            SimpleNamespace(encoder=encoder, initial_checkpoint=checkpoint),
            SimpleNamespace(encoder_pretraining=pretraining),
            world_size=1,
        )
    identity = SimpleNamespace(
        encoder=SimpleNamespace(type="identity"), initial_checkpoint=None
    )
    train_experiments._validate_encoder_request(
        identity, SimpleNamespace(encoder_pretraining=_settings(epochs=0)), world_size=2
    )


def test_an_extension_encoder_is_checked_once_built():
    from dendritic_modeling.networks.architectures.factory import (
        builds_autoencoder,
        get_architecture,
    )
    from dendritic_modeling.networks.architectures.registry import (
        register_architecture,
        unregister_architecture,
    )

    register_architecture(
        "test_extension_autoencoder",
        lambda parameters, **_: _autoencoder(),
    )
    try:
        assert builds_autoencoder("test_extension_autoencoder") is None
        model = SimpleNamespace(
            encoder=SimpleNamespace(type="test_extension_autoencoder"),
            initial_checkpoint=None,
        )
        # Only building the extension tells that it is an autoencoder.
        train_experiments._validate_encoder_request(
            model,
            SimpleNamespace(encoder_pretraining=_settings(epochs=0)),
            world_size=1,
        )
        with pytest.raises(ValueError, match="must be trained"):
            _encoder_datasets(
                get_architecture("test_extension_autoencoder", {}),
                pretraining=_settings(epochs=0),
            )
    finally:
        unregister_architecture("test_extension_autoencoder")
