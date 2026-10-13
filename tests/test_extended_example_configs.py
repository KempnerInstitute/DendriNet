"""Smoke tests for extended recurrent example configs."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from dendritic_modeling.config import load_config
from dendritic_modeling.config.conversion import to_plain_dict
from dendritic_modeling.datasets import get_unified_datasets
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

EXTENDED_EXAMPLES = [
    "configs/examples/population_network_recurrent_pyr_pv_sst.yaml",
    "configs/examples/population_network_synapse_types_ampa_nmda_gaba.yaml",
    "configs/examples/population_network_spiking.yaml",
    "configs/examples/population_network_soma_feedback.yaml",
    "configs/examples/population_network_taus_and_delays.yaml",
]

POPULATION_NETWORK_EXAMPLES = [
    "configs/examples/population_network_feedforward.yaml",
    "configs/examples/population_network_heterogeneous_feedforward.yaml",
    "configs/examples/population_network_two_layer.yaml",
    "configs/examples/population_network_cifar10.yaml",
    "configs/examples/population_network_spatial_morphology_cifar10.yaml",
    "configs/examples/population_network_dense_to_sparse_cifar10.yaml",
    "configs/examples/population_network_indexed_rewire_mnist.yaml",
    "configs/examples/population_network_normalized_additive_mnist.yaml",
    "configs/examples/population_network_local_learning_mnist.yaml",
]


@pytest.mark.parametrize("config_path", EXTENDED_EXAMPLES)
def test_extended_recurrent_example_config_builds_and_starts(config_path):
    cfg = load_config(config_path)
    model, _ = initialize_model(cfg.model)

    batch = 2
    seq_len = 4
    input_dim = int(cfg.model.encoder.params.input_dim)
    output_dim = int(cfg.model.decoder.params.output_dim)
    y = model(torch.rand(batch, seq_len, input_dim))

    assert Path(config_path).exists()
    assert y.shape == (batch, output_dim)
    assert torch.isfinite(y).all()
    assert model._is_recurrent_core


@pytest.mark.parametrize("config_path", POPULATION_NETWORK_EXAMPLES)
def test_population_network_example_config_builds_and_starts(config_path):
    cfg = load_config(config_path)
    model, _ = initialize_model(cfg.model)

    batch = 2
    input_dim = int(model.encoder_network.output_dim)
    output_dim = int(cfg.model.decoder.params.output_dim)
    y = model(torch.rand(batch, input_dim))

    assert Path(config_path).exists()
    assert y.shape == (batch, output_dim)
    assert torch.isfinite(y).all()
    assert not model._is_recurrent_core


@pytest.mark.parametrize("config_path", POPULATION_NETWORK_EXAMPLES)
def test_population_network_examples_make_reactivation_policy_explicit(config_path):
    cfg = load_config(config_path)
    layers = cfg.model.core.population_network["layers"]

    missing = []
    for layer in layers:
        defaults = layer.get("population_defaults", {})
        if defaults.get("reactivate", False) and not defaults.get(
            "reactivation_init_policy"
        ):
            missing.append(layer.get("name"))

    assert missing == []


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/examples/population_network_recurrent_pyr_pv_sst.yaml",
    ],
)
def test_recurrent_population_examples_drive_inhibitory_gradients(config_path):
    torch.manual_seed(0)
    cfg = load_config(config_path)
    model, _ = initialize_model(cfg.model)
    model.train()

    batch = 2
    seq_len = 5
    input_dim = int(cfg.model.encoder.params.input_dim)
    y = model(torch.rand(batch, seq_len, input_dim))
    loss = y.square().mean()
    loss.backward()

    watched = {
        name: param.grad
        for name, param in model.named_parameters()
        if param.requires_grad
        and ("core_network" in name)
        and (".i_population." in name or ".populations.i" in name or ".pv" in name)
    }
    assert watched, "No inhibitory population parameters found"
    dead = [
        name
        for name, grad in watched.items()
        if grad is None or not torch.isfinite(grad).all() or grad.abs().sum() == 0
    ]
    assert dead == []


def test_spiking_example_drives_core_gradients_on_configured_data():
    """The public spiking example should not be a silent readout-only model."""
    config_path = "configs/examples/population_network_spiking.yaml"
    torch.manual_seed(0)
    cfg = load_config(config_path)
    model, _ = initialize_model(cfg.model)
    model.train()

    dataset_params = to_plain_dict(cfg.data.dataset_params).get(
        cfg.data.dataset_name, {}
    )
    processing_params = to_plain_dict(cfg.data.processing)
    train_ds, _, _ = get_unified_datasets(
        {
            "dataset": cfg.data.dataset_name,
            "data_path": cfg.data.base_dir,
            "train_valid_split": cfg.experiment.train_valid_split,
            "parameters": {**processing_params, **dataset_params},
        }
    )
    x, y = next(iter(DataLoader(train_ds, batch_size=32, shuffle=False)))[:2]

    logits = model(x)
    logits = logits[0] if isinstance(logits, (tuple, list)) else logits
    loss = torch.nn.functional.cross_entropy(logits, y.long() % logits.shape[-1])
    loss.backward()

    core_active = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad
        and "core_network" in name
        and param.grad is not None
        and torch.isfinite(param.grad).all()
        and param.grad.abs().sum() > 0
    ]
    assert core_active
