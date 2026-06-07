"""Smoke tests for extended recurrent example configs."""

from pathlib import Path

import pytest
import torch

from dendritic_modeling.config import load_config
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

EXTENDED_EXAMPLES = [
    "configs/examples/recurrent_population_graph_pyr_pv_sst.yaml",
    "configs/examples/recurrent_synapse_types_ampa_nmda_gaba.yaml",
    "configs/examples/recurrent_structured_connectivity.yaml",
    "configs/examples/recurrent_spiking_population_graph.yaml",
]

MULTI_POPULATION_EXAMPLES = [
    "configs/examples/multi_population_ei_feedforward.yaml",
    "configs/examples/multi_population_ei_independent_transfer.yaml",
    "configs/examples/multi_population_ei_two_layer.yaml",
]

REPAIRED_RECURRENT_GRADIENT_EXAMPLES = [
    "configs/examples/recurrent_context_decision.yaml",
    "configs/examples/recurrent_dms.yaml",
]


@pytest.mark.parametrize("config_path", EXTENDED_EXAMPLES)
def test_extended_recurrent_example_config_builds_and_starts(config_path):
    load_config.cache_clear()
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


@pytest.mark.parametrize("config_path", MULTI_POPULATION_EXAMPLES)
def test_multi_population_example_config_builds_and_starts(config_path):
    load_config.cache_clear()
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


@pytest.mark.parametrize("config_path", REPAIRED_RECURRENT_GRADIENT_EXAMPLES)
def test_repaired_recurrent_examples_connect_input_projection_and_i_cells(config_path):
    load_config.cache_clear()
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
        and ("core_network.input_projection" in name or ".i_population." in name)
    }
    assert watched, "No input-projection or inhibitory parameters found"
    dead = [
        name
        for name, grad in watched.items()
        if grad is None or not torch.isfinite(grad).all() or grad.abs().sum() == 0
    ]
    assert dead == []
