"""Packaged examples work without locating a repository checkout."""

from __future__ import annotations

import sys

import pytest
import torch
import yaml

from dendritic_modeling.config import config_from_mapping
from dendritic_modeling.config.recurrence import is_recurrent_core_config
from dendritic_modeling.examples import example_names, example_text
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.scripts.tools.export_example import main


@pytest.mark.parametrize("name", example_names())
def test_example_builds_and_backpropagates(name):
    raw = yaml.safe_load(example_text(name))
    config = config_from_mapping(raw)
    model, _ = initialize_model(config.model)
    width = config.model.encoder.params.input_dim
    # Recurrent examples read sequences: [batch, time, features].
    recurrent = is_recurrent_core_config(raw["model"]["core"])
    x = torch.rand(2, 5, width) if recurrent else torch.rand(2, width)
    logits = model(x)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    assert logits.shape == (2, config.model.decoder.params.output_dim)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert any(p.grad is not None for p in model.parameters())
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_examples_cover_the_model_kinds_studio_documents():
    """Feedforward and recurrent cores, both tasks and a non-population family."""
    kinds = set()
    for name in example_names():
        raw = yaml.safe_load(example_text(name))
        core = raw["model"]["core"]
        kinds.add("recurrent" if is_recurrent_core_config(core) else "feedforward")
        kinds.add(raw["model"]["task"])
        kinds.add(core["type"])
        assert raw["outputs"]["run_name"] == name
        assert example_text(name).startswith("# "), "examples start with a title line"
    assert {"feedforward", "recurrent", "classification", "regression"} <= kinds
    assert "population_network" in kinds and "gru" in kinds
    assert kinds & {"dendritic_shunting", "einet", "dendritic_additive"}


def test_export_never_overwrites_existing_file(tmp_path, monkeypatch):
    target = tmp_path / "example.yaml"
    target.write_text("keep me")
    monkeypatch.setattr(
        sys,
        "argv",
        ["dendritic-example", "synthetic_population", "--output", str(target)],
    )
    with pytest.raises(SystemExit):
        main()
    assert target.read_text() == "keep me"


def test_unknown_resource_is_rejected():
    with pytest.raises(ValueError):
        example_text("../../pyproject.toml")
