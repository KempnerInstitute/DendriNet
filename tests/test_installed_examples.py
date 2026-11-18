"""Packaged examples work without locating a repository checkout."""

from __future__ import annotations

import sys

import pytest
import torch
import yaml

from dendritic_modeling.config import config_from_mapping
from dendritic_modeling.examples import example_names, example_text
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.scripts.tools.export_example import main


@pytest.mark.parametrize("name", example_names())
def test_example_builds_and_backpropagates(name):
    config = config_from_mapping(yaml.safe_load(example_text(name)))
    model, _ = initialize_model(config.model)
    x = torch.rand(2, config.model.encoder.params.input_dim)
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
