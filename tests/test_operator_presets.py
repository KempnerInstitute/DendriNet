"""Golden tests: every registered operator preset builds and runs forward."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from dendritic_modeling.config.operator_card import build_operator_card
from dendritic_modeling.networks.architectures.excitation_inhibition.ei_network import (
    ConfigurableEINetwork,
)

_PRESET_DIR = (
    Path(__file__).resolve().parents[1] / "configs" / "examples" / "operator_presets"
)
_PRESETS = sorted(p.name for p in _PRESET_DIR.glob("*.yaml"))


def test_preset_ladder_is_complete() -> None:
    assert _PRESETS == [
        "00_mlp_flat.yaml",
        "01_signed_sparse_tree.yaml",
        "02_positive_ei_flat.yaml",
        "03_positive_ei_tree.yaml",
        "04_shunting_tree.yaml",
    ]


@pytest.mark.parametrize("preset", _PRESETS)
def test_preset_builds_and_runs_forward(preset: str) -> None:
    config = yaml.safe_load((_PRESET_DIR / preset).read_text())
    torch.manual_seed(0)
    network = ConfigurableEINetwork(config=config, input_dim=64)

    signed_weights = config["morphology"]["weight_transform"] == "identity"
    x = torch.randn(4, 64) if signed_weights else torch.rand(4, 64)
    with torch.no_grad():
        out = network(x)

    assert out.shape == (4, 32)
    assert torch.isfinite(out).all()
    if signed_weights:
        assert (out < 0).any(), "signed presets must be able to emit negatives"
    else:
        assert (out >= 0).all()

    card = build_operator_card(config)
    assert card["weight_domain"]["signed"] == signed_weights
    assert card["schema_version"] == "dendritic_operator_card_v2"
