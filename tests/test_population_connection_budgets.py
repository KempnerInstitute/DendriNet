"""Declared population connections reach synapses, or say why they do not."""

from __future__ import annotations

import logging

import pytest
import torch

from dendritic_modeling.networks.architectures.factory import get_architecture


def _layer(defaults, connections, *, e_override=None, branch_factors=(2,)):
    population_defaults = {
        "ff_excitatory_synapses": 2,
        "ff_inhibitory_synapses": 1,
        "somatic_synapses": True,
        "reactivate": True,
        "dbl_init_method": "default",
        "weight_transform": "exp",
        **defaults,
    }
    return {
        "biological_neuron": False,
        "population_network": {
            "output_mode": "last",
            "layers": [
                {
                    "name": "layer0",
                    "recurrent": True,
                    "readout_population": "e",
                    "population_defaults": population_defaults,
                    "populations": [
                        {
                            "name": "i",
                            "polarity": "inhibitory",
                            "n_neurons": 3,
                            "branch_factors": [1],
                            "population": {"ff_inhibitory_synapses": 0},
                        },
                        {
                            "name": "e",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": list(branch_factors),
                            "population": dict(e_override or {}),
                        },
                    ],
                    "connections": [
                        {"source": "input", "target": "i"},
                        {"source": "input", "target": "e"},
                        *connections,
                    ],
                }
            ],
        },
    }


def _build(cfg):
    return get_architecture("population_network", cfg, input_dim=5)


_DELAYED_EE = {"source": "e", "target": "e", "timing": "delayed"}
_DELAYED_IE = {"source": "i", "target": "e", "timing": "delayed"}


@pytest.mark.parametrize(
    ("connection", "field"),
    [
        (_DELAYED_EE, "rec_excitatory_synapses"),
        (_DELAYED_IE, "rec_inhibitory_synapses"),
    ],
)
def test_a_delayed_connection_with_its_default_zero_budget_raises(connection, field):
    with pytest.raises(ValueError, match=f"{field} is unset and defaults to 0"):
        _build(_layer({}, [connection]))


def test_a_delayed_connection_with_a_budget_builds_a_recurrent_bank():
    net = _build(
        _layer(
            {"rec_excitatory_synapses": 1, "rec_inhibitory_synapses": 1},
            [_DELAYED_EE, _DELAYED_IE],
        )
    )
    y = net(torch.randn(2, 3, 5))
    assert torch.isfinite(y).all()


def test_a_written_zero_budget_is_a_lesion_that_builds(caplog):
    """An explicit zero keeps a declared edge as a dose-zero arm."""
    cfg = _layer(
        {"ff_inhibitory_synapses": 0},
        [{"source": "i", "target": "e"}],
    )
    with caplog.at_level(logging.INFO):
        net = _build(cfg)
    assert "reaches no synapses: ff_inhibitory is written as zero" in caplog.text
    assert torch.isfinite(net(torch.randn(2, 5))).all()


def test_a_connection_to_the_soma_alone_needs_somatic_synapses():
    cfg = _layer(
        {"somatic_synapses": False},
        [{"source": "i", "target": "e", "target_levels": ["soma"]}],
    )
    with pytest.raises(ValueError, match="somatic_synapses is false"):
        _build(cfg)


def test_target_levels_with_a_default_zero_budget_raise():
    cfg = _layer(
        {},
        [{**_DELAYED_IE, "target_levels": ["distal"]}],
    )
    with pytest.raises(ValueError, match="rec_inhibitory_synapses is unset"):
        _build(cfg)


def test_soma_entries_are_ignored_without_somatic_synapses_when_dendrites_are_reached():
    """The documented RNN layout: by-level soma entries > 0, somatic_synapses false."""
    cfg = _layer(
        {
            "somatic_synapses": False,
            "rec_excitatory_synapses_by_level": [2, 1],
            "rec_inhibitory_synapses": 1,
        },
        [_DELAYED_EE],
    )
    assert torch.isfinite(_build(cfg)(torch.randn(2, 3, 5))).all()


def test_written_contacts_only_at_the_soma_without_somatic_synapses_raise():
    cfg = _layer(
        {"somatic_synapses": False, "rec_excitatory_synapses_by_level": [0, 2]},
        [_DELAYED_EE],
    )
    with pytest.raises(ValueError, match="are soma contacts, and somatic_synapses"):
        _build(cfg)


def test_a_population_override_counts_as_written():
    cfg = _layer(
        {},
        [_DELAYED_EE],
        e_override={"rec_excitatory_synapses": 0},
    )
    _build(cfg)


def test_a_point_population_without_somatic_synapses_is_rejected():
    from dendritic_modeling.networks.architectures.recurrent import (
        PopulationConfig,
        StatefulDendriNet,
    )

    config = PopulationConfig(
        n_neurons=3,
        branch_factors=[],
        ff_excitatory_synapses=2,
        somatic_synapses=False,
        weight_transform="exp",
    )
    with pytest.raises(ValueError, match="has no synapses at any level"):
        StatefulDendriNet(pop_config=config, excitatory_input_dim=4)


@pytest.mark.parametrize(
    ("input_transform", "signed", "warns"),
    [("identity", True, True), ("identity", False, False), ("relu", True, False)],
)
def test_shunting_populations_warn_once_on_signed_external_input(
    caplog, input_transform, signed, warns
):
    cfg = _layer({}, [])
    cfg["population_network"]["input_transform"] = input_transform
    cfg["population_network"]["layers"][0]["recurrent"] = False
    net = _build(cfg)
    x = torch.randn(4, 5) if signed else torch.rand(4, 5)
    with caplog.at_level(logging.WARNING):
        net(x)
        net(x)
    messages = [r for r in caplog.records if "read signed external input" in r.message]
    assert len(messages) == (1 if warns else 0)
