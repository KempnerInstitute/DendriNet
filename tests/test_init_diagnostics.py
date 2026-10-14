"""Tests for post-build gate diagnostics."""

from __future__ import annotations

import json

from torch import nn

from dendritic_modeling.networks.activations.parametric import ParametricTanh
from dendritic_modeling.scripts.script_utils.init_diagnostics import (
    collect_reactivation_stats,
    dump_init_gate_stats,
)


class _DummyModel(nn.Module):
    def __init__(self, output_dim: int, init_m: float, init_b: float):
        super().__init__()
        self.reactivation = ParametricTanh(
            output_dim=output_dim, init_m=init_m, init_b=init_b
        )


def test_collect_reactivation_stats_reports_vector_shapes():
    model = _DummyModel(output_dim=4, init_m=1.5, init_b=0.5)

    stats = collect_reactivation_stats(model)

    assert "reactivation" in stats
    assert stats["reactivation"]["m"]["n"] == 4
    assert stats["reactivation"]["m"]["n_unique"] == 1
    assert stats["reactivation"]["b"]["n"] == 4
    assert stats["reactivation"]["b"]["n_unique"] == 1


def test_collect_reactivation_stats_preserves_nested_module_order():
    class NestedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = _DummyModel(output_dim=2, init_m=1.25, init_b=0.4)
            self.unrelated = nn.Identity()
            self.second = _DummyModel(output_dim=3, init_m=1.5, init_b=0.5)

    stats = collect_reactivation_stats(NestedModel())

    assert list(stats) == ["first.reactivation", "second.reactivation"]


def test_dump_init_gate_stats_flags_clipped_saturated_pattern(tmp_path):
    model = _DummyModel(output_dim=8, init_m=50.0, init_b=0.97)

    payload = dump_init_gate_stats(model, tmp_path)

    output_path = tmp_path / "init_gate_stats.json"
    assert output_path.exists()

    saved = json.loads(output_path.read_text())
    assert saved["n_reactivation_modules"] == 1
    assert payload["warnings"]
    assert any("calibration slope limit" in warning for warning in payload["warnings"])
    assert any("identical (m, b)" in warning for warning in payload["warnings"])


def test_dump_init_gate_stats_accepts_normalized_additive_fixed_gate(tmp_path):
    model = _DummyModel(output_dim=8, init_m=1.0, init_b=0.0)

    payload = dump_init_gate_stats(model, tmp_path)

    assert payload["warnings"] == []


def test_gate_stats_record_the_policy_each_layer_applied(tmp_path):
    import torch

    from dendritic_modeling.networks.architectures.factory import get_architecture

    population = {"branch_factors": [2], "n_neurons": 3}
    layer = {
        "name": "layer0",
        "readout_population": "pyr",
        "population_defaults": {
            "ff_excitatory_synapses": 2,
            "ff_inhibitory_synapses": 0,
        },
        "populations": [
            # Additive populations without a written policy calibrate their
            # gates on data; shunting ones keep the analytical gate.
            {
                "name": "pyr",
                "polarity": "excitatory",
                **population,
                "population": {"use_shunting": False},
            },
            {"name": "pv", "polarity": "excitatory", **population},
        ],
        "connections": [
            {"source": "input", "target": "pyr"},
            {"source": "input", "target": "pv"},
        ],
    }
    torch.manual_seed(0)
    network = get_architecture(
        "population_network", {"population_network": {"layers": [layer]}}, input_dim=4
    )
    payload = dump_init_gate_stats(network, tmp_path)
    saved = json.loads((tmp_path / "init_gate_stats.json").read_text())
    assert saved["init_policies"] == payload["init_policies"]
    entries = saved["reactivation_modules"]
    assert entries and all("init_policy" in entry for entry in entries.values())
    by_population = {
        name: {entry["init_policy"] for key, entry in entries.items() if name in key}
        for name in ("pyr", "pv")
    }
    assert by_population == {"pyr": {"occupancy_quantile"}, "pv": {"analytical"}}
    assert all(
        entry["data_driven"] == (entry["init_policy"] == "occupancy_quantile")
        for entry in entries.values()
    )
    assert set(saved["init_policies"]) == {"analytical", "occupancy_quantile"}


def test_unified_additive_aliases_record_the_analytical_policy(tmp_path):
    import torch

    from dendritic_modeling.networks.architectures.factory import get_architecture

    population = {"n_neurons": 3, "branch_factors": [2], "ff_excitatory_synapses": 2}
    torch.manual_seed(0)
    network = get_architecture(
        "rnn_dendritic_additive",
        {
            "unified_ei": {
                "layers": [
                    {
                        "excitatory": {**population, "rec_excitatory_synapses": 1},
                        "inhibitory": None,
                        "recurrent": True,
                    }
                ]
            }
        },
        input_dim=4,
    )
    payload = dump_init_gate_stats(network, tmp_path)
    assert set(payload["init_policies"]) == {"analytical"}
