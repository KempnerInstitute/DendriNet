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


def test_dump_init_gate_stats_flags_clipped_saturated_pattern(tmp_path):
    model = _DummyModel(output_dim=8, init_m=50.0, init_b=0.97)

    payload = dump_init_gate_stats(model, tmp_path)

    output_path = tmp_path / "init_gate_stats.json"
    assert output_path.exists()

    saved = json.loads(output_path.read_text())
    assert saved["n_reactivation_modules"] == 1
    assert payload["warnings"]
    assert any("calibration clip" in warning for warning in payload["warnings"])
    assert any("identical (m, b)" in warning for warning in payload["warnings"])


def test_dump_init_gate_stats_accepts_normalized_additive_fixed_gate(tmp_path):
    model = _DummyModel(output_dim=8, init_m=1.0, init_b=0.0)

    payload = dump_init_gate_stats(model, tmp_path)

    assert payload["warnings"] == []
