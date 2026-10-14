"""Tests for the ReactivationDynamicsAnalyzer.

The analyzer snapshots each ParametricTanh layer on every epoch-style call.
These tests verify the snapshot structure, parameter-tracking, and
per-layer saturation measurement — not the ultimate diagnosis, which is
empirical.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from dendritic_modeling.analysis.tools.reactivation_dynamics import (
    ReactivationDynamicsAnalyzer,
)
from dendritic_modeling.config.analysis import ReactivationDynamicsAnalysisParams
from dendritic_modeling.networks.activations.parametric import (
    ParametricReLU,
    ParametricTanh,
    ParametricTanhOnlyM,
)


class _StackedReactivationModel(nn.Module):
    """Minimal model with two ParametricTanh layers in sequence, so the
    analyzer has something to enumerate and snapshot."""

    def __init__(self, dim: int = 16, hidden: int = 8):
        super().__init__()
        self.encoder = nn.Linear(dim, hidden)
        self.r1 = ParametricTanh(output_dim=hidden, init_m=1.5, init_b=0.5)
        self.hidden = nn.Linear(hidden, hidden)
        self.r2 = ParametricTanh(output_dim=hidden, init_m=2.0, init_b=0.25)
        self.head = nn.Linear(hidden, 4)

    def forward(self, x):
        h = self.r1(self.encoder(x))
        h = self.r2(self.hidden(h))
        return self.head(h)


def _make_dataset(n: int = 64, dim: int = 16, n_classes: int = 4) -> TensorDataset:
    torch.manual_seed(0)
    x = torch.rand(n, dim)  # non-negative so shunting stays well-defined
    y = torch.randint(0, n_classes, (n,))
    return TensorDataset(x, y)


def test_analyzer_lists_parametric_gates_in_module_order():
    model = _StackedReactivationModel()
    analyzer = ReactivationDynamicsAnalyzer(
        ReactivationDynamicsAnalysisParams(max_samples=8)
    )

    layers = analyzer._list_gate_layers(model)

    assert layers == [("r1", model.r1), ("r2", model.r2)]


def test_analyzer_snapshots_all_reactivation_layers():
    model = _StackedReactivationModel()
    model.train()
    analyzer = ReactivationDynamicsAnalyzer(
        ReactivationDynamicsAnalysisParams(max_samples=32)
    )
    ds = _make_dataset()

    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot = analyzer.analyze(
            model=model,
            train_ds=ds,
            valid_ds=ds,
            test_ds=ds,
            device="cpu",
            save_path=tmpdir,
            filename="epoch0",
            training=True,
        )

        assert model.training
        assert snapshot["filename"] == "epoch0"
        assert snapshot["training"] is True
        assert snapshot["n_layers"] == 2, "should find both ParametricTanh layers"
        assert set(snapshot["layers"].keys()) == {"r1", "r2"}

        for _name, entry in snapshot["layers"].items():
            assert "V" in entry
            assert "params" in entry
            assert "reactivation_output" in entry
            v = entry["V"]
            assert v["n"] > 0
            assert "mean" in v and "std" in v
            assert "median" in v and "mad_std" in v
            assert "q10" in v and "q90" in v
            p = entry["params"]
            assert "m_mean" in p and "b_mean" in p and "log_m_mean" in p
            r = entry["reactivation_output"]
            assert 0.0 <= r["mean"] <= 1.0
            assert 0.0 <= r["sat_lo"] <= 1.0
            assert 0.0 <= r["sat_hi"] <= 1.0

        # File was saved under <tmpdir>/epochs/epoch0.json
        path = os.path.join(tmpdir, "epochs", "epoch0.json")
        assert os.path.exists(path)
        with open(path) as f:
            on_disk = json.load(f)
        assert on_disk["n_layers"] == 2


def test_analyzer_param_tracks_initial_mb():
    """Immediately after init, the snapshot m/b should match the ctor values."""
    model = _StackedReactivationModel()
    analyzer = ReactivationDynamicsAnalyzer(
        ReactivationDynamicsAnalysisParams(max_samples=16)
    )
    ds = _make_dataset()
    snap = analyzer.analyze(
        model=model,
        train_ds=ds,
        test_ds=ds,
        device="cpu",
        save_path=None,
        filename="epoch0",
        training=True,
    )
    r1 = snap["layers"]["r1"]
    r2 = snap["layers"]["r2"]
    assert r1["params"]["m_mean"] == pytest.approx(1.5, rel=1e-5)
    assert r1["params"]["b_mean"] == pytest.approx(0.5, rel=1e-5)
    assert r2["params"]["m_mean"] == pytest.approx(2.0, rel=1e-5)
    assert r2["params"]["b_mean"] == pytest.approx(0.25, rel=1e-5)


def test_analyzer_saturation_fractions_are_exclusive():
    """sat_lo + sat_hi + active_frac should sum to 1."""
    model = _StackedReactivationModel()
    analyzer = ReactivationDynamicsAnalyzer(
        ReactivationDynamicsAnalysisParams(max_samples=64)
    )
    ds = _make_dataset(n=128)
    snap = analyzer.analyze(
        model=model,
        train_ds=ds,
        test_ds=ds,
        device="cpu",
        save_path=None,
        filename="final",
        training=False,
    )
    for entry in snap["layers"].values():
        r = entry["reactivation_output"]
        total = r["sat_lo"] + r["sat_hi"] + r["active_frac"]
        assert total == pytest.approx(1.0, abs=1e-5)


def test_analyzer_no_op_without_parametric_gates():
    """A model with no parametric gates returns an empty snapshot."""

    class NoReactModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 4)

        def forward(self, x):
            return self.linear(x)

    model = NoReactModel()
    analyzer = ReactivationDynamicsAnalyzer(
        ReactivationDynamicsAnalysisParams(max_samples=8)
    )
    ds = TensorDataset(torch.rand(16, 8), torch.randint(0, 4, (16,)))
    snap = analyzer.analyze(
        model=model,
        train_ds=ds,
        test_ds=ds,
        device="cpu",
        filename="epoch0",
        training=True,
    )
    assert snap == {}


def test_analyzer_removes_hooks_after_failed_hook_registration(monkeypatch):
    model = _StackedReactivationModel()
    analyzer = ReactivationDynamicsAnalyzer(
        ReactivationDynamicsAnalysisParams(max_samples=8)
    )
    captured: dict[str, torch.Tensor] = {}
    layers = [("r1", model.r1), ("r2", model.r2)]
    original_register = model.r2.register_forward_pre_hook

    def _failing_register(*args, **kwargs):
        raise RuntimeError("reactivation hook registration failed")

    monkeypatch.setattr(model.r2, "register_forward_pre_hook", _failing_register)

    with pytest.raises(RuntimeError, match="reactivation hook registration failed"):
        analyzer._attach_capture_hooks(layers, captured)

    monkeypatch.setattr(model.r2, "register_forward_pre_hook", original_register)
    assert sum(len(module._forward_pre_hooks) for module in model.modules()) == 0


def test_analyzer_tracks_drift_after_parameter_modification():
    """If we manually mutate log_m, the next snapshot should reflect the change.

    This is the minimal check that the analyzer actually reads live
    parameter values instead of caching the init-time snapshot.
    """
    model = _StackedReactivationModel()
    analyzer = ReactivationDynamicsAnalyzer(
        ReactivationDynamicsAnalysisParams(max_samples=16)
    )
    ds = _make_dataset()

    snap_before = analyzer.analyze(
        model=model,
        train_ds=ds,
        test_ds=ds,
        device="cpu",
        filename="before",
        training=True,
    )
    m_before = snap_before["layers"]["r1"]["params"]["m_mean"]

    with torch.no_grad():
        model.r1.log_m.data.add_(1.0)  # m becomes m * e

    snap_after = analyzer.analyze(
        model=model,
        train_ds=ds,
        test_ds=ds,
        device="cpu",
        filename="after",
        training=True,
    )
    m_after = snap_after["layers"]["r1"]["params"]["m_mean"]

    import math

    assert m_after == pytest.approx(m_before * math.e, rel=1e-4)


def test_snapshot_records_soma_relative_depth_and_polarity():
    """branch_layers runs distal to soma, so depth is not the list index."""
    from test_biological_constraints import _population
    from test_trainer_configuration import IN_DIM, _wrap

    from dendritic_modeling.analysis.utils.dendritic_depth import (
        soma_relative_dendritic_depth,
    )
    from dendritic_modeling.networks.architectures.factory import get_architecture

    core = _population()
    for population in core["population_network"]["layers"][0]["populations"]:
        population["branch_factors"] = [2, 2]
    torch.manual_seed(0)
    model = _wrap(get_architecture("population_network", core, input_dim=IN_DIM))
    analyzer = ReactivationDynamicsAnalyzer(ReactivationDynamicsAnalysisParams())
    layers = analyzer._list_gate_layers(model)

    coordinates = analyzer._gate_coordinates(model, layers)

    e_distal = "core_network.layers.0.populations.e.branch_layers.0.reactivation"
    e_soma = "core_network.layers.0.populations.e.branch_layers.2.reactivation"
    i_soma = "core_network.layers.0.populations.i.branch_layers.2.reactivation"
    assert coordinates[e_distal]["soma_relative_dendritic_depth"] == 2
    assert coordinates[e_soma]["soma_relative_dendritic_depth"] == 0
    assert coordinates[e_soma]["population_polarity"] == "excitatory"
    assert coordinates[i_soma]["population_polarity"] == "inhibitory"
    parent = model.get_submodule(e_distal.rpartition(".")[0])
    assert soma_relative_dendritic_depth(parent) == 2


class _RecurrentGateModel(nn.Module):
    """Calls each gate once per step, as a stateful core's unroll does."""

    def __init__(self, gate: nn.Module, dim: int = 6):
        super().__init__()
        self.inp = nn.Linear(dim, dim)
        self.gate = gate

    def forward(self, x):
        h = torch.zeros(x.shape[0], x.shape[-1])
        for t in range(x.shape[1]):
            h = self.gate(0.5 * h + self.inp(x[:, t]))
        return h


def _snapshot(model, shape=(8, 3, 6)):
    ds = TensorDataset(torch.rand(*shape), torch.zeros(shape[0], dtype=torch.long))
    analyzer = ReactivationDynamicsAnalyzer(
        ReactivationDynamicsAnalysisParams(max_samples=8)
    )
    return analyzer.analyze(model=model, test_ds=ds, device="cpu", training=False)


def test_every_call_of_a_gate_is_recorded():
    torch.manual_seed(0)
    snapshot = _snapshot(_RecurrentGateModel(ParametricTanh(6, init_m=1.5, init_b=0.5)))
    entry = snapshot["layers"]["gate"]

    assert entry["gate"] == "ParametricTanh"
    assert entry["n_calls"] == 3
    assert entry["V"]["n"] == 3 * 8 * 6  # every timestep, not only the first


@pytest.mark.parametrize(
    "gate, unit_range",
    [
        (ParametricTanhOnlyM(6, init_m=1.5, fixed_b=0.25), True),
        (ParametricReLU(6), False),
    ],
)
def test_other_parametric_gates_are_tracked(gate, unit_range):
    entry = _snapshot(_RecurrentGateModel(gate))["layers"]["gate"]

    assert entry["params"]["m_mean"] == pytest.approx(gate.log_m.exp().mean().item())
    if unit_range:
        assert entry["params"]["b_mean"] == pytest.approx(0.25)
        output = entry["reactivation_output"]
        assert output["sat_lo"] + output["sat_hi"] + output["active_frac"] == (
            pytest.approx(1.0)
        )
    else:
        assert "sat_lo" not in entry["reactivation_output"]
        assert entry["reactivation_output"]["min"] >= 0.0  # m * relu(V - b)


def test_forward_failures_propagate():
    class Broken(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = ParametricTanh(6)

        def forward(self, x):
            raise RuntimeError("forward failed")

    with pytest.raises(RuntimeError, match="forward failed"):
        _snapshot(Broken(), shape=(8, 6))
