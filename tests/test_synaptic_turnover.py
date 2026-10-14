"""Synapse turnover counts the contacts that form and disappear in training."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from test_biological_constraints import _einet, _population
from test_trainer_configuration import IN_DIM, _data, _wrap

from dendritic_modeling.analysis import AnalysisManager
from dendritic_modeling.analysis.tools.synaptic_turnover import (
    SynapticTurnoverAnalyzer,
    turnover_counts,
)
from dendritic_modeling.config.analysis import AnalysisConfig
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.factory import get_architecture


def _model(core_type="einet", core=None):
    torch.manual_seed(0)
    core = _einet() if core is None else core
    return _wrap(get_architecture(core_type, core, input_dim=IN_DIM))


def _first_branch(model):
    return next(m for m in model.modules() if isinstance(m, DendriticBranchLayer))


def _move_one_contact(bank):
    """Activate the strongest inactive input of row 0, displacing one contact."""
    with torch.no_grad():
        inactive = (bank.weight_mask()[0] == 0).nonzero().flatten()
        bank.pre_w[0, inactive[0]] = bank.pre_w.max() + 1.0


def test_turnover_counts_follow_mask_semantics():
    reference = torch.tensor([[1, 0, 1], [0, 0, 1]])
    current = torch.tensor([[1, 1, 0], [0, 1, 1]])
    assert turnover_counts(reference, current) == {
        "active": 4,
        "stable": 2,
        "formed": 2,
        "eliminated": 1,
    }


@pytest.mark.parametrize(
    "core_type, core", [("einet", _einet()), ("population_network", _population())]
)
def test_moved_contacts_are_counted_incrementally_and_cumulatively(core_type, core):
    model = _model(core_type, core)
    analyzer = SynapticTurnoverAnalyzer(model, SimpleNamespace(snapshot_interval=1))
    _move_one_contact(_first_branch(model).branch_excitation)
    analyzer.analyze(model, training=True)
    analyzer.analyze(model, training=True)

    results = analyzer.analyze(model, training=False)
    assert results["snapshot_epochs"] == [0, 1, 2]
    assert results["incremental"]["exc"]["formed"] == [1, 0]
    assert results["incremental"]["exc"]["eliminated"] == [1, 0]
    assert results["cumulative"]["exc"]["formed"] == [1, 1]
    assert results["cumulative"]["all"]["eliminated"] == [1, 1]
    first = next(iter(results["modules"].values()))
    assert set(first["incremental"]) <= {"exc", "inh", "rec_exc", "rec_inh"}


def test_inhibitory_populations_are_labelled():
    model = _model("population_network", _population())
    analyzer = SynapticTurnoverAnalyzer(model, SimpleNamespace(snapshot_interval=1))
    analyzer.analyze(model, training=True)
    populations = {
        name: module["population"]
        for name, module in analyzer.results()["modules"].items()
    }
    assert {
        (".populations.i." in name, label) for name, label in populations.items()
    } == {(True, "explicit_inhibitory"), (False, "excitatory")}


def test_snapshots_copy_masks_that_banks_update_in_place():
    class LiveMaskBank(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mask", torch.tensor([[True, False, True]]))

        def weight_mask(self):
            return self.mask

    model = _model()
    branch = _first_branch(model)
    branch.branch_excitation = LiveMaskBank()
    analyzer = SynapticTurnoverAnalyzer(model, SimpleNamespace(snapshot_interval=1))
    branch.branch_excitation.mask[0, 1] = True
    analyzer.analyze(model, training=True)
    name = next(name for name, module in model.named_modules() if module is branch)
    module = analyzer.results()["modules"][name]
    assert module["incremental"]["exc"]["formed"] == [1]


def test_snapshot_interval_and_the_missing_training_snapshot():
    model = _model()
    analyzer = SynapticTurnoverAnalyzer(model, SimpleNamespace(snapshot_interval=2))
    with pytest.raises(ValueError, match="no training snapshot"):
        analyzer.results()
    for _ in range(5):
        analyzer.analyze(model, training=True)
    assert analyzer.results()["snapshot_epochs"] == [0, 2, 4]


def test_training_mask_noise_does_not_enter_snapshots():
    model = _model()
    for module in model.modules():
        if hasattr(module, "noise_level"):
            module.noise_level = 5.0
    model.train()
    analyzer = SynapticTurnoverAnalyzer(model, SimpleNamespace(snapshot_interval=1))
    analyzer.analyze(model, training=True)
    assert analyzer.results()["incremental"]["all"]["formed"] == [0]
    assert model.training


def test_turnover_must_run_during_training(tmp_path):
    config = AnalysisConfig()
    config.synapse_turnover.enabled = True
    config.synapse_turnover.training = False
    data = {"train": _data(), "valid": _data(seed=1), "test": _data(seed=2)}
    with pytest.raises(ValueError, match="taken during training"):
        AnalysisManager(
            model=_model(), data=data, analysis_config=config, save_root=str(tmp_path)
        )


def test_retired_turnover_options_load_with_a_warning(tmp_path, caplog):
    from dendritic_modeling.config.config import load_analysis_config

    path = tmp_path / "analysis.yaml"
    path.write_text(
        "analysis:\n"
        "  synapse_turnover:\n"
        "    enabled: true\n"
        "    params:\n"
        "      snapshot_interval: 3\n"
        "      synaptic_analysis: true\n"
        "      save_synaptic_report: true\n"
    )
    with caplog.at_level(logging.WARNING):
        config = load_analysis_config(str(path))
    assert config.synapse_turnover.params.snapshot_interval == 3
    assert config.synapse_turnover.training
    assert "synaptic_analysis" in caplog.text
