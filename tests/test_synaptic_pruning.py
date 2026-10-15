"""Synapse pruning analysis counts realized contacts before and after pruning."""

from __future__ import annotations

import json

import pytest
import torch
from test_biological_constraints import _einet, _population
from test_trainer_configuration import IN_DIM, _data, _wrap

from dendritic_modeling.analysis import AnalysisManager
from dendritic_modeling.analysis.tools.synaptic_pruning import SynapticPruningAnalyzer
from dendritic_modeling.analysis.utils.connectivity_census import (
    census_changes,
    connectivity_census,
)
from dendritic_modeling.config.analysis import AnalysisConfig
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.factory import get_architecture


def _model(core_type="einet", core=None):
    torch.manual_seed(0)
    core = (
        _einet(**{"morphology.weight_transform": "identity"}) if core is None else core
    )
    return _wrap(
        get_architecture(
            core_type, {**core, "biological_neuron": False}, input_dim=IN_DIM
        )
    )


def _first_bank(model):
    layer = next(m for m in model.modules() if isinstance(m, DendriticBranchLayer))
    return layer.branch_excitation


def _zero_row(bank):
    """Zero every candidate of row 0, as magnitude pruning of a weak row would.

    A Top-K bank re-selects its K largest candidates, so its contacts vanish
    only when fewer than K nonzero candidates remain; the mask then still
    selects K zeroed entries. Returns the number of contacts lost.
    """
    k = int(bank.weight_mask()[0].sum())
    with torch.no_grad():
        bank.pre_w[0].zero_()
    return k


@pytest.mark.parametrize(
    "core_type, core", [("einet", None), ("population_network", _population())]
)
def test_census_counts_every_selected_contact_of_an_unpruned_model(core_type, core):
    census = connectivity_census(_model(core_type, core))
    assert census["modules"]
    for module in census["modules"].values():
        for counts in module["pathways"].values():
            assert counts["realized"] == counts["selected"]
            assert sum(counts["realized_per_branch"]) == counts["realized"]
    assert census["totals"]["all"]["realized"] == sum(
        counts["realized"]
        for module in census["modules"].values()
        for counts in module["pathways"].values()
    )


def test_zeroed_contacts_the_mask_still_selects_are_not_realized():
    model = _model()
    before = connectivity_census(model)
    bank = _first_bank(model)
    selected_before = int(bank.weight_mask().sum())
    removed = _zero_row(bank)
    assert int(bank.weight_mask().sum()) == selected_before
    after = connectivity_census(model)
    changes = census_changes(before, after)
    assert removed > 0
    assert changes["totals"]["all"]["removed"] == removed
    assert changes["totals"]["exc"]["removed"] == removed


def test_zeroing_a_few_candidates_moves_top_k_contacts_rather_than_removing_them():
    model = _model()
    before = connectivity_census(model)
    bank = _first_bank(model)
    with torch.no_grad():
        bank.pre_w[0, bank.weight_mask()[0].nonzero().flatten()[0]] = 0.0
    changes = census_changes(before, connectivity_census(model))
    assert changes["totals"]["all"]["removed"] == 0


def test_a_pruning_step_is_recorded_and_reported(tmp_path):
    model = _model()
    analyzer = SynapticPruningAnalyzer()
    analyzer.record_pruning_start(model)
    removed = _zero_row(_first_bank(model))
    analyzer.record_pruning_end(
        model,
        {
            "pruning_stats": {"pruned_params": 2},
            "pre_pruning_performance": {"train_loss": 1.0, "valid_loss": 1.1},
            "post_pruning_performance": {"train_loss": 1.2, "valid_loss": 1.3},
        },
    )
    results = analyzer.analyze(model, save_path=str(tmp_path))
    (event,) = results["pruning_events"]
    assert event["changes"]["totals"]["all"]["removed"] == removed
    assert event["performance"]["after"]["valid_loss"] == 1.3
    assert results["connectivity"] == event["connectivity_after"]
    assert json.loads((tmp_path / "final").read_text())["pruning_events"]
    assert f"removed {removed} of" in (tmp_path / "final_summary.txt").read_text()
    assert (tmp_path / "final_active_synapses_per_branch.png").exists()
    assert (tmp_path / "final_active_synapses_by_layer.png").exists()


def test_ending_a_pruning_step_that_never_started_raises():
    with pytest.raises(RuntimeError, match="without record_pruning_start"):
        SynapticPruningAnalyzer().record_pruning_end(_model(), {})


def test_the_manager_records_pruning_and_runs_the_census(tmp_path):
    config = AnalysisConfig()
    config.synapse_pruning.enabled = True
    config.synapse_pruning.params.save_figures = False
    model = _model()
    data = {"train": _data(), "valid": _data(seed=1), "test": _data(seed=2)}
    manager = AnalysisManager(
        model=model, data=data, analysis_config=config, save_root=str(tmp_path)
    )
    manager.record_pruning_start()
    removed = _zero_row(_first_bank(model))
    manager.record_pruning_end({"pruning_stats": {}})
    manager._run_synapse_pruning_analysis(filename="final", training=False)
    saved = json.loads((tmp_path / "synapse_pruning_analysis" / "final").read_text())
    assert saved["pruning_events"][0]["changes"]["totals"]["all"]["removed"] == removed


def test_pruning_analysis_is_final_only(tmp_path):
    config = AnalysisConfig()
    config.synapse_pruning.enabled = True
    config.synapse_pruning.training = True
    data = {"train": _data(), "valid": _data(seed=1), "test": _data(seed=2)}
    with pytest.raises(ValueError, match="final-only"):
        AnalysisManager(
            model=_model(), data=data, analysis_config=config, save_root=str(tmp_path)
        )


def test_magnitude_pruning_removes_exactly_the_weak_realized_contacts():
    from dendritic_modeling.analysis.utils.connectivity_census import SYNAPSE_BANKS
    from dendritic_modeling.analysis.utils.effective_synapses import (
        effective_synapse_snapshot,
    )
    from dendritic_modeling.config.regularization import PruningConfig
    from dendritic_modeling.training.regularization import PruningManager

    model = _model()
    model.eval()
    threshold = 0.05
    weak = 0
    with torch.no_grad():
        for layer in model.modules():
            if not isinstance(layer, DendriticBranchLayer):
                continue
            for attribute in SYNAPSE_BANKS.values():
                bank = getattr(layer, attribute, None)
                if bank is None:
                    continue
                snapshot = effective_synapse_snapshot(bank)
                realized = snapshot.active_mask.ne(0) & snapshot.effective_weight.ne(0)
                weak += int(
                    (realized & (snapshot.effective_weight.abs() < threshold)).sum()
                )
    assert weak > 0

    analyzer = SynapticPruningAnalyzer()
    analyzer.record_pruning_start(model)
    manager = PruningManager(PruningConfig(enabled=True, threshold=threshold))
    stats = manager.prune_model(model)
    analyzer.record_pruning_end(model, {"pruning_stats": stats})

    (event,) = analyzer.pruning_events
    assert event["changes"]["totals"]["all"]["removed"] == weak
    # The manager counts every zeroed candidate weight, realized or not.
    assert stats["pruned_params"] >= weak
