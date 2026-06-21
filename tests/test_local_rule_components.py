"""Tests for local-rule component analysis helpers."""

import json

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from dendritic_modeling.analysis.tools.local_rule_components import (
    LocalRuleComponentAnalyzer,
)
from dendritic_modeling.config.analysis import LocalRuleComponentAnalysisParams
from dendritic_modeling.networks import DendriticBranchLayer, TopKLinear


class _FeedforwardBranchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branch = DendriticBranchLayer(
            output_dim=2,
            excitatory_input_dim=3,
            excitatory_synapses_per_branch=2,
            inhibitory_input_dim=3,
            inhibitory_synapses_per_branch=2,
            use_shunting=False,
            reactivate=False,
            topk_noise_level=0.0,
            weight_transform="identity",
        )

    def forward(self, x: torch.Tensor, inhibitory_input: torch.Tensor) -> torch.Tensor:
        return self.branch(x, inhibitory_input=inhibitory_input)


def test_local_rule_component_recorders_preserve_topk_keys() -> None:
    torch.manual_seed(41)
    model = _FeedforwardBranchModel()
    analyzer = LocalRuleComponentAnalyzer(
        LocalRuleComponentAnalysisParams(include_broadcast=False)
    )
    x = torch.tensor([[0.2, 0.6, 0.1], [0.4, 0.3, 0.7]], dtype=torch.float32)
    inhibitory_input = torch.tensor(
        [[0.5, 0.1, 0.4], [0.3, 0.9, 0.2]], dtype=torch.float32
    )

    records, handles, decoder_cache = analyzer._attach_recorders(model)
    try:
        _ = model(x, inhibitory_input)
    finally:
        for handle in handles:
            handle.remove()

    assert decoder_cache == {}
    assert len(records) == 1
    rec = records[0]
    assert rec["module_name"] == "branch"
    assert rec["population"] == "excitatory"
    assert rec["branch_module_index"] == -1
    assert torch.equal(rec["x_exc"], x)
    assert torch.equal(rec["x_inh"], inhibitory_input)
    assert isinstance(rec["exc_module"], TopKLinear)
    assert isinstance(rec["inh_module"], TopKLinear)
    assert rec["exc_out"].shape == (2, 2)
    assert rec["inh_out"].shape == (2, 2)
    assert rec["exc_mask"].shape == model.branch.branch_excitation.pre_w.shape
    assert rec["inh_mask"].shape == model.branch.branch_inhibition.pre_w.shape


def test_local_rule_component_recorders_remove_hooks_after_failed_registration(
    monkeypatch,
) -> None:
    model = _FeedforwardBranchModel()
    analyzer = LocalRuleComponentAnalyzer(
        LocalRuleComponentAnalysisParams(include_broadcast=False)
    )

    def _failing_register(*args, **kwargs):
        raise RuntimeError("local-rule hook registration failed")

    monkeypatch.setattr(
        model.branch.branch_inhibition,
        "register_forward_hook",
        _failing_register,
    )

    with pytest.raises(RuntimeError, match="local-rule hook registration failed"):
        analyzer._attach_recorders(model)

    assert sum(len(module._forward_hooks) for module in model.modules()) == 0


def test_local_rule_component_dataset_selection_priority() -> None:
    analyzer = LocalRuleComponentAnalyzer(LocalRuleComponentAnalysisParams())
    train_ds = TensorDataset(torch.ones(1, 1), torch.zeros(1))
    valid_ds = TensorDataset(torch.ones(2, 1), torch.zeros(2))
    test_ds = TensorDataset(torch.ones(3, 1), torch.zeros(3))

    assert (
        analyzer._select_dataset(
            training=True,
            test_dataset=test_ds,
            train_ds=train_ds,
            valid_ds=valid_ds,
        )
        is train_ds
    )
    assert (
        analyzer._select_dataset(
            training=False,
            test_dataset=test_ds,
            train_ds=train_ds,
            valid_ds=valid_ds,
        )
        is test_ds
    )
    assert (
        analyzer._select_dataset(
            training=False,
            test_dataset=None,
            train_ds=train_ds,
            valid_ds=valid_ds,
        )
        is valid_ds
    )


def test_local_rule_component_topk_cache_reset_clears_masks() -> None:
    model = _FeedforwardBranchModel()
    modules = LocalRuleComponentAnalyzer._topk_modules(model)
    assert modules

    LocalRuleComponentAnalyzer._set_topk_mask_cache(modules, enabled=True)
    for module in modules:
        assert module.cache_mask
        module._last_forward_weight_mask = torch.ones_like(module.pre_w)

    LocalRuleComponentAnalyzer._set_topk_mask_cache(modules, enabled=False)
    for module in modules:
        assert not module.cache_mask
        assert module._last_forward_weight_mask is None


def test_local_rule_component_save_snapshot_writes_expected_files(tmp_path) -> None:
    analyzer = LocalRuleComponentAnalyzer(LocalRuleComponentAnalysisParams())
    snapshot = {
        "filename": "epoch_1",
        "training": True,
        "n_rows": 1,
        "rows": [
            {
                "filename": "epoch_1",
                "training": True,
                "component": "branch_voltage",
                "mean": 1.0,
            }
        ],
    }

    analyzer._save_snapshot(
        snapshot,
        save_path=str(tmp_path),
        filename="epoch_1",
        training=True,
    )

    json_path = tmp_path / "epochs" / "epoch_1.json"
    csv_path = tmp_path / "epochs" / "epoch_1.csv"
    trajectory_path = tmp_path / "trajectory.csv"
    assert json_path.exists()
    assert csv_path.exists()
    assert trajectory_path.exists()
    with open(json_path) as f:
        payload = json.load(f)
    assert payload["n_rows"] == 1
