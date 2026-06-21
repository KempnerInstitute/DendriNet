"""Tests for local-rule component analysis helpers."""

import json

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from dendritic_modeling.analysis.tools.local_rule_components import (
    LocalRuleComponentAnalyzer,
    _activation_derivative_for_record,
    _branch_record_for_module,
    _depth_labels_for_records,
    _population_label_for_module_name,
)
from dendritic_modeling.config.analysis import LocalRuleComponentAnalysisParams
from dendritic_modeling.networks import DendriticBranchLayer, TopKLinear
from dendritic_modeling.networks.activations.parametric import ParametricTanh


class _LayerWithReactivation(nn.Module):
    def __init__(self, reactivation: nn.Module | None) -> None:
        super().__init__()
        if reactivation is not None:
            self.reactivation = reactivation


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


def _branch_at_depth(depth: int) -> DendriticBranchLayer:
    layer = DendriticBranchLayer(
        output_dim=1,
        excitatory_input_dim=2,
        excitatory_synapses_per_branch=1,
        inhibitory_input_dim=2,
        inhibitory_synapses_per_branch=1,
        use_shunting=False,
        reactivate=False,
        topk_noise_level=0.0,
        weight_transform="identity",
    )
    layer.layer_idx = depth
    return layer


def test_local_rule_component_depth_labels_preserve_depth_ordering() -> None:
    records = [
        {"layer": _branch_at_depth(2)},
        {"layer": _branch_at_depth(0)},
        {"layer": _branch_at_depth(4)},
        {"layer": _branch_at_depth(1)},
        {"layer": nn.Linear(1, 1)},
    ]

    labels = _depth_labels_for_records(records)

    assert labels == {
        0: "soma",
        1: "proximal",
        2: "intermediate",
        4: "distal",
    }


def test_local_rule_component_depth_labels_preserve_single_positive_depth() -> None:
    records = [{"layer": _branch_at_depth(3)}]

    assert _depth_labels_for_records(records) == {3: "distal"}


def test_local_rule_component_depth_labels_preserve_negative_depth_cases() -> None:
    assert _depth_labels_for_records([{"layer": _branch_at_depth(-1)}]) == {
        -1: "dendritic"
    }
    assert _depth_labels_for_records(
        [{"layer": _branch_at_depth(-1)}, {"layer": _branch_at_depth(2)}]
    ) == {-1: "intermediate", 2: "distal"}


def test_local_rule_component_depth_labels_method_uses_shared_helper() -> None:
    analyzer = LocalRuleComponentAnalyzer(LocalRuleComponentAnalysisParams())
    records = [{"layer": _branch_at_depth(0)}, {"layer": _branch_at_depth(2)}]

    assert analyzer._depth_labels(records) == _depth_labels_for_records(records)


def test_local_rule_component_population_label_helper_preserves_name_rules() -> None:
    assert _population_label_for_module_name("branch_layers.0") == "excitatory"
    assert (
        _population_label_for_module_name("core.inhibitory_cells.branch_layers.2")
        == "explicit_inhibitory"
    )


def test_local_rule_component_branch_record_helper_preserves_metadata() -> None:
    layer = _branch_at_depth(5)

    record = _branch_record_for_module("core.branch_layers.7", layer)

    assert record == {
        "layer": layer,
        "module_name": "core.branch_layers.7",
        "population": "excitatory",
        "branch_module_index": 7,
        "dendritic_depth": 5,
    }


def test_local_rule_component_branch_record_method_uses_shared_helpers() -> None:
    analyzer = LocalRuleComponentAnalyzer(LocalRuleComponentAnalysisParams())
    layer = _branch_at_depth(3)

    assert analyzer._branch_record(
        "core.inhibitory_cells.branch_layers.2", layer
    ) == _branch_record_for_module("core.inhibitory_cells.branch_layers.2", layer)
    assert (
        analyzer._population_label("core.inhibitory_cells.branch_layers.2")
        == "explicit_inhibitory"
    )


def test_local_rule_component_activation_derivative_preserves_empty_fallback() -> None:
    derivative = _activation_derivative_for_record(
        {"layer": _LayerWithReactivation(nn.Identity()), "v_n": None}
    )

    assert derivative.numel() == 0


def test_local_rule_component_activation_derivative_preserves_identity_and_relu() -> (
    None
):
    v_n = torch.tensor([[-1.0, 0.0, 2.0]], dtype=torch.float32)

    identity = _activation_derivative_for_record(
        {"layer": _LayerWithReactivation(nn.Identity()), "v_n": v_n}
    )
    relu = _activation_derivative_for_record(
        {"layer": _LayerWithReactivation(nn.ReLU()), "v_n": v_n}
    )
    no_reactivation = _activation_derivative_for_record(
        {"layer": _LayerWithReactivation(None), "v_n": v_n}
    )

    assert torch.equal(identity, torch.ones_like(v_n))
    assert torch.equal(relu, torch.tensor([[0.0, 0.0, 1.0]]))
    assert torch.equal(no_reactivation, torch.ones_like(v_n))


def test_local_rule_component_activation_derivative_uses_cached_sigmoid_output() -> (
    None
):
    v_n = torch.tensor([[-20.0, 20.0]], dtype=torch.float32)
    v_out = torch.tensor([[0.25, 0.75]], dtype=torch.float32)

    derivative = _activation_derivative_for_record(
        {
            "layer": _LayerWithReactivation(nn.Sigmoid()),
            "v_n": v_n,
            "v_out": v_out,
        }
    )

    assert torch.equal(derivative, v_out * (1.0 - v_out))


def test_local_rule_component_activation_derivative_uses_cached_tanh_output() -> None:
    v_n = torch.tensor([[-20.0, 20.0]], dtype=torch.float32)
    v_out = torch.tensor([[-0.25, 0.75]], dtype=torch.float32)

    derivative = _activation_derivative_for_record(
        {
            "layer": _LayerWithReactivation(nn.Tanh()),
            "v_n": v_n,
            "v_out": v_out,
        }
    )

    assert torch.equal(derivative, 1.0 - v_out.square())


def test_local_rule_component_activation_derivative_preserves_parametric_tanh() -> None:
    reactivation = ParametricTanh(output_dim=2, init_m=1.5, init_b=0.1)
    v_n = torch.tensor([[0.2, -0.4], [0.5, 0.1]], dtype=torch.float32)
    v_out = reactivation(v_n)
    m = reactivation.log_m.detach().exp().to(device=v_n.device, dtype=v_n.dtype)
    expected = 0.5 * m.unsqueeze(0) * (1.0 - (2.0 * v_out - 1.0).square())

    derivative = _activation_derivative_for_record(
        {
            "layer": _LayerWithReactivation(reactivation),
            "v_n": v_n,
            "v_out": v_out,
        }
    )

    assert torch.allclose(derivative, expected)


def test_local_rule_component_activation_derivative_method_uses_shared_helper() -> None:
    analyzer = LocalRuleComponentAnalyzer(LocalRuleComponentAnalysisParams())
    rec = {
        "layer": _LayerWithReactivation(nn.ReLU()),
        "v_n": torch.tensor([[-1.0, 2.0]], dtype=torch.float32),
    }

    assert torch.equal(
        analyzer._activation_derivative(rec),
        _activation_derivative_for_record(rec),
    )


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
