"""Tests for local-rule component analysis helpers."""

import json

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from dendritic_modeling.analysis.tools.local_rule_components import (
    LocalRuleComponentAnalyzer,
    _activation_derivative_for_record,
    _active_presynaptic_values_for_layer,
    _add_component_values,
    _branch_record_for_module,
    _component_key_from_record,
    _ComponentKey,
    _delta_out_for_outputs,
    _depth_labels_for_records,
    _population_label_for_module_name,
    _total_conductance_for_record,
)
from dendritic_modeling.config.analysis import LocalRuleComponentAnalysisParams
from dendritic_modeling.networks import DendriticBranchLayer, TopKLinear
from dendritic_modeling.networks.activations.parametric import ParametricTanh


class _LayerWithReactivation(nn.Module):
    def __init__(self, reactivation: nn.Module | None) -> None:
        super().__init__()
        if reactivation is not None:
            self.reactivation = reactivation


class _LayerWithShunting(nn.Module):
    def __init__(self, use_shunting: bool) -> None:
        super().__init__()
        self.use_shunting = use_shunting


class _BlockConductanceModule(nn.Module):
    def __init__(self, conductances: torch.Tensor) -> None:
        super().__init__()
        self.conductances = nn.Parameter(conductances)

    def sum_conductances(self) -> torch.Tensor:
        return self.conductances


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


def _component_key(component: str = "branch_voltage") -> _ComponentKey:
    return _ComponentKey(
        module_name="branch",
        population="excitatory",
        branch_module_index=0,
        dendritic_depth=1,
        depth_label="distal",
        synapse_type="branch",
        component=component,
        used_in_rule=True,
    )


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


def test_local_rule_component_total_conductance_preserves_non_shunting() -> None:
    v_n = torch.zeros(2, 3)
    rec = {
        "layer": _LayerWithShunting(False),
        "exc_out": torch.tensor([[10.0, 0.0, -1.0], [2.0, 3.0, 4.0]]),
        "inh_out": torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]),
    }

    conductance = _total_conductance_for_record(rec, v_n)

    assert torch.equal(conductance, torch.ones_like(v_n))


def test_local_rule_component_total_conductance_preserves_shunting_terms() -> None:
    v_n = torch.zeros(2, 3)
    exc_out = torch.tensor([[0.5, -0.5, 2.0], [1.0, -2.0, 3.0]])
    inh_out = torch.tensor([[-1.0, 0.25, 0.0], [4.0, -5.0, 0.5]])
    rec = {
        "layer": _LayerWithShunting(True),
        "exc_out": exc_out,
        "inh_out": inh_out,
    }

    conductance = _total_conductance_for_record(rec, v_n)

    assert torch.equal(
        conductance, torch.ones_like(v_n) + exc_out.relu() + inh_out.relu()
    )


def test_local_rule_component_total_conductance_broadcasts_block_conductances() -> None:
    v_n = torch.zeros(2, 3)
    block = _BlockConductanceModule(torch.tensor([-1.0, 0.5, 2.0]))
    rec = {
        "layer": _LayerWithShunting(True),
        "blk_module": block,
    }

    conductance = _total_conductance_for_record(rec, v_n)

    expected = torch.ones_like(v_n) + torch.tensor([[0.0, 0.5, 2.0]]).expand_as(v_n)
    assert torch.equal(conductance, expected)
    assert not conductance.requires_grad


def test_local_rule_component_total_conductance_method_uses_shared_helper() -> None:
    analyzer = LocalRuleComponentAnalyzer(LocalRuleComponentAnalysisParams())
    v_n = torch.zeros(1, 2)
    rec = {
        "layer": _LayerWithShunting(True),
        "exc_out": torch.tensor([[0.25, 0.5]]),
    }

    assert torch.equal(
        analyzer._total_conductance(rec, v_n),
        _total_conductance_for_record(rec, v_n),
    )


def test_local_rule_component_active_presynaptic_values_preserve_topk_order() -> None:
    layer = TopKLinear(in_features=4, out_features=2, K=2)
    x = torch.tensor([[10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[0.0, 2.0, 0.0, 1.0], [1.0, 0.0, 3.0, 0.0]])

    active_values, active_mask = _active_presynaptic_values_for_layer(x, layer, mask)

    expected_values = torch.tensor(
        [
            [[20.0, 40.0], [30.0, 10.0]],
            [[2.0, 4.0], [3.0, 1.0]],
        ]
    )
    assert torch.equal(active_values, expected_values)
    assert torch.equal(active_mask, torch.ones(2, 2, dtype=torch.bool))


def test_local_rule_component_active_presynaptic_values_preserve_inactive_padding() -> (
    None
):
    layer = TopKLinear(in_features=4, out_features=2, K=2)
    x = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    mask = torch.tensor([[0.0, 2.0, 0.0, 1.0], [0.0, 0.0, 3.0, 0.0]])

    active_values, active_mask = _active_presynaptic_values_for_layer(x, layer, mask)

    assert active_values.shape == (1, 2, 2)
    assert torch.equal(active_values[:, 0], torch.tensor([[20.0, 40.0]]))
    assert torch.equal(active_values[:, 1, 0], torch.tensor([30.0]))
    assert torch.equal(
        active_mask,
        torch.tensor([[True, True], [True, False]]),
    )


def test_local_rule_component_active_presynaptic_values_flattens_inputs() -> None:
    layer = TopKLinear(in_features=4, out_features=1, K=2)
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    mask = torch.tensor([[0.0, 1.0, 0.0, 2.0]])

    active_values, active_mask = _active_presynaptic_values_for_layer(x, layer, mask)

    assert torch.equal(active_values, torch.tensor([[[4.0, 2.0]]]))
    assert torch.equal(active_mask, torch.tensor([[True, True]]))


def test_local_rule_component_active_presynaptic_values_handles_missing_active_inputs() -> (
    None
):
    layer = TopKLinear(in_features=4, out_features=2, K=2)
    x = torch.ones(2, 4)

    active_values, active_mask = _active_presynaptic_values_for_layer(
        x, layer, torch.zeros(2, 4)
    )
    assert active_values is None
    assert active_mask is None

    active_values, active_mask = _active_presynaptic_values_for_layer(
        torch.ones(2, 3), layer, torch.ones(2, 4)
    )
    assert active_values is None
    assert active_mask is None


def test_local_rule_component_active_presynaptic_method_uses_shared_helper() -> None:
    analyzer = LocalRuleComponentAnalyzer(LocalRuleComponentAnalysisParams())
    layer = TopKLinear(in_features=4, out_features=1, K=2)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[0.0, 2.0, 0.0, 1.0]])

    method_values, method_mask = analyzer._active_presynaptic_values(x, layer, mask)
    helper_values, helper_mask = _active_presynaptic_values_for_layer(x, layer, mask)

    assert torch.equal(method_values, helper_values)
    assert torch.equal(method_mask, helper_mask)


def test_local_rule_component_add_component_values_preserves_detached_cpu_append() -> (
    None
):
    key = _component_key()
    accum: dict[_ComponentKey, list[torch.Tensor]] = {}
    values = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    _add_component_values(accum, key, values)

    assert list(accum) == [key]
    assert torch.equal(accum[key][0], values.detach())
    assert accum[key][0].device.type == "cpu"
    assert not accum[key][0].requires_grad


def test_local_rule_component_add_component_values_preserves_noop_cases() -> None:
    key = _component_key()
    accum: dict[_ComponentKey, list[torch.Tensor]] = {}

    _add_component_values(accum, key, None)
    _add_component_values(accum, key, torch.empty(0))
    _add_component_values(accum, key, torch.tensor([1.0]), torch.tensor([False]))

    assert accum == {}


def test_local_rule_component_add_component_values_preserves_expanded_mask() -> None:
    key = _component_key()
    accum: dict[_ComponentKey, list[torch.Tensor]] = {}
    values = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
    mask = torch.tensor([[True, False], [False, True]])

    _add_component_values(accum, key, values, mask)

    assert torch.equal(accum[key][0], torch.tensor([0.0, 3.0, 4.0, 7.0]))


def test_local_rule_component_add_component_values_preserves_broadcast_mask() -> None:
    key = _component_key()
    accum: dict[_ComponentKey, list[torch.Tensor]] = {}
    values = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    mask = torch.tensor([True, False, True])

    _add_component_values(accum, key, values, mask)

    assert torch.equal(accum[key][0], torch.tensor([0.0, 2.0, 3.0, 5.0]))


def test_local_rule_component_add_component_method_uses_shared_helper() -> None:
    analyzer = LocalRuleComponentAnalyzer(LocalRuleComponentAnalysisParams())
    key = _component_key()
    method_accum: dict[_ComponentKey, list[torch.Tensor]] = {}
    helper_accum: dict[_ComponentKey, list[torch.Tensor]] = {}
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.tensor([True, False])

    analyzer._add_component(method_accum, key, values, mask)
    _add_component_values(helper_accum, key, values, mask)

    assert torch.equal(method_accum[key][0], helper_accum[key][0])


def test_local_rule_component_key_helper_preserves_record_fields() -> None:
    rec = {
        "module_name": "core.branch_layers.3",
        "population": "explicit_inhibitory",
        "branch_module_index": 3,
        "dendritic_depth": 2,
    }

    key = _component_key_from_record(
        rec,
        {2: "distal"},
        synapse_type="excitatory",
        component="local_update_factor",
        used_in_rule=1,
    )

    assert key == _ComponentKey(
        module_name="core.branch_layers.3",
        population="explicit_inhibitory",
        branch_module_index=3,
        dendritic_depth=2,
        depth_label="distal",
        synapse_type="excitatory",
        component="local_update_factor",
        used_in_rule=True,
    )


def test_local_rule_component_key_helper_preserves_missing_field_defaults() -> None:
    key = _component_key_from_record(
        {},
        {},
        synapse_type="branch",
        component="total_conductance",
        used_in_rule=False,
    )

    assert key == _ComponentKey(
        module_name="",
        population="unknown",
        branch_module_index=-1,
        dendritic_depth=-1,
        depth_label="unknown",
        synapse_type="branch",
        component="total_conductance",
        used_in_rule=False,
    )


def test_local_rule_component_key_method_uses_shared_helper() -> None:
    analyzer = LocalRuleComponentAnalyzer(LocalRuleComponentAnalysisParams())
    rec = {
        "module_name": "branch",
        "population": "excitatory",
        "branch_module_index": -1,
        "dendritic_depth": 0,
    }

    assert analyzer._base_key(
        rec,
        {0: "soma"},
        "branch",
        "branch_voltage",
        True,
    ) == _component_key_from_record(
        rec,
        {0: "soma"},
        "branch",
        "branch_voltage",
        True,
    )


def test_local_rule_component_delta_out_preserves_none_mode() -> None:
    y_hat = torch.tensor([[1.0, -1.0]])
    y = torch.tensor([[0.0, 1.0]])

    assert _delta_out_for_outputs(y_hat, y, "none") is None


def test_local_rule_component_delta_out_preserves_mse_modes() -> None:
    y_hat = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    y = torch.tensor([[0.5, 1.0], [2.0, 3.0]], dtype=torch.float64)

    expected = y_hat - y.to(dtype=y_hat.dtype)

    assert torch.equal(_delta_out_for_outputs(y_hat, y, "mse"), expected)
    assert torch.equal(_delta_out_for_outputs(y_hat, y, "auto"), expected)


def test_local_rule_component_delta_out_preserves_soft_target_ce() -> None:
    y_hat = torch.tensor([[1.0, 2.0], [3.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([[0.25, 0.75], [1.0, 0.0]], dtype=torch.float32)

    delta = _delta_out_for_outputs(y_hat, y, "ce")

    assert torch.equal(delta, torch.softmax(y_hat, dim=-1) - y)


def test_local_rule_component_delta_out_preserves_index_label_fallback() -> None:
    y_hat = torch.tensor([[1.0, 2.0, 3.0], [0.5, -0.5, 1.5]])
    y = torch.tensor([2, 0])
    expected = torch.softmax(y_hat, dim=-1) - torch.nn.functional.one_hot(
        y, num_classes=3
    ).to(y_hat.dtype)

    assert torch.equal(_delta_out_for_outputs(y_hat, y, "auto"), expected)
    assert torch.equal(_delta_out_for_outputs(y_hat, y[:, None], "ce"), expected)
    assert torch.equal(_delta_out_for_outputs(y_hat, y, "custom"), expected)


def test_local_rule_component_delta_out_preserves_same_shape_fallback() -> None:
    y_hat = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    y = torch.tensor([[1, 0]], dtype=torch.int64)

    delta = _delta_out_for_outputs(y_hat, y, "custom")

    assert torch.equal(delta, y_hat - y.to(dtype=y_hat.dtype))


def test_local_rule_component_delta_out_preserves_none_for_incompatible_shapes() -> (
    None
):
    y_hat = torch.tensor([[1.0, 2.0]])
    y = torch.tensor([3])

    assert _delta_out_for_outputs(y_hat, y, "auto") is None


def test_local_rule_component_delta_out_method_uses_shared_helper() -> None:
    analyzer = LocalRuleComponentAnalyzer(
        LocalRuleComponentAnalysisParams(error_mode="ce")
    )
    y_hat = torch.tensor([[1.0, 2.0]])
    y = torch.tensor([1])

    assert torch.equal(
        analyzer._compute_delta_out(y_hat, y),
        _delta_out_for_outputs(y_hat, y, "ce"),
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
