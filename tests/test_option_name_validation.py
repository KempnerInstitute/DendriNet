"""Gradient-scaling, reactivation-policy and DeepST option names are validated."""

from __future__ import annotations

import logging
from copy import deepcopy

import pytest
import torch
from test_operator_config_contract import _config

from dendritic_modeling.config.model import (
    BlockLinearConfig,
    ReactivationConfig,
    SparsityConfig,
)
from dendritic_modeling.config.reactivation import (
    normalize_reactivation_init_policy,
    resolve_reactivation_calibration_mode,
)
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.gradient_scaling import (
    GradientScaler,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.deepst import (
    DeepstLinear,
)
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    PopulationConfig,
)


def _einet(rule="shunting", **sections):
    config = _config("einet", rule, "param_tanh")
    for name, values in sections.items():
        config.setdefault(name, {}).update(values)
    return config


def _build(config):
    torch.manual_seed(5)
    return get_architecture(config["type"], deepcopy(config), input_dim=6)


@pytest.mark.parametrize(
    "config_cls, value, message",
    [
        (SparsityConfig, "distal_upweight_by_idx", "sparsity.gradient_scaling"),
        (SparsityConfig, "conductance-dyn", "sparsity.gradient_scaling"),
        (BlockLinearConfig, "conductance_dynamc", "did you mean"),
        (ReactivationConfig, "invrse", "reactivation.gradient_scaling"),
    ],
)
def test_unknown_gradient_scaling_names_raise_in_config(config_cls, value, message):
    with pytest.raises(ValueError, match=message):
        config_cls(gradient_scaling=value)


@pytest.mark.parametrize(
    "value", ["inverse", "conductance_dynamic", "voltage_dynamic", "scaled"]
)
def test_legacy_reactivation_names_mean_straight_through(value, caplog):
    with caplog.at_level(logging.WARNING):
        assert ReactivationConfig(gradient_scaling=value).gradient_scaling == (
            "straight_through"
        )
    assert "deprecated name for 'straight_through'" in caplog.text


def test_scaler_rejects_topk_distal_upweighting():
    with pytest.raises(ValueError, match="topk gradient_scaling"):
        GradientScaler(topk_strategy="distal_upweight_by_idx")
    scaler = GradientScaler(
        blocklinear_strategy="distal_upweight_by_idx",
        reactivation_strategy="straight_through",
    )
    assert scaler.blocklinear_strategy == "distal_upweight_by_idx"


def test_straight_through_alias_builds_the_same_gate_gradient():
    canonical = _build(_einet(reactivation={"gradient_scaling": "straight_through"}))
    legacy = _build(_einet(reactivation={"gradient_scaling": "inverse"}))
    x = torch.rand(4, 6)
    for model in (canonical, legacy):
        model.train()
        model.zero_grad(set_to_none=True)
        model(x).sum().backward()
    for (name, a), (_, b) in zip(
        canonical.named_parameters(), legacy.named_parameters()
    ):
        assert torch.equal(a.grad, b.grad), name


@pytest.mark.parametrize(
    "section, message",
    [
        ({"sparsity": {"gradient_scaling": "conductance_dynamic"}}, "topk_strategy"),
        (
            {"reactivation": {"gradient_scaling": "straight_through"}},
            "reactivation_strategy",
        ),
    ],
)
def test_shunting_only_strategies_raise_on_additive_layers(section, message):
    with pytest.raises(ValueError, match=message):
        _build(_einet(rule="raw", **section))
    branches = [
        module
        for module in _build(_einet(rule="shunting", **section)).modules()
        if isinstance(module, DendriticBranchLayer)
    ]
    assert branches


@pytest.mark.parametrize(
    "policy, canonical",
    [
        ("analytical", "analytical"),
        ("auto", "analytical"),
        ("manual", "fixed"),
        ("Occupancy-Quantile", "occupancy_quantile"),
    ],
)
def test_reactivation_policy_aliases_map(policy, canonical):
    assert normalize_reactivation_init_policy(policy) == canonical


def test_misspelled_reactivation_policy_raises_everywhere():
    with pytest.raises(ValueError, match="did you mean 'empirical'"):
        normalize_reactivation_init_policy("emprical")
    with pytest.raises(ValueError, match="init_policy"):
        ReactivationConfig(init_policy="occupancy_quantle")
    with pytest.raises(ValueError, match="init_policy"):
        PopulationConfig(n_neurons=2, reactivation_init_policy="occupancy_quantle")
    config = _einet()
    config["reactivation"]["init_policy"] = "emprical"
    with pytest.raises(ValueError, match="init_policy"):
        _build(config)


def test_calibration_mode_resolution():
    assert resolve_reactivation_calibration_mode("median_mad") == "median_mad"
    assert resolve_reactivation_calibration_mode("empirical") == "median_mad"
    with pytest.warns(DeprecationWarning):
        assert resolve_reactivation_calibration_mode("quantile") == "occupancy_quantile"
    with pytest.raises(ValueError, match="neither a calibration mode"):
        resolve_reactivation_calibration_mode("analytical")
    with pytest.raises(ValueError, match="not supported"):
        resolve_reactivation_calibration_mode("median-madd")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"rewiring_mode": "per_branch"}, "rewiring_mode"),
        ({"rewiring_mode": "Local"}, None),
        ({"param_space": "logit"}, "param_space"),
        ({"target_density": 1.5}, "target_density"),
        ({"sigma": -0.1}, "sigma"),
        ({"weight_threshold": -1e-3}, "weight_threshold"),
        (
            {"rewiring_mode": "constant_branch", "synapses_per_branch": 0},
            "synapses_per_branch",
        ),
    ],
)
def test_deepst_options_are_validated(kwargs, message):
    def build():
        return DeepstLinear(in_features=4, out_features=2, **kwargs)

    if message is None:
        assert build().rewiring_mode == "local"
    else:
        with pytest.raises(ValueError, match=message):
            build()


def test_population_rewiring_mode_is_validated():
    with pytest.raises(ValueError, match="rewiring_mode"):
        PopulationConfig(n_neurons=2, rewiring_mode="globl")
    assert PopulationConfig(
        n_neurons=2, rewiring_mode="Constant-Branch"
    ).rewiring_mode == ("constant_branch")
