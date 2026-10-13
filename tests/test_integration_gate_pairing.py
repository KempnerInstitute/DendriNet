"""The integration/gate pairing warning flags only the two fragile settings."""

from __future__ import annotations

import warnings

import pytest
from golden_cases import _einet_core, _full_model

from dendritic_modeling.config.reactivation import warn_integration_gate_pairing
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
    DendriticBranchLayer,
)


def _pairing(**overrides):
    settings = {
        "weight_transform": "softplus",
        "use_shunting": False,
        "additive_mode": "raw",
        "reactivation_init_policy": "analytical",
        "reactivate": True,
    }
    settings.update(overrides)
    return warn_integration_gate_pairing(**settings)


def test_signed_shunting_warns_about_the_denominator():
    with pytest.warns(UserWarning, match="shunting denominator"):
        message = _pairing(weight_transform="identity", use_shunting=True)
    assert "identity" in message


@pytest.mark.parametrize(
    "policy, uses",
    [
        ("analytical", "slope and midpoint"),
        ("fixed", "init_m and init_b"),
        ("analytical_slope_occupancy_center", "the slope"),
        ("occupancy_slope_analytical_center", "the midpoint"),
    ],
)
@pytest.mark.parametrize("mode", ["raw", "conductance_normalized"])
def test_uncalibrated_gates_on_positive_additive_drive_warn(policy, uses, mode):
    with pytest.warns(UserWarning, match="saturated or flat"):
        message = _pairing(additive_mode=mode, reactivation_init_policy=policy)
    assert uses in message
    assert repr(policy) in message
    assert "'occupancy_quantile'" in message


@pytest.mark.parametrize(
    "overrides",
    [
        {"reactivation_init_policy": "occupancy_quantile"},
        {"reactivation_init_policy": "empirical"},
        {"use_shunting": True},
        {"weight_transform": "identity"},
        {"reactivate": False},
        {"additive_mode": "tangent_matched"},
        {"additive_normalization": True},
    ],
    ids=lambda overrides: "-".join(f"{k}={v}" for k, v in overrides.items()),
)
def test_calibrated_or_bounded_settings_stay_silent(overrides):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _pairing(**overrides) is None


def _layer(**options):
    return DendriticBranchLayer(
        output_dim=2,
        excitatory_input_dim=4,
        excitatory_synapses_per_branch=2,
        inhibitory_input_dim=2,
        inhibitory_synapses_per_branch=1,
        **options,
    )


def test_every_branch_layer_checks_its_own_settings():
    # The branch layer is shared by the feedforward, population and recurrent
    # families, so each checks the settings it was built with.
    with pytest.warns(UserWarning, match="saturated or flat"):
        _layer(
            use_shunting=False,
            weight_transform="softplus",
            reactivate=True,
            reactivation_init_policy="analytical",
        )
    with pytest.warns(UserWarning, match="shunting denominator"):
        _layer(use_shunting=True, weight_transform="identity", reactivate=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _layer(
            use_shunting=False,
            weight_transform="softplus",
            reactivate=True,
            reactivation_init_policy="occupancy_quantile",
        )


def test_additive_cores_default_to_a_calibrated_gate():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _full_model(_einet_core("dendritic_additive"))
    with pytest.warns(UserWarning, match="saturated or flat"):
        _full_model(
            _einet_core(
                "dendritic_additive", reactivation={"init_policy": "analytical"}
            )
        )
