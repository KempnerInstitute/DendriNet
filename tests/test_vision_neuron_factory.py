"""Tests for visual dendritic neuron template dispatch."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.vision_factories.neuron_factory import (
    _build_neuron_from_template,
    build_neuron_from_yaml,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.vision_factories.vision_units import (
    create_center_surround_unit,
    create_end_stopped_unit,
    create_gabor_unit,
    create_motion_unit,
)
from dendritic_modeling.networks.layers.dendritic_conv2d import DendriteConv2d


def _assert_same_push_pull_weights(
    actual: DendriteConv2d,
    expected: DendriteConv2d,
) -> None:
    assert actual.kernel_size == expected.kernel_size
    assert actual.padding == expected.padding
    torch.testing.assert_close(actual.weight_E_raw, expected.weight_E_raw)
    torch.testing.assert_close(actual.weight_I_raw, expected.weight_I_raw)


@pytest.mark.parametrize(
    ("template", "params", "expected_builder"),
    [
        (
            "gabor",
            {"orientation": 30, "wavelength": 6, "sigma": 2.5},
            lambda params: create_gabor_unit(
                orientation=params["orientation"],
                wavelength=params["wavelength"],
                sigma=params["sigma"],
            ),
        ),
        (
            "gabor_pair",
            {"orientation": 45, "wavelength": 7, "sigma": 2.0},
            lambda params: create_gabor_unit(
                orientation=params["orientation"],
                wavelength=params["wavelength"],
                sigma=params["sigma"],
            ),
        ),
        (
            "center_surround",
            {"cell_type": "off-center", "center_size": 3, "surround_size": 9},
            lambda params: create_center_surround_unit(
                kind=params["cell_type"],
                center_size=params["center_size"],
                surround_size=params["surround_size"],
            ),
        ),
        (
            "motion_ds",
            {"preferred_direction": "rightward", "kernel_size": 9},
            lambda params: create_motion_unit(
                direction=params["preferred_direction"],
                kernel_size=params["kernel_size"],
            ),
        ),
        (
            "end_stopped",
            {"preferred_length": 16, "orientation": 60},
            lambda params: create_end_stopped_unit(
                preferred_length=params["preferred_length"],
                orientation=params["orientation"],
            ),
        ),
    ],
)
def test_template_registry_preserves_direct_factory_weights(
    template: str,
    params: dict[str, object],
    expected_builder: Callable[[dict[str, object]], DendriteConv2d],
) -> None:
    actual = _build_neuron_from_template(template, params)
    expected = expected_builder(params)

    _assert_same_push_pull_weights(actual, expected)


def test_build_neuron_from_yaml_uses_template_registry() -> None:
    config = {
        "retina_off": {
            "weight_template": "center_surround",
            "params": {"cell_type": "off-center", "center_size": 3},
        }
    }

    actual = build_neuron_from_yaml("retina_off", config)
    expected = create_center_surround_unit(kind="off-center", center_size=3)

    _assert_same_push_pull_weights(actual, expected)


def test_motion_unit_supports_leftward_odd_kernel() -> None:
    params = {"preferred_direction": "leftward", "kernel_size": 9}

    actual = _build_neuron_from_template("motion_ds", params)
    expected = create_motion_unit(direction="leftward", kernel_size=9)

    _assert_same_push_pull_weights(actual, expected)
    assert actual.weight_E_raw.shape[-2:] == (9, 9)
    assert actual.weight_I_raw.shape[-2:] == (9, 9)
    assert torch.isfinite(actual.weight_E_raw).all()
    assert torch.isfinite(actual.weight_I_raw).all()


def test_build_neuron_from_yaml_rejects_unknown_template() -> None:
    config = {"bad": {"weight_template": "unknown", "params": {}}}

    with pytest.raises(ValueError, match="Unknown weight template: unknown"):
        build_neuron_from_yaml("bad", config)
