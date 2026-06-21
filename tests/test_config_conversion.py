"""Tests for shared config conversion helpers."""

from dataclasses import dataclass

from omegaconf import OmegaConf

import dendritic_modeling.config.config as config_module
from dendritic_modeling.config.conversion import (
    has_enabled_synapse_types,
    to_plain_dict,
)
from dendritic_modeling.networks.architectures import (
    factory,
    factory_recurrent,
    factory_spatial,
    factory_unified,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    structured_mask,
)
from dendritic_modeling.networks.architectures.recurrent import (
    structured_recurrent_sections,
    synapse_types,
)
from dendritic_modeling.networks.architectures.transformer import (
    utils as transformer_utils,
)
from dendritic_modeling.scripts.script_utils import setup_utils
from dendritic_modeling.training import replacement_common


@dataclass
class _ToyConfig:
    value: int
    label: str = "test"


class _AsDictConfig:
    def asdict(self) -> dict[str, int]:
        return {"value": 3}


def test_plain_mapping_helpers_match_shared_converter():
    """Private config-boundary helpers should share one conversion contract."""

    helpers = (
        replacement_common._to_plain_mapping,
        factory._to_plain_mapping,
        factory_spatial._to_plain_mapping,
        factory_recurrent._to_plain_mapping,
        factory_unified._to_plain_mapping,
        transformer_utils._to_plain_mapping,
        structured_recurrent_sections._to_plain_mapping,
        synapse_types._to_plain_mapping,
        structured_mask._to_plain_mapping,
        config_module._to_plain_dict,
        setup_utils._config_to_plain_dict,
    )
    inputs = (
        {"value": 1},
        OmegaConf.create({"value": 2}),
        _AsDictConfig(),
        _ToyConfig(value=4),
    )

    for helper in helpers:
        for payload in inputs:
            assert helper(payload) == to_plain_dict(payload)


def test_has_enabled_synapse_types_respects_explicit_false():
    assert (
        has_enabled_synapse_types(
            {
                "enabled": False,
                "excitatory": [{"name": "AMPA", "fraction": 1.0}],
                "inhibitory": [{"name": "GABA_A", "fraction": 1.0}],
            }
        )
        is False
    )


def test_has_enabled_synapse_types_keeps_legacy_implicit_lists():
    assert (
        has_enabled_synapse_types(
            {
                "excitatory": [{"name": "AMPA", "fraction": 1.0}],
                "inhibitory": [{"name": "GABA_A", "fraction": 1.0}],
            }
        )
        is True
    )
