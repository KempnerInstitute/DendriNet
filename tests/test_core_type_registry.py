"""The factory and the configuration layer share one registry of core types."""

from __future__ import annotations

import pytest

from dendritic_modeling.config import recurrence
from dendritic_modeling.config.recurrence import is_recurrent_core_config
from dendritic_modeling.networks.architectures import factory, factory_recurrent


def test_factory_uses_the_configuration_registries():
    assert factory._EINET_TYPES is recurrence._STRUCTURED_EI_TYPES
    assert factory._UNIFIED_EI_TYPES is recurrence._UNIFIED_EI_TYPES
    assert factory._POPULATION_NETWORK_TYPES is recurrence._POPULATION_NETWORK_TYPES
    assert set(factory_recurrent._BASELINE_RNN_TYPES) == recurrence._BASELINE_RNN_TYPES


@pytest.mark.parametrize("core_type", sorted(recurrence._STRUCTURED_EI_TYPES))
def test_every_structured_type_detects_recurrent_connectivity(core_type):
    core = {
        "type": core_type,
        "connectivity": {"rec_ee_synapses_per_branch": [2]},
    }
    assert is_recurrent_core_config(core) is True
    core["connectivity"] = {"rec_ee_synapses_per_branch": [0]}
    assert is_recurrent_core_config(core) is False
