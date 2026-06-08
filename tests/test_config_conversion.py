"""Tests for shared config conversion helpers."""

from dendritic_modeling.config.conversion import has_enabled_synapse_types


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
