"""Regression tests for load-time structural config validation."""

import pytest
import yaml

from dendritic_modeling.config import ConfigValidationError, load_config


def _write_config(tmp_path, payload):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    load_config.cache_clear()
    return path


def _base_config(core):
    return {
        "data": {"dataset_name": "mnist"},
        "model": {"core": core},
        "training": {"main": {"common": {"epochs": 1}}},
    }


def test_load_config_rejects_invalid_legacy_synapse_list_length(tmp_path):
    path = _write_config(
        tmp_path,
        _base_config(
            {
                "type": "einet",
                "architecture": {"excitatory_layer_sizes": [8]},
                "connectivity": {
                    "ee_synapses_per_branch_per_layer": [4, 4],
                    "ei_synapses_per_branch_per_layer": [0],
                    "ie_synapses_per_branch_per_layer": [2],
                    "ii_synapses_per_branch_per_layer": [0],
                },
            }
        ),
    )

    with pytest.raises(ConfigValidationError, match="ee_synapses_per_branch"):
        load_config(str(path))


def test_load_config_rejects_population_network_bad_endpoint(tmp_path):
    path = _write_config(
        tmp_path,
        _base_config(
            {
                "type": "population_network",
                "population_network": {
                    "layers": [
                        {
                            "name": "L0",
                            "populations": [
                                {
                                    "name": "e",
                                    "polarity": "excitatory",
                                    "n_neurons": 4,
                                    "branch_factors": [1],
                                }
                            ],
                            "connections": [{"source": "input", "target": "missing"}],
                        }
                    ]
                },
            }
        ),
    )

    with pytest.raises(ConfigValidationError, match="target 'missing'"):
        load_config(str(path))


def test_load_config_rejects_population_network_delayed_without_recurrent(tmp_path):
    path = _write_config(
        tmp_path,
        _base_config(
            {
                "type": "population_network",
                "population_network": {
                    "layers": [
                        {
                            "name": "L0",
                            "populations": [
                                {
                                    "name": "e",
                                    "polarity": "excitatory",
                                    "n_neurons": 4,
                                    "branch_factors": [1],
                                }
                            ],
                            "connections": [
                                {"source": "input", "target": "e"},
                                {
                                    "source": "e",
                                    "target": "e",
                                    "timing": "delayed",
                                },
                            ],
                        }
                    ]
                },
            }
        ),
    )

    with pytest.raises(ConfigValidationError, match="requires recurrent: true"):
        load_config(str(path))


def test_load_config_rejects_population_graph_bad_endpoint(tmp_path):
    path = _write_config(
        tmp_path,
        _base_config(
            {
                "type": "population_graph",
                "population_graph": {
                    "cell_types": [
                        {
                            "name": "pyr",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [1],
                        }
                    ],
                    "connections": [{"source": "pv", "target": "pyr"}],
                },
            }
        ),
    )

    with pytest.raises(ConfigValidationError, match="source 'pv'"):
        load_config(str(path))
