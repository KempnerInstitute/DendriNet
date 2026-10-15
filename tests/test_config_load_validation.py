"""Regression tests for load-time structural config validation."""

import pytest
import yaml

from dendritic_modeling.config import ConfigValidationError, load_config


def _write_config(tmp_path, payload):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
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
                "biological_neuron": False,
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


@pytest.mark.parametrize(
    "core_type",
    [
        "dendritic_signed",
        "flat_signed",
        "flat_mlp",
        "tensor_map",
        "hierarchical_dendritic_tensor_map",
        "hierarchical_dendritic_conv",
        "sparse_active_matched_point_conv",
        "sparse_active_matched_point",
        "point_mlp",
        "ss_mlp",
    ],
)
def test_every_type_built_from_the_structured_sections_checks_its_lists(
    tmp_path, core_type
):
    # These types build (or match) a structured E/I network from the same
    # sections, which would ignore entries past the number of layers.
    core = {
        "biological_neuron": False,
        "type": core_type,
        "architecture": {"excitatory_layer_sizes": [8], "inhibitory_layer_sizes": []},
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [2],
            "ie_synapses_per_branch_per_layer": [0],
        },
    }
    load_config(str(_write_config(tmp_path, _base_config(core))))
    core["connectivity"]["ee_synapses_per_branch_per_layer"] = [2, 2, 2]
    with pytest.raises(ConfigValidationError, match="has 3 entries for 1 layer"):
        load_config(str(_write_config(tmp_path, _base_config(core))))


def test_load_config_rejects_population_network_bad_endpoint(tmp_path):
    path = _write_config(
        tmp_path,
        _base_config(
            {
                "biological_neuron": False,
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
                "biological_neuron": False,
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


def test_load_config_rejects_removed_population_graph_type(tmp_path):
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

    with pytest.raises(ConfigValidationError, match=r"population_graph.*removed"):
        load_config(str(path))
