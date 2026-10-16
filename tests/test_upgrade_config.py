"""dendritic-upgrade-config writes configs that load cleanly and mean the same run."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import pytest
import torch
import yaml

from dendritic_modeling.config import CONFIG_VERSION, config_from_mapping
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.scripts.tools.upgrade_config import main, upgrade_mapping

_VERSION_NOTE = (
    f"config_version: wrote {CONFIG_VERSION} (every key now has this library's meaning)"
)


def _rules(changes: list[str]) -> list[str]:
    """The changes the upgrade rules made; every upgrade ends by writing the version."""
    assert changes[-1] == _VERSION_NOTE
    # This file isolates older migration rules; format-4 declarations and
    # idempotence are tested independently in test_biological_default.py.
    return [
        change
        for change in changes[:-1]
        if not change.startswith("model.core.biological_neuron:")
    ]


_POPULATION = {
    "type": "population_network",
    "population_network": {
        "layers": [
            {
                "name": "layer0",
                "readout_population": "e",
                "population_defaults": {
                    "ff_excitatory_synapses": 4,
                    "ff_inhibitory_synapses": 0,
                    "topk_strategy": "none",
                },
                "populations": [
                    {
                        "name": "e",
                        "polarity": "excitatory",
                        "n_neurons": 6,
                        "branch_factors": [2],
                    }
                ],
                "connections": [{"source": "input", "target": "e"}],
            }
        ]
    },
}


def _old_config() -> dict:
    return {
        "paper_contract": {"study_id": "s1"},
        "experiment": {"seed": 3, "enable_hooks": True},
        "data": {"dataset_name": "mnist"},
        "model": {
            "task": "classification",
            "encoder": {"type": "identity", "params": {"input_dim": 16}},
            "core": _POPULATION,
            "decoder": {"type": "MLP", "params": {"hidden_dims": [], "output_dim": 3}},
        },
        "training": {"main": {"common": {"epochs": 2, "weight_decay_rate": 0.02}}},
        "wandb": {"enabled": True, "use_wandb": False},
    }


def test_an_upgraded_config_loads_as_the_same_run_without_warnings(caplog):
    upgraded, changes, unresolved = upgrade_mapping(_old_config())
    assert unresolved == []
    assert upgraded["metadata"] == {"paper_contract": {"study_id": "s1"}}
    assert "weight_decay_rate" not in upgraded["training"]["main"]["common"]
    defaults = upgraded["model"]["core"]["population_network"]["layers"][0]
    assert defaults["population_defaults"]["topk_gradient_scaling"] == "none"
    assert any("enable_hooks" in change for change in changes)

    with caplog.at_level(logging.WARNING):
        new = config_from_mapping(upgraded)
    assert caplog.records == []
    # The original, without the annotation and the logging request the loader
    # rejects, loads with warnings to the same typed configuration.
    original = _old_config()
    original.pop("paper_contract")
    original.pop("wandb")
    original["model"]["core"]["biological_neuron"] = False
    old = config_from_mapping(original)
    assert new.training.main.common.synapse_decay_rate == 0.02
    assert dataclasses.asdict(new.training) == dataclasses.asdict(old.training)
    # Population payload names are resolved when the model is built.
    models = []
    for config in (new, old):
        torch.manual_seed(0)
        models.append(initialize_model(config.model)[0].state_dict())
    assert models[0].keys() == models[1].keys()
    assert all(torch.equal(models[0][key], models[1][key]) for key in models[0])


def test_sweep_override_paths_take_the_new_names():
    sweep = {
        "base_config": _old_config(),
        "sweep_config": {"training.main.common.weight_decay_rate": [0.0, 0.01]},
        "sweep_variants": [
            {
                "name": "a",
                "overrides": {
                    "model.core.population_network.layers.0.population_defaults.topk_strategy": "none"
                },
            }
        ],
    }
    upgraded, _changes, _unresolved = upgrade_mapping(sweep)
    assert list(upgraded["sweep_config"]) == ["training.main.common.synapse_decay_rate"]
    assert list(upgraded["sweep_variants"][0]["overrides"]) == [
        "model.core.population_network.layers.0.population_defaults.topk_gradient_scaling"
    ]
    assert "metadata" in upgraded["base_config"]


def test_the_command_writes_copies_and_reports_unresolved_keys(tmp_path):
    source = tmp_path / "old.yaml"
    old = _old_config()
    old["experiment"]["checkpointing"] = {"enabled": False}
    old["task"] = "classification"  # a pre-2025 layout key that cannot be replayed
    source.write_text(yaml.safe_dump(old))
    out = tmp_path / "out"
    assert (
        main(
            [
                str(source),
                "--output-dir",
                str(out),
                "--report",
                str(tmp_path / "r.json"),
            ]
        )
        == 1
    )
    written = yaml.safe_load((out / "old.yaml").read_text())
    assert written["metadata"]["paper_contract"] == {"study_id": "s1"}
    assert "task" in written  # left for the user to decide


def _experiment(core: dict) -> dict:
    return {
        "model": {
            "task": "classification",
            "encoder": {"type": "identity", "params": {"input_dim": 16}},
            "core": core,
            "decoder": {"type": "MLP", "params": {"hidden_dims": [], "output_dim": 3}},
        }
    }


def test_unset_recurrent_counts_of_delayed_connections_become_explicit_lesions():
    core = {
        "type": "population_network",
        "population_network": {
            "layers": [
                {
                    "name": "L0",
                    "readout_population": "pyr",
                    "recurrent": True,
                    "population_defaults": {"ff_excitatory_synapses": 4},
                    "populations": [
                        {"name": "pyr", "n_neurons": 6, "branch_factors": [2]},
                        {
                            "name": "sst",
                            "polarity": "inhibitory",
                            "n_neurons": 2,
                            "branch_factors": [1],
                        },
                    ],
                    "connections": [
                        {"source": "input", "target": "pyr"},
                        {"source": "input", "target": "sst"},
                        {"source": "sst", "target": "pyr", "timing": "delayed"},
                    ],
                }
            ]
        },
    }
    upgraded, changes, unresolved = upgrade_mapping(_experiment(core))
    layer = upgraded["model"]["core"]["population_network"]["layers"][0]
    pyr = next(pop for pop in layer["populations"] if pop["name"] == "pyr")
    assert pyr["population"] == {"rec_inhibitory_synapses": 0}
    assert unresolved == [] and len(_rules(changes)) == 1
    initialize_model(config_from_mapping(upgraded).model)


def _structured(
    inhibitory_sizes: list[int], transfer: dict | None = None, **connectivity
) -> dict:
    core = {
        "type": "einet",
        "architecture": {
            "excitatory_layer_sizes": [8, 4],
            "inhibitory_layer_sizes": inhibitory_sizes,
            "excitatory_branch_factors": [2],
            "inhibitory_branch_factors": [1],
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [4, 2],
            "ei_synapses_per_branch_per_layer": [2],
            "ie_synapses_per_branch_per_layer": [1],
            "ii_synapses_per_branch_per_layer": [0],
            **connectivity,
        },
    }
    if transfer is not None:
        core["transfer"] = transfer
    return core


@pytest.mark.parametrize(
    "inhibitory_sizes, transfer, recurrent, expected_sizes, expected_ei, expected_ie",
    [
        # No inhibitory cells: the first layer reads the transferred stream.
        ([], None, {}, [], [0, 0], [1, 0]),
        # input_mode 1, the former feedforward default, built no inhibitory cells.
        ([2], None, {}, [0, 0], [0, 0], [1, 0]),
        ([2, 2], {"input_mode": 1}, {}, [0, 0], [0, 0], [1, 0]),
        # input_mode 0 built them, but none past the end of the list.
        ([2], {"input_mode": 0}, {}, [2, 0], [2, 0], [1, 0]),
        # The recurrent EINet repeated the last size for later layers.
        ([2], None, {"rec_ee_synapses_per_branch": [2]}, [2, 2], [2], [1]),
        # Recurrent input_mode 0 ran first-layer excitatory cells without ie.
        (
            [0, 2],
            {"input_mode": 0},
            {"rec_ee_synapses_per_branch": [2]},
            [0, 2],
            [0, 2],
            [0, 1],
        ),
    ],
)
def test_structured_inhibitory_sizes_and_counts_match_the_built_network(
    inhibitory_sizes, transfer, recurrent, expected_sizes, expected_ei, expected_ie
):
    core = _structured(inhibitory_sizes, transfer, **recurrent)
    upgraded, _changes, unresolved = upgrade_mapping(_experiment(core))
    new_core = upgraded["model"]["core"]
    connectivity = new_core["connectivity"]
    assert new_core["architecture"]["inhibitory_layer_sizes"] == expected_sizes
    assert connectivity["ei_synapses_per_branch_per_layer"] == expected_ei
    assert connectivity["ie_synapses_per_branch_per_layer"] == expected_ie
    assert "transfer" not in new_core
    assert unresolved == []
    initialize_model(config_from_mapping(upgraded).model)


def test_routing_the_current_rule_cannot_express_is_reported():
    core = _structured([2, 2], {"input_mode": 1, "inhibitory_mode": "all"})
    core["connectivity"]["rec_ee_synapses_per_branch"] = [2]
    core["connectivity"]["ii_synapses_per_branch_per_layer"] = [0, 1]
    upgraded, _changes, unresolved = upgrade_mapping(_experiment(core))
    assert upgraded["model"]["core"]["transfer"]["inhibitory_mode"] == "all"
    assert len(unresolved) == 1 and "inhibitory_mode='all'" in unresolved[0]


def test_zero_inhibitory_layer_sizes_load_and_build():
    core = _structured([2, 0])
    core["biological_neuron"] = False
    core["connectivity"]["ei_synapses_per_branch_per_layer"] = [2, 0]
    core["connectivity"]["ie_synapses_per_branch_per_layer"] = [1, 0]
    model, _encoder = initialize_model(config_from_mapping(_experiment(core)).model)
    assert any(p.requires_grad for p in model.parameters())


def test_output_scale_mode_wins_over_the_retired_learned_flag():
    config = _experiment(_structured([2, 0]))
    config["model"]["core"]["connectivity"]["ei_synapses_per_branch_per_layer"] = [2, 0]
    config["model"]["core"]["connectivity"]["ie_synapses_per_branch_per_layer"] = [1, 0]
    config["model"].update(learned_output_scale=False, output_scale_mode="global")
    config["model"]["decoder"] = {"type": "identity", "params": {}}
    upgraded, _changes, unresolved = upgrade_mapping(config)
    assert "learned_output_scale" not in upgraded["model"]
    assert upgraded["model"]["output_scale_mode"] == "global"
    del config["model"]["output_scale_mode"]
    upgraded, _changes, unresolved = upgrade_mapping(config)
    assert upgraded["model"]["output_scale_mode"] == "fixed"
    assert unresolved == []


def test_unread_decoder_params_and_derived_input_dims_are_removed():
    core = {
        "type": "population_network",
        "population_network": {
            "input_dim": 3072,
            "layers": [
                {
                    "name": "L0",
                    "readout_population": "e",
                    "population_defaults": {"ff_excitatory_synapses": 4},
                    "populations": [{"name": "e", "n_neurons": 6}],
                    "connections": [{"source": "input", "target": "e"}],
                }
            ],
        },
    }
    config = _experiment(core)
    config["model"]["decoder"] = {"type": "Identity", "params": {"hidden_dims": [64]}}
    upgraded, changes, unresolved = upgrade_mapping(config)
    assert "input_dim" not in upgraded["model"]["core"]["population_network"]
    assert "hidden_dims" not in upgraded["model"]["decoder"]["params"]
    assert unresolved == [] and len(_rules(changes)) == 2
    initialize_model(config_from_mapping(upgraded).model)


def test_sweep_axes_that_change_what_a_rule_reads_block_it():
    base = _experiment(_structured([2], {"input_mode": 0}))
    sweep = {
        "base_config": base,
        "sweep_config": {
            "model.core.architecture.inhibitory_layer_sizes": [[2], [2, 2]],
            "training.main.common.epochs": [1, 2],
        },
    }
    upgraded, _changes, unresolved = upgrade_mapping(sweep)
    connectivity = upgraded["base_config"]["model"]["core"]["connectivity"]
    assert connectivity["ei_synapses_per_branch_per_layer"] == [2]  # unchanged
    assert unresolved and all(
        "model.core.architecture.inhibitory_layer_sizes" in item for item in unresolved
    )
    # An axis that does not touch the rule's settings leaves it applied.
    sweep["sweep_config"] = {"training.main.common.epochs": [1, 2]}
    upgraded, _changes, unresolved = upgrade_mapping(sweep)
    connectivity = upgraded["base_config"]["model"]["core"]["connectivity"]
    assert connectivity["ei_synapses_per_branch_per_layer"] == [2, 0]
    assert unresolved == []


def test_a_layer_indexed_seed_the_populations_replaced_is_removed():
    core = {
        "type": "population_network",
        "population_network": {
            "layers": [
                {
                    "name": "L0",
                    "readout_population": "e",
                    "population_defaults": {
                        "ff_excitatory_synapses": 4,
                        "topk_type": "indexed",
                        "indexed_seed": 42,
                    },
                    "populations": [{"name": "e", "n_neurons": 6}],
                    "connections": [{"source": "input", "target": "e"}],
                }
            ]
        },
    }
    upgraded, changes, unresolved = upgrade_mapping(_experiment(core))
    layer = upgraded["model"]["core"]["population_network"]["layers"][0]
    assert "indexed_seed" not in layer["population_defaults"]
    assert len(_rules(changes)) == 1 and unresolved == []
    # Standard Top-K never read the seed; it is left as written.
    core["population_network"]["layers"][0]["population_defaults"][
        "topk_type"
    ] = "standard"
    upgraded, changes, _unresolved = upgrade_mapping(_experiment(core))
    layer = upgraded["model"]["core"]["population_network"]["layers"][0]
    assert layer["population_defaults"]["indexed_seed"] == 42
    assert _rules(changes) == []


def test_step_options_the_training_loop_never_applied_are_removed():
    config = _experiment(_structured([], None))
    config["training"] = {
        "main": {
            "strategy": "local_ca",
            "common": {"epochs": 1, "enable_nan_checking": True},
        }
    }
    upgraded, changes, unresolved = upgrade_mapping(config)
    assert unresolved == []
    assert "enable_nan_checking" not in upgraded["training"]["main"]["common"]
    assert any("enable_nan_checking" in change for change in changes)
    config["training"]["main"]["strategy"] = "standard"
    upgraded, changes, _unresolved = upgrade_mapping(config)
    assert upgraded["training"]["main"]["common"]["enable_nan_checking"] is True


def test_processing_flags_the_loader_never_applied_are_removed():
    config = _experiment(_structured([], None))
    config["data"] = {
        "dataset_name": "block_correlated",
        "processing": {"flatten": True, "normalize": True},
    }
    upgraded, changes, unresolved = upgrade_mapping(config)
    assert upgraded["data"]["processing"] == {"flatten": True}
    assert unresolved == [] and any("normalize" in change for change in changes)
    config["data"] = {"dataset_name": "mnist", "processing": {"flatten": True}}
    upgraded, _changes, _unresolved = upgrade_mapping(config)
    assert upgraded["data"]["processing"] == {"flatten": True}


def test_unified_layers_without_an_inhibitory_entry_get_the_default_population():
    core = {
        "type": "ei_unified",
        "unified_ei": {
            "output_mode": "last",
            "layers": [
                {
                    "excitatory": {
                        "n_neurons": 6,
                        "branch_factors": [2],
                        "ff_excitatory_synapses": 3,
                        "ff_inhibitory_synapses": 0,
                        "rec_excitatory_synapses": 2,
                    },
                    "recurrent": True,
                    "dt": 1.0,
                }
            ],
        },
    }
    upgraded, changes, unresolved = upgrade_mapping(_experiment(core))
    layer = upgraded["model"]["core"]["unified_ei"]["layers"][0]
    assert layer["inhibitory"] == {"n_neurons": 16}
    assert len(_rules(changes)) == 1 and unresolved == []


def test_unified_inhibitory_counts_without_an_input_become_zero():
    population = {"n_neurons": 3, "branch_factors": [1], "ff_excitatory_synapses": 3}
    core = {
        "type": "ei_unified",
        "unified_ei": {
            "output_mode": "last",
            "layers": [
                {
                    "excitatory": {
                        "n_neurons": 6,
                        "branch_factors": [2],
                        "ff_excitatory_synapses": 3,
                        "rec_excitatory_synapses": 2,
                    },
                    "inhibitory": {**population, "ff_inhibitory_synapses": 4},
                    "recurrent": True,
                }
            ],
        },
    }
    upgraded, changes, unresolved = upgrade_mapping(_experiment(core))
    layer = upgraded["model"]["core"]["unified_ei"]["layers"][0]
    assert layer["inhibitory"]["ff_inhibitory_synapses"] == 0
    assert unresolved == [] and any("use_transfer is false" in c for c in changes)
    # The upgraded file builds the network the original built (no synapses).
    reference = yaml.safe_load(yaml.safe_dump(upgraded))
    reference["model"]["core"]["unified_ei"]["layers"][0]["inhibitory"].pop(
        "ff_inhibitory_synapses"
    )
    models = []
    for config in (upgraded, reference):
        torch.manual_seed(0)
        models.append(initialize_model(config_from_mapping(config).model)[0])
    states = [model.state_dict() for model in models]
    assert states[0].keys() == states[1].keys()
    assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])
    # A source keeps the count.
    core["unified_ei"]["use_transfer"] = True
    upgraded, _changes, _unresolved = upgrade_mapping(_experiment(core))
    layer = upgraded["model"]["core"]["unified_ei"]["layers"][0]
    assert layer["inhibitory"]["ff_inhibitory_synapses"] == 4


def test_recurrent_baseline_input_dims_and_cells_set_elsewhere_are_removed():
    core = {
        "type": "lstm",
        "baseline_rnn": {"cell_type": "gru", "input_dim": 9, "hidden_dim": 8},
    }
    upgraded, changes, unresolved = upgrade_mapping(_experiment(core))
    assert upgraded["model"]["core"]["baseline_rnn"] == {"hidden_dim": 8}
    assert len(_rules(changes)) == 2 and unresolved == []
    from dendritic_modeling.networks.architectures.recurrent import BaselineRNN

    model, _encoder = initialize_model(config_from_mapping(upgraded).model)
    cores = [m for m in model.modules() if isinstance(m, BaselineRNN)]
    assert [core.config.cell_type for core in cores] == ["lstm"]


def test_a_sweep_rewrite_that_holds_for_every_condition_is_applied():
    base = _experiment(_structured([2], {"input_mode": 1}))
    sweep = {
        "base_config": base,
        "sweep_config": {
            "model.core.type": ["dendritic_additive", "dendritic_shunting"],
        },
    }
    upgraded, _changes, unresolved = upgrade_mapping(sweep)
    core = upgraded["base_config"]["model"]["core"]
    # Both core types are feedforward EINets that built no inhibitory cells.
    assert core["architecture"]["inhibitory_layer_sizes"] == [0, 0]
    assert "transfer" not in core and unresolved == []


def test_settings_that_had_no_effect_are_rewritten_to_their_effect():
    config = _experiment(_structured([], None))
    config["model"]["output_scale_mode"] = "global"
    config["model"]["core"]["morphology"] = {
        "use_shunting": True,
        "use_additive_normalization": True,
    }
    config["model"]["core"]["recurrent_ei"] = {"tau_base": 5.0, "enabled": False}
    config["training"] = {
        "main": {
            "strategy": "standard",
            "common": {"epochs": 1, "reactivation_update_mode": "quantile"},
        }
    }
    upgraded, _changes, unresolved = upgrade_mapping(config)
    model = upgraded["model"]
    assert model["output_scale_mode"] == "fixed"
    assert model["core"]["morphology"] == {"use_shunting": True}
    assert model["core"]["recurrent_ei"] == {"enabled": False}
    assert (
        upgraded["training"]["main"]["common"]["reactivation_update_mode"] == "frozen"
    )
    assert unresolved == []
    initialize_model(config_from_mapping(upgraded).model)


def test_additive_options_of_shunting_populations_are_neutralized():
    core = {
        "type": "population_network",
        "population_network": {
            "layers": [
                {
                    "name": "L0",
                    "readout_population": "a",
                    "population_defaults": {
                        "ff_excitatory_synapses": 4,
                        "additive_mode": "tangent_matched",
                    },
                    "populations": [
                        {
                            "name": "a",
                            "n_neurons": 6,
                            "population": {"use_shunting": False},
                        },
                        {"name": "s", "n_neurons": 6},
                    ],
                    "connections": [
                        {"source": "input", "target": "a"},
                        {"source": "input", "target": "s"},
                    ],
                }
            ]
        },
    }
    upgraded, _changes, unresolved = upgrade_mapping(_experiment(core))
    layer = upgraded["model"]["core"]["population_network"]["layers"][0]
    additive, shunting = layer["populations"]
    assert layer["population_defaults"]["additive_mode"] == "tangent_matched"
    assert "additive_mode" not in additive["population"]
    assert shunting["population"] == {"additive_mode": "raw"}
    assert unresolved == []


def test_local_ca_on_unmodeled_features_gets_the_explicit_opt_in():
    core = {
        "type": "population_network",
        "population_network": {
            "layers": [
                {
                    "name": "L0",
                    "readout_population": "e",
                    "population_defaults": {"ff_excitatory_synapses": 4},
                    "populations": [
                        {
                            "name": "e",
                            "n_neurons": 6,
                            "branch_factors": [2, 2],
                            "population": {"cross_level_mode": "parallel_readout"},
                        }
                    ],
                    "connections": [{"source": "input", "target": "e"}],
                }
            ]
        },
    }
    config = _experiment(core)
    config["training"] = {"main": {"strategy": "local_ca", "common": {"epochs": 1}}}
    upgraded, changes, unresolved = upgrade_mapping(config)
    rules = upgraded["training"]["main"]["learning_strategy_config"]
    assert rules == {"allow_unmodeled_forward_features": True}
    assert any("parallel_readout" in change for change in changes) and unresolved == []
    config["training"]["main"]["strategy"] = "standard"
    upgraded, _changes, _unresolved = upgrade_mapping(config)
    assert "learning_strategy_config" not in upgraded["training"]["main"]


@pytest.mark.parametrize("core_type", ["dendritic_mlp", "flat_mlp"])
def test_mlp_synapse_mode_keeps_its_merged_bank_size(core_type):
    core = _structured([2, 1])
    core["type"] = core_type
    core["connectivity"]["ie_synapses_per_branch_per_layer"] = [1, 1]
    upgraded, _changes, unresolved = upgrade_mapping(_experiment(core))
    connectivity = upgraded["model"]["core"]["connectivity"]
    # Layer 1 has no inhibitory input; its ie contact stays in the merged bank.
    assert connectivity["ee_synapses_per_branch_per_layer"] == [4, 3]
    assert connectivity["ie_synapses_per_branch_per_layer"] == [1, 0]
    assert unresolved == []
    initialize_model(config_from_mapping(upgraded).model)


def test_sparse_options_a_population_type_never_read_are_removed():
    core = {
        "type": "population_network",
        "population_network": {
            "layers": [
                {
                    "name": "L0",
                    "readout_population": "e",
                    "population_defaults": {
                        "ff_excitatory_synapses": 4,
                        "dense_to_sparse_end_step": 4000,
                        "release_max_conductance": 10000.0,
                    },
                    "populations": [{"name": "e", "n_neurons": 6}],
                    "connections": [{"source": "input", "target": "e"}],
                }
            ]
        },
    }
    upgraded, changes, unresolved = upgrade_mapping(_experiment(core))
    population = upgraded["model"]["core"]["population_network"]["layers"][0][
        "populations"
    ][0]
    assert set(population["population"]) == {
        "dense_to_sparse_end_step",
        "release_max_conductance",
    }
    assert len(_rules(changes)) == 2 and unresolved == []
    initialize_model(config_from_mapping(upgraded).model)


def test_path_based_local_errors_get_the_scalar_fallback_opt_in():
    config = _experiment(_structured([], None))
    config["training"] = {
        "main": {
            "strategy": "local_ca",
            "common": {"epochs": 1},
            "learning_strategy_config": {"error_broadcast_mode": "path_transport"},
        }
    }
    upgraded, changes, unresolved = upgrade_mapping(config)
    rules = upgraded["training"]["main"]["learning_strategy_config"]
    assert rules["allow_scalar_fallback"] is True and unresolved == []
    assert any("allow_scalar_fallback" in change for change in changes)
    config["training"]["main"]["learning_strategy_config"] = {
        "error_broadcast_mode": "per_soma_shared"
    }
    upgraded, _changes, _unresolved = upgrade_mapping(config)
    assert (
        "allow_scalar_fallback"
        not in upgraded["training"]["main"]["learning_strategy_config"]
    )


def test_analysis_annotations_move_under_metadata_and_typos_are_reported():
    config = _experiment(_structured([], None))
    config["analysis"] = {
        "study_cell": {"cell_id": "c0"},
        "performence": {"enabled": True},
    }
    upgraded, _changes, unresolved = upgrade_mapping(config)
    assert upgraded["metadata"]["analysis"] == {"study_cell": {"cell_id": "c0"}}
    assert upgraded["analysis"] == {"performence": {"enabled": True}}
    assert len(unresolved) == 1 and "performance" in unresolved[0]


def test_alias_overridden_morphology_and_sweep_filter_tolerances():
    core = _structured([], None)
    core["type"] = "dendritic_additive"
    core["morphology"] = {"use_additive_normalization": True}
    sweep = {
        "base_config": _experiment(core),
        "sweep_config": {"training.main.common.epochs": [1]},
        "filter_config": {"mode": "ei-sum", "value": 8},
    }
    upgraded, _changes, unresolved = upgrade_mapping(sweep)
    assert upgraded["base_config"]["model"]["core"]["morphology"] == {}
    assert upgraded["filter_config"]["tolerance"] == 0 and unresolved == []


def test_core_sections_the_core_type_never_reads_are_removed():
    population = dict(
        _POPULATION,
        morphology={"weight_transform": "identity", "use_shunting": False},
    )
    with pytest.raises(ValueError, match=r"does not read model\.core\.morphology"):
        config_from_mapping(_experiment(population))
    upgraded, changes, unresolved = upgrade_mapping(_experiment(population))
    assert "morphology" not in upgraded["model"]["core"] and unresolved == []
    assert any("model.core.morphology" in change for change in changes)
    config_from_mapping(upgraded)

    structured = _structured([2, 1])
    structured["population_network"] = _POPULATION["population_network"]
    structured["morphology"] = {"use_shunting": True}
    upgraded, _changes, _unresolved = upgrade_mapping(_experiment(structured))
    assert "population_network" not in upgraded["model"]["core"]
    assert upgraded["model"]["core"]["morphology"] == {"use_shunting": True}


def test_a_second_upgrade_changes_nothing():
    # input_mode 0 built the inhibitory cells; the first pass removes the
    # retired key, and a second pass must not read its absence as the former
    # default input_mode 1, which built none.
    legacy = _experiment(_structured([2], {"input_mode": 0}))
    once, changes, unresolved = upgrade_mapping(legacy)
    assert next(iter(once)) == "config_version"
    assert once["config_version"] == CONFIG_VERSION
    assert once["model"]["core"]["architecture"]["inhibitory_layer_sizes"] == [2, 0]
    assert _rules(changes) and unresolved == []

    twice, second_changes, second_unresolved = upgrade_mapping(once)
    assert twice == once
    assert second_changes == [] and second_unresolved == []


@pytest.mark.parametrize("version", [0, CONFIG_VERSION + 1, "2", True])
def test_an_unknown_config_version_is_reported_and_left_unchanged(version):
    legacy = _experiment(_structured([2], {"input_mode": 0}))
    legacy["config_version"] = version
    upgraded, changes, unresolved = upgrade_mapping(legacy)
    assert upgraded == legacy and changes == []
    assert len(unresolved) == 1 and "config_version" in unresolved[0]


def test_an_explicit_first_version_is_upgraded():
    legacy = _experiment(_structured([2], {"input_mode": 0}))
    legacy["config_version"] = 1
    upgraded, changes, _unresolved = upgrade_mapping(legacy)
    assert upgraded["config_version"] == CONFIG_VERSION
    assert upgraded["model"]["core"]["architecture"]["inhibitory_layer_sizes"] == [2, 0]
    assert _rules(changes)


def test_sweep_base_configs_get_the_version():
    sweep = {
        "base_config": _old_config(),
        "sweep_config": {"training.main.common.epochs": [1, 2]},
    }
    upgraded, _changes, _unresolved = upgrade_mapping(sweep)
    assert upgraded["base_config"]["config_version"] == CONFIG_VERSION
    assert "config_version" not in upgraded


_SHIPPED = sorted(
    path
    for path in (Path(__file__).resolve().parents[1] / "configs").rglob("*.yaml")
    if isinstance(data := yaml.safe_load(path.read_text()), dict)
    and ("base_config" in data or {"model", "training", "experiment"} & set(data))
)


@pytest.mark.parametrize("path", _SHIPPED, ids=lambda path: path.name)
def test_shipped_configs_have_the_current_version(path):
    data = yaml.safe_load(path.read_text())
    upgraded, changes, unresolved = upgrade_mapping(data)
    assert upgraded == data
    assert changes == [] and unresolved == []


def test_omitted_stochastic_temperature_keeps_the_former_default():
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.stochastic import (
        StochasticTopKLinear,
    )

    core = _structured([0, 0])
    core["sparsity"] = {"type": "stochastic"}
    upgraded, changes, unresolved = upgrade_mapping(_experiment(core))
    assert upgraded["model"]["core"]["sparsity"]["temperature"] == 0.0
    assert unresolved == [] and any("temperature" in change for change in changes)
    model, _encoder = initialize_model(config_from_mapping(upgraded).model)
    layers = [m for m in model.modules() if isinstance(m, StochasticTopKLinear)]
    assert layers and {layer.temperature for layer in layers} == {0.0}


def test_stochastic_selection_that_was_exactly_standard_is_reported():
    core = _structured([0, 0])
    core["sparsity"] = {"type": "stochastic", "ultrafast": False}
    _upgraded, _changes, unresolved = upgrade_mapping(_experiment(core))
    assert len(unresolved) == 1 and "standard Top-K" in unresolved[0]


def test_indexed_dynamic_stochastic_selection_at_temperature_zero_becomes_standard():
    core = _structured([0, 0])
    core["sparsity"] = {
        "type": "indexed_dynamic",
        "indexed": {"selection": "stochastic"},
    }
    upgraded, _changes, unresolved = upgrade_mapping(_experiment(core))
    assert upgraded["model"]["core"]["sparsity"]["indexed"]["selection"] == "standard"
    assert unresolved == []


@pytest.mark.parametrize(
    "options, written",
    [
        ({"topk_ultrafast": True}, {"topk_temperature": 0.0}),
        ({"topk_temperature": 0.3}, {"topk_ultrafast": False}),
        (
            {"topk_type": "indexed_dynamic", "indexed_selection": "stochastic"},
            {"indexed_selection": "standard"},
        ),
    ],
)
def test_omitted_population_stochastic_options_keep_the_former_defaults(
    options, written
):
    core = yaml.safe_load(yaml.safe_dump(_POPULATION))
    defaults = core["population_network"]["layers"][0]["population_defaults"]
    defaults.pop("topk_strategy")
    defaults.update({"topk_type": "stochastic", **options})
    upgraded, _changes, unresolved = upgrade_mapping(_experiment(core))
    new_defaults = upgraded["model"]["core"]["population_network"]["layers"][0][
        "population_defaults"
    ]
    assert unresolved == []
    for key, value in written.items():
        assert new_defaults[key] == value


def test_population_stochastic_selection_that_was_exactly_standard_is_reported():
    core = yaml.safe_load(yaml.safe_dump(_POPULATION))
    defaults = core["population_network"]["layers"][0]["population_defaults"]
    defaults.pop("topk_strategy")
    defaults["topk_type"] = "stochastic"
    _upgraded, _changes, unresolved = upgrade_mapping(_experiment(core))
    assert len(unresolved) == 1 and "standard Top-K" in unresolved[0]


def test_a_top_level_fsdp_block_is_removed_only_when_it_has_no_effect():
    config = _old_config()
    config["distributed"] = {
        "mode": "fsdp",
        "fsdp": {"sharding_strategy": "FULL_SHARD"},
    }
    config["fsdp"] = {"sharding_strategy": "FULL_SHARD"}
    upgraded, _changes, unresolved = upgrade_mapping(config)
    assert "fsdp" not in upgraded and unresolved == []

    config["fsdp"] = {"mixed_precision_dtype": "float32"}
    upgraded, _changes, unresolved = upgrade_mapping(config)
    assert upgraded["fsdp"] == {"mixed_precision_dtype": "float32"}
    assert len(unresolved) == 1 and "distributed.fsdp" in unresolved[0]

    # The retired switch off: the run did not use FSDP.
    config.pop("distributed")
    config["fsdp"] = {"use_fsdp": False, "sharding_strategy": "FULL_SHARD"}
    upgraded, changes, unresolved = upgrade_mapping(config)
    assert "fsdp" not in upgraded and unresolved == []
    assert any("use_fsdp: false" in change for change in changes)
    config_from_mapping(upgraded)


def test_the_command_keeps_relative_paths_of_same_named_inputs(tmp_path):
    for name, seed in (("a", 1), ("b", 2)):
        (tmp_path / name).mkdir()
        config = _old_config()
        config["experiment"]["seed"] = seed
        (tmp_path / name / "same.yaml").write_text(yaml.safe_dump(config))
    out = tmp_path / "out"
    main(
        [
            str(tmp_path / "a/same.yaml"),
            str(tmp_path / "b/same.yaml"),
            "--output-dir",
            str(out),
        ]
    )
    seeds = {
        name: yaml.safe_load((out / name / "same.yaml").read_text())["experiment"][
            "seed"
        ]
        for name in ("a", "b")
    }
    assert seeds == {"a": 1, "b": 2}


def test_the_command_refuses_inputs_that_would_share_an_output(tmp_path):
    source = tmp_path / "old.yaml"
    source.write_text(yaml.safe_dump(_old_config()))
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="would both be written"):
        main([str(source), str(source), "--output-dir", str(out)])
    assert not out.exists()


def _legacy_recurrent_stack() -> dict:
    core = _structured([0, 0])
    core["recurrent"] = {"enabled": True}  # the cell stack removed on 2026-02-27
    return _experiment(core)


def test_a_partial_upgrade_stays_unresolved_until_it_is_resolved():
    once, _changes, unresolved = upgrade_mapping(_legacy_recurrent_stack())
    assert len(unresolved) == 1 and "legacy recurrent cell stack" in unresolved[0]
    assert list(once)[:2] == ["config_version", "unresolved_migration"]
    assert once["unresolved_migration"] == unresolved
    with pytest.raises(ValueError, match="only partly upgraded"):
        config_from_mapping(once)

    # A retry rewrites nothing and reports the same items.
    again, again_changes, again_unresolved = upgrade_mapping(once)
    assert again == once
    assert again_changes == [] and again_unresolved == unresolved

    # Resolved by hand, the file loads and a further upgrade changes nothing.
    resolved = yaml.safe_load(yaml.safe_dump(once))
    del resolved["unresolved_migration"]
    del resolved["model"]["core"]["recurrent"]
    config_from_mapping(resolved)
    assert upgrade_mapping(resolved) == (resolved, [], [])


def test_a_partial_sweep_upgrade_is_listed_in_its_base_config():
    sweep = {
        "base_config": _legacy_recurrent_stack(),
        "sweep_config": {"training.main.common.epochs": [1, 2]},
    }
    once, _changes, unresolved = upgrade_mapping(sweep)
    assert unresolved and all(item.startswith("base_config: ") for item in unresolved)
    assert once["base_config"]["unresolved_migration"]
    again, again_changes, again_unresolved = upgrade_mapping(once)
    assert again == once and again_changes == [] and again_unresolved == unresolved


def test_the_command_keeps_failing_on_a_partial_upgrade(tmp_path):
    source = tmp_path / "old.yaml"
    source.write_text(yaml.safe_dump(_legacy_recurrent_stack()))
    first, second = tmp_path / "first", tmp_path / "second"
    assert main([str(source), "--output-dir", str(first)]) == 1
    assert main([str(first / "old.yaml"), "--output-dir", str(second)]) == 1
    assert (second / "old.yaml").read_text() == (first / "old.yaml").read_text()
