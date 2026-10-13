"""config_version 3 and the upgrade of version-2 structured-connectivity rules.

Version 3 reads a structured E/I ``distance`` rule without a kernel as
gaussian, lets a pathway entry's second spelling of a key override the root's
key, and seeds unseeded population and unified E/I rules from the topology
seed. ``dendritic-upgrade-config`` writes the version-2 meaning into version-2
files; the models those files built are recorded below as the state hashes
that library revision 945bf481c (config_version 2) produced.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest
import torch

from dendritic_modeling.config import CONFIG_VERSION, config_from_mapping
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.scripts.script_utils.setup_utils import (
    _apply_topology_seed_defaults,
    initialize_model,
)
from dendritic_modeling.scripts.tools.upgrade_config import upgrade_mapping

_INPUTS, _COMPARTMENTS = 20, 10
_PRE = [[float(index)] for index in range(_INPUTS)]
_POST = [[2.0 * index] for index in range(_COMPARTMENTS)]


def _structured(structured: dict | None, version: int = 2) -> dict:
    connectivity = {
        "ee_synapses_per_branch_per_layer": [_INPUTS],
        "ie_synapses_per_branch_per_layer": [0],
    }
    if structured is not None:
        connectivity["structured"] = structured
    return {
        "config_version": version,
        "model": {
            "task": "classification",
            "encoder": {"type": "identity", "params": {"input_dim": _INPUTS}},
            "core": {
                "type": "dendritic_shunting",
                "architecture": {
                    "excitatory_layer_sizes": [_COMPARTMENTS],
                    "inhibitory_layer_sizes": [],
                    "excitatory_branch_factors": [1],
                },
                "connectivity": connectivity,
            },
            "decoder": {"type": "identity", "params": {"hidden_dims": []}},
        },
    }


def _population(structured: dict, version: int = 2, **defaults) -> dict:
    population_defaults = {
        "ff_excitatory_synapses": _INPUTS,
        "ff_inhibitory_synapses": 0,
        "structured_connectivity": structured,
        **defaults,
    }
    layer = {
        "name": "layer0",
        "readout_population": "e",
        "population_defaults": population_defaults,
        "populations": [
            {
                "name": "e",
                "polarity": "excitatory",
                "n_neurons": _COMPARTMENTS,
                "branch_factors": [1],
            }
        ],
        "connections": [{"source": "input", "target": "e"}],
    }
    return {
        "config_version": version,
        "model": {
            "task": "classification",
            "encoder": {"type": "identity", "params": {"input_dim": _INPUTS}},
            "core": {
                "type": "population_network",
                "population_network": {"layers": [layer]},
            },
            "decoder": {"type": "identity", "params": {"hidden_dims": []}},
        },
    }


def _distance(**options) -> dict:
    return {
        "method": "distance",
        "pre_positions": _PRE,
        "post_positions": _POST,
        **options,
    }


_SPATIAL = {
    "enabled": True,
    "method": "spatial_morphology",
    "spatial": {"input_shape": [1, 4, 5], "split_axes": ["height"]},
}

# Version-2 files and the state hashes of the models 945bf481c built from them
# (the same with and without the training seed defaults).
_V2_CASES = {
    # The typed default kernel was "none": base_probability everywhere.
    "typed_distance_without_kernel": (
        _structured(
            {
                "enabled": True,
                "seed": 7,
                "pathways": {"ee": _distance(base_probability=0.6)},
            }
        ),
        "52c440d641b91e1761f6f157af5504c6fb325706fc7223d687fe227e355a7773",
    ),
    # Second spellings in a pathway lost to the typed root's keys.
    "typed_distance_shadowed_aliases": (
        _structured(
            {
                "enabled": True,
                "seed": 7,
                "pathways": {
                    "ee": _distance(kernel="gaussian", sigma=3.0, base_probability=0.6)
                },
            }
        ),
        "52c440d641b91e1761f6f157af5504c6fb325706fc7223d687fe227e355a7773",
    ),
    # Unseeded population masks drew from the global random stream.
    "population_without_seed": (
        _population({"enabled": True, "probability": 0.5}),
        "ee076a6241a9487996357444bcc04261754a9a89166ff81a91336daca8963e3e",
    ),
    # Unseeded spatial contacts used seed 0.
    "population_spatial_without_seed": (
        _population(_SPATIAL, topk_type="indexed"),
        "faf6b0ec707c7f66d0cb981d366f26b5253514923ad288056594ac8dc0f84278",
    ),
    # A canonical root key shadowed a pathway's second spelling.
    "population_shadowed_alias": (
        _population(
            {
                "enabled": True,
                "seed": 5,
                "distance_kernel": "exponential",
                "distance_sigma": 2.0,
                "pathways": {
                    "ee": _distance(kernel="gaussian", sigma=0.5, base_probability=0.9)
                },
            }
        ),
        "446c75d777b5ebf9c31c53c0cb315bf90b81aa3365adbec58e68c24eae34b76e",
    ),
    # Neither: only the version changes.
    "neither": (
        _structured(None),
        "e5d401d1155a3f46d47385fd07d1011458351a4411b938f9599f3eec5e99190f",
    ),
}


def _state_hash(model) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _build(data: dict, *, topology_seed: int | None = None, torch_seed: int = 0):
    config = config_from_mapping(deepcopy(data))
    if topology_seed is not None:
        _apply_topology_seed_defaults(config.model, topology_seed)
    torch.manual_seed(torch_seed)
    return initialize_model(config.model)[0]


def _masks(model) -> list[torch.Tensor]:
    return [
        layer.branch_excitation.connection_mask.clone()
        for layer in model.modules()
        if isinstance(layer, DendriticBranchLayer)
        and getattr(layer.branch_excitation, "connection_mask", None) is not None
    ]


def test_the_current_format_is_version_3():
    assert CONFIG_VERSION == 3
    with pytest.raises(ValueError, match="dendritic-upgrade-config"):
        config_from_mapping(_V2_CASES["neither"][0])


@pytest.mark.parametrize("name", sorted(_V2_CASES))
@pytest.mark.parametrize("training", [False, True], ids=["built", "trained"])
def test_the_upgrade_rebuilds_the_version_2_model_byte_for_byte(name, training):
    data, v2_hash = _V2_CASES[name]
    upgraded, _changes, unresolved = upgrade_mapping(deepcopy(data))
    assert unresolved == [] and upgraded["config_version"] == 3
    model = _build(upgraded, topology_seed=17 if training else None)
    assert _state_hash(model) == v2_hash
    twice, again, still = upgrade_mapping(deepcopy(upgraded))
    assert twice == upgraded and again == [] and still == []


def test_the_upgrade_writes_the_version_2_values():
    def core(name):
        upgraded, _, _ = upgrade_mapping(deepcopy(_V2_CASES[name][0]))
        return upgraded["model"]["core"]

    typed = core("typed_distance_shadowed_aliases")["connectivity"]["structured"]
    assert typed["distance_kernel"] == "none"
    assert set(typed["pathways"]["ee"]) == {
        "method",
        "pre_positions",
        "post_positions",
        "base_probability",
    }

    def rule(name):
        layer = core(name)["population_network"]["layers"][0]
        return layer["population_defaults"]["structured_connectivity"]

    assert rule("population_without_seed")["seed"] == "global_rng"
    assert rule("population_spatial_without_seed")["seed"] == 0
    assert "kernel" not in rule("population_shadowed_alias")["pathways"]["ee"]
    assert core("neither") == _V2_CASES["neither"][0]["model"]["core"]


def test_version_3_reads_an_omitted_kernel_as_gaussian():
    rule = {
        "enabled": True,
        "seed": 7,
        "pathways": {"ee": _distance(base_probability=1.0)},
    }
    gaussian = deepcopy(rule)
    gaussian["pathways"]["ee"]["distance_kernel"] = "gaussian"
    for builder in (_structured, _population):
        omitted = _masks(_build(builder(rule, version=3)))
        written = _masks(_build(builder(gaussian, version=3)))
        assert omitted and all(torch.equal(a, b) for a, b in zip(omitted, written))
        # Far inputs get no contact, so the positions are used.
        assert not any(bool(mask.all()) for mask in omitted)


@pytest.mark.parametrize("where", ["population", "unified"])
def test_version_3_seeds_unseeded_rules_from_the_topology_seed(where):
    if where == "population":
        data = _population({"enabled": True, "probability": 0.5}, version=3)
    else:
        population = {
            "n_neurons": _COMPARTMENTS,
            "branch_factors": [1],
            "ff_excitatory_synapses": _INPUTS,
            "ff_inhibitory_synapses": 0,
            "structured_connectivity": {"enabled": True, "probability": 0.5},
        }
        data = _structured(None, version=3)
        data["model"]["core"] = {
            "type": "unified_ei",
            "unified_ei": {"layers": [{"excitatory": population, "inhibitory": None}]},
        }
    first = _masks(_build(data, topology_seed=17, torch_seed=0))
    again = _masks(_build(data, topology_seed=17, torch_seed=1))
    other = _masks(_build(data, topology_seed=18, torch_seed=0))
    assert first and all(torch.equal(a, b) for a, b in zip(first, again))
    assert not all(torch.equal(a, b) for a, b in zip(first, other))
    # Built outside a training run the rule has no seed.
    unseeded = _masks(_build(data, torch_seed=1))
    assert not all(torch.equal(a, b) for a, b in zip(first, unseeded))


def test_a_layer_default_seed_is_inherited_and_a_null_override_is_filled():
    data = _population({"enabled": True, "probability": 0.5}, version=3)
    layer = data["model"]["core"]["population_network"]["layers"][0]
    second = deepcopy(layer["populations"][0])
    second.update(
        name="f",
        population={"structured_connectivity": {"enabled": True, "seed": None}},
    )
    layer["populations"].append(second)
    layer["connections"].append({"source": "input", "target": "f"})
    config = config_from_mapping(data)
    _apply_topology_seed_defaults(config.model, 17)
    layer = config.model.core.population_network["layers"][0]
    assert layer["population_defaults"]["structured_connectivity"]["seed"] == 17
    assert "seed" not in layer["populations"][0]["population"].get(
        "structured_connectivity", {}
    )
    assert (
        layer["populations"][1]["population"]["structured_connectivity"]["seed"] == 17
    )


def test_version_1_rules_leave_version_2_files_alone():
    # The stochastic Top-K default changed between versions 1 and 2.
    data = _structured(None)
    data["model"]["core"]["sparsity"] = {"type": "stochastic"}
    upgraded, changes, unresolved = upgrade_mapping(deepcopy(data))
    assert "temperature" not in upgraded["model"]["core"]["sparsity"]
    assert unresolved == [] and len(changes) == 1
    legacy = deepcopy(data)
    del legacy["config_version"]
    upgraded, _changes, _unresolved = upgrade_mapping(legacy)
    assert upgraded["model"]["core"]["sparsity"]["temperature"] == 0.0


def test_keys_retired_after_version_2_leave_version_2_files():
    # batch_process was never read; a version-2 file loads with it (with a
    # warning), and the upgrade to version 3 removes it as it does for older files.
    data = _structured(None)
    data["analysis"] = {
        "correlation": {
            "enabled": False,
            "params": {"compute": {"batch_process": True}},
        }
    }
    upgraded, changes, unresolved = upgrade_mapping(deepcopy(data))
    assert unresolved == []
    assert upgraded["config_version"] == CONFIG_VERSION
    assert upgraded["analysis"]["correlation"]["params"]["compute"] == {}
    assert any("batch_process" in change for change in changes)


def test_a_rule_the_upgrade_cannot_keep_is_reported():
    # One unseeded root serves a spatial rule (seed 0) and random masks.
    rule = {**_SPATIAL, "pathways": {"ie": {"method": "bernoulli", "probability": 0.5}}}
    upgraded, _changes, unresolved = upgrade_mapping(
        _population(rule, topk_type="indexed")
    )
    assert unresolved and all("version 2 read" in item for item in unresolved)
    assert upgraded["unresolved_migration"] == unresolved


def test_unified_layers_get_their_own_filled_structured_seed():
    # Unified layers of equal shape draw equal masks from one seed, so each
    # later layer gets a seed of its own.
    population = {
        "n_neurons": 6,
        "branch_factors": [1],
        "ff_excitatory_synapses": 6,
        "ff_inhibitory_synapses": 0,
        "structured_connectivity": {"enabled": True, "probability": 0.5},
    }
    data = _structured(None, version=3)
    data["model"]["encoder"]["params"]["input_dim"] = 6
    data["model"]["core"] = {
        "type": "unified_ei",
        "unified_ei": {
            "layers": [
                {"excitatory": deepcopy(population), "inhibitory": None}
                for _ in range(3)
            ]
        },
    }
    config = config_from_mapping(deepcopy(data))
    _apply_topology_seed_defaults(config.model, 17)
    layers = config.model.core.unified_ei["layers"]
    seeds = [layer["excitatory"]["structured_connectivity"]["seed"] for layer in layers]
    assert seeds[0] == 17 and len(set(seeds)) == 3
    model = _build(data, topology_seed=17)
    per_layer = [
        _masks(layer) for layer in model.modules() if type(layer).__name__ == "EILayer"
    ]
    assert len(per_layer) == 3
    assert not all(torch.equal(a, b) for a, b in zip(per_layer[1], per_layer[2]))
    assert not all(torch.equal(a, b) for a, b in zip(per_layer[0], per_layer[1]))
