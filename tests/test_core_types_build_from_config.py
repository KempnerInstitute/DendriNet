"""Every built-in core type builds from a config mapping as ``dendritic-train`` does.

``initialize_model`` hands the typed ``CoreConfig`` to the architecture
factory, while most other tests call the factory with plain dicts. This test
builds each core type through ``config_from_mapping`` and ``initialize_model``
so a builder that only accepts mappings fails here. Encoder/decoder building
blocks raise a clear error when ``model.core`` names them.
"""

from __future__ import annotations

import math
import re
import warnings
from copy import deepcopy

import pytest
import torch

from dendritic_modeling.config import config_from_mapping
from dendritic_modeling.networks.architectures.factory import (
    _BUILTIN_ARCHITECTURE_BUILDERS,
)
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

_STRUCTURED = {
    "biological_neuron": False,
    "architecture": {
        "excitatory_layer_sizes": [8, 4],
        "inhibitory_layer_sizes": [2, 1],
        "excitatory_branch_factors": [2],
        "inhibitory_branch_factors": [1],
    },
    "connectivity": {
        "ee_synapses_per_branch_per_layer": [4, 2],
        "ei_synapses_per_branch_per_layer": [2, 1],
        "ie_synapses_per_branch_per_layer": [1, 1],
        "ii_synapses_per_branch_per_layer": [0, 0],
    },
}
_INDEXED_SEED = {**_STRUCTURED, "sparsity": {"indexed": {"seed": 3}}}


def _unified(branch_factors: list[int]) -> dict:
    return {
        "biological_neuron": False,
        "unified_ei": {
            "output_mode": "last",
            "layers": [
                {
                    "excitatory": {
                        "n_neurons": 6,
                        "branch_factors": branch_factors,
                        "ff_excitatory_synapses": 3,
                        "rec_excitatory_synapses": 2,
                        # Calibrated gates for the additive aliases.
                        "reactivation_init_policy": "occupancy_quantile",
                    },
                    "inhibitory": None,
                    "recurrent": True,
                    "dt": 1.0,
                }
            ],
        },
    }


_POPULATION = {
    "biological_neuron": False,
    "population_network": {
        "layers": [
            {
                "name": "layer0",
                "readout_population": "e0",
                "population_defaults": {
                    "ff_excitatory_synapses": 4,
                    "ff_inhibitory_synapses": 0,
                },
                "populations": [
                    {
                        "name": "e0",
                        "polarity": "excitatory",
                        "n_neurons": 6,
                        "branch_factors": [2],
                    }
                ],
                "connections": [{"source": "input", "target": "e0"}],
            }
        ]
    },
}
_HETEROGENEOUS = {
    "heterogeneous_leak_ctrnn": {
        "input_dim": 16,
        "input_projection_dims": [3],
        "n_excitatory_outputs": 2,
        "n_inhibitory_outputs": 1,
        "channels_per_output": 4,
        "tau_values": [9.0, 3.0],
        "owner_tau_patterns": [
            {"owner_count": 3, "tau_counts": [2, 2], "excitatory_owner_count": 2}
        ],
        "input_contacts": 30,
        "recurrent_excitatory_contacts": 18,
        "recurrent_inhibitory_contacts": 9,
        "readout_contacts": 4,
        "input_scale": 0.7,
        "recurrent_excitatory_scale": 0.5,
        "recurrent_inhibitory_scale": 0.4,
        "readout_scale": 1.0,
        "topology_seed": 711,
        "initialization_seed": 919,
    }
}
_LEGENDRE = {
    "legendre_memory": {
        "output_dim": 2,
        "memory_order": 4,
        "n_memory_channels": 2,
        "theta": 8.0,
        "input_contacts": 5,
        "readout_contacts": 9,
    }
}
_AFFINE = {"affine_mlp": {"dimensions": [16, 7, 5]}}
_SHARED_PROJECTION = {
    "shared_projection_shunting": {
        "input_dim": 16,
        "output_dim": 5,
        "supports": [[0, 2], [3, 5]],
        "nodes": [0.1, 1.0],
    }
}
_ALEXNET = {
    "biological_neuron": False,
    "architecture": {
        "excitatory_branch_factors": [2],
        "inhibitory_branch_factors": [1],
    },
    "connectivity": {
        "ee_synapses_per_branch_per_layer": [4],
        "ei_synapses_per_branch_per_layer": [3],
        "ie_synapses_per_branch_per_layer": [1],
        "ii_synapses_per_branch_per_layer": [0],
    },
    "transfer": {"output_activation": "relu"},
    "vision": {
        "input_channels": 3,
        "input_transform": "signed_split",
        "channels": [4, 4, 4, 4, 4],
        "kernels": [3, 3, 3, 3, 3],
        "strides": [1, 1, 1, 1, 1],
        "paddings": [1, 1, 1, 1, 1],
        "pool_after": [0, 1, 4],
        "pool_kernel": 2,
        "pool_stride": 2,
        "adaptive_pool_shape": [1, 1],
        "classifier_widths": [6, 6],
        "inhibitory_ratio": 0.25,
        "num_classes": 3,
    },
}

# core type -> (core sections, encoder input size or image shape)
CORE_TYPES = {
    **dict.fromkeys(
        (
            "einet",
            "dendritic_shunting",
            "dendritic_additive",
            "dendritic_normalized_additive",
            "dendritic_signed",
            "dendritic_mlp",
            "flat_shunting",
            "flat_additive",
            "flat_normalized_additive",
            "flat_signed",
            "flat_mlp",
            "point_mlp",
            "total_param_mlp",
            "active_param_mlp",
            "ss_mlp",
            "ss_mlp_flat",
            "hierarchical_dendritic_tensor_map",
            "dendritic_tensor_map",
            "tensor_map",
        ),
        (_STRUCTURED, 16),
    ),
    # Spatial cores read images: an input shape [C, H, W].
    **dict.fromkeys(
        ("hierarchical_dendritic_conv", "spatial_dendritic_conv"),
        (_STRUCTURED, [3, 8, 8]),
    ),
    "sparse_active_matched_point": (_INDEXED_SEED, 64),
    **dict.fromkeys(
        ("sparse_active_matched_point_conv", "spatial_sparse_active_matched_point"),
        (_INDEXED_SEED, [8, 6, 6]),
    ),
    **{
        name: (_unified([2]), 16)
        for name in (
            "unified_ei",
            "ei_unified",
            "ei_net",
            "unified_einet",
            "rnn_dendritic_shunting",
            "rnn_dendritic_additive",
            "rnn_dendritic_normalized_additive",
        )
    },
    **{
        name: (_unified([1]), 16)
        for name in (
            "rnn_flat_shunting",
            "rnn_flat_additive",
            "rnn_flat_normalized_additive",
        )
    },
    **{
        name: ({"baseline_rnn": {"hidden_dim": 4}}, 16)
        for name in ("gru", "lstm", "rnn", "vanilla_rnn")
    },
    **dict.fromkeys(
        ("heterogeneous_leak_ctrnn", "heterogeneous_ctrnn"), (_HETEROGENEOUS, 16)
    ),
    "population_network": (_POPULATION, 16),
    "legendre_memory": (_LEGENDRE, 3),
    "affine_mlp": (_AFFINE, 16),
    "routed_ei": ({"routed_ei": {"features": 8, "branches": 3, "output_dim": 5}}, 16),
    "shared_projection_shunting": (_SHARED_PROJECTION, 16),
    "dendritic_alexnet": (_ALEXNET, [3, 16, 16]),
    "identity": ({}, 16),
    "none": ({}, 16),
}

# Encoder/decoder building blocks take flat parameters that model.core cannot
# express; ``mlp`` points to point_mlp and affine_mlp instead.
BUILDING_BLOCKS = {
    "alexnet": "encoder/decoder building block",
    "cnn": "encoder/decoder building block",
    "cnn_down": "encoder/decoder building block",
    "cnn_downsample": "encoder/decoder building block",
    "cnndownsample": "encoder/decoder building block",
    "cnn_up": "encoder/decoder building block",
    "cnn_upsample": "encoder/decoder building block",
    "cnnupsample": "encoder/decoder building block",
    "mlpautoencoder": "encoder/decoder building block",
    "mlpvariationalautoencoder": "encoder/decoder building block",
    "cnnautoencoder": "encoder/decoder building block",
    "cnnvariationalautoencoder": "encoder/decoder building block",
    "pathway_router": "encoder/decoder building block",
    "router": "encoder/decoder building block",
    "effectivematchedparammlp": "encoder/decoder building block",
    "mlp": "point_mlp",
}


def _experiment(core_type: str, sections: dict, input_dim: int | list[int]) -> dict:
    encoder_params = {"input_dim": input_dim}
    if isinstance(input_dim, list):
        encoder_params = {"input_dim": math.prod(input_dim), "input_shape": input_dim}
    return {
        "experiment": {"seed": 0},
        "data": {"dataset_name": "mnist"},
        "model": {
            "task": "classification",
            "encoder": {"type": "identity", "params": encoder_params},
            "core": {"type": core_type, **deepcopy(sections)},
            "decoder": {
                "type": "MLP",
                "params": {"hidden_dims": [], "output_dim": 3},
            },
        },
    }


def test_every_builtin_type_is_covered():
    covered = set(CORE_TYPES) | set(BUILDING_BLOCKS) | {"direct_active_matched_point"}
    assert covered == set(_BUILTIN_ARCHITECTURE_BUILDERS)


@pytest.mark.parametrize("core_type", sorted(CORE_TYPES))
def test_core_type_builds_from_a_typed_config(core_type):
    sections, input_dim = CORE_TYPES[core_type]
    config = config_from_mapping(_experiment(core_type, sections, input_dim))
    torch.manual_seed(0)
    model, _encoder = initialize_model(config.model)
    assert sum(parameter.numel() for parameter in model.parameters()) >= 0


def test_direct_matched_point_builds_when_its_bottleneck_fits():
    sections = deepcopy(_STRUCTURED)
    sections["architecture"]["excitatory_layer_sizes"] = [8, 64]
    sections["architecture"]["inhibitory_layer_sizes"] = [2, 2]
    config = config_from_mapping(
        _experiment("direct_active_matched_point", sections, 16)
    )
    model, _encoder = initialize_model(config.model)
    assert any(True for _ in model.parameters())


@pytest.mark.parametrize("core_type", sorted(BUILDING_BLOCKS))
def test_building_blocks_are_rejected_as_core_types(core_type):
    config = config_from_mapping(_experiment(core_type, _STRUCTURED, 16))
    with pytest.raises(ValueError, match=BUILDING_BLOCKS[core_type]):
        initialize_model(config.model)


@pytest.mark.parametrize(
    "core_type, sections, path, typo",
    [
        (
            "population_network",
            {
                "biological_neuron": False,
                "population_network": {
                    **_POPULATION["population_network"],
                    "layers": [
                        {
                            **_POPULATION["population_network"]["layers"][0],
                            "population_defaults": {
                                "ff_excitatory_synapses": 4,
                                "ff_excitatory_synapse": 4,
                            },
                        }
                    ],
                },
            },
            "populations[e0]",
            "ff_excitatory_synapse",
        ),
        (
            "population_network",
            {
                "biological_neuron": False,
                "population_network": {
                    **_POPULATION["population_network"],
                    "layers": [
                        {
                            **_POPULATION["population_network"]["layers"][0],
                            "readout": "e0",
                        }
                    ],
                },
            },
            "population_network.layers[0]",
            "readout",
        ),
        (
            "unified_ei",
            {
                "biological_neuron": False,
                "unified_ei": {
                    "layers": [
                        {
                            **_unified([2])["unified_ei"]["layers"][0],
                            "recurent": True,
                        }
                    ]
                },
            },
            "unified_ei.layers[0]",
            "recurent",
        ),
    ],
)
def test_payload_typos_name_their_section(core_type, sections, path, typo):
    config = config_from_mapping(_experiment(core_type, sections, 16))
    with pytest.raises(ValueError, match=re.escape(path)) as error:
        initialize_model(config.model)
    assert typo in str(error.value)


@pytest.mark.parametrize(
    "core_type",
    ["dendritic_additive", "dendritic_normalized_additive", "dendritic_shunting"],
)
def test_recurrent_structured_aliases_build_without_alias_warnings(core_type):
    sections = {
        "biological_neuron": False,
        "architecture": {
            "excitatory_layer_sizes": [8],
            "inhibitory_layer_sizes": [],
            "excitatory_branch_factors": [3, 3],
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [4],
            "ie_synapses_per_branch_per_layer": [2],
            "rec_ee_synapses_per_branch": [2],
        },
    }
    config = config_from_mapping(_experiment(core_type, sections, 16))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model, _encoder = initialize_model(config.model)
    assert any(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("core_type", ["ss_mlp", "ss_mlp_flat"])
def test_sparse_structured_mlp_builds_without_inhibitory_branches(core_type):
    sections = deepcopy(_STRUCTURED)
    sections["architecture"]["inhibitory_layer_sizes"] = [0, 0]
    sections["connectivity"]["ei_synapses_per_branch_per_layer"] = [0, 0]
    sections["connectivity"]["ie_synapses_per_branch_per_layer"] = [1, 0]
    config = config_from_mapping(_experiment(core_type, sections, 16))
    model, _encoder = initialize_model(config.model)
    assert any(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize(
    "core_type", ["dendritic_signed", "flat_signed", "dendritic_mlp", "flat_mlp"]
)
@pytest.mark.parametrize(
    "morphology",
    [{"use_shunting": True}, {"weight_transform": "softplus"}],
    ids=["shunting", "softplus"],
)
def test_signed_and_mlp_types_reject_settings_they_cannot_build(core_type, morphology):
    """These types build additive branches with signed (identity) weights."""
    data = _experiment(core_type, deepcopy(_STRUCTURED), 16)
    data["model"]["core"]["morphology"] = morphology
    with pytest.raises(ValueError, match=f"model.core.type={core_type!r} fixes"):
        config_from_mapping(data)


@pytest.mark.parametrize(
    "core_type", ["dendritic_signed", "flat_signed", "dendritic_mlp", "flat_mlp"]
)
def test_signed_and_mlp_types_record_what_they_build(core_type):
    from dendritic_modeling.networks import DendriticBranchLayer
    from dendritic_modeling.networks.architectures.transformer.dendritic_ffn import (
        build_dendritic_ffn_kwargs_from_core_config,
    )

    data = _experiment(core_type, deepcopy(_STRUCTURED), 16)
    config = config_from_mapping(data)
    morphology = config.model.core.morphology
    assert (morphology.use_shunting, morphology.weight_transform) == (
        False,
        "identity",
    )
    model, _encoder = initialize_model(config.model)
    built = {
        (layer.use_shunting, layer.weight_transform)
        for layer in model.modules()
        if isinstance(layer, DendriticBranchLayer)
    }
    assert built == {(False, "identity")}
    # Transformer feed-forward translations of the type build the same cells.
    for core in (config.model.core, data["model"]["core"]):
        kwargs = build_dendritic_ffn_kwargs_from_core_config(core)
        assert (kwargs["use_shunting"], kwargs["weight_transform"]) == (
            False,
            "identity",
        )
