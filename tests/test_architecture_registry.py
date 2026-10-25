"""Every registered architecture type is classified, and aliases build one network.

A new type added to the factory must be classified here, so it is placed in a
family with tests (or on the non-dendritic list) rather than left untested.
"""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from golden_cases import _spatial_einet_config, _unified_rnn
from test_heterogeneous_leak_ctrnn import _small_config

from dendritic_modeling.networks.architectures.factory import (
    _BUILTIN_ARCHITECTURE_BUILDERS,
    get_architecture,
)

FAMILIES = {
    "einet": [
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
    ],
    "unified_ei": [
        "unified_ei",
        "ei_unified",
        "ei_net",
        "unified_einet",
        "rnn_dendritic_shunting",
        "rnn_dendritic_additive",
        "rnn_dendritic_normalized_additive",
        "rnn_flat_shunting",
        "rnn_flat_additive",
        "rnn_flat_normalized_additive",
    ],
    "population_network": ["population_network"],
    "spatial_tensor_map": [
        "hierarchical_dendritic_tensor_map",
        "dendritic_tensor_map",
        "tensor_map",
    ],
    "spatial_conv": ["hierarchical_dendritic_conv", "spatial_dendritic_conv"],
    "spatial_point_controls": [
        "sparse_active_matched_point_conv",
        "spatial_sparse_active_matched_point",
    ],
    "dendritic_alexnet": ["dendritic_alexnet"],
    "shared_projection_shunting": ["shared_projection_shunting"],
    "routed_ei": ["routed_ei"],
    "feature_readout": ["feature_readout"],
    # Non-dendritic baselines and controls.
    "baseline_rnn": ["gru", "lstm", "rnn", "vanilla_rnn"],
    "heterogeneous_leak_ctrnn": ["heterogeneous_leak_ctrnn", "heterogeneous_ctrnn"],
    "legendre_memory": ["legendre_memory"],
    "matched_parameter_controls": [
        "point_mlp",
        "total_param_mlp",
        "active_param_mlp",
        "direct_active_matched_point",
        "sparse_active_matched_point",
        "effectivematchedparammlp",
    ],
    "mlp": ["mlp", "affine_mlp", "ss_mlp", "ss_mlp_flat"],
    "cnn": [
        "alexnet",
        "cnn",
        "cnn_down",
        "cnn_downsample",
        "cnndownsample",
        "cnn_up",
        "cnn_upsample",
        "cnnupsample",
    ],
    "autoencoder": [
        "mlpautoencoder",
        "cnnautoencoder",
        "mlpvariationalautoencoder",
        "cnnvariationalautoencoder",
    ],
    "router": ["pathway_router", "router"],
    "identity": ["identity", "none"],
}


def test_every_registered_type_is_classified_once():
    classified = [name for names in FAMILIES.values() for name in names]
    assert len(classified) == len(set(classified))
    assert set(_BUILTIN_ARCHITECTURE_BUILDERS) == set(classified)


_CNN_DOWN = {
    "input_shape": [1, 8, 8],
    "hidden_dims": [2],
    "kernel_sizes": [3],
    "strides": [2],
    "output_dim": 4,
}
_CNN_UP = {"input_shape": [2, 2, 2], "target_shape": [1, 4, 4], "hidden_dims": [2]}

# canonical type, its aliases, parameters, input_dim and an input shape
ALIASES = [
    (
        "cnn_downsample",
        ["cnn", "cnn_down", "cnndownsample"],
        _CNN_DOWN,
        None,
        (2, 1, 8, 8),
    ),
    ("cnn_upsample", ["cnn_up", "cnnupsample"], _CNN_UP, None, (2, 2, 2, 2)),
    (
        "hierarchical_dendritic_tensor_map",
        ["tensor_map", "dendritic_tensor_map"],
        _spatial_einet_config(),
        12,
        (2, 12),
    ),
    (
        "hierarchical_dendritic_conv",
        ["spatial_dendritic_conv"],
        _spatial_einet_config(),
        3,
        (2, 3, 5, 5),
    ),
    ("pathway_router", ["router"], {"input_dim": 6, "pathway_dim": 3}, None, (2, 6)),
    ("identity", ["none"], {"input_dim": 5}, None, (2, 5)),
    ("rnn", ["vanilla_rnn"], {"baseline_rnn": {"hidden_dim": 4}}, 3, (2, 5, 3)),
    (
        "heterogeneous_leak_ctrnn",
        ["heterogeneous_ctrnn"],
        {"heterogeneous_leak_ctrnn": _small_config().__dict__},
        2,
        (2, 5, 2),
    ),
]


def _unified_parameters() -> dict:
    """The unified recurrent golden case's parameters, without its type."""
    return {
        "biological_neuron": False,
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
                    "inhibitory": None,
                    "recurrent": True,
                    "dt": 1.0,
                }
            ],
        },
    }


ALIASES.append(
    (
        "unified_ei",
        ["ei_unified", "ei_net", "unified_einet"],
        _unified_parameters(),
        8,
        (2, 4, 8),
    )
)


def _build(name, parameters, input_dim):
    torch.manual_seed(5)
    return get_architecture(name, deepcopy(parameters), input_dim=input_dim)


@pytest.mark.parametrize(
    "canonical, alias, parameters, input_dim, shape",
    [
        (canonical, alias, parameters, input_dim, shape)
        for canonical, aliases, parameters, input_dim, shape in ALIASES
        for alias in aliases
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_alias_builds_the_same_network_as_its_canonical_type(
    canonical, alias, parameters, input_dim, shape
):
    reference = _build(canonical, parameters, input_dim).eval()
    network = _build(alias, parameters, input_dim).eval()

    assert type(network) is type(reference)
    state, expected = network.state_dict(), reference.state_dict()
    assert state.keys() == expected.keys()
    assert all(torch.equal(state[key], expected[key]) for key in expected)
    x = torch.rand(shape, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        output, reference_output = network(x), reference(x)
    if isinstance(output, tuple):
        output, reference_output = output[0], reference_output[0]
    assert torch.equal(output, reference_output)


def test_the_unified_recurrent_golden_builds_through_its_named_type():
    # rnn_dendritic_shunting fixes the integration rule on top of unified_ei.
    network = _unified_rnn()
    assert torch.isfinite(network(torch.rand(2, 4, 8))[0]).all()


@pytest.mark.parametrize(
    "name, parameters, shape",
    [
        (
            "mlpautoencoder",
            {
                "input_dim": 6,
                "encoder_hidden_dims": [4],
                "latent_dim": 2,
                "decoder_hidden_dims": [4],
            },
            (3, 6),
        ),
        (
            "mlpvariationalautoencoder",
            {
                "input_dim": 6,
                "encoder_hidden_dims": [4],
                "latent_dim": 2,
                "decoder_hidden_dims": [4],
            },
            (3, 6),
        ),
        (
            "cnnautoencoder",
            {
                "input_shape": [1, 8, 8],
                "encoder_hidden_dims": [2, 3],
                "latent_shape": [2, 2, 2],
                "decoder_hidden_dims": [3, 2],
            },
            (3, 1, 8, 8),
        ),
        (
            "cnnvariationalautoencoder",
            {
                "input_shape": [1, 8, 8],
                "encoder_hidden_dims": [2, 3],
                "latent_shape": [2, 2, 2],
                "decoder_hidden_dims": [3, 2],
            },
            (3, 1, 8, 8),
        ),
    ],
)
def test_autoencoders_reconstruct_their_input_shape(name, parameters, shape):
    torch.manual_seed(0)
    network = get_architecture(name, parameters)
    output = network(torch.rand(shape))
    reconstruction = output[0] if isinstance(output, tuple) else output
    assert reconstruction.shape == shape
    assert torch.isfinite(reconstruction).all()
