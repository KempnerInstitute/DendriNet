import torch

from dendritic_modeling.networks import DendriticSynapseConfig
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic import (
    DendriNet,
    DendriticBranchLayer,
)


def _assert_state_dict_exact(left, right):
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0.0, atol=0.0)


def test_dendritic_synapse_config_round_trips_legacy_kwargs():
    legacy_kwargs = {
        "topk_type": "indexed_rewire",
        "topk_init_method": "xavier_normal",
        "topk_gamma": 0.75,
        "indexed_seed": 17,
        "indexed_output_chunk_size": 64,
        "indexed_rewire_frequency": 9,
        "indexed_rewire_quantile": 0.2,
        "dense_to_sparse_initial_density": 0.4,
        "reactivate": True,
        "reactivation_type": "param_tanh",
        "reactivation_init_policy": "fixed",
        "reactivation_init_m": 1.2,
        "reactivation_init_b": 0.3,
        "use_shunting": False,
        "use_additive_normalization": True,
        "weight_transform": "softplus",
        "dbl_init_method": "default",
        "adaptive_initialization": False,
        "adaptive_initialization_policy": "preserve_shunting_center",
        "adaptive_target_conductance": 4.0,
        "branch_factors": [2, 1],
        "unknown_activation_kwarg": "kept out of config",
    }

    config, extra = DendriticSynapseConfig.split_kwargs(legacy_kwargs)
    round_trip = config.to_kwargs()

    assert extra == {"unknown_activation_kwarg": "kept out of config"}
    for key, value in legacy_kwargs.items():
        if key in extra:
            continue
        assert round_trip[key] == value


def test_branch_layer_typed_config_matches_legacy_kwargs_numerically():
    geometry_kwargs = {
        "output_dim": 4,
        "excitatory_input_dim": 7,
        "excitatory_synapses_per_branch": 3,
        "inhibitory_input_dim": 5,
        "inhibitory_synapses_per_branch": 2,
        "input_branch_factor": 2,
        "layer_idx": 1,
    }
    legacy_synapse_kwargs = {
        "topk_type": "standard",
        "topk_init_method": "xavier_normal",
        "topk_gamma": 0.6,
        "reactivate": True,
        "reactivation_type": "param_tanh",
        "reactivation_init_policy": "fixed",
        "reactivation_init_m": 1.3,
        "reactivation_init_b": 0.25,
        "use_shunting": True,
        "weight_transform": "exp",
        "dbl_init_method": "default",
        "adaptive_initialization": False,
        "branch_factors": [2],
    }

    torch.manual_seed(123)
    legacy_layer = DendriticBranchLayer(
        **geometry_kwargs,
        **legacy_synapse_kwargs,
    )
    torch.manual_seed(123)
    typed_layer = DendriticBranchLayer(
        **geometry_kwargs,
        synapse_config=DendriticSynapseConfig.from_kwargs(legacy_synapse_kwargs),
    )

    _assert_state_dict_exact(legacy_layer.state_dict(), typed_layer.state_dict())

    x = torch.rand(3, 7)
    inhibitory = torch.rand(3, 5)
    branch_input = torch.rand(3, 8)

    torch.testing.assert_close(
        legacy_layer(x, inhibitory, branch_input),
        typed_layer(x, inhibitory, branch_input),
        rtol=0.0,
        atol=0.0,
    )


def test_dendrinet_passes_legacy_synapse_kwargs_through_typed_config():
    net = DendriNet(
        n_soma=3,
        branch_factors=[2],
        excitatory_input_dim=6,
        excitatory_synapses_per_branch=2,
        inhibitory_input_dim=4,
        inhibitory_synapses_per_branch=1,
        somatic_synapses=True,
        topk_type="indexed_rewire",
        indexed_seed=31,
        indexed_output_chunk_size=16,
        indexed_rewire_frequency=7,
        indexed_rewire_quantile=0.15,
        reactivate=False,
        use_additive_normalization=True,
        weight_transform="softplus",
    )

    assert len(net.branch_layers) == 2
    for layer in net.branch_layers:
        assert layer.topk_type == "indexed_rewire"
        assert layer.indexed_seed == 31
        assert layer.indexed_output_chunk_size == 16
        assert layer.indexed_rewire_frequency == 7
        assert layer.indexed_rewire_quantile == 0.15
        assert layer.use_additive_normalization is True
        assert layer.weight_transform == "softplus"
