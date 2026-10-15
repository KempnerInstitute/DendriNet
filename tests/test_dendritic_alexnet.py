"""Contract tests for the end-to-end dendritic AlexNet."""

import torch

from dendritic_modeling.networks.architectures.factory import (
    get_architecture,
    get_available_architectures,
)


def _tiny_config() -> dict:
    return {
        "biological_neuron": False,
        "vision": {
            "input_transform": "signed_split",
            "channels": [2, 3, 4, 4, 4],
            "kernels": [3, 3, 3, 3, 3],
            "strides": [2, 1, 1, 1, 1],
            "paddings": [1, 1, 1, 1, 1],
            "pool_after": [0, 1, 4],
            "adaptive_pool_shape": [1, 1],
            "classifier_widths": [5, 5],
            "inhibitory_ratio": 0.5,
            "num_classes": 3,
            "dropout": 0.0,
            "seed": 11,
        },
        "architecture": {
            "excitatory_layer_sizes": [2],
            "inhibitory_layer_sizes": [1],
            "excitatory_branch_factors": [1],
            "inhibitory_branch_factors": [1],
            "inhibitory_network_type": "dendritic",
            "mlp_inhibitory_network_params": {
                "hidden_dims": [],
                "activation": "relu",
            },
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [2],
            "ei_synapses_per_branch_per_layer": [1],
            "ie_synapses_per_branch_per_layer": [1],
            "ii_synapses_per_branch_per_layer": [0],
        },
        "transfer": {
            "independent_pathways": False,
            "excitatory_dim": None,
            "inhibitory_dim": None,
            "output_activation": "relu",
        },
        "morphology": {
            "somatic_synapses": False,
            "use_shunting": True,
            "weight_transform": "softplus",
            "dbl_init_method": "analytical_expectation",
        },
        "sparsity": {
            "type": "standard",
            "init_method": "xavier_normal",
            "noise_level": 0.0,
            "weight_norm_order": None,
            "gamma": 1.0,
            "temperature": 0.0,
            "gradient_scaling": "none",
            "indexed": {"seed": 11},
        },
        "reactivation": {
            "enabled": True,
            "type": "param_tanh",
            "init_policy": "analytical",
            "init_m": 1.5,
            "init_b": 0.5,
            "gradient_scaling": "none",
        },
        "blocklinear": {"gradient_scaling": "none", "efficient": True},
        "implementation": {
            "adaptive_initialization": False,
            "print_hooks": False,
        },
    }


def test_factory_exposes_dendritic_alexnet():
    assert "dendritic_alexnet" in get_available_architectures()


def test_dendritic_alexnet_preserves_channel_and_class_contracts():
    model = get_architecture("dendritic_alexnet", _tiny_config(), input_dim=3)

    output = model(torch.randn(2, 3, 64, 64))

    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()
    assert model.architecture_contract() == {
        "input_layout": "BCHW",
        "input_transform": "signed_split",
        "input_channels": 3,
        "adapted_input_channels": 6,
        "spatial_channel_mapping": "one_excitatory_soma_per_output_channel",
        "channel_widths": [2, 3, 4, 4, 4],
        "class_mapping": "one_excitatory_soma_per_class",
        "num_classes": 3,
        "propagated_population": "excitatory",
        "inter_block_output_adapter": "identity",
        "population_reduction": "one_to_one",
        "spatial_somas_per_channel": [1, 1, 1, 1, 1],
        "classifier_somas_per_unit": [1, 1],
        "component_seeds": [11, 12, 13, 14, 15, 16, 17, 18],
        "topology_seeds": [11, 12, 13, 14, 15, 16, 17, 18],
        "structured_seeds": [11, 12, 13, 14, 15, 16, 17, 18],
    }


def test_dendritic_alexnet_assigns_distinct_component_seeds():
    model = get_architecture("dendritic_alexnet", _tiny_config(), input_dim=3)

    assert model.component_seeds == (11, 12, 13, 14, 15, 16, 17, 18)


def test_dendritic_alexnet_can_learn_positive_per_channel_output_scales():
    config = _tiny_config()
    config["vision"]["output_adapter"] = {
        "mode": "learned_per_channel",
        "spatial_initial_scale": 2.0,
        "classifier_initial_scale": [3.0, 4.0],
    }
    model = get_architecture("dendritic_alexnet", config, input_dim=3)

    assert model.architecture_contract()["inter_block_output_adapter"] == (
        "learned_per_channel"
    )
    assert len(model.spatial_output_adapters) == 5
    assert len(model.feedforward_output_adapters) == 3
    torch.testing.assert_close(
        model.spatial_output_adapters[0].log_scale.exp(),
        torch.full((2,), 2.0),
    )
    torch.testing.assert_close(
        model.feedforward_output_adapters[0].log_scale.exp(),
        torch.full((5,), 3.0),
    )
    assert torch.isfinite(model(torch.randn(2, 3, 64, 64))).all()


def test_dendritic_alexnet_can_apply_featurewise_threshold_relu_adapters():
    config = _tiny_config()
    config["vision"]["output_adapter"] = {
        "mode": "learned_per_channel_threshold_relu",
        "spatial_initial_scale": 1.0,
        "spatial_initial_threshold": 0.25,
        "classifier_initial_scale": 1.0,
        "classifier_initial_threshold": 0.25,
    }
    model = get_architecture("dendritic_alexnet", config, input_dim=3)

    assert model.architecture_contract()["inter_block_output_adapter"] == (
        "learned_per_channel_threshold_relu"
    )
    assert all(
        adapter.log_threshold.numel() == width
        for adapter, width in zip(
            model.spatial_output_adapters,
            model.channel_widths,
            strict=True,
        )
    )
    assert torch.isfinite(model(torch.randn(2, 3, 64, 64))).all()


def test_dendritic_alexnet_supports_multiple_somas_per_output_feature():
    config = _tiny_config()
    config["vision"]["output_organization"] = {
        "mode": "learned_positive",
        "spatial_somas_per_channel": [2, 2, 2, 2, 2],
        "classifier_somas_per_unit": [2, 2],
    }

    model = get_architecture("dendritic_alexnet", config, input_dim=3)
    output = model(torch.randn(2, 3, 64, 64))

    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()
    contract = model.architecture_contract()
    assert contract["spatial_channel_mapping"] == (
        "fixed_excitatory_soma_group_per_output_channel"
    )
    assert contract["population_reduction"] == "learned_positive"
    assert contract["spatial_somas_per_channel"] == [2, 2, 2, 2, 2]
    assert contract["classifier_somas_per_unit"] == [2, 2]
    assert all(
        reducer.logits is not None for reducer in model.spatial_population_reducers
    )
    assert model.feedforward_population_reducers[-1].logits is None


def test_block_topology_follows_the_topology_seed_and_unknown_vision_keys_raise():
    import pytest

    from dendritic_modeling.config.keys import UnknownConfigKeyError

    config = _tiny_config()
    config["sparsity"].setdefault("indexed", {})["seed"] = 100
    model = get_architecture("dendritic_alexnet", config)
    assert model.topology_seeds == tuple(range(100, 108))
    assert model.component_seeds != model.topology_seeds

    config = _tiny_config()
    config["vision"]["chanels"] = [4, 4, 4, 4, 4]
    with pytest.raises(UnknownConfigKeyError, match="channels"):
        get_architecture("dendritic_alexnet", config)
