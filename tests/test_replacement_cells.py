import torch

from dendritic_modeling.networks.architectures.replacement import (
    LayerwiseTokenReplacementStack,
    NonNegativeInputAdapter,
    PopulationToFeatureReduction,
    PositiveFeatureScale,
    SpatialNonNegativeInputAdapter,
    TokenReplacementCell,
)
from dendritic_modeling.networks.architectures.transformer import (
    DendriticFFNReplacement,
    LayerwiseEIStackReplacement,
)


def test_signed_split_adapter_maps_signed_features_to_nonnegative_channels():
    adapter = NonNegativeInputAdapter(3, "signed_split")
    x = torch.tensor([[-2.0, 0.0, 3.0]])

    y = adapter(x)

    assert y.shape == (1, 6)
    assert torch.all(y >= 0)
    assert torch.allclose(y, torch.tensor([[0.0, 0.0, 3.0, 2.0, 0.0, 0.0]]))


def test_spatial_signed_split_preserves_bchw_channel_identity():
    adapter = SpatialNonNegativeInputAdapter(3, "signed_split")
    x = torch.randn(2, 3, 5, 7)

    y = adapter(x)

    assert y.shape == (2, 6, 5, 7)
    assert torch.all(y >= 0)
    torch.testing.assert_close(y[:, :3] - y[:, 3:], x)


def test_positive_feature_scale_preserves_layout_and_nonnegativity():
    adapter = PositiveFeatureScale(
        3,
        feature_axis=1,
        initial_scale=2.0,
        per_feature=True,
    )
    x = torch.rand(2, 3, 5, 7)

    y = adapter(x)

    assert y.shape == x.shape
    assert torch.all(y >= 0)
    torch.testing.assert_close(y, 2.0 * x)
    y.mean().backward()
    assert adapter.log_scale.grad is not None


def test_population_reduction_uses_feature_local_simplexes():
    adapter = PopulationToFeatureReduction(
        2,
        somas_per_feature=3,
        feature_axis=1,
        mode="learned_positive",
    )
    x = torch.tensor([[[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]], [[6.0]]]])

    y = adapter(x)

    assert y.shape == (1, 2, 1, 1)
    torch.testing.assert_close(y.flatten(), torch.tensor([2.0, 5.0]))
    y.sum().backward()
    assert adapter.logits.grad is not None


def test_fixed_population_reduction_does_not_mix_output_features():
    adapter = PopulationToFeatureReduction(
        2,
        somas_per_feature=2,
        feature_axis=-1,
        mode="fixed_mean",
    )

    y = adapter(torch.tensor([[1.0, 3.0, 10.0, 14.0]]))

    torch.testing.assert_close(y, torch.tensor([[2.0, 12.0]]))


def test_transformer_ffn_uses_shared_cell_without_state_dict_key_migration():
    replacement = DendriticFFNReplacement(
        hidden_size=4,
        dendritic_units=3,
        branch_factors=[1],
        synapses_per_branch=2,
        input_transform="signed_split",
        topk_type="indexed",
        indexed_seed=17,
        use_shunting=False,
    )

    assert isinstance(replacement, TokenReplacementCell)
    state_keys = set(replacement.state_dict())
    assert "output_projection.weight" in state_keys
    assert any(key.startswith("core.") for key in state_keys)
    assert not any(key.startswith("cell.") for key in state_keys)

    x = torch.randn(2, 5, 4)
    adapted = replacement.encode_input(x)
    y = replacement(x)

    assert adapted.shape == (2, 5, 8)
    assert torch.all(adapted >= 0)
    assert y.shape == x.shape


def test_layerwise_ei_stack_uses_shared_layerwise_cell_contract():
    stack = LayerwiseEIStackReplacement(
        hidden_size=4,
        num_layers=2,
        excitatory_cells=3,
        inhibitory_cells=2,
        excitatory_branch_factors=[1],
        inhibitory_branch_factors=[1],
        ee_synapses_per_branch=2,
        ei_synapses_per_branch=1,
        ie_synapses_per_branch=1,
        ii_synapses_per_branch=1,
        input_transform="signed_split",
        topk_type="standard",
        use_shunting=False,
    )

    assert isinstance(stack, LayerwiseTokenReplacementStack)
    x = torch.randn(2, 5, 4)
    adapted = stack.encode_layer_input(1, x)
    y = stack.forward_layer(1, x)

    assert adapted.shape == (2, 5, 8)
    assert torch.all(adapted >= 0)
    assert y.shape == x.shape
