"""Tests for DendriteConv2d and DendriteAttention layers."""

import pytest
import torch
import torch.nn.functional as functional
from torch import nn

from dendritic_modeling.networks.convert.cnn_to_dendrite import conv2d_to_dendrite
from dendritic_modeling.networks.layers.dendritic_conv2d import (
    DendriteAttention,
    DendriteConv2d,
)

# ---------------------------------------------------------------------------
# DendriteConv2d
# ---------------------------------------------------------------------------


class TestDendriteConv2d:
    def _make_layer(self, **kwargs):
        defaults = {"in_channels": 3, "out_channels": 8, "kernel_size": 3, "padding": 1}
        defaults.update(kwargs)
        return DendriteConv2d(**defaults)

    def test_forward_output_shape(self):
        layer = self._make_layer()
        x = torch.randn(2, 3, 16, 16)
        out = layer(x)
        assert out.shape == (2, 8, 16, 16)

    def test_forward_with_stride(self):
        layer = self._make_layer(stride=2)
        x = torch.randn(2, 3, 16, 16)
        out = layer(x)
        assert out.shape == (2, 8, 8, 8)

    def test_forward_with_padding(self):
        layer = self._make_layer(kernel_size=5, padding=2)
        x = torch.randn(2, 3, 12, 12)
        out = layer(x)
        assert out.shape == (2, 8, 12, 12)

    def test_weights_positive(self):
        layer = self._make_layer()
        assert (layer.weight_e > 0).all()
        assert (layer.weight_i > 0).all()

    def test_output_bounded(self):
        layer = self._make_layer()
        # Use strictly positive input so E and I are positive
        x = torch.rand(4, 3, 8, 8) + 0.1
        out = layer(x)
        assert (out >= 0).all()
        assert (out < 1).all()

    def test_divisive_normalization(self):
        layer = self._make_layer()
        x = torch.randn(1, 3, 8, 8)

        # Manually compute E, I, and v_out
        E = functional.conv2d(
            x,
            layer.weight_e,
            layer.bias_E,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
        )
        inh = functional.conv2d(
            x,
            layer.weight_i,
            layer.bias_I,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
        )
        expected = E / (E + inh + layer.epsilon)

        out = layer(x)
        torch.testing.assert_close(out, expected)

    def test_forward_without_v_in(self):
        layer = self._make_layer()
        x = torch.randn(2, 3, 8, 8)
        out = layer(x, v_in=None)
        assert out.shape == (2, 8, 8, 8)

    def test_forward_with_v_in_global(self):
        layer = self._make_layer(v_in_channels=8)
        x = torch.randn(2, 3, 8, 8)
        v_in = torch.randn(2, 8)
        out = layer(x, v_in=v_in)
        assert out.shape == (2, 8, 8, 8)

    def test_forward_with_v_in_spatial(self):
        layer = self._make_layer(v_in_channels=8)
        x = torch.randn(2, 3, 8, 8)
        v_in = torch.randn(2, 8, 8, 8)
        out = layer(x, v_in=v_in)
        assert out.shape == (2, 8, 8, 8)

    def test_no_bias(self):
        layer = self._make_layer(bias=False)
        assert layer.bias_E is None
        assert layer.bias_I is None

    def test_gradient_flow(self):
        layer = self._make_layer()
        x = torch.randn(2, 3, 8, 8)
        out = layer(x)
        out.sum().backward()
        assert layer.weight_E_raw.grad is not None
        assert layer.weight_I_raw.grad is not None

    def test_init_softplus_normal(self):
        layer = self._make_layer(init_method="softplus_normal")
        assert torch.isfinite(layer.weight_E_raw).all()
        assert torch.isfinite(layer.weight_I_raw).all()

    def test_init_kaiming(self):
        layer = self._make_layer(init_method="kaiming")
        assert torch.isfinite(layer.weight_E_raw).all()
        assert torch.isfinite(layer.weight_I_raw).all()

    def test_extra_repr(self):
        layer = self._make_layer()
        s = layer.extra_repr()
        assert "3" in s  # in_channels
        assert "8" in s  # out_channels
        assert "epsilon" in s

    def test_morphology_synapse_counts(self):
        layer = self._make_layer(
            n_e_dendrites=3,
            n_i_dendrites=2,
            e_synapses_per_dendrite=5,
            i_synapses_per_dendrite=7,
            morphology_seed=123,
        )
        assert layer.weight_E_raw.shape[:2] == (8, 3)
        assert layer.weight_I_raw.shape[:2] == (8, 2)

        e_counts = layer.weight_E_mask.view(8, 3, -1).sum(dim=-1)
        i_counts = layer.weight_I_mask.view(8, 2, -1).sum(dim=-1)
        assert torch.all(e_counts == 5)
        assert torch.all(i_counts == 7)

    @pytest.mark.parametrize("aggregation", ["sum", "mean", "max"])
    def test_forward_branch_aggregation_modes(self, aggregation):
        layer = self._make_layer(
            n_e_dendrites=3,
            n_i_dendrites=3,
            e_synapses_per_dendrite=7,
            i_synapses_per_dendrite=7,
            branch_aggregation=aggregation,
        )
        x = torch.randn(2, 3, 16, 16)
        out = layer(x)
        assert out.shape == (2, 8, 16, 16)

    def test_invalid_synapse_count_raises(self):
        # For 3x3 kernels with 3 input channels, max synapses per dendrite is 27.
        with pytest.raises(ValueError):
            self._make_layer(e_synapses_per_dendrite=28)


class TestConv2dToDendrite:
    def test_generalized_conversion_shape_and_masks(self):
        conv = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=True)
        layer = conv2d_to_dendrite(
            conv,
            copy_bias=True,
            n_e_dendrites=2,
            n_i_dendrites=4,
            e_synapses_per_dendrite=6,
            i_synapses_per_dendrite=9,
            branch_aggregation="sum",
            morphology_seed=0,
        )
        x = torch.randn(2, 3, 16, 16)
        out = layer(x)
        assert out.shape == (2, 8, 8, 8)

        e_counts = layer.weight_E_mask.view(8, 2, -1).sum(dim=-1)
        i_counts = layer.weight_I_mask.view(8, 4, -1).sum(dim=-1)
        assert torch.all(e_counts == 6)
        assert torch.all(i_counts == 9)

    @pytest.mark.parametrize("dendrites", [1, 3])
    def test_conversion_reproduces_the_kernel(self, dendrites):
        torch.manual_seed(0)
        conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        layer = conv2d_to_dendrite(
            conv,
            n_e_dendrites=dendrites,
            n_i_dendrites=dendrites,
            branch_aggregation="sum",
            baseline_inhib=0.0,
        )
        weight = conv.weight.detach()
        assert torch.allclose(layer.weight_e, weight.clamp_min(0), atol=1e-5)
        assert torch.allclose(layer.weight_i, (-weight).clamp_min(0), atol=1e-5)

    def test_unknown_init_method_raises(self):
        with pytest.raises(ValueError, match="init_method must be one of"):
            DendriteConv2d(1, 1, 3, init_method="xavier")


# ---------------------------------------------------------------------------
# DendriteAttention
# ---------------------------------------------------------------------------


class TestDendriteAttention:
    def _make_attn(self, embed_dim=16, num_heads=4, **kwargs):
        return DendriteAttention(embed_dim=embed_dim, num_heads=num_heads, **kwargs)

    def test_output_shape(self):
        attn = self._make_attn()
        q = k = v = torch.randn(2, 5, 16)
        out, _weights = attn(q, k, v)
        assert out.shape == (2, 5, 16)

    def test_weights_shape(self):
        attn = self._make_attn(embed_dim=16, num_heads=4)
        q = k = v = torch.randn(2, 5, 16)
        _, weights = attn(q, k, v)
        assert weights.shape == (2, 4, 5, 5)

    def test_with_mask(self):
        attn = self._make_attn()
        q = k = v = torch.randn(1, 4, 16)
        # Causal mask: lower triangle
        mask = torch.tril(torch.ones(4, 4)).unsqueeze(0)
        out, _weights = attn(q, k, v, attn_mask=mask)
        assert out.shape == (1, 4, 16)

    def test_gradient_flow(self):
        attn = self._make_attn()
        q = k = v = torch.randn(2, 3, 16)
        out, _ = attn(q, k, v)
        out.sum().backward()
        assert attn.q_proj.weight.grad is not None
        assert attn.k_proj.weight.grad is not None
        assert attn.v_proj.weight.grad is not None
        assert attn.out_proj.weight.grad is not None
        assert attn.i_proj.weight.grad is not None

    def test_embed_dim_divisibility(self):
        with pytest.raises(AssertionError):
            DendriteAttention(embed_dim=15, num_heads=4)
