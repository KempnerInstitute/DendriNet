"""Tests for CNN downsample/upsample architectures."""

import pytest
import torch

from dendritic_modeling.networks.architectures.classical.cnn.cnn import (
    CNNDownsample,
    CNNUpsample,
)

# ---------------------------------------------------------------------------
# CNNDownsample
# ---------------------------------------------------------------------------


class TestCNNDownsample:
    def test_output_shape_with_output_dim(self):
        net = CNNDownsample(
            input_shape=(1, 28, 28),
            hidden_dims=[16, 32],
            kernel_sizes=[3, 3],
            strides=[2, 2],
            output_dim=64,
        )
        x = torch.randn(4, 1, 28, 28)
        out = net(x)
        assert out.shape == (4, 64)

    def test_output_shape_without_output_dim(self):
        net = CNNDownsample(
            input_shape=(1, 28, 28),
            hidden_dims=[16, 32],
            kernel_sizes=[3, 3],
            strides=[2, 2],
            output_dim=None,
        )
        x = torch.randn(2, 1, 28, 28)
        out = net(x)
        # Without output_dim, returns spatial output
        assert out.dim() == 4  # (B, C, H, W)
        assert out.shape[1] == 32  # last hidden_dim

    def test_3d_input_auto_batch(self):
        net = CNNDownsample(
            input_shape=(1, 16, 16),
            hidden_dims=[8],
            kernel_sizes=[3],
            strides=[1],
            output_dim=10,
        )
        x = torch.randn(1, 16, 16)  # no batch dim → code adds batch dim
        out = net(x)
        assert out.shape == (1, 10)  # batch=1 preserved

    def test_gradient_flow(self):
        net = CNNDownsample(
            input_shape=(1, 16, 16),
            hidden_dims=[8],
            kernel_sizes=[3],
            strides=[1],
            output_dim=10,
        )
        x = torch.randn(2, 1, 16, 16)
        out = net(x)
        out.sum().backward()
        for p in net.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_different_activations(self):
        for act in ("relu", "sigmoid", "tanh"):
            net = CNNDownsample(
                input_shape=(1, 16, 16),
                hidden_dims=[8],
                kernel_sizes=[3],
                strides=[1],
                output_dim=10,
                activation=act,
            )
            x = torch.randn(2, 1, 16, 16)
            out = net(x)
            assert out.shape == (2, 10)


# ---------------------------------------------------------------------------
# CNNUpsample
# ---------------------------------------------------------------------------


class TestCNNUpsample:
    def test_output_shape_exact_target(self):
        net = CNNUpsample(
            input_shape=(16, 4, 4),
            target_shape=(1, 28, 28),
            hidden_dims=[8],
        )
        x = torch.randn(2, 16, 4, 4)
        out = net(x)
        assert out.shape == (2, 1, 28, 28)

    def test_same_size(self):
        net = CNNUpsample(
            input_shape=(8, 4, 4),
            target_shape=(1, 4, 4),
            hidden_dims=[4],
        )
        x = torch.randn(2, 8, 4, 4)
        out = net(x)
        assert out.shape == (2, 1, 4, 4)

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square spatial"):
            CNNUpsample(
                input_shape=(8, 4, 6),
                target_shape=(1, 28, 28),
                hidden_dims=[4],
            )

    def test_input_larger_than_target_raises(self):
        with pytest.raises(ValueError):
            CNNUpsample(
                input_shape=(8, 16, 16),
                target_shape=(1, 4, 4),
                hidden_dims=[4],
            )

    def test_gradient_flow(self):
        net = CNNUpsample(
            input_shape=(8, 4, 4),
            target_shape=(1, 16, 16),
            hidden_dims=[4],
        )
        x = torch.randn(2, 8, 4, 4)
        out = net(x)
        out.sum().backward()
        for p in net.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_3d_input_auto_batch(self):
        net = CNNUpsample(
            input_shape=(8, 4, 4),
            target_shape=(1, 16, 16),
            hidden_dims=[4],
        )
        x = torch.randn(8, 4, 4)  # no batch dim → code adds batch dim
        out = net(x)
        assert out.shape == (1, 1, 16, 16)  # batch=1 preserved


# ---------------------------------------------------------------------------
# _calculate_transpose_params
# ---------------------------------------------------------------------------


class TestCalculateTransposeParams:
    def _calc(self, input_size, target_size, num_layers):
        """Helper to call the private method."""
        net = CNNUpsample.__new__(CNNUpsample)
        return net._calculate_transpose_params(input_size, target_size, num_layers)

    def test_identity(self):
        kernels, strides, paddings = self._calc(4, 4, 2)
        assert kernels == [1, 1]
        assert strides == [1, 1]
        assert paddings == [0, 0]

    def test_simple_upscale(self):
        kernels, strides, paddings = self._calc(4, 8, 1)
        # Verify the params produce exactly 8
        size = 4
        for k, s, p in zip(kernels, strides, paddings):
            size = (size - 1) * s - 2 * p + k
        assert size == 8

    def test_multi_layer(self):
        kernels, strides, paddings = self._calc(2, 8, 2)
        size = 2
        for k, s, p in zip(kernels, strides, paddings):
            size = (size - 1) * s - 2 * p + k
        assert size == 8

    def test_realistic_vae_params(self):
        # Typical VAE: 4x4 latent → 28x28 output in 2 layers
        kernels, strides, paddings = self._calc(4, 28, 2)
        size = 4
        for k, s, p in zip(kernels, strides, paddings):
            size = (size - 1) * s - 2 * p + k
        assert size == 28

    def test_input_gt_target_raises(self):
        with pytest.raises(ValueError):
            self._calc(10, 4, 1)
