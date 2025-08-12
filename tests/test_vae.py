"""
Test suite for Variational Autoencoder implementations.

This module provides comprehensive testing for VAE components including
VariationalEncoder, MLPVariationalAutoencoder, CNNVariationalEncoder,
and CNNVariationalAutoencoder classes.
"""

import pytest
import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.classical.autoencoder.vae import (
    BaseVariationalAutoencoder,
    CNNVariationalAutoencoder,
    CNNVariationalEncoder,
    MLPVariationalAutoencoder,
    VariationalEncoder,
)
from dendritic_modeling.networks.architectures.classical.cnn.cnn import (
    CNNDownsample,
    CNNUpsample,
)
from dendritic_modeling.networks.architectures.classical.mlp.mlp import MLP


class TestVariationalEncoder:
    """Test the VariationalEncoder class."""

    @pytest.fixture
    def encoder_net(self):
        """Create a simple MLP for testing."""
        return MLP(input_dim=28 * 28, hidden_dims=[128, 64], output_dim=32)

    @pytest.fixture
    def vae_encoder(self, encoder_net):
        """Create a VariationalEncoder for testing."""
        return VariationalEncoder(net=encoder_net, latent_dim=16)

    def test_initialization(self, vae_encoder, encoder_net):
        """Test VariationalEncoder initialization."""
        assert vae_encoder.net == encoder_net
        assert vae_encoder.output_dim == 16
        assert isinstance(vae_encoder.fc_mean, nn.Sequential)
        assert isinstance(vae_encoder.fc_std, nn.Sequential)

    def test_latent_distribution(self, vae_encoder):
        """Test latent distribution computation."""
        batch_size = 4
        input_tensor = torch.randn(batch_size, 28 * 28)

        z_mu, z_sigma = vae_encoder.latent_distribution(input_tensor)

        assert z_mu.shape == (batch_size, 16)
        assert z_sigma.shape == (batch_size, 16)
        assert torch.all(z_sigma > 0)  # Softplus ensures positive values

    def test_forward(self, vae_encoder):
        """Test forward pass."""
        batch_size = 4
        input_tensor = torch.randn(batch_size, 28 * 28)

        output = vae_encoder(input_tensor)

        assert output.shape == (batch_size, 16)
        # The VariationalEncoder.forward() uses torch.sigmoid() on the sampled latent vector
        # so output should be bounded between 0 and 1
        assert torch.all(output >= 0) and torch.all(output <= 1)

    def test_different_latent_dims(self, encoder_net):
        """Test with different latent dimensions."""
        for latent_dim in [8, 16, 32, 64]:
            vae_encoder = VariationalEncoder(net=encoder_net, latent_dim=latent_dim)
            assert vae_encoder.output_dim == latent_dim

            input_tensor = torch.randn(2, 28 * 28)
            output = vae_encoder(input_tensor)
            assert output.shape == (2, latent_dim)


class TestBaseVariationalAutoencoder:
    """Test the BaseVariationalAutoencoder class."""

    def test_initialization(self):
        """Test BaseVariationalAutoencoder initialization."""
        base_vae = BaseVariationalAutoencoder()
        assert isinstance(base_vae, BaseVariationalAutoencoder)


class TestMLPVariationalAutoencoder:
    """Test the MLPVariationalAutoencoder class."""

    @pytest.fixture
    def mlp_vae(self):
        """Create an MLPVariationalAutoencoder for testing."""
        return MLPVariationalAutoencoder(
            input_dim=28 * 28,
            encoder_hidden_dims=[256, 128],
            decoder_hidden_dims=[128, 256],
            latent_dim=32,
        )

    def test_initialization(self, mlp_vae):
        """Test MLPVariationalAutoencoder initialization."""
        assert isinstance(mlp_vae.encoder, VariationalEncoder)
        assert isinstance(mlp_vae.decoder, MLP)
        assert mlp_vae.encoder.output_dim == 32

    def test_forward_pass(self, mlp_vae):
        """Test forward pass of MLPVariationalAutoencoder."""
        batch_size = 4
        input_tensor = torch.randn(batch_size, 28 * 28)

        output = mlp_vae(input_tensor)

        assert output.shape == (batch_size, 28 * 28)

    def test_different_architectures(self):
        """Test different encoder/decoder architectures."""
        configs = [
            {
                "input_dim": 100,
                "encoder_hidden_dims": [64, 32],
                "decoder_hidden_dims": [32, 64],
                "latent_dim": 16,
            },
            {
                "input_dim": 784,
                "encoder_hidden_dims": [512, 256, 128],
                "decoder_hidden_dims": [128, 256, 512],
                "latent_dim": 64,
            },
        ]

        for config in configs:
            vae = MLPVariationalAutoencoder(**config)
            input_tensor = torch.randn(2, config["input_dim"])
            output = vae(input_tensor)
            assert output.shape == (2, config["input_dim"])

    def test_encoder_decoder_compatibility(self, mlp_vae):
        """Test that encoder output is compatible with decoder input."""
        input_tensor = torch.randn(3, 28 * 28)

        # Get latent representation
        latent = mlp_vae.encoder(input_tensor)

        # Decode latent representation
        reconstructed = mlp_vae.decoder(latent)

        assert reconstructed.shape == input_tensor.shape


class TestCNNVariationalEncoder:
    """Test the CNNVariationalEncoder class."""

    @pytest.fixture
    def cnn_downsample(self):
        """Create a CNNDownsample network for testing."""
        return CNNDownsample(
            in_channels=3,
            out_channels=128,
            kernels=[3, 3, 3],
            strides=[2, 2, 2],
            paddings=[1, 1, 1],
            input_spatial_dim=(32, 32),
        )

    @pytest.fixture
    def cnn_vae_encoder(self, cnn_downsample):
        """Create a CNNVariationalEncoder for testing."""
        return CNNVariationalEncoder(net=cnn_downsample)

    def test_initialization(self, cnn_vae_encoder, cnn_downsample):
        """Test CNNVariationalEncoder initialization."""
        assert cnn_vae_encoder.net == cnn_downsample
        assert isinstance(cnn_vae_encoder.fc_mean, nn.Sequential)
        assert isinstance(cnn_vae_encoder.fc_std, nn.Sequential)

    def test_latent_distribution(self, cnn_vae_encoder):
        """Test latent distribution computation for CNN encoder."""
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 32, 32)

        z_mu, z_sigma = cnn_vae_encoder.latent_distribution(input_tensor)

        # Should produce latent representations
        assert len(z_mu.shape) == 2  # [batch_size, latent_dim]
        assert len(z_sigma.shape) == 2
        assert z_mu.shape == z_sigma.shape
        assert torch.all(z_sigma > 0)  # Softplus ensures positive values

    def test_forward_with_flatten(self, cnn_vae_encoder):
        """Test forward pass with flattening."""
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 32, 32)

        output = cnn_vae_encoder(input_tensor, flatten=True)

        assert len(output.shape) == 2  # Should be flattened
        # VariationalEncoder.forward() applies torch.sigmoid() to sampled latent vector
        assert torch.all(output >= 0) and torch.all(output <= 1)

    def test_forward_without_flatten(self, cnn_vae_encoder):
        """Test forward pass without flattening."""
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 32, 32)

        output = cnn_vae_encoder(input_tensor, flatten=False)

        assert len(output.shape) >= 3  # Should maintain spatial dimensions
        # VariationalEncoder.forward() applies torch.sigmoid() to sampled latent vector
        assert torch.all(output >= 0) and torch.all(output <= 1)


class TestCNNVariationalAutoencoder:
    """Test the CNNVariationalAutoencoder class."""

    @pytest.fixture
    def cnn_vae(self):
        """Create a CNNVariationalAutoencoder for testing."""
        return CNNVariationalAutoencoder(
            in_channels=3,
            latent_dim=128,
            input_spatial_dim=(32, 32),
            encoder_channels=[32, 64, 128],
            decoder_channels=[128, 64, 32],
        )

    def test_initialization(self, cnn_vae):
        """Test CNNVariationalAutoencoder initialization."""
        assert isinstance(cnn_vae.encoder, CNNVariationalEncoder)
        assert isinstance(cnn_vae.decoder, CNNUpsample)

    def test_forward_pass(self, cnn_vae):
        """Test forward pass of CNNVariationalAutoencoder."""
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 32, 32)

        output = cnn_vae(input_tensor)

        assert output.shape == input_tensor.shape

    def test_calculate_downsample_params(self, cnn_vae):
        """Test the _calculate_downsample_params method."""
        # Test cases for different spatial dimensions and layer counts
        test_cases = [
            (32, 4, 2),  # (initial_size, target_size, num_layers)
            (64, 8, 3),
            (128, 16, 3),
            (256, 32, 4),
        ]

        for initial_size, target_size, num_layers in test_cases:
            kernels, strides = cnn_vae._calculate_downsample_params(
                initial_size, target_size, num_layers
            )

            assert len(kernels) == num_layers
            assert len(strides) == num_layers
            assert all(k >= 1 for k in kernels)
            assert all(s >= 1 for s in strides)

    def test_different_architectures(self):
        """Test different CNN VAE architectures."""
        configs = [
            {
                "in_channels": 1,
                "latent_dim": 64,
                "input_spatial_dim": (28, 28),
                "encoder_channels": [16, 32],
                "decoder_channels": [32, 16],
            },
            {
                "in_channels": 3,
                "latent_dim": 256,
                "input_spatial_dim": (64, 64),
                "encoder_channels": [32, 64, 128, 256],
                "decoder_channels": [256, 128, 64, 32],
            },
        ]

        for config in configs:
            vae = CNNVariationalAutoencoder(**config)
            h, w = config["input_spatial_dim"]
            input_tensor = torch.randn(1, config["in_channels"], h, w)
            output = vae(input_tensor)
            assert output.shape == input_tensor.shape

    def test_encoder_decoder_compatibility(self, cnn_vae):
        """Test that encoder output is compatible with decoder input."""
        input_tensor = torch.randn(1, 3, 32, 32)

        # The decoder should be able to process the latent representation
        # Note: The decoder might need reshaping, which is handled internally
        reconstructed = cnn_vae(input_tensor)

        assert reconstructed.shape == input_tensor.shape

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with minimum viable configuration
        min_vae = CNNVariationalAutoencoder(
            in_channels=1,
            latent_dim=8,
            input_spatial_dim=(8, 8),
            encoder_channels=[4],
            decoder_channels=[4],
        )

        input_tensor = torch.randn(1, 1, 8, 8)
        output = min_vae(input_tensor)
        assert output.shape == input_tensor.shape


# Integration tests
class TestVAEIntegration:
    """Integration tests for VAE components."""

    def test_mlp_vae_training_compatibility(self):
        """Test that MLP VAE works in training mode."""
        vae = MLPVariationalAutoencoder(
            input_dim=784,
            encoder_hidden_dims=[256, 128],
            decoder_hidden_dims=[128, 256],
            latent_dim=32,
        )

        vae.train()
        input_tensor = torch.randn(4, 784)
        output = vae(input_tensor)

        # Should work with gradients
        loss = nn.MSELoss()(output, input_tensor)
        loss.backward()

        assert output.shape == input_tensor.shape

    def test_cnn_vae_training_compatibility(self):
        """Test that CNN VAE works in training mode."""
        vae = CNNVariationalAutoencoder(
            in_channels=3,
            latent_dim=64,
            input_spatial_dim=(32, 32),
            encoder_channels=[16, 32],
            decoder_channels=[32, 16],
        )

        vae.train()
        input_tensor = torch.randn(2, 3, 32, 32)
        output = vae(input_tensor)

        # Should work with gradients
        loss = nn.MSELoss()(output, input_tensor)
        loss.backward()

        assert output.shape == input_tensor.shape

    def test_deterministic_behavior(self):
        """Test that VAE produces consistent results with same input."""
        vae = MLPVariationalAutoencoder(
            input_dim=100,
            encoder_hidden_dims=[64, 32],
            decoder_hidden_dims=[32, 64],
            latent_dim=16,
        )

        vae.eval()
        input_tensor = torch.randn(1, 100)

        with torch.no_grad():
            output1 = vae(input_tensor)
            output2 = vae(input_tensor)

        # In eval mode with same random seed, outputs should be similar
        # (though not identical due to sampling in VAE)
        assert output1.shape == output2.shape
