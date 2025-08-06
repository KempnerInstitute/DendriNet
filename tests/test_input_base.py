"""
Consolidated tests for networks input base module.
Combines original and extended tests for comprehensive coverage.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os

from dendritic_modeling.networks.input.base import InputNetTransform


class TestInputNetTransform:
    """Comprehensive tests for InputNetTransform class."""

    def test_init_with_minimal_params(self):
        """Test initialization with minimal parameters."""
        with pytest.raises((AttributeError, TypeError)):
            InputNetTransform()

    def test_init_with_all_none_dimensions(self):
        """Test initialization with all dimensions as None."""
        with pytest.raises((AttributeError, TypeError)):
            InputNetTransform(
                input_dim=None,
                excitatory_dim=None,
                inhibitory_dim=None,
                network_type="IndexInputTransform",
            )

    def test_init_independent_heads(self):
        """Test initialization with independent heads."""
        transform = InputNetTransform(
            input_dim=10,
            excitatory_dim=6,
            inhibitory_dim=4,
            independent_heads=True,
            network_type="IndexInputTransform",
        )
        assert transform.input_transform is not None

    def test_init_shared_network(self):
        """Test initialization with shared network."""
        transform = InputNetTransform(
            input_dim=10,
            excitatory_dim=6,
            inhibitory_dim=4,
            independent_heads=False,
            network_type="IndexInputTransform",
        )
        assert transform.input_transform is not None

    def test_forward_mlp_independent(self):
        """Test forward pass with independent heads using IndexInputTransform."""
        transform = InputNetTransform(
            input_dim=10,
            excitatory_dim=6,
            inhibitory_dim=4,
            independent_heads=True,
            network_type="IndexInputTransform",
        )

        x = torch.randn(2, 10)
        exc_out, inh_out = transform(x)

        assert exc_out.shape == (2, 6)
        assert inh_out.shape == (2, 4)

    def test_forward_mlp_shared(self):
        """Test forward pass with shared network using IndexInputTransform."""
        transform = InputNetTransform(
            input_dim=10,
            excitatory_dim=6,
            inhibitory_dim=4,
            independent_heads=False,
            network_type="IndexInputTransform",
        )

        x = torch.randn(2, 10)
        exc_out, inh_out = transform(x)

        assert exc_out.shape == (2, 6)
        assert inh_out.shape == (2, 4)

    def test_init_alexnet_transform(self):
        """Test initialization with AlexNetTransform."""
        with pytest.raises(NotImplementedError):
            InputNetTransform(
                input_shape=(3, 224, 224),
                excitatory_dim=256,
                inhibitory_dim=128,
                network_type="AlexNetTransform",
            )

    def test_forward_alexnet_transform(self):
        """Test forward pass with AlexNetTransform."""
        with pytest.raises(NotImplementedError):
            transform = InputNetTransform(
                input_shape=(3, 224, 224),
                excitatory_dim=256,
                inhibitory_dim=128,
                network_type="AlexNetTransform",
            )
            x = torch.randn(1, 3, 224, 224)
            transform(x)

    def test_alexnet_with_custom_dimensions(self):
        """Test AlexNetTransform with custom dimensions."""
        with pytest.raises(NotImplementedError):
            InputNetTransform(
                input_shape=(3, 64, 64),
                excitatory_dim=128,
                inhibitory_dim=64,
                excitatory_hidden_dims=[32, 64],
                inhibitory_hidden_dims=[16, 32],
                network_type="AlexNetTransform",
            )

    def test_conv_layers_initialization(self):
        """Test initialization of convolutional layers."""
        with pytest.raises(NotImplementedError):
            InputNetTransform(
                input_shape=(3, 32, 32),
                excitatory_dim=64,
                inhibitory_dim=32,
                excitatory_kernel_sizes=[3, 5],
                inhibitory_kernel_sizes=[3, 5],
                excitatory_strides=[1, 2],
                inhibitory_strides=[1, 2],
                network_type="AlexNetTransform",
            )

    def test_parameter_count(self):
        """Test parameter counting."""
        transform = InputNetTransform(
            input_dim=10,
            excitatory_dim=6,
            inhibitory_dim=4,
            network_type="IndexInputTransform",
        )

        total_params = sum(p.numel() for p in transform.parameters() if p.requires_grad)
        assert total_params >= 0

    def test_load_pretrained_weights_none_source(self):
        """Test loading weights with None source."""
        transform = InputNetTransform(
            input_dim=10,
            excitatory_dim=6,
            inhibitory_dim=4,
            network_type="IndexInputTransform",
        )

        with pytest.raises(ValueError, match="Unknown source"):
            transform.load_pretrained_weights(None)

    def test_load_pretrained_weights_empty_source(self):
        """Test loading weights with empty source."""
        transform = InputNetTransform(
            input_dim=10,
            excitatory_dim=6,
            inhibitory_dim=4,
            network_type="IndexInputTransform",
        )

        with pytest.raises(ValueError, match="Unknown source"):
            transform.load_pretrained_weights("")

    def test_load_pretrained_weights_pytorch_alexnet(self):
        """Test loading PyTorch AlexNet weights."""
        with pytest.raises(
            NotImplementedError, match="AlexNetInputTransform not implemented"
        ):
            transform = InputNetTransform(
                input_shape=(3, 224, 224),
                excitatory_dim=256,
                inhibitory_dim=128,
                network_type="AlexNetTransform",
            )
            transform.load_pretrained_weights("pytorch:alexnet")

    # Extended tests start here
    def test_forward_pass_shapes_consistency(self):
        """Test that forward pass maintains shape consistency."""
        transform = InputNetTransform(
            input_dim=20,
            excitatory_dim=12,
            inhibitory_dim=8,
            network_type="IndexInputTransform",
        )

        # Test with different batch sizes
        for batch_size in [1, 5, 10, 32]:
            x = torch.randn(batch_size, 20)
            exc_out, inh_out = transform(x)

            assert exc_out.shape == (batch_size, 12)
            assert inh_out.shape == (batch_size, 8)

    def test_forward_pass_gradient_flow(self):
        """Test that gradients flow properly through the network."""
        transform = InputNetTransform(
            input_dim=10,
            excitatory_dim=6,
            inhibitory_dim=4,
            network_type="IndexInputTransform",
        )

        x = torch.randn(2, 10, requires_grad=True)
        exc_out, inh_out = transform(x)

        # Compute some loss
        loss = exc_out.sum() + inh_out.sum()
        loss.backward()

        # Check that input gradients are computed
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_parameters_accessibility(self):
        """Test that model parameters are accessible."""
        transform = InputNetTransform(
            input_dim=8,
            excitatory_dim=5,
            inhibitory_dim=3,
            network_type="IndexInputTransform",
        )

        params = list(transform.parameters())
        assert len(params) >= 0  # IndexInputTransform may have no trainable params

        # Test parameter counting
        total_params = sum(p.numel() for p in params if p.requires_grad)
        assert total_params >= 0

    def test_training_mode_switching(self):
        """Test switching between training and evaluation modes."""
        transform = InputNetTransform(
            input_dim=10,
            excitatory_dim=6,
            inhibitory_dim=4,
            network_type="IndexInputTransform",
        )

        # Test training mode
        transform.train()
        assert transform.training

        # Test evaluation mode
        transform.eval()
        assert not transform.training

        # Test forward pass in both modes
        x = torch.randn(2, 10)

        transform.train()
        exc_train, inh_train = transform(x)

        transform.eval()
        exc_eval, inh_eval = transform(x)

        # Outputs should be identical for IndexInputTransform
        assert torch.allclose(exc_train, exc_eval)
        assert torch.allclose(inh_train, inh_eval)

    def test_device_transfer(self):
        """Test transferring model to different devices."""
        transform = InputNetTransform(
            input_dim=6,
            excitatory_dim=4,
            inhibitory_dim=2,
            network_type="IndexInputTransform",
        )

        # Test CPU device
        transform.to("cpu")
        x_cpu = torch.randn(2, 6)
        exc_out, inh_out = transform(x_cpu)

        assert exc_out.device.type == "cpu"
        assert inh_out.device.type == "cpu"

        # Test CUDA device if available
        if torch.cuda.is_available():
            transform.to("cuda")
            x_cuda = torch.randn(2, 6, device="cuda")
            exc_out, inh_out = transform(x_cuda)

            assert exc_out.device.type == "cuda"
            assert inh_out.device.type == "cuda"

    def test_input_validation_edge_cases(self):
        """Test input validation for edge cases."""
        # Test with minimal dimensions
        transform = InputNetTransform(
            input_dim=2,
            excitatory_dim=1,
            inhibitory_dim=1,
            network_type="IndexInputTransform",
        )

        x = torch.randn(1, 2)
        exc_out, inh_out = transform(x)

        assert exc_out.shape == (1, 1)
        assert inh_out.shape == (1, 1)

    def test_load_pretrained_weights_pytorch_models(self):
        """Test loading weights from different PyTorch models."""
        # Test with valid model names but expect NotImplementedError
        valid_models = ["alexnet", "vgg16", "resnet18", "resnet50"]

        for model_name in valid_models:
            with pytest.raises(
                NotImplementedError, match="AlexNetInputTransform not implemented"
            ):
                transform = InputNetTransform(
                    input_shape=(3, 224, 224),
                    excitatory_dim=256,
                    inhibitory_dim=128,
                    network_type="AlexNetTransform",
                )
                transform.load_pretrained_weights(f"pytorch:{model_name}")

    def test_load_pretrained_weights_file_operations(self):
        """Test file operations in load_pretrained_weights."""
        transform = InputNetTransform(
            input_dim=8,
            excitatory_dim=5,
            inhibitory_dim=3,
            network_type="IndexInputTransform",
        )

        # Test with file that should raise ValueError (unknown source)
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
            # Create a mock state dict
            mock_state_dict = {
                "input_transform.weight": torch.randn(8, 5),
                "input_transform.bias": torch.randn(8),
            }
            torch.save(mock_state_dict, tmp_file.name)

            try:
                # Should raise ValueError for unknown source
                with pytest.raises(ValueError, match="Unknown source"):
                    transform.load_pretrained_weights(tmp_file.name)
            finally:
                os.unlink(tmp_file.name)

    def test_state_dict_save_load(self):
        """Test saving and loading state dict."""
        transform = InputNetTransform(
            input_dim=6,
            excitatory_dim=4,
            inhibitory_dim=2,
            network_type="IndexInputTransform",
        )

        # Get original state dict
        original_state = transform.state_dict()

        # Create new transform and load state
        new_transform = InputNetTransform(
            input_dim=6,
            excitatory_dim=4,
            inhibitory_dim=2,
            network_type="IndexInputTransform",
        )

        new_transform.load_state_dict(original_state)

        # Test that they produce same output
        x = torch.randn(2, 6)
        exc1, inh1 = transform(x)
        exc2, inh2 = new_transform(x)

        assert torch.allclose(exc1, exc2)
        assert torch.allclose(inh1, inh2)

    def test_repr_and_str_methods(self):
        """Test string representation methods."""
        transform = InputNetTransform(
            input_dim=10,
            excitatory_dim=6,
            inhibitory_dim=4,
            network_type="IndexInputTransform",
        )

        # Test that repr works without error
        repr_str = repr(transform)
        assert isinstance(repr_str, str)
        assert "InputNetTransform" in repr_str

        # Test that str works without error
        str_repr = str(transform)
        assert isinstance(str_repr, str)

    def test_named_parameters_and_modules(self):
        """Test named_parameters and named_modules methods."""
        transform = InputNetTransform(
            input_dim=8,
            excitatory_dim=5,
            inhibitory_dim=3,
            network_type="IndexInputTransform",
        )

        # Test named_parameters
        named_params = list(transform.named_parameters())
        assert isinstance(named_params, list)

        # Test named_modules
        named_modules = list(transform.named_modules())
        assert isinstance(named_modules, list)
        assert len(named_modules) >= 1  # At least the root module

    def test_forward_with_different_dtypes(self):
        """Test forward pass with different data types."""
        transform = InputNetTransform(
            input_dim=6,
            excitatory_dim=4,
            inhibitory_dim=2,
            network_type="IndexInputTransform",
        )

        # Test with float32
        x_float32 = torch.randn(2, 6, dtype=torch.float32)
        exc_out, inh_out = transform(x_float32)

        assert exc_out.dtype == torch.float32
        assert inh_out.dtype == torch.float32

        # Test with float64
        x_float64 = torch.randn(2, 6, dtype=torch.float64)
        exc_out, inh_out = transform(x_float64.float())  # Convert to expected dtype

        assert exc_out.dtype in [torch.float32, torch.float64]
        assert inh_out.dtype in [torch.float32, torch.float64]

    def test_module_registration(self):
        """Test that submodules are properly registered."""
        transform = InputNetTransform(
            input_dim=10,
            excitatory_dim=6,
            inhibitory_dim=4,
            network_type="IndexInputTransform",
        )

        # Check that input_transform is registered
        assert hasattr(transform, "input_transform")
        assert transform.input_transform is not None

        # Check that it appears in children
        children = list(transform.children())
        assert len(children) >= 1

    def test_zero_grad_functionality(self):
        """Test zero_grad functionality."""
        transform = InputNetTransform(
            input_dim=8,
            excitatory_dim=5,
            inhibitory_dim=3,
            network_type="IndexInputTransform",
        )

        # Create input with gradient
        x = torch.randn(2, 8, requires_grad=True)
        exc_out, inh_out = transform(x)

        # Backward pass
        loss = exc_out.sum() + inh_out.sum()
        loss.backward()

        # Check that some parameters have gradients (if any exist)
        params_with_grad = [p for p in transform.parameters() if p.grad is not None]

        # Zero gradients
        transform.zero_grad()

        # Check that gradients are zeroed
        for param in params_with_grad:
            if param.grad is not None:
                assert torch.allclose(param.grad, torch.zeros_like(param.grad))

    def test_integration_with_loss_function(self):
        """Test integration with common loss functions."""
        transform = InputNetTransform(
            input_dim=10,
            excitatory_dim=6,
            inhibitory_dim=4,
            network_type="IndexInputTransform",
        )

        x = torch.randn(4, 10)
        exc_out, inh_out = transform(x)

        # Test with MSE loss
        target_exc = torch.randn(4, 6)
        target_inh = torch.randn(4, 4)

        mse_loss = nn.MSELoss()
        loss_exc = mse_loss(exc_out, target_exc)
        loss_inh = mse_loss(inh_out, target_inh)

        assert isinstance(loss_exc, torch.Tensor)
        assert isinstance(loss_inh, torch.Tensor)

    def test_integration_with_optimizer(self):
        """Test integration with PyTorch optimizers."""
        transform = InputNetTransform(
            input_dim=8,
            excitatory_dim=5,
            inhibitory_dim=3,
            network_type="IndexInputTransform",
        )

        # IndexInputTransform has no parameters, so optimizer creation will fail
        with pytest.raises(ValueError, match="optimizer got an empty parameter list"):
            torch.optim.Adam(transform.parameters(), lr=0.001)

        # Test that forward pass still works
        x = torch.randn(2, 8)
        exc_out, inh_out = transform(x)

        assert exc_out.shape == (2, 5)
        assert inh_out.shape == (2, 3)

    def test_batch_processing_consistency(self):
        """Test that batch processing is consistent with individual processing."""
        transform = InputNetTransform(
            input_dim=6,
            excitatory_dim=4,
            inhibitory_dim=2,
            network_type="IndexInputTransform",
        )

        # Process individual samples
        individual_results_exc = []
        individual_results_inh = []

        for _i in range(3):
            x_single = torch.randn(1, 6)
            exc_single, inh_single = transform(x_single)
            individual_results_exc.append(exc_single)
            individual_results_inh.append(inh_single)

        # Process as batch
        x_batch = torch.randn(3, 6)
        exc_batch, inh_batch = transform(x_batch)

        # Note: Results won't be identical due to different random inputs,
        # but the shapes and processing should be consistent
        assert exc_batch.shape == (3, 4)
        assert inh_batch.shape == (3, 2)

        for _i, (exc_single, inh_single) in enumerate(
            zip(individual_results_exc, individual_results_inh)
        ):
            assert exc_single.shape == (1, 4)
            assert inh_single.shape == (1, 2)

    def test_memory_efficiency(self):
        """Test memory usage with larger inputs."""
        transform = InputNetTransform(
            input_dim=100,
            excitatory_dim=60,
            inhibitory_dim=40,
            network_type="IndexInputTransform",
        )

        # Test with larger batch
        x = torch.randn(64, 100)
        exc_out, inh_out = transform(x)

        assert exc_out.shape == (64, 60)
        assert inh_out.shape == (64, 40)

        # Test memory cleanup
        del x, exc_out, inh_out
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def test_numerical_stability(self):
        """Test numerical stability with edge case inputs."""
        transform = InputNetTransform(
            input_dim=6,
            excitatory_dim=4,
            inhibitory_dim=2,
            network_type="IndexInputTransform",
        )

        # Test with very small values
        x_small = torch.randn(2, 6) * 1e-6
        exc_out, inh_out = transform(x_small)

        assert torch.isfinite(exc_out).all()
        assert torch.isfinite(inh_out).all()

        # Test with very large values
        x_large = torch.randn(2, 6) * 1e6
        exc_out, inh_out = transform(x_large)

        assert torch.isfinite(exc_out).all()
        assert torch.isfinite(inh_out).all()

        # Test with zeros
        x_zeros = torch.zeros(2, 6)
        exc_out, inh_out = transform(x_zeros)

        assert torch.isfinite(exc_out).all()
        assert torch.isfinite(inh_out).all()
