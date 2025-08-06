"""
Tests for training FSDP utils module to improve coverage.
"""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from dendritic_modeling.training.fsdp_utils import (
    _apply_gradient_checkpointing,
    create_fsdp_wrap_policy,
    estimate_memory_requirements,
    get_fsdp_config,
    get_recommended_gpu_count,
    load_fsdp_checkpoint,
    save_fsdp_checkpoint,
    wrap_model_with_fsdp,
)


class MockDendriticBranchLayer(nn.Module):
    """Mock dendritic branch layer for testing."""

    def __init__(self, num_params=1000):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_params, 1))

    def forward(self, x):
        return x


class MockExcitationInhibitionLayer(nn.Module):
    """Mock excitation-inhibition layer for testing."""

    def __init__(self, num_params=2000):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_params, 1))

    def forward(self, x):
        return x


class TestCreateFSDPWrapPolicy:
    """Test create_fsdp_wrap_policy function."""

    def test_wrap_policy_basic(self):
        """Test basic wrap policy creation."""
        policy = create_fsdp_wrap_policy(min_num_params=1000)

        assert callable(policy)

    def test_wrap_policy_dendritic_layers(self):
        """Test policy wraps dendritic layers."""
        policy = create_fsdp_wrap_policy(min_num_params=1000)

        # Mock dendritic layer should be wrapped
        dendritic_layer = MockDendriticBranchLayer()
        result = policy(dendritic_layer, None, 500)  # Below threshold but should wrap
        assert result is True

        # Mock EI layer should be wrapped
        ei_layer = MockExcitationInhibitionLayer()
        result = policy(ei_layer, None, 500)
        assert result is True

    def test_wrap_policy_parameter_threshold(self):
        """Test policy wraps based on parameter count."""
        policy = create_fsdp_wrap_policy(min_num_params=1000)

        # Regular module above threshold should be wrapped
        regular_module = nn.Linear(100, 100)  # ~10k params
        result = policy(regular_module, None, 2000)
        assert result is True

        # Regular module below threshold should not be wrapped
        result = policy(regular_module, None, 500)
        assert result is False

    def test_wrap_policy_recurse_true(self):
        """Test policy with recurse=True."""
        policy = create_fsdp_wrap_policy(min_num_params=1000)

        regular_module = nn.Linear(10, 10)
        result = policy(regular_module, True, 500)
        assert result is True  # Should wrap when recurse is True

    def test_wrap_policy_different_thresholds(self):
        """Test policy with different parameter thresholds."""
        # High threshold
        policy_high = create_fsdp_wrap_policy(min_num_params=10000)
        regular_module = nn.Linear(50, 50)  # ~2.5k params
        result = policy_high(regular_module, None, 2500)
        assert result is False

        # Low threshold
        policy_low = create_fsdp_wrap_policy(min_num_params=100)
        result = policy_low(regular_module, None, 2500)
        assert result is True


class TestGetFSDPConfig:
    """Test get_fsdp_config function."""

    def test_get_fsdp_config_defaults(self):
        """Test get_fsdp_config with default parameters."""
        config = get_fsdp_config()

        assert isinstance(config, dict)
        assert "auto_wrap_policy" in config
        assert "mixed_precision" in config
        assert "cpu_offload" in config
        assert "sharding_strategy" in config
        assert "sync_module_states" in config

    def test_get_fsdp_config_small_model(self):
        """Test FSDP config for small model."""
        config = get_fsdp_config(model_size="small")

        # Small models might have different defaults
        assert config["auto_wrap_policy"] is not None

    def test_get_fsdp_config_large_model(self):
        """Test FSDP config for large model."""
        config = get_fsdp_config(model_size="large")

        assert config["auto_wrap_policy"] is not None

    def test_get_fsdp_config_xlarge_model(self):
        """Test FSDP config for extra large model."""
        config = get_fsdp_config(model_size="xlarge")

        assert config["auto_wrap_policy"] is not None

    def test_get_fsdp_config_mixed_precision(self):
        """Test FSDP config with mixed precision."""
        config = get_fsdp_config(mixed_precision=True)

        assert config["mixed_precision"] is not None

        config_no_mp = get_fsdp_config(mixed_precision=False)
        assert config_no_mp["mixed_precision"] is None

    def test_get_fsdp_config_cpu_offload(self):
        """Test FSDP config with CPU offload."""
        config = get_fsdp_config(cpu_offload=True)

        assert config["cpu_offload"] is not None

        config_no_offload = get_fsdp_config(cpu_offload=False)
        assert config_no_offload["cpu_offload"] is None

    def test_get_fsdp_config_sharding_strategies(self):
        """Test different sharding strategies."""
        strategies = ["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"]

        for strategy in strategies:
            config = get_fsdp_config(sharding_strategy=strategy)
            assert config["sharding_strategy"] is not None

    def test_get_fsdp_config_with_dict_override(self):
        """Test FSDP config with dictionary override."""
        override_dict = {
            "mixed_precision": False,
            "cpu_offload": True,
            "sharding_strategy": "NO_SHARD",
            "min_num_params": 5000,
            "sync_module_states": False,
        }

        config = get_fsdp_config(fsdp_config_dict=override_dict)

        # Should use override values
        assert config["cpu_offload"] is not None  # True from override
        assert config["sync_module_states"] is False
        assert config["mixed_precision"] is None  # False from override

    def test_get_fsdp_config_communication_overhead(self):
        """Test communication overhead optimization."""
        with patch("dendritic_modeling.training.fsdp_utils.logger") as mock_logger:
            get_fsdp_config(reduce_communication_overhead=True)

            # Should log optimization info
            mock_logger.info.assert_called_once()

    def test_get_fsdp_config_backward_prefetch_mapping(self):
        """Test backward prefetch strategy mapping."""
        config = get_fsdp_config()

        # Should have valid backward prefetch config
        assert "backward_prefetch" in config


class TestWrapModelWithFSDP:
    """Test wrap_model_with_fsdp function."""

    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        return nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

    @pytest.fixture
    def mock_fsdp_config(self):
        """Create mock FSDP config."""
        return {
            "auto_wrap_policy": create_fsdp_wrap_policy(),
            "mixed_precision": None,
            "cpu_offload": None,
            "sharding_strategy": "FULL_SHARD",
            "sync_module_states": True,
            "use_orig_params": True,
            "backward_prefetch": "BACKWARD_PRE",
            "limit_all_gathers": True,
            "forward_prefetch": True,
            "gradient_checkpointing": False,
        }

    @patch("torch.distributed.is_initialized")
    @patch("dendritic_modeling.training.fsdp_utils.FullyShardedDataParallel")
    def test_wrap_model_basic(
        self, mock_fsdp, mock_dist_init, simple_model, mock_fsdp_config
    ):
        """Test basic model wrapping with FSDP."""
        mock_dist_init.return_value = True
        mock_fsdp.return_value = simple_model

        wrapped_model = wrap_model_with_fsdp(simple_model, mock_fsdp_config)

        mock_fsdp.assert_called_once()
        assert wrapped_model == simple_model

    @patch("torch.distributed.is_initialized")
    def test_wrap_model_no_distributed(
        self, mock_dist_init, simple_model, mock_fsdp_config
    ):
        """Test model wrapping when distributed is not initialized."""
        mock_dist_init.return_value = False

        with pytest.raises(RuntimeError, match="Distributed training not initialized"):
            wrap_model_with_fsdp(simple_model, mock_fsdp_config)

    @patch("torch.distributed.is_initialized")
    @patch("dendritic_modeling.training.fsdp_utils.FullyShardedDataParallel")
    def test_wrap_model_with_gradient_checkpointing(
        self, mock_fsdp, mock_dist_init, simple_model
    ):
        """Test model wrapping with gradient checkpointing."""
        mock_dist_init.return_value = True
        mock_fsdp.return_value = simple_model

        config = get_fsdp_config()
        config["gradient_checkpointing"] = True

        with patch(
            "dendritic_modeling.training.fsdp_utils._apply_gradient_checkpointing"
        ) as mock_gc:
            wrap_model_with_fsdp(simple_model, config)

            mock_gc.assert_called_once_with(simple_model)

    @patch("torch.distributed.is_initialized")
    @patch("dendritic_modeling.training.fsdp_utils.FullyShardedDataParallel")
    def test_wrap_model_logs_info(
        self, mock_fsdp, mock_dist_init, simple_model, mock_fsdp_config
    ):
        """Test that model wrapping logs information."""
        mock_dist_init.return_value = True
        mock_fsdp.return_value = simple_model

        with patch("dendritic_modeling.training.fsdp_utils.logger") as mock_logger:
            wrap_model_with_fsdp(simple_model, mock_fsdp_config)

            mock_logger.info.assert_called()


class TestApplyGradientCheckpointing:
    """Test _apply_gradient_checkpointing function."""

    def test_apply_gradient_checkpointing(self):
        """Test applying gradient checkpointing to model."""
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

        with patch("dendritic_modeling.training.fsdp_utils.logger") as mock_logger:
            _apply_gradient_checkpointing(model)

            # Should log application
            mock_logger.info.assert_called()
            assert "gradient checkpointing" in mock_logger.info.call_args[0][0].lower()

    def test_apply_gradient_checkpointing_nested_model(self):
        """Test gradient checkpointing on nested model."""
        model = nn.ModuleDict(
            {
                "encoder": nn.Sequential(nn.Linear(10, 5), nn.ReLU()),
                "decoder": nn.Sequential(nn.Linear(5, 2), nn.Sigmoid()),
            }
        )

        with patch("dendritic_modeling.training.fsdp_utils.logger") as mock_logger:
            _apply_gradient_checkpointing(model)

            mock_logger.info.assert_called()


class TestFSDPCheckpointing:
    """Test FSDP checkpoint save/load functions."""

    @pytest.fixture
    def mock_fsdp_model(self):
        """Create mock FSDP model."""
        model = Mock()
        model.state_dict.return_value = {"weight": torch.randn(5, 5)}
        return model

    @pytest.fixture
    def mock_optimizer(self):
        """Create mock optimizer."""
        optimizer = Mock()
        optimizer.state_dict.return_value = {"lr": 0.001}
        return optimizer

    def test_save_fsdp_checkpoint(self, mock_fsdp_model, mock_optimizer):
        """Test saving FSDP checkpoint."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "checkpoint.pt")

            with patch("torch.save") as mock_save:
                save_fsdp_checkpoint(
                    mock_fsdp_model,
                    mock_optimizer,
                    epoch=10,
                    loss=0.5,
                    checkpoint_path=checkpoint_path,
                )

                mock_save.assert_called_once()
                saved_data = mock_save.call_args[0][0]
                assert "model_state_dict" in saved_data
                assert "optimizer_state_dict" in saved_data
                assert "epoch" in saved_data
                assert "loss" in saved_data
                assert saved_data["epoch"] == 10
                assert saved_data["loss"] == 0.5

    def test_save_fsdp_checkpoint_with_metadata(self, mock_fsdp_model, mock_optimizer):
        """Test saving checkpoint with additional metadata."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "checkpoint.pt")
            metadata = {"config": {"lr": 0.001}, "best_score": 0.95}

            with patch("torch.save") as mock_save:
                save_fsdp_checkpoint(
                    mock_fsdp_model,
                    mock_optimizer,
                    epoch=5,
                    loss=0.3,
                    checkpoint_path=checkpoint_path,
                    **metadata,
                )

                saved_data = mock_save.call_args[0][0]
                assert "config" in saved_data
                assert "best_score" in saved_data
                assert saved_data["config"]["lr"] == 0.001

    def test_load_fsdp_checkpoint(self, mock_fsdp_model, mock_optimizer):
        """Test loading FSDP checkpoint."""
        checkpoint_data = {
            "model_state_dict": {"weight": torch.randn(5, 5)},
            "optimizer_state_dict": {"lr": 0.001},
            "epoch": 15,
            "loss": 0.25,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "checkpoint.pt")

            with patch("torch.load", return_value=checkpoint_data) as mock_load:
                loaded_data = load_fsdp_checkpoint(
                    mock_fsdp_model, mock_optimizer, checkpoint_path
                )

                mock_load.assert_called_once_with(checkpoint_path, map_location="cpu")
                mock_fsdp_model.load_state_dict.assert_called_once()
                mock_optimizer.load_state_dict.assert_called_once()

                assert loaded_data["epoch"] == 15
                assert loaded_data["loss"] == 0.25

    def test_load_fsdp_checkpoint_model_only(self, mock_fsdp_model):
        """Test loading checkpoint with model only."""
        checkpoint_data = {
            "model_state_dict": {"weight": torch.randn(5, 5)},
            "epoch": 20,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "checkpoint.pt")

            with patch("torch.load", return_value=checkpoint_data):
                loaded_data = load_fsdp_checkpoint(
                    mock_fsdp_model, None, checkpoint_path  # No optimizer
                )

                mock_fsdp_model.load_state_dict.assert_called_once()
                assert loaded_data["epoch"] == 20

    def test_load_fsdp_checkpoint_nonexistent_file(
        self, mock_fsdp_model, mock_optimizer
    ):
        """Test loading checkpoint from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_fsdp_checkpoint(
                mock_fsdp_model, mock_optimizer, "/nonexistent/checkpoint.pt"
            )


class TestMemoryEstimation:
    """Test memory estimation functions."""

    def test_estimate_memory_requirements_small_model(self):
        """Test memory estimation for small model."""
        memory_gb = estimate_memory_requirements(
            model_params=1e6, batch_size=32, sequence_length=512  # 1M parameters
        )

        assert isinstance(memory_gb, float)
        assert memory_gb > 0

    def test_estimate_memory_requirements_large_model(self):
        """Test memory estimation for large model."""
        memory_gb = estimate_memory_requirements(
            model_params=1e9, batch_size=16, sequence_length=1024  # 1B parameters
        )

        assert isinstance(memory_gb, float)
        assert memory_gb > 10  # Should be substantial for 1B param model

    def test_estimate_memory_requirements_with_fsdp(self):
        """Test memory estimation with FSDP enabled."""
        memory_gb = estimate_memory_requirements(
            model_params=1e8, batch_size=64, sequence_length=256, use_fsdp=True
        )

        memory_gb_no_fsdp = estimate_memory_requirements(
            model_params=1e8, batch_size=64, sequence_length=256, use_fsdp=False
        )

        # FSDP should reduce memory requirements
        assert memory_gb < memory_gb_no_fsdp

    def test_estimate_memory_requirements_different_precision(self):
        """Test memory estimation with different precision."""
        memory_fp32 = estimate_memory_requirements(
            model_params=1e7, batch_size=32, sequence_length=512, mixed_precision=False
        )

        memory_fp16 = estimate_memory_requirements(
            model_params=1e7, batch_size=32, sequence_length=512, mixed_precision=True
        )

        # Mixed precision should reduce memory
        assert memory_fp16 < memory_fp32

    def test_estimate_memory_requirements_gradient_checkpointing(self):
        """Test memory estimation with gradient checkpointing."""
        memory_normal = estimate_memory_requirements(
            model_params=1e7,
            batch_size=32,
            sequence_length=512,
            gradient_checkpointing=False,
        )

        memory_gc = estimate_memory_requirements(
            model_params=1e7,
            batch_size=32,
            sequence_length=512,
            gradient_checkpointing=True,
        )

        # Gradient checkpointing should reduce memory
        assert memory_gc < memory_normal

    def test_get_recommended_gpu_count(self):
        """Test GPU count recommendation."""
        gpu_count = get_recommended_gpu_count(
            memory_per_gpu_gb=16.0, total_memory_gb=64.0
        )

        assert gpu_count == 4  # 64 / 16 = 4

    def test_get_recommended_gpu_count_minimum(self):
        """Test GPU count recommendation with minimum."""
        gpu_count = get_recommended_gpu_count(
            memory_per_gpu_gb=32.0, total_memory_gb=8.0  # Less than per-GPU memory
        )

        assert gpu_count == 1  # Should return at least 1

    def test_get_recommended_gpu_count_fractional(self):
        """Test GPU count recommendation with fractional result."""
        gpu_count = get_recommended_gpu_count(
            memory_per_gpu_gb=24.0, total_memory_gb=100.0  # 100/24 = 4.17
        )

        assert gpu_count == 5  # Should round up


class TestFSDPUtilsEdgeCases:
    """Test edge cases for FSDP utilities."""

    def test_wrap_policy_with_none_values(self):
        """Test wrap policy with None values."""
        policy = create_fsdp_wrap_policy()

        # Should handle None module gracefully
        result = policy(None, None, 1000)
        assert result is False

    def test_get_fsdp_config_invalid_strategy(self):
        """Test FSDP config with invalid sharding strategy."""
        # Should handle gracefully, using default mapping
        config = get_fsdp_config(sharding_strategy="INVALID_STRATEGY")

        # Should still return a valid config
        assert isinstance(config, dict)

    def test_memory_estimation_edge_cases(self):
        """Test memory estimation with edge cases."""
        # Zero parameters
        memory = estimate_memory_requirements(
            model_params=0, batch_size=1, sequence_length=1
        )
        assert memory >= 0

        # Very large parameters
        memory = estimate_memory_requirements(
            model_params=1e12, batch_size=1, sequence_length=1  # 1T parameters
        )
        assert memory > 0

    def test_checkpoint_save_directory_creation(self, mock_fsdp_model, mock_optimizer):
        """Test that checkpoint save creates directory if needed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nested_path = os.path.join(tmp_dir, "nested", "checkpoint.pt")

            with patch("torch.save"):
                with patch("os.makedirs") as mock_makedirs:
                    save_fsdp_checkpoint(
                        mock_fsdp_model,
                        mock_optimizer,
                        epoch=1,
                        loss=0.1,
                        checkpoint_path=nested_path,
                    )

                    mock_makedirs.assert_called_once()

    def test_estimate_memory_with_all_optimizations(self):
        """Test memory estimation with all memory optimizations enabled."""
        memory = estimate_memory_requirements(
            model_params=1e8,
            batch_size=32,
            sequence_length=512,
            use_fsdp=True,
            mixed_precision=True,
            gradient_checkpointing=True,
            cpu_offload=True,
        )

        # Should be significantly reduced with all optimizations
        assert isinstance(memory, float)
        assert memory > 0
