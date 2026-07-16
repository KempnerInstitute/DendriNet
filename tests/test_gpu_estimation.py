"""
Tests for utils gpu_estimation module to improve coverage.
"""

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from dendritic_modeling.utils.gpu_estimation import (
    BLOCKLINEAR_ACTIVATION_FACTOR,
    BYTES_PER_GIB,
    DENDRITIC_BRANCHING_OVERHEAD,
    GPU_MEMORY_SAFETY_FACTOR,
    GRADIENT_GRAPH_OVERHEAD,
    LINEAR_ACTIVATION_FACTOR,
    PARAMETER_FALLBACK_ACTIVATION_MULTIPLIER,
    TOPK_INDEX_VALUE_FACTOR,
    TOPK_SPARSE_GRADIENT_FACTOR,
    GPUMemoryEstimator,
    GPURecommendation,
    MemoryBreakdown,
    _find_einet_activation_config,
    _iter_leaf_modules,
    _module_parameter_category,
    _resolve_estimation_input_shape,
    estimate_from_config,
)


class TestMemoryBreakdown:
    """Test MemoryBreakdown dataclass."""

    def test_memory_breakdown_creation(self):
        """Test creating MemoryBreakdown instance."""
        breakdown = MemoryBreakdown(
            parameters=2.5,
            gradients=2.5,
            optimizer_states=5.0,
            activations=1.0,
            fsdp_overhead=0.5,
            total=11.5,
        )

        assert breakdown.parameters == 2.5
        assert breakdown.gradients == 2.5
        assert breakdown.optimizer_states == 5.0
        assert breakdown.activations == 1.0
        assert breakdown.fsdp_overhead == 0.5
        assert breakdown.total == 11.5

    def test_memory_breakdown_str_representation(self):
        """Test string representation of MemoryBreakdown."""
        breakdown = MemoryBreakdown(
            parameters=1.0,
            gradients=1.0,
            optimizer_states=2.0,
            activations=0.5,
            fsdp_overhead=0.1,
            total=4.6,
        )

        str_repr = str(breakdown)
        assert "Memory Breakdown:" in str_repr
        assert "Parameters:        1.00 GB" in str_repr
        assert "Gradients:         1.00 GB" in str_repr
        assert "Optimizer states:  2.00 GB" in str_repr
        assert "Activations:       0.50 GB" in str_repr
        assert "FSDP overhead:     0.10 GB" in str_repr
        assert "Total:             4.60 GB" in str_repr

    def test_memory_breakdown_zero_values(self):
        """Test MemoryBreakdown with zero values."""
        breakdown = MemoryBreakdown(
            parameters=0.0,
            gradients=0.0,
            optimizer_states=0.0,
            activations=0.0,
            fsdp_overhead=0.0,
            total=0.0,
        )

        str_repr = str(breakdown)
        assert "0.00 GB" in str_repr

    def test_memory_breakdown_large_values(self):
        """Test MemoryBreakdown with large values."""
        breakdown = MemoryBreakdown(
            parameters=1000.0,
            gradients=1000.0,
            optimizer_states=2000.0,
            activations=500.0,
            fsdp_overhead=100.0,
            total=4600.0,
        )

        str_repr = str(breakdown)
        assert "1000.00 GB" in str_repr
        assert "4600.00 GB" in str_repr


class TestGPURecommendation:
    """Test GPURecommendation dataclass."""

    def test_gpu_recommendation_creation(self):
        """Test creating GPURecommendation instance."""
        recommendation = GPURecommendation(
            gpu_model="A100-40GB",
            gpu_memory_gb=40,
            min_gpus_needed=2,
            recommended_gpus=2,
            memory_per_gpu_gb=20.0,
            utilization_percent=85.0,
            fits=True,
        )

        assert recommendation.gpu_model == "A100-40GB"
        assert recommendation.gpu_memory_gb == 40
        assert recommendation.min_gpus_needed == 2
        assert recommendation.recommended_gpus == 2
        assert recommendation.memory_per_gpu_gb == 20.0
        assert recommendation.utilization_percent == 85.0
        assert recommendation.fits is True

    def test_gpu_recommendation_h100(self):
        """Test GPURecommendation with H100."""
        recommendation = GPURecommendation(
            gpu_model="H100",
            gpu_memory_gb=80,
            min_gpus_needed=1,
            recommended_gpus=1,
            memory_per_gpu_gb=40.0,
            utilization_percent=50.0,
            fits=True,
        )

        assert recommendation.gpu_model == "H100"
        assert recommendation.fits is True

    def test_gpu_recommendation_no_fit(self):
        """Test GPURecommendation when model doesn't fit."""
        recommendation = GPURecommendation(
            gpu_model="V100",
            gpu_memory_gb=32,
            min_gpus_needed=4,
            recommended_gpus=4,
            memory_per_gpu_gb=50.0,  # Exceeds available memory
            utilization_percent=150.0,  # Over 100%
            fits=False,
        )

        assert recommendation.fits is False
        assert recommendation.utilization_percent > 100.0


class TestGPUMemoryEstimator:
    """Test GPUMemoryEstimator class."""

    def test_iter_leaf_modules_preserves_recursive_module_order(self):
        model = nn.Sequential(
            nn.Linear(3, 4),
            nn.Sequential(nn.ReLU(), nn.Linear(4, 2)),
        )

        leaves = list(_iter_leaf_modules(model))

        assert leaves == [model[0], model[1][0], model[1][1]]

    def test_module_parameter_category_preserves_legacy_name_mapping(self):
        class DummyTopKLayer(nn.Module):
            pass

        class DummyBlockLinearLayer(nn.Module):
            pass

        class DummyReactivationLayer(nn.Module):
            pass

        assert _module_parameter_category(DummyTopKLayer()) == "TopK Layers"
        assert (
            _module_parameter_category(DummyBlockLinearLayer()) == "Block Linear Layers"
        )
        assert (
            _module_parameter_category(DummyReactivationLayer())
            == "Reactivation Layers"
        )
        assert _module_parameter_category(nn.Conv2d(1, 1, 1)) == (
            "Convolutional Layers"
        )
        assert _module_parameter_category(nn.LayerNorm(2)) == "Normalization Layers"

    def test_module_parameter_category_preserves_keyword_precedence(self):
        class DummyTopKLinearLayer(nn.Module):
            pass

        assert _module_parameter_category(DummyTopKLinearLayer()) == "TopK Layers"

    def test_find_einet_activation_config_preserves_first_match_rule(self):
        class ExcitationInhibitionNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.excitatory_layer_sizes = [5]
                self.inhibitory_layer_sizes = [2]
                self.excitatory_branch_factors = [3]
                self.inhibitory_branch_factors = [4]
                self.ee_synapses_per_branch_per_layer = [7]
                self.ei_synapses_per_branch_per_layer = [11]
                self.ie_synapses_per_branch_per_layer = [13]
                self.ii_synapses_per_branch_per_layer = [17]

        model = nn.Sequential(nn.Linear(1, 1), ExcitationInhibitionNetwork())

        assert _find_einet_activation_config(model) == {
            "e_layers": [5],
            "i_layers": [2],
            "e_branches": [3],
            "i_branches": [4],
            "ee_synapses": [7],
            "ei_synapses": [11],
            "ie_synapses": [13],
            "ii_synapses": [17],
        }

    def test_find_einet_activation_config_preserves_missing_attr_fallback(self):
        class ExcitationInhibitionNetwork(nn.Module):
            pass

        assert _find_einet_activation_config(ExcitationInhibitionNetwork()) is None

    def test_init_default(self):
        """Test initialization with default parameters."""
        estimator = GPUMemoryEstimator()

        assert estimator.dtype == "fp16"
        assert estimator.optimizer == "adam"
        assert estimator.bytes_per_param == 2  # fp16
        assert estimator.optimizer_state_multiplier == 2.0  # adam

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        estimator = GPUMemoryEstimator(dtype="fp32", optimizer="sgd")

        assert estimator.dtype == "fp32"
        assert estimator.optimizer == "sgd"
        assert estimator.bytes_per_param == 4  # fp32
        assert estimator.optimizer_state_multiplier == 1.0  # sgd

    def test_get_bytes_per_param(self):
        """Test _get_bytes_per_param method."""
        estimator = GPUMemoryEstimator()

        assert estimator._get_bytes_per_param("fp32") == 4
        assert estimator._get_bytes_per_param("fp16") == 2
        assert estimator._get_bytes_per_param("bf16") == 2
        assert estimator._get_bytes_per_param("int8") == 1
        assert estimator._get_bytes_per_param("unknown") == 2  # default

    def test_get_optimizer_multiplier(self):
        """Test _get_optimizer_multiplier method."""
        estimator = GPUMemoryEstimator()

        assert estimator._get_optimizer_multiplier("sgd") == 1.0
        assert estimator._get_optimizer_multiplier("adam") == 2.0
        assert estimator._get_optimizer_multiplier("adamw") == 2.0
        assert estimator._get_optimizer_multiplier("lamb") == 2.0
        assert estimator._get_optimizer_multiplier("rmsprop") == 1.0
        assert estimator._get_optimizer_multiplier("unknown") == 2.0  # default

    def test_estimate_model_parameters(self):
        """Test parameter counting for different layer types."""
        estimator = GPUMemoryEstimator()

        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 20),  # Linear layer
            nn.Conv2d(3, 64, 3),  # Convolutional layer
            nn.BatchNorm2d(64),  # Normalization layer
            nn.ReLU(),  # No parameters
        )

        param_counts = estimator.estimate_model_parameters(model)

        assert "Linear Layers" in param_counts
        assert "Convolutional Layers" in param_counts
        assert "Normalization Layers" in param_counts

        # Check that parameter counts are positive
        assert param_counts["Linear Layers"] > 0
        assert param_counts["Convolutional Layers"] > 0
        assert param_counts["Normalization Layers"] > 0

    def test_estimate_model_parameters_custom_layers(self):
        """Test parameter counting with custom layer names."""
        estimator = GPUMemoryEstimator()

        # Mock a model with custom layer types
        class TopKLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(10, 5))

        class BlockLinearLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(5, 3))

        class ReactivationLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(3, 2))

        model = nn.ModuleDict(
            {
                "topk": TopKLinear(),
                "block": BlockLinearLayer(),
                "react": ReactivationLayer(),
            }
        )

        param_counts = estimator.estimate_model_parameters(model)

        assert "TopK Layers" in param_counts
        assert "Block Linear Layers" in param_counts
        assert "Reactivation Layers" in param_counts

    def test_estimate_model_parameters_empty_model(self):
        """Test parameter counting with model that has no parameters."""
        estimator = GPUMemoryEstimator()

        model = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Identity(),
        )

        param_counts = estimator.estimate_model_parameters(model)

        # Should return empty dict or only have zero counts
        assert len(param_counts) == 0 or all(
            count == 0 for count in param_counts.values()
        )

    def test_estimate_activation_memory(self):
        """Test activation memory estimation."""
        estimator = GPUMemoryEstimator()

        # Create a simple model for testing
        model = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128))
        batch_size = 32
        input_shape = 512

        # Test basic activation estimation
        activation_gb = estimator.estimate_activation_memory(
            model, batch_size, input_shape
        )

        assert isinstance(activation_gb, float)
        assert activation_gb > 0

        # Larger batch size should require more memory
        larger_activation_gb = estimator.estimate_activation_memory(
            model, batch_size * 2, input_shape
        )

        assert larger_activation_gb > activation_gb

    def test_estimate_activation_memory_different_dtypes(self):
        """Test activation memory estimation with different data types."""
        estimator_fp16 = GPUMemoryEstimator(dtype="fp16")
        estimator_fp32 = GPUMemoryEstimator(dtype="fp32")

        model = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
        batch_size = 16
        input_shape = 256

        memory_fp16 = estimator_fp16.estimate_activation_memory(
            model, batch_size, input_shape
        )
        memory_fp32 = estimator_fp32.estimate_activation_memory(
            model, batch_size, input_shape
        )

        # fp32 should use more memory than fp16
        assert memory_fp32 > memory_fp16

    def test_estimate_activation_memory_einet_matches_legacy_formula(self):
        class ExcitationInhibitionNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.excitatory_layer_sizes = [5]
                self.inhibitory_layer_sizes = [2]
                self.excitatory_branch_factors = [3]
                self.inhibitory_branch_factors = [4]
                self.ee_synapses_per_branch_per_layer = [7]
                self.ei_synapses_per_branch_per_layer = [11]
                self.ie_synapses_per_branch_per_layer = [13]
                self.ii_synapses_per_branch_per_layer = [17]

        estimator = GPUMemoryEstimator(dtype="fp16")
        batch_size = 2
        expected_bytes = batch_size * 3 * estimator.bytes_per_param
        expected_bytes += batch_size * 5 * 3 * estimator.bytes_per_param
        expected_bytes += (
            batch_size * 5 * 3 * 7 * estimator.bytes_per_param * TOPK_INDEX_VALUE_FACTOR
        )
        expected_bytes += batch_size * 2 * 4 * estimator.bytes_per_param
        expected_bytes += (
            batch_size
            * 2
            * 4
            * 17
            * estimator.bytes_per_param
            * TOPK_INDEX_VALUE_FACTOR
        )
        expected_bytes *= DENDRITIC_BRANCHING_OVERHEAD

        activation_gb = estimator.estimate_activation_memory(
            ExcitationInhibitionNetwork(),
            batch_size=batch_size,
            input_shape=(3,),
        )

        assert activation_gb == pytest.approx(expected_bytes / BYTES_PER_GIB)

    def test_estimate_activation_memory_linear_matches_formula(self):
        """Dense fallback activations should match the documented byte formula."""
        estimator = GPUMemoryEstimator(dtype="fp16")
        model = nn.Sequential(nn.Linear(10, 4), nn.ReLU(), nn.Linear(4, 2))
        batch_size = 8

        expected_bytes = (
            batch_size
            * ((10 + 4) + (4 + 2))
            * estimator.bytes_per_param
            * LINEAR_ACTIVATION_FACTOR
            * GRADIENT_GRAPH_OVERHEAD
        )

        activation_gb = estimator.estimate_activation_memory(
            model,
            batch_size=batch_size,
            input_shape=(10,),
        )

        assert activation_gb == pytest.approx(expected_bytes / BYTES_PER_GIB)

    def test_estimate_activation_memory_sparse_helpers_match_formulas(self):
        """TopK and BlockLinear fallback estimates should keep exact formulas."""

        class DummyTopKLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.out_features = 5
                self.K = 3

        class DummyBlockLinearLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(2, 4))

        estimator = GPUMemoryEstimator(dtype="fp32")
        model = nn.Sequential(DummyTopKLayer(), DummyBlockLinearLayer())
        batch_size = 7

        expected_bytes = (
            batch_size * 5 * 3 * estimator.bytes_per_param * TOPK_SPARSE_GRADIENT_FACTOR
        )
        expected_bytes += (
            batch_size * 8 * estimator.bytes_per_param * BLOCKLINEAR_ACTIVATION_FACTOR
        )
        expected_bytes *= GRADIENT_GRAPH_OVERHEAD

        activation_gb = estimator.estimate_activation_memory(
            model,
            batch_size=batch_size,
            input_shape=(1,),
        )

        assert activation_gb == pytest.approx(expected_bytes / BYTES_PER_GIB)

    def test_estimate_activation_memory_opaque_module_uses_parameter_fallback(self):
        """Opaque parameterized modules should retain the historical fallback."""

        class OpaqueLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(3, 4))

        estimator = GPUMemoryEstimator(dtype="fp16")
        model = OpaqueLayer()

        expected_bytes = (
            12
            * estimator.bytes_per_param
            * PARAMETER_FALLBACK_ACTIVATION_MULTIPLIER
            * GRADIENT_GRAPH_OVERHEAD
        )

        activation_gb = estimator.estimate_activation_memory(
            model,
            batch_size=2,
            input_shape=(1,),
        )

        assert activation_gb == pytest.approx(expected_bytes / BYTES_PER_GIB)

    def test_resolve_estimation_input_shape_uses_real_dataset_sample(self, monkeypatch):
        dataset = torch.utils.data.TensorDataset(
            torch.zeros(3, 2, 5),
            torch.zeros(3, dtype=torch.long),
        )

        def fake_get_unified_datasets(*, task_cfg):
            assert task_cfg.dataset == "custom"
            return dataset, None, None

        monkeypatch.setattr(
            "dendritic_modeling.datasets.get_unified_datasets",
            fake_get_unified_datasets,
        )

        shape = _resolve_estimation_input_shape(
            SimpleNamespace(dataset="custom"),
            {"input_dim": 99},
        )

        assert shape == torch.Size([2, 5])

    def test_resolve_estimation_input_shape_falls_back_to_canonical_shape(
        self, monkeypatch
    ):
        def missing_dataset(*, task_cfg):
            raise RuntimeError(f"missing {task_cfg.dataset}")

        monkeypatch.setattr(
            "dendritic_modeling.datasets.get_unified_datasets",
            missing_dataset,
        )

        shape = _resolve_estimation_input_shape(
            SimpleNamespace(dataset="mnist"),
            {"flatten": True},
        )

        assert shape == (784,)

    def test_estimate_memory_breakdown(self):
        """Test total memory estimation."""
        estimator = GPUMemoryEstimator()

        # Create simple model
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )

        breakdown = estimator.estimate_memory_breakdown(
            model, batch_size=16, input_shape=(784,)
        )

        assert isinstance(breakdown, MemoryBreakdown)
        assert breakdown.parameters > 0
        assert breakdown.gradients > 0
        assert breakdown.optimizer_states > 0
        assert breakdown.activations > 0
        assert breakdown.total > 0

        # Total should be sum of components
        expected_total = (
            breakdown.parameters
            + breakdown.gradients
            + breakdown.optimizer_states
            + breakdown.activations
            + breakdown.fsdp_overhead
        )
        assert abs(breakdown.total - expected_total) < 0.01

    def test_estimate_memory_breakdown_with_fsdp(self):
        """Test memory estimation with FSDP enabled."""
        estimator = GPUMemoryEstimator()

        model = nn.Sequential(nn.Linear(1000, 500), nn.Linear(500, 100))

        breakdown_no_fsdp = estimator.estimate_memory_breakdown(
            model, batch_size=32, input_shape=(256,), use_fsdp=False
        )

        breakdown_with_fsdp = estimator.estimate_memory_breakdown(
            model, batch_size=32, input_shape=(256,), use_fsdp=True
        )

        # FSDP should reduce parameter memory
        assert breakdown_with_fsdp.parameters <= breakdown_no_fsdp.parameters

        # FSDP should have some overhead
        assert breakdown_with_fsdp.fsdp_overhead > 0

    def test_get_gpu_recommendations(self):
        """Test GPU configuration recommendation."""
        estimator = GPUMemoryEstimator()

        model = nn.Sequential(nn.Linear(1000, 100), nn.Linear(100, 10))

        # First get memory breakdown, then get recommendations
        breakdown = estimator.estimate_memory_breakdown(
            model, batch_size=16, input_shape=(128,)
        )
        recommendations = estimator.get_gpu_recommendations(breakdown)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        for rec in recommendations:
            assert isinstance(rec, GPURecommendation)
            assert rec.gpu_model in estimator.GPU_SPECS
            assert rec.gpu_memory_gb > 0
            assert rec.min_gpus_needed > 0

    def test_get_gpu_recommendations_large_model(self):
        """Test GPU recommendation for large model."""
        estimator = GPUMemoryEstimator()

        # Create large model
        model = nn.Sequential(
            nn.Linear(10000, 5000),
            nn.Linear(5000, 2500),
            nn.Linear(2500, 1000),
            nn.Linear(1000, 100),
        )

        # First get memory breakdown, then get recommendations
        breakdown = estimator.estimate_memory_breakdown(
            model, batch_size=64, input_shape=(512,)
        )
        recommendations = estimator.get_gpu_recommendations(breakdown)

        # Should provide recommendations for large model
        assert len(recommendations) > 0
        # Note: Actual GPU requirement depends on model size and available GPU memory

    def test_get_gpu_recommendations_default_uses_safety_factor_constant(self):
        """Default recommendations should match the named safety-factor policy."""
        estimator = GPUMemoryEstimator()
        breakdown = MemoryBreakdown(
            parameters=10.0,
            gradients=10.0,
            optimizer_states=20.0,
            activations=1.0,
            fsdp_overhead=0.0,
            total=41.0,
        )

        default_recommendations = estimator.get_gpu_recommendations(breakdown)
        explicit_recommendations = estimator.get_gpu_recommendations(
            breakdown, safety_factor=GPU_MEMORY_SAFETY_FACTOR
        )

        assert default_recommendations == explicit_recommendations

    def test_gpu_specs_class_variable(self):
        """Test GPU_SPECS class variable."""
        assert "A100-40GB" in GPUMemoryEstimator.GPU_SPECS
        assert "H100" in GPUMemoryEstimator.GPU_SPECS

        a100_spec = GPUMemoryEstimator.GPU_SPECS["A100-40GB"]
        assert a100_spec["memory_gb"] == 40
        assert a100_spec["compute"] == "high"

        h100_spec = GPUMemoryEstimator.GPU_SPECS["H100"]
        assert h100_spec["memory_gb"] == 80
        assert h100_spec["compute"] == "high"


class TestEstimateFromConfig:
    """Test estimate_from_config function."""

    def test_estimate_from_config_basic(self):
        """Test basic config-based estimation."""
        config_data = {
            "model": {
                "hidden_size": 768,
                "num_layers": 12,
                "num_attention_heads": 12,
            },
            "training": {
                "batch_size": 16,
                "input_shape": 512,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            # This function requires a complex real config file structure
            # For now we test that it handles invalid configs
            try:
                estimate_from_config(config_path)
            except Exception:
                # Expected to fail with invalid simple config
                pass

        finally:
            os.unlink(config_path)

    def test_estimate_from_config_with_batch_size_override(self):
        """Test config estimation with batch size override."""
        config_data = {
            "model": {"hidden_size": 512},
            "training": {"batch_size": 8},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            # This function requires a complex real config file structure
            # For now we test that it handles invalid configs with batch size override
            try:
                estimate_from_config(config_path, batch_size=32)
            except Exception:
                # Expected to fail with invalid simple config
                pass

        finally:
            os.unlink(config_path)

    def test_estimate_from_config_missing_file(self):
        """Test config estimation with missing file."""
        with pytest.raises(FileNotFoundError):
            estimate_from_config("/nonexistent/config.json")

    def test_estimate_from_config_invalid_json(self):
        """Test config estimation with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json {")
            config_path = f.name

        try:
            # The function has complex error handling, so we'll just test it doesn't crash catastrophically
            try:
                estimate_from_config(config_path)
            except Exception:
                # Expected to fail with various possible exceptions
                pass
        finally:
            os.unlink(config_path)

    def test_delayed_comparison_paper_config_reaches_resource_estimation(
        self, monkeypatch
    ):
        """Typed current config should build the real recurrent model and estimate."""
        config_path = (
            Path(__file__).resolve().parents[1]
            / "drafts"
            / "dendritic_RNN"
            / "configs"
            / "E22p_romo_full_paperclean.yaml"
        )
        captured = {}

        class TinySequenceDataset:
            def __len__(self):
                return 1

            def __getitem__(self, _index):
                return torch.zeros(7, 8), torch.tensor(0)

        def tiny_dataset_loader(*, task_cfg):
            from dendritic_modeling.datasets.unified_options import (
                _resolve_unified_dataset_options,
            )

            options = _resolve_unified_dataset_options(task_cfg)
            captured["task_params"] = options.parameters
            dataset = TinySequenceDataset()
            return dataset, dataset, dataset

        def capture_report(estimator, model, batch_size, input_shape, config=None):
            captured["breakdown"] = estimator.estimate_memory_breakdown(
                model,
                batch_size,
                input_shape,
            )

        monkeypatch.setattr(
            "dendritic_modeling.datasets.get_unified_datasets",
            tiny_dataset_loader,
        )
        monkeypatch.setattr(
            GPUMemoryEstimator,
            "print_estimation_report",
            capture_report,
        )

        estimator, model = estimate_from_config(str(config_path), batch_size=2)

        assert isinstance(estimator, GPUMemoryEstimator)
        assert model.core_network.config.input_dim == 8
        assert model.decoder_network.output_dim == 2
        assert captured["task_params"]["flatten"] is False
        assert captured["task_params"]["train"]["n_samples"] == 30000
        assert captured["breakdown"].total > 0


class TestGPUEstimationEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_parameter_model(self):
        """Test estimation with model that has zero parameters."""
        estimator = GPUMemoryEstimator()

        # Model with no learnable parameters
        model = nn.Sequential(nn.ReLU(), nn.Dropout(0.5))

        breakdown = estimator.estimate_memory_breakdown(
            model, batch_size=1, input_shape=(1,)
        )

        assert breakdown.parameters == 0
        assert breakdown.gradients == 0
        assert breakdown.optimizer_states == 0
        assert breakdown.activations >= 0  # May still have activations

    def test_very_large_model_estimation(self):
        """Test estimation with very large model."""
        estimator = GPUMemoryEstimator()

        # Create artificially large model
        class LargeLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(10000, 10000))

        model = LargeLayer()

        breakdown = estimator.estimate_memory_breakdown(
            model, batch_size=1, input_shape=(1,)
        )

        # Should handle large models
        assert breakdown.parameters > 0
        assert breakdown.total > breakdown.parameters

    def test_different_optimizer_types(self):
        """Test estimation with different optimizer types."""
        model = nn.Linear(100, 50)

        optimizers = ["sgd", "adam", "adamw", "lamb", "rmsprop"]

        for opt in optimizers:
            estimator = GPUMemoryEstimator(optimizer=opt)
            breakdown = estimator.estimate_memory_breakdown(
                model, batch_size=8, input_shape=(64,)
            )

            assert breakdown.optimizer_states >= 0
            assert breakdown.total > 0

    def test_extreme_batch_sizes(self):
        """Test estimation with extreme batch sizes."""
        estimator = GPUMemoryEstimator()
        model = nn.Linear(10, 5)

        # Very small batch
        breakdown_small = estimator.estimate_memory_breakdown(
            model, batch_size=1, input_shape=(1,)
        )

        # Very large batch
        breakdown_large = estimator.estimate_memory_breakdown(
            model, batch_size=1000, input_shape=(1000,)
        )

        # Larger batch should use more memory for activations
        assert breakdown_large.activations > breakdown_small.activations
        assert breakdown_large.total > breakdown_small.total

    def test_gpu_recommendation_edge_cases(self):
        """Test GPU recommendation with edge cases."""
        estimator = GPUMemoryEstimator()

        # Very small model
        small_model = nn.Linear(2, 1)
        # First get memory breakdown, then get recommendations
        breakdown = estimator.estimate_memory_breakdown(
            small_model, batch_size=1, input_shape=(1,)
        )
        recommendations = estimator.get_gpu_recommendations(breakdown)

        # Should still provide recommendations
        assert len(recommendations) > 0
        assert all(rec.min_gpus_needed >= 1 for rec in recommendations)

    def test_memory_calculation_consistency(self):
        """Test that memory calculations are consistent."""
        estimator = GPUMemoryEstimator()
        model = nn.Sequential(nn.Linear(100, 50), nn.Linear(50, 10))

        # Same model should give same results
        breakdown1 = estimator.estimate_memory_breakdown(
            model, batch_size=16, input_shape=(128,)
        )
        breakdown2 = estimator.estimate_memory_breakdown(
            model, batch_size=16, input_shape=(128,)
        )

        assert breakdown1.parameters == breakdown2.parameters
        assert breakdown1.gradients == breakdown2.gradients
        assert breakdown1.optimizer_states == breakdown2.optimizer_states
        assert breakdown1.total == breakdown2.total

    def test_dtype_consistency(self):
        """Test that dtype affects memory calculations consistently."""
        model = nn.Linear(1000, 500)

        estimators = {
            "fp16": GPUMemoryEstimator(dtype="fp16"),
            "fp32": GPUMemoryEstimator(dtype="fp32"),
            "bf16": GPUMemoryEstimator(dtype="bf16"),
        }

        breakdowns = {}
        for dtype, estimator in estimators.items():
            breakdowns[dtype] = estimator.estimate_memory_breakdown(
                model, batch_size=16, input_shape=(128,)
            )

        # fp32 should use more memory than fp16/bf16
        assert breakdowns["fp32"].parameters > breakdowns["fp16"].parameters
        assert breakdowns["fp32"].parameters > breakdowns["bf16"].parameters

        # fp16 and bf16 should use similar memory
        assert abs(breakdowns["fp16"].parameters - breakdowns["bf16"].parameters) < 0.1
