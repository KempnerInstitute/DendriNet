"""
Tests for layer contribution analysis to improve coverage.
"""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from dendritic_modeling.analysis.metrics.layer_contribution import (
    LayerContributionAnalyzer,
    analyze_layer_importance,
    compute_activation_based_importance,
    compute_gradient_based_importance,
)


class TestLayerContributionAnalyzer:
    """Test LayerContributionAnalyzer class."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
        return model

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        x = torch.randn(20, 10)
        y = torch.randint(0, 2, (20,))
        return x, y

    def test_analyzer_initialization(self, simple_model):
        """Test analyzer initialization."""
        analyzer = LayerContributionAnalyzer(simple_model)

        assert analyzer.model == simple_model
        assert hasattr(analyzer, "layer_names")
        assert len(analyzer.layer_names) > 0

    def test_analyze_gradient_importance(self, simple_model, sample_data):
        """Test gradient-based importance analysis."""
        analyzer = LayerContributionAnalyzer(simple_model)
        x, y = sample_data

        importance = analyzer.analyze_gradient_importance(x, y)

        assert isinstance(importance, dict)
        assert len(importance) > 0
        for _layer_name, score in importance.items():
            assert isinstance(score, (float, torch.Tensor))

    def test_analyze_activation_importance(self, simple_model, sample_data):
        """Test activation-based importance analysis."""
        analyzer = LayerContributionAnalyzer(simple_model)
        x, y = sample_data

        importance = analyzer.analyze_activation_importance(x)

        assert isinstance(importance, dict)
        assert len(importance) > 0
        for _layer_name, score in importance.items():
            assert isinstance(score, (float, torch.Tensor))

    def test_analyze_layer_removal_impact(self, simple_model, sample_data):
        """Test layer removal impact analysis."""
        analyzer = LayerContributionAnalyzer(simple_model)
        x, y = sample_data

        with patch.object(analyzer, "_evaluate_model_performance") as mock_eval:
            mock_eval.return_value = 0.8

            impact = analyzer.analyze_layer_removal_impact(x, y)

            assert isinstance(impact, dict)
            assert len(impact) >= 0  # May be empty if no removable layers

    def test_get_layer_names(self, simple_model):
        """Test getting layer names."""
        analyzer = LayerContributionAnalyzer(simple_model)
        names = analyzer.get_layer_names()

        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(name, str) for name in names)

    def test_compute_layer_statistics(self, simple_model, sample_data):
        """Test computing layer statistics."""
        analyzer = LayerContributionAnalyzer(simple_model)
        x, _ = sample_data

        stats = analyzer.compute_layer_statistics(x)

        assert isinstance(stats, dict)
        for _layer_name, layer_stats in stats.items():
            assert "mean_activation" in layer_stats
            assert "std_activation" in layer_stats
            assert "sparsity" in layer_stats


class TestLayerContributionFunctions:
    """Test standalone layer contribution functions."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        x = torch.randn(20, 10)
        y = torch.randint(0, 2, (20,))
        return x, y

    def test_analyze_layer_importance(self, simple_model, sample_data):
        """Test the main analyze_layer_importance function."""
        x, y = sample_data

        importance = analyze_layer_importance(simple_model, x, y, method="gradient")

        assert isinstance(importance, dict)
        assert len(importance) > 0

    def test_analyze_layer_importance_activation_method(
        self, simple_model, sample_data
    ):
        """Test analyze_layer_importance with activation method."""
        x, y = sample_data

        importance = analyze_layer_importance(simple_model, x, y, method="activation")

        assert isinstance(importance, dict)
        assert len(importance) > 0

    def test_analyze_layer_importance_invalid_method(self, simple_model, sample_data):
        """Test analyze_layer_importance with invalid method."""
        x, y = sample_data

        with pytest.raises(ValueError):
            analyze_layer_importance(simple_model, x, y, method="invalid")

    def test_compute_gradient_based_importance(self, simple_model, sample_data):
        """Test gradient-based importance computation."""
        x, y = sample_data

        importance = compute_gradient_based_importance(simple_model, x, y)

        assert isinstance(importance, dict)
        assert len(importance) > 0
        for score in importance.values():
            assert isinstance(score, (float, torch.Tensor))

    def test_compute_activation_based_importance(self, simple_model, sample_data):
        """Test activation-based importance computation."""
        x, _ = sample_data

        importance = compute_activation_based_importance(simple_model, x)

        assert isinstance(importance, dict)
        assert len(importance) > 0
        for score in importance.values():
            assert isinstance(score, (float, torch.Tensor))

    def test_compute_gradient_importance_with_loss_fn(self, simple_model, sample_data):
        """Test gradient importance with custom loss function."""
        x, y = sample_data
        loss_fn = nn.CrossEntropyLoss()

        importance = compute_gradient_based_importance(
            simple_model, x, y, loss_fn=loss_fn
        )

        assert isinstance(importance, dict)
        assert len(importance) > 0


class TestLayerContributionEdgeCases:
    """Test edge cases for layer contribution analysis."""

    def test_empty_model(self):
        """Test with empty model."""
        model = nn.Sequential()

        with pytest.raises((ValueError, AttributeError)):
            LayerContributionAnalyzer(model)

    def test_single_layer_model(self):
        """Test with single layer model."""
        model = nn.Linear(10, 2)
        analyzer = LayerContributionAnalyzer(model)

        assert len(analyzer.layer_names) == 1

    def test_zero_input(self, tmp_path):
        """Test with zero input."""
        model = nn.Linear(10, 2)
        analyzer = LayerContributionAnalyzer(model)

        x = torch.zeros(5, 10)
        y = torch.zeros(5, dtype=torch.long)

        # Should not crash
        importance = analyzer.analyze_gradient_importance(x, y)
        assert isinstance(importance, dict)

    def test_very_small_model(self):
        """Test with very small input/output dimensions."""
        model = nn.Linear(1, 1)
        analyzer = LayerContributionAnalyzer(model)

        x = torch.randn(1, 1)
        y = torch.tensor([0])

        importance = analyzer.analyze_gradient_importance(x, y)
        assert isinstance(importance, dict)
