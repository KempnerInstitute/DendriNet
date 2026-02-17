"""
Tests for layer contribution analysis to improve coverage.
"""

from unittest.mock import Mock

import pytest
import torch

from dendritic_modeling.analysis.metrics.layer_contribution import (
    SingleLayerContributionAnalyzer,
)


class TestSingleLayerContributionAnalyzer:
    """Test SingleLayerContributionAnalyzer class."""

    @pytest.fixture
    def mock_params(self):
        """Create mock parameters for testing."""

        class MockParams:
            excitation_contribution = True
            inhibition_contribution = True
            both_contribution = True
            accuracy = True
            auc = True
            categorical_loglikelihood = True
            mse = True
            cosine_similarity = True

        return MockParams()

    @pytest.fixture
    def minimal_params(self):
        """Create minimal parameters for testing."""

        class MinimalParams:
            pass

        return MinimalParams()

    @pytest.fixture
    def sample_dataset(self):
        """Create sample TensorDataset for testing."""
        x = torch.randn(20, 10)
        y = torch.randint(0, 2, (20,))
        return torch.utils.data.TensorDataset(x, y)

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        return Mock()

    def test_analyzer_initialization(self, mock_params):
        """Test analyzer initialization with full parameters."""
        analyzer = SingleLayerContributionAnalyzer(mock_params)

        assert analyzer.excitation_contribution
        assert analyzer.inhibition_contribution
        assert analyzer.both_contribution
        assert analyzer.accuracy
        assert analyzer.auc
        assert analyzer.categorical_ll
        assert analyzer.mse
        assert analyzer.cosine_similarity
        assert hasattr(analyzer, "logger")

    def test_analyzer_with_minimal_params(self, minimal_params):
        """Test analyzer with minimal parameters."""
        analyzer = SingleLayerContributionAnalyzer(minimal_params)

        # Should use default values (False)
        assert not analyzer.excitation_contribution
        assert not analyzer.inhibition_contribution
        assert not analyzer.both_contribution
        assert not analyzer.accuracy
        assert not analyzer.auc
        assert not analyzer.categorical_ll
        assert not analyzer.mse
        assert not analyzer.cosine_similarity

    def test_analyzer_attributes(self, mock_params):
        """Test analyzer has all expected attributes."""
        analyzer = SingleLayerContributionAnalyzer(mock_params)

        expected_attrs = [
            "excitation_contribution",
            "inhibition_contribution",
            "both_contribution",
            "accuracy",
            "auc",
            "categorical_ll",
            "mse",
            "cosine_similarity",
            "logger",
        ]

        for attr in expected_attrs:
            assert hasattr(analyzer, attr)

    def test_analyzer_logger_setup(self, mock_params):
        """Test that logger is properly set up."""
        analyzer = SingleLayerContributionAnalyzer(mock_params)

        assert (
            analyzer.logger.name
            == "dendritic_modeling.analysis.metrics.layer_contribution"
        )

    def test_analyze_method_exists(self, mock_params):
        """Test that the analyze method exists."""
        analyzer = SingleLayerContributionAnalyzer(mock_params)

        assert hasattr(analyzer, "analyze")
        assert callable(analyzer.analyze)

    def test_evaluate_layer_contribution_method_exists(self, mock_params):
        """Test that layer evaluation method exists."""
        analyzer = SingleLayerContributionAnalyzer(mock_params)

        assert hasattr(analyzer, "evaluate_layer_contribution")
        assert callable(analyzer.evaluate_layer_contribution)

    def test_selective_parameter_access(self):
        """Test accessing parameters that may not exist."""

        class PartialParams:
            accuracy = True
            # Missing other parameters

        params = PartialParams()
        analyzer = SingleLayerContributionAnalyzer(params)

        assert analyzer.accuracy
        assert not analyzer.auc  # Should default to False
        assert not analyzer.mse  # Should default to False

    def test_boolean_parameter_handling(self, mock_params):
        """Test that boolean parameters are handled correctly."""
        analyzer = SingleLayerContributionAnalyzer(mock_params)

        # All should be boolean values
        assert isinstance(analyzer.excitation_contribution, bool)
        assert isinstance(analyzer.inhibition_contribution, bool)
        assert isinstance(analyzer.both_contribution, bool)
        assert isinstance(analyzer.accuracy, bool)
        assert isinstance(analyzer.auc, bool)
        assert isinstance(analyzer.categorical_ll, bool)
        assert isinstance(analyzer.mse, bool)
        assert isinstance(analyzer.cosine_similarity, bool)

    def test_parameter_getattr_fallback(self):
        """Test getattr with default values works correctly."""

        class EmptyParams:
            pass

        params = EmptyParams()
        analyzer = SingleLayerContributionAnalyzer(params)

        # All should be False due to getattr default
        assert not analyzer.excitation_contribution
        assert not analyzer.inhibition_contribution
        assert not analyzer.both_contribution

    def test_mixed_parameter_configuration(self):
        """Test with mixed parameter configuration."""

        class MixedParams:
            excitation_contribution = True
            accuracy = True
            mse = False
            # Missing other parameters

        params = MixedParams()
        analyzer = SingleLayerContributionAnalyzer(params)

        assert analyzer.excitation_contribution
        assert analyzer.accuracy
        assert not analyzer.mse
        assert not analyzer.inhibition_contribution  # Default
        assert not analyzer.auc  # Default

    def test_string_parameter_handling(self):
        """Test handling of string parameters that should be boolean."""

        class StringParams:
            categorical_loglikelihood = "true"  # String instead of bool

        params = StringParams()
        analyzer = SingleLayerContributionAnalyzer(params)

        # getattr should return the string, not convert to bool
        assert analyzer.categorical_ll == "true"

    def test_none_parameter_handling(self):
        """Test handling of None parameters."""

        class NoneParams:
            accuracy = None
            auc = True

        params = NoneParams()
        analyzer = SingleLayerContributionAnalyzer(params)

        assert analyzer.accuracy is None
        assert analyzer.auc
        assert not analyzer.mse  # Default for missing param

    def test_analyzer_initialization_edge_cases(self):
        """Test analyzer initialization with edge case parameters."""

        # Test with numeric values
        class NumericParams:
            accuracy = 1
            auc = 0
            mse = 1.0

        params = NumericParams()
        analyzer = SingleLayerContributionAnalyzer(params)

        assert analyzer.accuracy == 1
        assert analyzer.auc == 0
        assert analyzer.mse == 1.0
