"""
Tests for analysis core base module to improve coverage.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.models.base import BaseModel


class ConcreteAnalyzer(AbstractAnalyzer):
    """Concrete implementation of AbstractAnalyzer for testing."""

    def analyze(
        self,
        model: BaseModel,
        data: torch.utils.data.Dataset,
        device: str = "cpu",
        **kwargs,
    ) -> dict[str, any]:
        """Simple implementation that returns mock analysis results."""
        return {
            "accuracy": 0.85,
            "loss": 0.15,
            "predictions": torch.randn(10, 5),
            "metadata": {"device": device, "num_samples": len(data)},
        }


class TestAbstractAnalyzer:
    """Test AbstractAnalyzer class."""

    def test_init_default_logger(self):
        """Test initialization with default logger name."""
        analyzer = ConcreteAnalyzer()

        assert analyzer.logger.name == "ConcreteAnalyzer"
        assert isinstance(analyzer.logger, logging.Logger)

    def test_init_custom_logger(self):
        """Test initialization with custom logger name."""
        analyzer = ConcreteAnalyzer(logger_name="custom_analyzer")

        assert analyzer.logger.name == "custom_analyzer"

    def test_analyze_method_implementation(self):
        """Test that concrete implementation works."""
        analyzer = ConcreteAnalyzer()
        mock_model = Mock(spec=BaseModel)
        mock_data = Mock()
        mock_data.__len__ = Mock(return_value=100)

        results = analyzer.analyze(mock_model, mock_data, device="cuda")

        assert isinstance(results, dict)
        assert "accuracy" in results
        assert "loss" in results
        assert "predictions" in results
        assert "metadata" in results
        assert results["metadata"]["device"] == "cuda"
        assert results["metadata"]["num_samples"] == 100

    def test_abstract_methods_not_implemented(self):
        """Test that abstract methods raise TypeError when not implemented."""
        with pytest.raises(TypeError):
            AbstractAnalyzer()

    def test_save_results_basic(self):
        """Test basic results saving functionality."""
        analyzer = ConcreteAnalyzer()
        results = {
            "accuracy": 0.92,
            "loss": 0.08,
            "epoch": 50,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "results.json")

            with patch.object(analyzer.logger, "info") as mock_info:
                analyzer.save_results(results, save_path)

                mock_info.assert_called_once()
                assert "Results saved to" in mock_info.call_args[0][0]

            # Verify file was created and contains correct data
            assert os.path.exists(save_path)
            with open(save_path) as f:
                saved_data = json.load(f)
                assert saved_data == results

    def test_save_results_creates_directory(self):
        """Test that save_results creates parent directories."""
        analyzer = ConcreteAnalyzer()
        results = {"test": "data"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            nested_path = os.path.join(tmp_dir, "nested", "deep", "results.json")

            analyzer.save_results(results, nested_path)

            assert os.path.exists(nested_path)
            with open(nested_path) as f:
                saved_data = json.load(f)
                assert saved_data == results

    def test_save_results_with_tensors(self):
        """Test saving results containing torch tensors."""
        analyzer = ConcreteAnalyzer()
        tensor_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        results = {
            "weights": tensor_data,
            "scalar": 0.5,
            "list": [1, 2, 3],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "tensor_results.json")

            analyzer.save_results(results, save_path)

            with open(save_path) as f:
                saved_data = json.load(f)
                assert saved_data["weights"] == [[1.0, 2.0], [3.0, 4.0]]
                assert saved_data["scalar"] == 0.5
                assert saved_data["list"] == [1, 2, 3]

    def test_save_results_error_handling(self):
        """Test error handling in save_results."""
        analyzer = ConcreteAnalyzer()
        results = {"test": "data"}

        with patch.object(analyzer.logger, "error") as mock_error:
            with pytest.raises((OSError, IOError, PermissionError)):
                analyzer.save_results(results, "/invalid/path/that/cannot/be/created")

            mock_error.assert_called_once()
            assert "Failed to save results" in mock_error.call_args[0][0]

    def test_make_json_serializable_tensor(self):
        """Test tensor conversion to JSON-serializable format."""
        analyzer = ConcreteAnalyzer()
        tensor = torch.tensor([1.0, 2.0, 3.0])

        result = analyzer._make_json_serializable(tensor)

        assert isinstance(result, list)
        assert result == [1.0, 2.0, 3.0]

    def test_make_json_serializable_dict(self):
        """Test dictionary conversion with nested tensors."""
        analyzer = ConcreteAnalyzer()
        data = {
            "tensor": torch.tensor([1.0, 2.0]),
            "scalar": 5,
            "nested": {"inner_tensor": torch.tensor([[1, 2], [3, 4]])},
        }

        result = analyzer._make_json_serializable(data)

        assert result["tensor"] == [1.0, 2.0]
        assert result["scalar"] == 5
        assert result["nested"]["inner_tensor"] == [[1, 2], [3, 4]]

    def test_make_json_serializable_list_tuple(self):
        """Test list and tuple conversion."""
        analyzer = ConcreteAnalyzer()
        data = [torch.tensor([1.0]), (torch.tensor([2.0]), 3)]

        result = analyzer._make_json_serializable(data)

        assert result == [[1.0], [[2.0], 3]]

    def test_make_json_serializable_custom_object(self):
        """Test custom object conversion."""
        analyzer = ConcreteAnalyzer()

        class CustomObject:
            def __init__(self):
                self.value = torch.tensor([1.0, 2.0])
                self.name = "test"

        obj = CustomObject()
        result = analyzer._make_json_serializable(obj)

        assert result["value"] == [1.0, 2.0]
        assert result["name"] == "test"

    def test_make_json_serializable_primitive_types(self):
        """Test that primitive types pass through unchanged."""
        analyzer = ConcreteAnalyzer()

        primitives = [
            42,
            3.14,
            "string",
            True,
            None,
        ]

        for primitive in primitives:
            result = analyzer._make_json_serializable(primitive)
            assert result == primitive

    def test_log_analysis_start_basic(self):
        """Test basic analysis start logging."""
        analyzer = ConcreteAnalyzer()

        with patch.object(analyzer.logger, "info") as mock_info:
            analyzer.log_analysis_start("test_analysis")

            mock_info.assert_called_once_with("Starting test_analysis analysis...")

    def test_log_analysis_start_with_params(self):
        """Test analysis start logging with parameters."""
        analyzer = ConcreteAnalyzer()

        with patch.object(analyzer.logger, "info") as mock_info:
            analyzer.log_analysis_start("detailed_analysis", param1="value1", param2=42)

            calls = mock_info.call_args_list
            assert len(calls) == 2
            assert "Starting detailed_analysis analysis..." in calls[0][0][0]
            assert "Analysis parameters:" in calls[1][0][0]
            assert "param1=value1" in calls[1][0][0]
            assert "param2=42" in calls[1][0][0]

    def test_log_analysis_start_empty_kwargs(self):
        """Test analysis start logging with empty kwargs."""
        analyzer = ConcreteAnalyzer()

        with patch.object(analyzer.logger, "info") as mock_info:
            analyzer.log_analysis_start("simple_analysis")

            mock_info.assert_called_once_with("Starting simple_analysis analysis...")

    def test_log_analysis_end_basic(self):
        """Test basic analysis end logging."""
        analyzer = ConcreteAnalyzer()

        with patch.object(analyzer.logger, "info") as mock_info:
            analyzer.log_analysis_end("test_analysis")

            mock_info.assert_called_once_with("Completed test_analysis analysis")

    def test_log_analysis_end_with_results_count(self):
        """Test analysis end logging with results count."""
        analyzer = ConcreteAnalyzer()

        with patch.object(analyzer.logger, "info") as mock_info:
            analyzer.log_analysis_end("test_analysis", num_results=150)

            mock_info.assert_called_once_with(
                "Completed test_analysis analysis with 150 results"
            )

    def test_log_analysis_end_zero_results(self):
        """Test analysis end logging with zero results."""
        analyzer = ConcreteAnalyzer()

        with patch.object(analyzer.logger, "info") as mock_info:
            analyzer.log_analysis_end("test_analysis", num_results=0)

            mock_info.assert_called_once_with(
                "Completed test_analysis analysis with 0 results"
            )


class TestAbstractAnalyzerEdgeCases:
    """Test edge cases for AbstractAnalyzer."""

    def test_save_results_complex_nested_structure(self):
        """Test saving deeply nested structures with mixed types."""
        analyzer = ConcreteAnalyzer()
        complex_data = {
            "level1": {
                "level2": {
                    "tensors": [
                        torch.tensor([1.0, 2.0]),
                        torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
                    ],
                    "mixed": (torch.tensor([7.0]), "string", 42),
                }
            },
            "arrays": [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "complex_results.json")

            analyzer.save_results(complex_data, save_path)

            with open(save_path) as f:
                saved_data = json.load(f)

            # Verify structure is preserved and tensors are converted
            assert saved_data["level1"]["level2"]["tensors"] == [
                [1.0, 2.0],
                [[3.0, 4.0], [5.0, 6.0]],
            ]
            assert saved_data["level1"]["level2"]["mixed"] == [[7.0], "string", 42]

    def test_make_json_serializable_circular_reference_protection(self):
        """Test handling of objects that might cause issues."""
        analyzer = ConcreteAnalyzer()

        # Test with mock object that has no __dict__
        mock_obj = Mock()
        mock_obj.__dict__ = {"value": 42}

        result = analyzer._make_json_serializable(mock_obj)
        assert isinstance(result, dict)

    def test_save_results_pathlib_path(self):
        """Test save_results with Path object."""
        analyzer = ConcreteAnalyzer()
        results = {"path_test": True}

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "pathlib_results.json"

            analyzer.save_results(results, str(save_path))

            assert save_path.exists()
            with open(save_path) as f:
                saved_data = json.load(f)
                assert saved_data == results

    def test_logger_hierarchy(self):
        """Test logger hierarchy and naming."""
        # Test with module-style naming
        analyzer = ConcreteAnalyzer(logger_name="module.submodule.analyzer")

        assert analyzer.logger.name == "module.submodule.analyzer"

        # Test logger propagation
        with patch("logging.getLogger") as mock_get_logger:
            ConcreteAnalyzer(logger_name="test.logger")
            mock_get_logger.assert_called_with("test.logger")

    def test_analyze_with_all_kwargs(self):
        """Test analyze method with various kwargs."""
        analyzer = ConcreteAnalyzer()
        mock_model = Mock(spec=BaseModel)
        mock_data = Mock()
        mock_data.__len__ = Mock(return_value=50)

        results = analyzer.analyze(
            mock_model,
            mock_data,
            device="cuda:1",
            batch_size=32,
            num_workers=4,
            custom_param="test_value",
        )

        assert results["metadata"]["device"] == "cuda:1"
        assert results["metadata"]["num_samples"] == 50

    def test_save_results_file_permissions(self):
        """Test save_results handles file permission issues gracefully."""
        analyzer = ConcreteAnalyzer()
        results = {"test": "data"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a read-only directory
            readonly_dir = os.path.join(tmp_dir, "readonly")
            os.makedirs(readonly_dir)
            os.chmod(readonly_dir, 0o444)  # Read-only

            save_path = os.path.join(readonly_dir, "results.json")

            with patch.object(analyzer.logger, "error") as mock_error:
                try:
                    with pytest.raises((OSError, IOError, PermissionError)):
                        analyzer.save_results(results, save_path)
                    mock_error.assert_called_once()
                finally:
                    # Restore permissions for cleanup
                    os.chmod(readonly_dir, 0o755)

    def test_make_json_serializable_numpy_arrays(self):
        """Test conversion of numpy arrays."""
        analyzer = ConcreteAnalyzer()
        data = {
            "numpy_array": np.array([1, 2, 3, 4]),
            "numpy_2d": np.array([[1, 2], [3, 4]]),
            "torch_tensor": torch.tensor([5, 6, 7]),
        }

        result = analyzer._make_json_serializable(data)

        # Numpy arrays don't get automatically converted, they remain as numpy arrays
        # The _make_json_serializable method only handles torch tensors directly
        assert isinstance(result["numpy_array"], np.ndarray)
        assert isinstance(result["numpy_2d"], np.ndarray)
        assert result["torch_tensor"] == [5, 6, 7]
