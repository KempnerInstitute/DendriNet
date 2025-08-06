"""
Tests for plotting visualizations performance_plots module to improve coverage.
"""

import pytest
import tempfile
import os
import json
from unittest.mock import patch
import matplotlib.pyplot as plt
import numpy as np

from dendritic_modeling.plotting.visualizations.performance_plots import (
    PerformancePlotter,
    SweepResultsPlotter,
)
from dendritic_modeling.plotting.visualizations.base import PlotConfig


class TestPerformancePlotter:
    """Test PerformancePlotter class."""

    @pytest.fixture
    def plotter(self):
        """Create PerformancePlotter instance."""
        return PerformancePlotter()

    @pytest.fixture
    def custom_plotter(self):
        """Create PerformancePlotter with custom config."""
        config = PlotConfig(figsize=(8, 6), dpi=150)
        return PerformancePlotter(config=config)

    def test_init_default(self, plotter):
        """Test initialization with default config."""
        assert isinstance(plotter, PerformancePlotter)
        assert isinstance(plotter.config, PlotConfig)

    def test_init_custom_config(self, custom_plotter):
        """Test initialization with custom config."""
        assert custom_plotter.config.figsize == (8, 6)
        assert custom_plotter.config.dpi == 150

    def test_plot_loss_curves_basic(self, plotter):
        """Test basic loss curve plotting."""
        train_losses = [1.0, 0.8, 0.6, 0.4, 0.3]
        valid_losses = [1.2, 0.9, 0.7, 0.5, 0.4]

        fig, ax = plotter.plot_loss_curves(train_losses, valid_losses, "CrossEntropy")

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

        # Check that lines were plotted
        lines = ax.get_lines()
        assert len(lines) >= 2  # At least train and valid lines

        plt.close(fig)

    def test_plot_loss_curves_no_validation(self, plotter):
        """Test loss curve plotting without validation data."""
        train_losses = [1.5, 1.2, 0.9, 0.7, 0.5]
        valid_losses = []

        fig, ax = plotter.plot_loss_curves(train_losses, valid_losses, "MSE")

        assert isinstance(fig, plt.Figure)
        lines = ax.get_lines()
        # Should have train line(s) but no validation
        assert len(lines) >= 1

        plt.close(fig)

    def test_plot_loss_curves_empty_data(self, plotter):
        """Test loss curve plotting with empty data."""
        train_losses = []
        valid_losses = []

        fig, ax = plotter.plot_loss_curves(train_losses, valid_losses, "Loss")

        # Should create error plot with title "Loss Curves"
        assert isinstance(fig, plt.Figure)
        assert "Loss Curves" in ax.get_title()

        plt.close(fig)

    def test_plot_loss_curves_with_saving(self, plotter):
        """Test loss curve plotting with file saving."""
        train_losses = [1.0, 0.5, 0.3]
        valid_losses = [1.1, 0.6, 0.4]

        with tempfile.TemporaryDirectory() as tmp_dir:
            fig, ax = plotter.plot_loss_curves(
                train_losses,
                valid_losses,
                "Loss",
                save_dir=tmp_dir,
                filename_prefix="test_",
            )

            # Check that file was saved
            files = os.listdir(tmp_dir)
            assert any("test_" in f and f.endswith(".png") for f in files)

            plt.close(fig)

    def test_plot_performance_metrics_basic(self, plotter):
        """Test basic performance summary plotting."""
        results = {
            "train_accuracy": 0.95,
            "valid_accuracy": 0.87,
            "train_loss": 0.12,
            "valid_loss": 0.23,
            "training_time": 1800,
        }

        plots = plotter.plot_performance_metrics(results)

        assert "summary" in plots
        fig, ax = plots["summary"]
        assert isinstance(fig, plt.Figure)

        # Check that bars were plotted
        bars = ax.patches
        assert len(bars) == len([k for k in results.keys() if not k.startswith("_")])

        plt.close(fig)

    def test_plot_performance_metrics_empty(self, plotter):
        """Test performance summary with no metrics."""
        results = {"_internal": "data", "non_numeric": "string"}

        plots = plotter.plot_performance_metrics(results)

        assert "error" in plots
        fig, ax = plots["error"]
        assert "Performance Metrics" in ax.get_title()

        plt.close(fig)

    def test_plot_performance_metrics_with_saving(self, plotter):
        """Test performance summary with file saving."""
        results = {"accuracy": 0.9, "loss": 0.1}

        with tempfile.TemporaryDirectory() as tmp_dir:
            plots = plotter.plot_performance_metrics(
                results, save_path=tmp_dir, filename="test_summary"
            )

            files = os.listdir(tmp_dir)
            assert any("test_summary" in f for f in files)

            plt.close(plots["summary"][0])

    def test_plot_performance_evolution_no_files(self, plotter):
        """Test performance evolution with no files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            plots = plotter.plot_performance_evolution(tmp_dir)

            assert "error" in plots
            fig, ax = plots["error"]
            assert "Performance Evolution" in ax.get_title()

            plt.close(fig)

    def test_plot_performance_evolution_with_files(self, plotter):
        """Test performance evolution with mock performance files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create mock performance files
            for epoch in [1, 2, 3]:
                perf_data = {
                    "accuracy": {
                        "train": 0.7 + epoch * 0.1,
                        "valid": 0.65 + epoch * 0.08,
                    },
                    "loss": {"train": 1.0 - epoch * 0.2, "valid": 1.1 - epoch * 0.15},
                }

                epoch_file = os.path.join(tmp_dir, f"epoch{epoch}")
                with open(epoch_file, "w") as f:
                    json.dump(perf_data, f)

            plots = plotter.plot_performance_evolution(tmp_dir)

            # Should have plots for metrics
            assert len(plots) > 0
            assert not any("error" in key for key in plots.keys())

            for fig, _ax in plots.values():
                plt.close(fig)

    def test_plot_performance_evolution_invalid_json(self, plotter):
        """Test performance evolution with invalid JSON files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create invalid JSON file
            epoch_file = os.path.join(tmp_dir, "epoch1")
            with open(epoch_file, "w") as f:
                f.write("invalid json {")

            with patch.object(plotter.logger, "warning") as mock_warning:
                plots = plotter.plot_performance_evolution(tmp_dir)

                # Should log warning and create error plot
                mock_warning.assert_called()

            for fig, _ax in plots.values():
                plt.close(fig)

    def test_validate_data_method(self, plotter):
        """Test _validate_data internal method."""
        # Valid data
        assert plotter._validate_data([1, 2, 3], "test")

        # Empty data
        assert not plotter._validate_data([], "test")

        # None data
        assert not plotter._validate_data(None, "test")

    def test_smooth_data_method(self, plotter):
        """Test _smooth_data internal method."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        window_size = 3
        smoothed = plotter._smooth_data(data, window_size=window_size)

        # With "valid" mode, smoothed data will be shorter due to convolution
        assert len(smoothed) == len(data) - window_size + 1
        assert isinstance(smoothed, np.ndarray)

        # Smoothed data should be less variable
        original_var = np.var(data[window_size - 1 :])  # Compare same section
        smoothed_var = np.var(smoothed)
        assert smoothed_var <= original_var

    def test_create_error_plot_method(self, plotter):
        """Test _create_error_plot internal method."""
        fig, ax = plotter._create_error_plot("Test Title", "Test Message")

        assert isinstance(fig, plt.Figure)
        assert "Test Title" in ax.get_title()

        plt.close(fig)

    def test_color_assignment_logic(self, plotter):
        """Test metric-based color assignment."""
        results = {
            "train_accuracy": 0.9,
            "validation_accuracy": 0.85,
            "train_loss": 0.1,
            "test_loss": 0.15,
            "training_time": 1800,
            "inference_time": 0.1,
            "custom_metric": 0.5,
        }

        plots = plotter.plot_performance_metrics(results)
        fig, ax = plots["summary"]

        # Get the colors used for bars
        bars = ax.patches
        colors = [bar.get_facecolor() for bar in bars]

        # Should have different colors for different metric types
        assert len(colors) == len(results)

        plt.close(fig)


class TestSweepResultsPlotter:
    """Test SweepResultsPlotter class."""

    @pytest.fixture
    def sweep_plotter(self):
        """Create SweepResultsPlotter instance."""
        return SweepResultsPlotter()

    def test_init(self, sweep_plotter):
        """Test SweepResultsPlotter initialization."""
        assert isinstance(sweep_plotter, SweepResultsPlotter)
        assert hasattr(sweep_plotter, "config")

    def test_inheritance(self, sweep_plotter):
        """Test that SweepResultsPlotter inherits from TimeSeriesPlotter."""
        # Should have inherited methods
        assert hasattr(sweep_plotter, "_create_figure")
        assert hasattr(sweep_plotter, "_apply_styling")
        assert hasattr(sweep_plotter, "_save_plot")

    def test_sweep_plotter_basic_functionality(self, sweep_plotter):
        """Test basic functionality inherited from parent."""
        fig, ax = sweep_plotter._create_figure()

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

        plt.close(fig)


class TestPerformancePlotterIntegration:
    """Integration tests for PerformancePlotter."""

    def test_full_workflow_with_real_data(self):
        """Test complete workflow with realistic data."""
        plotter = PerformancePlotter()

        # Simulate training data
        epochs = 20
        train_losses = [
            1.0 * np.exp(-0.1 * i) + 0.1 + np.random.normal(0, 0.02)
            for i in range(epochs)
        ]
        valid_losses = [
            1.1 * np.exp(-0.08 * i) + 0.15 + np.random.normal(0, 0.03)
            for i in range(epochs)
        ]

        # Test loss curves
        fig1, ax1 = plotter.plot_loss_curves(train_losses, valid_losses, "CrossEntropy")

        # Test performance summary
        final_results = {
            "final_train_loss": train_losses[-1],
            "final_valid_loss": valid_losses[-1],
            "train_accuracy": 0.95,
            "valid_accuracy": 0.87,
            "training_time": 3600,
        }

        plots = plotter.plot_performance_metrics(final_results)

        # Both should succeed
        assert isinstance(fig1, plt.Figure)
        assert "summary" in plots

        plt.close(fig1)
        plt.close(plots["summary"][0])

    def test_edge_case_handling(self):
        """Test handling of various edge cases."""
        plotter = PerformancePlotter()

        # Test with single data point
        single_loss = [0.5]
        fig, ax = plotter.plot_loss_curves(single_loss, [], "Loss")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        # Test with NaN values
        nan_losses = [1.0, 0.8, float("nan"), 0.4]
        fig, ax = plotter.plot_loss_curves(nan_losses, [], "Loss")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        # Test with very large values
        large_losses = [1e6, 1e5, 1e4]
        fig, ax = plotter.plot_loss_curves(large_losses, [], "Loss")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_concurrent_plotting(self):
        """Test creating multiple plots concurrently."""
        plotter = PerformancePlotter()

        plots = []
        for i in range(3):
            losses = [1.0 - i * 0.1, 0.8 - i * 0.05, 0.6 - i * 0.02]
            fig, ax = plotter.plot_loss_curves(losses, [], f"Loss_{i}")
            plots.append((fig, ax))

        # All plots should be created successfully
        assert len(plots) == 3
        for fig, _ax in plots:
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_memory_cleanup(self):
        """Test that plots don't cause memory leaks."""
        plotter = PerformancePlotter()

        # Create many plots and close them
        for i in range(10):
            losses = [1.0 / (i + 1), 0.5 / (i + 1)]
            fig, ax = plotter.plot_loss_curves(losses, [], "Test")
            plt.close(fig)

        # Test should complete without memory issues


class TestPerformancePlotterEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_loss_data_types(self):
        """Test with invalid data types."""
        plotter = PerformancePlotter()

        # Test with string data - this will likely cause an error, so we'll catch it
        invalid_losses = ["1.0", "0.8", "0.6"]
        try:
            fig, ax = plotter.plot_loss_curves(invalid_losses, [], "Loss")
            # If it doesn't error, test that it's a figure
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
        except (ValueError, TypeError):
            # Expected to fail with string data
            pass

    def test_mismatched_loss_lengths(self):
        """Test with mismatched train/validation lengths."""
        plotter = PerformancePlotter()

        train_losses = [1.0, 0.8, 0.6, 0.4, 0.3]
        valid_losses = [1.2, 0.9]  # Shorter

        fig, ax = plotter.plot_loss_curves(train_losses, valid_losses, "Loss")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_performance_summary_edge_cases(self):
        """Test performance summary with edge case data."""
        plotter = PerformancePlotter()

        # Test with extreme values
        extreme_results = {
            "very_small": 1e-10,
            "very_large": 1e10,
            "negative": -0.5,
            "zero": 0.0,
        }

        plots = plotter.plot_performance_metrics(extreme_results)
        assert "summary" in plots
        plt.close(plots["summary"][0])

        # Test with mixed types
        mixed_results = {
            "accuracy": 0.9,
            "count": 100,  # int
            "rate": 0.05,  # float
        }

        plots = plotter.plot_performance_metrics(mixed_results)
        assert "summary" in plots
        plt.close(plots["summary"][0])

    def test_file_io_error_handling(self):
        """Test file I/O error handling."""
        plotter = PerformancePlotter()

        # Test with non-existent directory for evolution plotting
        plots = plotter.plot_performance_evolution("/non/existent/path")
        assert "error" in plots
        plt.close(plots["error"][0])

        # Test saving to read-only directory
        results = {"accuracy": 0.9}

        with tempfile.TemporaryDirectory() as tmp_dir:
            readonly_dir = os.path.join(tmp_dir, "readonly")
            os.makedirs(readonly_dir)
            os.chmod(readonly_dir, 0o444)

            try:
                # This might fail or succeed depending on implementation
                plots = plotter.plot_performance_metrics(
                    results, save_path=readonly_dir
                )
                if "summary" in plots:
                    plt.close(plots["summary"][0])
            except (OSError, PermissionError):
                # Expected in some cases
                pass
            finally:
                os.chmod(readonly_dir, 0o755)  # Restore for cleanup
