"""
Tests for plotting utilities to improve coverage.
"""

import pytest
import matplotlib.pyplot as plt
from unittest.mock import patch

# Import plotting utilities
from dendritic_modeling.plotting.core.utils import (
    save_plot,
    create_figure_grid,
    format_axis_labels,
    apply_plot_style,
)


class TestPlottingUtils:
    """Test plotting utility functions."""

    def test_save_plot_png(self, tmp_path):
        """Test saving plot as PNG."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        save_path = tmp_path / "test_plot.png"
        save_plot(fig, str(save_path))

        assert save_path.exists()
        plt.close(fig)

    def test_save_plot_pdf(self, tmp_path):
        """Test saving plot as PDF."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        save_path = tmp_path / "test_plot.pdf"
        save_plot(fig, str(save_path))

        assert save_path.exists()
        plt.close(fig)

    def test_create_figure_grid_single(self):
        """Test creating a single subplot grid."""
        fig, axes = create_figure_grid(1, 1, figsize=(8, 6))

        assert fig is not None
        assert axes is not None
        assert fig.get_size_inches()[0] == 8
        assert fig.get_size_inches()[1] == 6
        plt.close(fig)

    def test_create_figure_grid_multiple(self):
        """Test creating multiple subplot grid."""
        fig, axes = create_figure_grid(2, 3, figsize=(12, 8))

        assert fig is not None
        assert axes.shape == (2, 3)
        plt.close(fig)

    def test_format_axis_labels(self):
        """Test formatting axis labels."""
        fig, ax = plt.subplots()

        format_axis_labels(ax, xlabel="X Axis", ylabel="Y Axis", title="Test Title")

        assert ax.get_xlabel() == "X Axis"
        assert ax.get_ylabel() == "Y Axis"
        assert ax.get_title() == "Test Title"
        plt.close(fig)

    def test_apply_plot_style_default(self):
        """Test applying default plot style."""
        with patch("matplotlib.pyplot.style.use") as mock_style:
            apply_plot_style()
            mock_style.assert_called_once()

    def test_apply_plot_style_custom(self):
        """Test applying custom plot style."""
        with patch("matplotlib.pyplot.style.use") as mock_style:
            apply_plot_style("seaborn")
            mock_style.assert_called_with("seaborn")


class TestPlottingUtilsEdgeCases:
    """Test edge cases and error handling for plotting utilities."""

    def test_save_plot_invalid_path(self):
        """Test saving plot to invalid path."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        with pytest.raises((OSError, IOError)):
            save_plot(fig, "/invalid/path/test.png")

        plt.close(fig)

    def test_create_figure_grid_zero_size(self):
        """Test creating grid with zero size."""
        with pytest.raises(ValueError):
            create_figure_grid(0, 1)

    def test_format_axis_labels_none_values(self):
        """Test formatting with None values."""
        fig, ax = plt.subplots()

        # Should not raise errors with None values
        format_axis_labels(ax, xlabel=None, ylabel=None, title=None)

        plt.close(fig)

    def test_apply_plot_style_invalid_style(self):
        """Test applying invalid plot style."""
        with patch("matplotlib.pyplot.style.use") as mock_style:
            mock_style.side_effect = OSError("Style not found")

            # Should handle the error gracefully
            apply_plot_style("nonexistent_style")
