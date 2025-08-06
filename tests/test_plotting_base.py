"""
Tests for plotting base module to improve coverage.
"""

import os
import tempfile

import matplotlib.pyplot as plt

from dendritic_modeling.plotting.visualizations.base import (
    BasePlotter,
    ComparisonPlotter,
    DistributionPlotter,
    PlotConfig,
    TimeSeriesPlotter,
)


class TestPlotConfig:
    """Test PlotConfig class."""

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        config = PlotConfig()

        assert config.figsize == (10, 6)
        assert config.dpi == 300
        assert isinstance(config.fontsize, dict)
        assert isinstance(config.colors, dict)
        assert isinstance(config.alpha, dict)
        assert isinstance(config.grid, dict)
        assert config.save_format == "png"
        assert config.bbox_inches == "tight"

    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        custom_config = {
            "figsize": (12, 8),
            "dpi": 150,
            "fontsize": {"title": 18, "label": 16},
            "colors": {"primary": "#ff0000"},
        }

        config = PlotConfig(**custom_config)

        assert config.figsize == (12, 8)
        assert config.dpi == 150
        assert config.fontsize["title"] == 18
        assert config.fontsize["label"] == 16
        assert config.colors["primary"] == "#ff0000"

    def test_config_merge_with_defaults(self):
        """Test that custom config merges with defaults."""
        config = PlotConfig(figsize=(8, 6))

        # Custom value should override default
        assert config.figsize == (8, 6)
        # Other defaults should remain
        assert config.dpi == 300
        assert "title" in config.fontsize

    def test_color_palette_access(self):
        """Test accessing color palette."""
        config = PlotConfig()

        assert "excitatory" in config.colors
        assert "inhibitory" in config.colors
        assert "combined" in config.colors
        assert "primary" in config.colors

    def test_fontsize_categories(self):
        """Test fontsize categories."""
        config = PlotConfig()

        assert "title" in config.fontsize
        assert "label" in config.fontsize
        assert "tick" in config.fontsize
        assert "legend" in config.fontsize
        assert "annotation" in config.fontsize

    def test_alpha_values(self):
        """Test alpha transparency values."""
        config = PlotConfig()

        assert "main" in config.alpha
        assert "fill" in config.alpha
        assert "annotation" in config.alpha

        # Alpha values should be between 0 and 1
        for alpha_val in config.alpha.values():
            assert 0 <= alpha_val <= 1


class TestBasePlotter:
    """Test BasePlotter class."""

    def test_init_default(self):
        """Test initialization with default config."""
        plotter = BasePlotter()

        assert isinstance(plotter.config, PlotConfig)
        assert hasattr(plotter, "logger")

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        custom_config = PlotConfig(figsize=(8, 6))
        plotter = BasePlotter(config=custom_config)

        assert plotter.config == custom_config

    def test_create_figure_default(self):
        """Test figure creation with default parameters."""
        plotter = BasePlotter()

        fig, ax = plotter._create_figure()

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert fig.get_size_inches().tolist() == list(plotter.config.figsize)
        plt.close(fig)

    def test_create_figure_custom(self):
        """Test figure creation with custom parameters."""
        plotter = BasePlotter()

        fig, ax = plotter._create_figure(figsize=(12, 8))

        assert fig.get_size_inches().tolist() == [12.0, 8.0]
        plt.close(fig)

    def test_create_figure_subplots(self):
        """Test figure creation with subplots."""
        plotter = BasePlotter()

        fig, axes = plotter._create_figure(nrows=2, ncols=2)

        assert isinstance(fig, plt.Figure)
        assert axes.shape == (2, 2)
        assert all(isinstance(ax, plt.Axes) for ax in axes.flat)
        plt.close(fig)

    def test_apply_styling(self):
        """Test applying plot styling."""
        plotter = BasePlotter()
        fig, ax = plotter._create_figure()

        plotter._apply_styling(
            ax, title="Test Title", xlabel="X Label", ylabel="Y Label"
        )

        assert ax.get_title() == "Test Title"
        assert ax.get_xlabel() == "X Label"
        assert ax.get_ylabel() == "Y Label"
        plt.close(fig)

    def test_save_plot(self):
        """Test saving plot to file."""
        plotter = BasePlotter()
        fig, ax = plotter._create_figure()
        ax.plot([1, 2, 3], [1, 4, 9])

        with tempfile.TemporaryDirectory() as tmp_dir:
            plotter._save_plot(fig, tmp_dir, "test_plot", close_after_save=False)

            expected_file = os.path.join(
                tmp_dir, f"test_plot.{plotter.config.save_format}"
            )
            assert os.path.exists(expected_file)

        plt.close(fig)


class TestComparisonPlotter:
    """Test ComparisonPlotter class."""

    def test_init_comparison_plotter(self):
        """Test initialization of ComparisonPlotter."""
        plotter = ComparisonPlotter()

        assert isinstance(plotter, BasePlotter)
        assert isinstance(plotter.config, PlotConfig)

    def test_comparison_plotter_with_config(self):
        """Test ComparisonPlotter with custom config."""
        config = PlotConfig(figsize=(12, 8))
        plotter = ComparisonPlotter(config=config)

        assert plotter.config == config


class TestTimeSeriesPlotter:
    """Test TimeSeriesPlotter class."""

    def test_init_timeseries_plotter(self):
        """Test initialization of TimeSeriesPlotter."""
        plotter = TimeSeriesPlotter()

        assert isinstance(plotter, BasePlotter)
        assert isinstance(plotter.config, PlotConfig)

    def test_timeseries_plotter_inheritance(self):
        """Test that TimeSeriesPlotter inherits from BasePlotter."""
        plotter = TimeSeriesPlotter()

        # Should have BasePlotter methods
        assert hasattr(plotter, "_create_figure")
        assert hasattr(plotter, "_apply_styling")
        assert hasattr(plotter, "_save_plot")


class TestDistributionPlotter:
    """Test DistributionPlotter class."""

    def test_init_distribution_plotter(self):
        """Test initialization of DistributionPlotter."""
        plotter = DistributionPlotter()

        assert isinstance(plotter, BasePlotter)
        assert isinstance(plotter.config, PlotConfig)

    def test_distribution_plotter_methods(self):
        """Test that DistributionPlotter has expected methods."""
        plotter = DistributionPlotter()

        # Should inherit BasePlotter functionality
        fig, ax = plotter._create_figure()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)


class TestPlottingBaseEdgeCases:
    """Test edge cases for plotting base functionality."""

    def test_save_plot_with_invalid_dir(self):
        """Test saving plot with invalid directory."""
        plotter = BasePlotter()
        fig, ax = plotter._create_figure()

        # Should handle gracefully without crashing
        try:
            plotter._save_plot(fig, "/nonexistent/directory", "test")
        except (OSError, FileNotFoundError):
            pass  # Expected for invalid directory

        plt.close(fig)

    def test_apply_styling_with_none_values(self):
        """Test applying styling with None values."""
        plotter = BasePlotter()
        fig, ax = plotter._create_figure()

        # Should not crash with None values
        plotter._apply_styling(ax, title=None)

        plt.close(fig)

    def test_create_figure_with_invalid_params(self):
        """Test figure creation with invalid parameters."""
        plotter = BasePlotter()

        # Should handle gracefully
        try:
            fig, ax = plotter._create_figure(figsize=(-1, -1))
            plt.close(fig)
        except (ValueError, TypeError):
            pass  # Expected for invalid parameters

    def test_plotter_inheritance_chain(self):
        """Test inheritance chain of plotters."""
        comparison = ComparisonPlotter()
        timeseries = TimeSeriesPlotter()
        distribution = DistributionPlotter()

        # All should be instances of BasePlotter
        assert isinstance(comparison, BasePlotter)
        assert isinstance(timeseries, BasePlotter)
        assert isinstance(distribution, BasePlotter)

    def test_multiple_plotter_configs(self):
        """Test that different plotters can have different configs."""
        config1 = PlotConfig(figsize=(8, 6))
        config2 = PlotConfig(figsize=(10, 8))

        plotter1 = ComparisonPlotter(config=config1)
        plotter2 = TimeSeriesPlotter(config=config2)

        assert plotter1.config != plotter2.config
        assert plotter1.config.figsize == (8, 6)
        assert plotter2.config.figsize == (10, 8)

    def test_config_with_invalid_types(self):
        """Test config with invalid types."""
        # PlotConfig accepts whatever kwargs are passed, so invalid types are stored as-is
        config = PlotConfig(figsize="invalid", dpi="invalid")

        # The values are stored as passed
        assert config.figsize == "invalid"
        assert config.dpi == "invalid"
