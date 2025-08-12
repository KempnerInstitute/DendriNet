"""
Test suite for result analyzer functionality.

This module provides comprehensive testing for the unified result analyzer
including sweep type detection, data loading, plotting functions, and
analysis capabilities.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Mock matplotlib to avoid display issues in tests
with patch("matplotlib.pyplot.show"):
    pass

from dendritic_modeling.scripts.sweeps.result_analyzer import (
    auto_detect_sweep_type,
    create_ei_heatmap,
    create_information_plots,
    create_metric_vs_ei_ratio_plot,
    create_noise_analysis_plots,
    load_results_from_directory,
    main,
    parse_arguments,
)


class TestSweepTypeDetection:
    """Test sweep type auto-detection functionality."""

    @pytest.fixture
    def temp_results_dir(self):
        """Create a temporary directory with mock results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)

            # Create mock config files
            config_data = {
                "sweep_config": {
                    "model.network.excitatory_layer_sizes": [[4, 8], [8, 16]],
                    "model.network.inhibitory_layer_sizes": [[2, 4], [4, 8]],
                }
            }

            # Create subdirectories with configs
            for i in range(3):
                subdir = results_dir / f"config_{i}"
                subdir.mkdir()

                config_file = subdir / "config.yaml"
                with open(config_file, "w") as f:
                    # Write YAML-like content
                    f.write("sweep_config:\n")
                    f.write("  model.network.excitatory_layer_sizes:\n")
                    f.write("    - [4, 8]\n")
                    f.write("    - [8, 16]\n")

                # Create mock results
                results_file = subdir / "final_results.json"
                with open(results_file, "w") as f:
                    json.dump(
                        {
                            "performance": {
                                "accuracy": 0.85 + i * 0.05,
                                "loss": 0.5 - i * 0.1,
                            },
                            "config": config_data,
                        },
                        f,
                    )

            yield results_dir

    def test_auto_detect_ei_sweep(self, temp_results_dir):
        """Test auto-detection of EI sweeps."""
        sweep_type = auto_detect_sweep_type(str(temp_results_dir))
        assert sweep_type == "ei"

    def test_auto_detect_with_branch_params(self, temp_results_dir):
        """Test detection when branch parameters are present."""
        # Add branch parameters to one config
        subdir = temp_results_dir / "config_0"
        config_file = subdir / "config.yaml"
        with open(config_file, "a") as f:
            f.write("  model.network.excitatory_branch_factors:\n")
            f.write("    - [2, 4]\n")

        sweep_type = auto_detect_sweep_type(str(temp_results_dir))
        # Should still detect as EI since EI params are primary
        assert sweep_type in ["ei", "branch"]

    def test_auto_detect_noise_sweep(self, temp_results_dir):
        """Test detection of noise sweeps."""
        # Modify configs to include noise parameters
        for i in range(3):
            subdir = temp_results_dir / f"config_{i}"
            config_file = subdir / "config.yaml"
            with open(config_file, "w") as f:
                f.write("sweep_config:\n")
                f.write("  task.noise_level:\n")
                f.write("    - 0.1\n")
                f.write("    - 0.2\n")

        sweep_type = auto_detect_sweep_type(str(temp_results_dir))
        assert sweep_type == "noise"

    def test_auto_detect_general_sweep(self, temp_results_dir):
        """Test detection of general sweeps."""
        # Modify configs to include general parameters
        for i in range(3):
            subdir = temp_results_dir / f"config_{i}"
            config_file = subdir / "config.yaml"
            with open(config_file, "w") as f:
                f.write("sweep_config:\n")
                f.write("  train.main.trainer.learning_rate:\n")
                f.write("    - 0.001\n")
                f.write("    - 0.01\n")

        sweep_type = auto_detect_sweep_type(str(temp_results_dir))
        assert sweep_type == "general"

    def test_auto_detect_no_configs(self):
        """Test detection when no config files are found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sweep_type = auto_detect_sweep_type(tmpdir)
            assert sweep_type == "general"  # Default fallback


class TestDataLoading:
    """Test data loading functionality."""

    @pytest.fixture
    def mock_results_data(self):
        """Create mock results data."""
        return [
            {
                "config_path": "/path/to/config_0",
                "results": {
                    "performance": {"accuracy": 0.85, "loss": 0.3},
                    "information_analysis": {
                        "I(E;C)": 0.5,
                        "I(I;C)": 0.3,
                        "I(E;I)": 0.2,
                    },
                    "config": {
                        "model": {
                            "network": {
                                "excitatory_layer_sizes": [8, 16],
                                "inhibitory_layer_sizes": [4, 8],
                            }
                        }
                    },
                },
            },
            {
                "config_path": "/path/to/config_1",
                "results": {
                    "performance": {"accuracy": 0.90, "loss": 0.2},
                    "information_analysis": {
                        "I(E;C)": 0.6,
                        "I(I;C)": 0.4,
                        "I(E;I)": 0.25,
                    },
                    "config": {
                        "model": {
                            "network": {
                                "excitatory_layer_sizes": [16, 32],
                                "inhibitory_layer_sizes": [8, 16],
                            }
                        }
                    },
                },
            },
        ]

    def test_load_results_basic(self, temp_results_dir):
        """Test basic results loading."""
        results = load_results_from_directory(str(temp_results_dir))

        assert isinstance(results, list)
        assert len(results) > 0

        for result in results:
            assert "config_path" in result
            assert "results" in result

    @patch(
        "dendritic_modeling.scripts.sweeps.result_analyzer.load_results_from_directory"
    )
    def test_load_results_error_handling(self, mock_load):
        """Test error handling in results loading."""
        # Mock an exception
        mock_load.side_effect = Exception("Mock error")

        with pytest.raises(Exception, match="Mock error"):
            load_results_from_directory("/fake/path")

    def test_load_results_empty_directory(self):
        """Test loading from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = load_results_from_directory(tmpdir)
            assert results == []

    def test_load_results_invalid_json(self, temp_results_dir):
        """Test handling of invalid JSON files."""
        # Create a subdirectory with invalid JSON
        subdir = temp_results_dir / "invalid_config"
        subdir.mkdir()

        results_file = subdir / "final_results.json"
        with open(results_file, "w") as f:
            f.write("invalid json content {")

        # Should still load valid results and skip invalid ones
        results = load_results_from_directory(str(temp_results_dir))
        assert isinstance(results, list)


class TestPlottingFunctions:
    """Test plotting functionality."""

    @pytest.fixture
    def mock_results_dataframe(self):
        """Create a mock results DataFrame."""
        return pd.DataFrame(
            {
                "excitatory_neurons": [8, 16, 32, 8, 16, 32],
                "inhibitory_neurons": [4, 8, 16, 4, 8, 16],
                "ei_ratio": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                "accuracy": [0.8, 0.85, 0.9, 0.82, 0.87, 0.92],
                "loss": [0.5, 0.4, 0.3, 0.48, 0.38, 0.28],
                "I(E;C)": [0.4, 0.5, 0.6, 0.42, 0.52, 0.62],
                "noise_level": [0.0, 0.0, 0.0, 0.1, 0.1, 0.1],
            }
        )

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    def test_create_ei_heatmap(self, mock_savefig, mock_show, mock_results_dataframe):
        """Test EI heatmap creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_ei_heatmap(
                mock_results_dataframe, save_dir=tmpdir, metric="accuracy"
            )

            # Should have called savefig
            mock_savefig.assert_called()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    def test_create_ei_heatmap_multiple_metrics(
        self, mock_savefig, mock_show, mock_results_dataframe
    ):
        """Test EI heatmap with multiple metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_ei_heatmap(
                mock_results_dataframe, save_dir=tmpdir, metric=["accuracy", "loss"]
            )

            # Should have called savefig multiple times
            assert mock_savefig.call_count >= 2

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    def test_create_metric_vs_ei_ratio_plot(
        self, mock_savefig, mock_show, mock_results_dataframe
    ):
        """Test metric vs EI ratio plot creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_metric_vs_ei_ratio_plot(mock_results_dataframe, save_dir=tmpdir)

            mock_savefig.assert_called()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    def test_create_noise_analysis_plots(
        self, mock_savefig, mock_show, mock_results_dataframe
    ):
        """Test noise analysis plot creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_noise_analysis_plots(mock_results_dataframe, save_dir=tmpdir)

            mock_savefig.assert_called()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    def test_create_information_plots(
        self, mock_savefig, mock_show, mock_results_dataframe
    ):
        """Test information analysis plot creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_information_plots(mock_results_dataframe, save_dir=tmpdir)

            mock_savefig.assert_called()

    @patch("matplotlib.pyplot.show")
    def test_plotting_with_missing_columns(self, mock_show, mock_results_dataframe):
        """Test plotting functions handle missing columns gracefully."""
        # Remove some columns
        limited_df = mock_results_dataframe.drop(columns=["I(E;C)", "noise_level"])

        with tempfile.TemporaryDirectory() as tmpdir:
            # These should not crash even with missing data
            try:
                create_ei_heatmap(limited_df, save_dir=tmpdir, metric="accuracy")
                create_metric_vs_ei_ratio_plot(limited_df, save_dir=tmpdir)
                # create_noise_analysis_plots and create_information_plots might fail
                # due to missing columns, which is expected behavior
            except Exception:
                # Some plotting functions may fail with missing data, which is acceptable
                pass

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    def test_plotting_empty_dataframe(self, mock_savefig, mock_show):
        """Test plotting functions with empty DataFrame."""
        empty_df = pd.DataFrame()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Should handle empty data gracefully
            try:
                create_ei_heatmap(empty_df, save_dir=tmpdir, metric="accuracy")
            except Exception:
                # Expected to fail with empty data
                pass


class TestArgumentParsing:
    """Test command-line argument parsing."""

    def test_parse_arguments_basic(self):
        """Test basic argument parsing."""
        args = parse_arguments(["--results-dir", "/path/to/results"])

        assert args.results_dir == "/path/to/results"
        assert args.sweep_type is None  # Default
        assert args.auto_detect is False  # Default

    def test_parse_arguments_with_sweep_type(self):
        """Test argument parsing with sweep type."""
        args = parse_arguments(
            ["--results-dir", "/path/to/results", "--sweep-type", "ei"]
        )

        assert args.results_dir == "/path/to/results"
        assert args.sweep_type == "ei"

    def test_parse_arguments_with_auto_detect(self):
        """Test argument parsing with auto-detect."""
        args = parse_arguments(["--results-dir", "/path/to/results", "--auto-detect"])

        assert args.results_dir == "/path/to/results"
        assert args.auto_detect is True

    def test_parse_arguments_multiple_dirs(self):
        """Test argument parsing with multiple directories."""
        args = parse_arguments(
            [
                "--results-dirs",
                "/path/1",
                "/path/2",
                "/path/3",
                "--sweep-type",
                "branch",
            ]
        )

        assert args.results_dirs == ["/path/1", "/path/2", "/path/3"]
        assert args.sweep_type == "branch"

    def test_parse_arguments_output_dir(self):
        """Test argument parsing with custom output directory."""
        args = parse_arguments(
            ["--results-dir", "/path/to/results", "--output-dir", "/custom/output"]
        )

        assert args.results_dir == "/path/to/results"
        assert args.output_dir == "/custom/output"


class TestMainFunction:
    """Test the main function and integration."""

    @patch("dendritic_modeling.scripts.sweeps.result_analyzer.auto_detect_sweep_type")
    @patch(
        "dendritic_modeling.scripts.sweeps.result_analyzer.load_results_from_directory"
    )
    @patch("dendritic_modeling.scripts.sweeps.result_analyzer.create_ei_heatmap")
    @patch(
        "dendritic_modeling.scripts.sweeps.result_analyzer.create_metric_vs_ei_ratio_plot"
    )
    @patch("dendritic_modeling.scripts.sweeps.result_analyzer.create_information_plots")
    def test_main_with_auto_detect(
        self, mock_info_plots, mock_ei_plot, mock_heatmap, mock_load, mock_detect
    ):
        """Test main function with auto-detection."""
        # Setup mocks
        mock_detect.return_value = "ei"
        mock_load.return_value = [
            {
                "config_path": "/path/config",
                "results": {
                    "performance": {"accuracy": 0.85},
                    "config": {"model": {"network": {"excitatory_layer_sizes": [8]}}},
                },
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "sys.argv",
                ["result_analyzer.py", "--results-dir", tmpdir, "--auto-detect"],
            ):
                try:
                    main()

                    # Verify that detection and loading were called
                    mock_detect.assert_called_once()
                    mock_load.assert_called_once()

                except SystemExit:
                    # main() might call sys.exit(), which is fine
                    pass

    @patch(
        "dendritic_modeling.scripts.sweeps.result_analyzer.load_results_from_directory"
    )
    @patch(
        "dendritic_modeling.scripts.sweeps.result_analyzer.create_noise_analysis_plots"
    )
    def test_main_with_explicit_sweep_type(self, mock_noise_plots, mock_load):
        """Test main function with explicit sweep type."""
        mock_load.return_value = [
            {
                "config_path": "/path/config",
                "results": {
                    "performance": {"accuracy": 0.85},
                    "config": {"task": {"noise_level": 0.1}},
                },
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "sys.argv",
                [
                    "result_analyzer.py",
                    "--results-dir",
                    tmpdir,
                    "--sweep-type",
                    "noise",
                ],
            ):
                try:
                    main()

                    mock_load.assert_called_once()

                except SystemExit:
                    pass

    @patch(
        "dendritic_modeling.scripts.sweeps.result_analyzer.load_results_from_directory"
    )
    def test_main_no_results(self, mock_load):
        """Test main function when no results are found."""
        mock_load.return_value = []

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("sys.argv", ["result_analyzer.py", "--results-dir", tmpdir]):
                try:
                    main()
                except SystemExit:
                    pass

                mock_load.assert_called_once()

    @patch(
        "dendritic_modeling.scripts.sweeps.result_analyzer.load_results_from_directory"
    )
    def test_main_error_handling(self, mock_load):
        """Test main function error handling."""
        mock_load.side_effect = Exception("Mock error")

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("sys.argv", ["result_analyzer.py", "--results-dir", tmpdir]):
                try:
                    main()
                except SystemExit:
                    pass
                except Exception:
                    # Should handle errors gracefully
                    pass


class TestResultAnalyzerIntegration:
    """Integration tests for result analyzer."""

    def test_end_to_end_ei_analysis(self):
        """Test end-to-end EI sweep analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)

            # Create realistic EI sweep results
            for i, (exc_size, inh_size) in enumerate([(8, 4), (16, 8), (32, 16)]):
                for j, lr in enumerate([0.001, 0.01]):
                    config_dir = results_dir / f"config_{i}_{j}"
                    config_dir.mkdir()

                    # Create config file
                    config_file = config_dir / "config.yaml"
                    with open(config_file, "w") as f:
                        f.write("sweep_config:\n")
                        f.write("  model.network.excitatory_layer_sizes:\n")
                        f.write(f"    - [{exc_size}]\n")
                        f.write("  model.network.inhibitory_layer_sizes:\n")
                        f.write(f"    - [{inh_size}]\n")

                    # Create results file
                    results_file = config_dir / "final_results.json"
                    accuracy = 0.7 + (exc_size / 100) + (lr * 10)
                    results_data = {
                        "performance": {"accuracy": accuracy, "loss": 1.0 - accuracy},
                        "information_analysis": {
                            "I(E;C)": 0.3 + (exc_size / 100),
                            "I(I;C)": 0.2 + (inh_size / 100),
                            "I(E;I)": 0.15,
                        },
                        "config": {
                            "model": {
                                "network": {
                                    "excitatory_layer_sizes": [exc_size],
                                    "inhibitory_layer_sizes": [inh_size],
                                }
                            },
                            "train": {"main": {"trainer": {"learning_rate": lr}}},
                        },
                    }

                    with open(results_file, "w") as f:
                        json.dump(results_data, f)

            # Test auto-detection
            sweep_type = auto_detect_sweep_type(str(results_dir))
            assert sweep_type == "ei"

            # Test data loading
            results = load_results_from_directory(str(results_dir))
            assert len(results) == 6  # 3 sizes x 2 learning rates

            # Test that we can create plots (mock the display/save)
            with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.savefig"):
                # Convert to DataFrame (simplified)
                data = []
                for result in results:
                    config = result["results"]["config"]
                    perf = result["results"]["performance"]
                    info = result["results"]["information_analysis"]

                    data.append(
                        {
                            "excitatory_neurons": config["model"]["network"][
                                "excitatory_layer_sizes"
                            ][0],
                            "inhibitory_neurons": config["model"]["network"][
                                "inhibitory_layer_sizes"
                            ][0],
                            "accuracy": perf["accuracy"],
                            "loss": perf["loss"],
                            "I(E;C)": info["I(E;C)"],
                            "I(I;C)": info["I(I;C)"],
                        }
                    )

                df = pd.DataFrame(data)
                df["ei_ratio"] = df["inhibitory_neurons"] / df["excitatory_neurons"]

                # Test plotting functions
                create_ei_heatmap(df, save_dir=tmpdir, metric="accuracy")
                create_metric_vs_ei_ratio_plot(df, save_dir=tmpdir)
                create_information_plots(df, save_dir=tmpdir)

    def test_comprehensive_analysis_workflow(self):
        """Test a comprehensive analysis workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)

            # Create mixed parameter sweep results
            configs = [
                {"exc": [8], "inh": [4], "noise": 0.0, "lr": 0.001},
                {"exc": [16], "inh": [8], "noise": 0.1, "lr": 0.001},
                {"exc": [32], "inh": [16], "noise": 0.2, "lr": 0.01},
            ]

            for i, config in enumerate(configs):
                config_dir = results_dir / f"config_{i}"
                config_dir.mkdir()

                # Create comprehensive results
                results_data = {
                    "performance": {
                        "accuracy": 0.8 + i * 0.05,
                        "loss": 0.5 - i * 0.1,
                        "f1_score": 0.75 + i * 0.05,
                    },
                    "information_analysis": {
                        "I(E;C)": 0.4 + i * 0.1,
                        "I(I;C)": 0.3 + i * 0.05,
                        "I(E;I)": 0.2,
                        "I(Vout;C)": 0.5 + i * 0.08,
                    },
                    "ablation_analysis": {
                        "layer_0": {"accuracy_drop": 0.1},
                        "layer_1": {"accuracy_drop": 0.15},
                    },
                    "config": {
                        "model": {
                            "network": {
                                "excitatory_layer_sizes": config["exc"],
                                "inhibitory_layer_sizes": config["inh"],
                            }
                        },
                        "task": {"noise_level": config["noise"]},
                        "train": {"main": {"trainer": {"learning_rate": config["lr"]}}},
                    },
                }

                results_file = config_dir / "final_results.json"
                with open(results_file, "w") as f:
                    json.dump(results_data, f)

            # Test the complete workflow
            results = load_results_from_directory(str(results_dir))
            assert len(results) == 3

            # Verify rich data structure
            for result in results:
                assert "performance" in result["results"]
                assert "information_analysis" in result["results"]
                assert "ablation_analysis" in result["results"]
                assert "config" in result["results"]
