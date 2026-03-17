"""Tests for sweep DataCollector."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dendritic_modeling.scripts.sweeps.collectors.data_collector import DataCollector


class TestDataCollector(unittest.TestCase):
    """Test DataCollector functionality."""

    def setUp(self):
        """Set up test fixtures with temp directory."""
        self.temp_dir = TemporaryDirectory()
        self.results_dir = Path(self.temp_dir.name)

        # Create directory structure
        (self.results_dir / "configs").mkdir()
        (self.results_dir / "results").mkdir()

        # Create sample config directory
        config_dir = self.results_dir / "results" / "config_0"
        config_dir.mkdir()

        # Create sample performance data
        perf_dir = config_dir / "performance"
        perf_dir.mkdir()
        with open(perf_dir / "final.json", "w") as f:
            json.dump({"test": {"accuracy": 0.95, "auc": 0.98}}, f)

        # Create sample config file
        with open(self.results_dir / "configs" / "config_0.yaml", "w") as f:
            f.write("experiment:\n  seed: 42\n")

    def tearDown(self):
        """Clean up temp directory."""
        self.temp_dir.cleanup()

    def test_get_config_dirs(self):
        """Test getting config directories."""
        collector = DataCollector(self.results_dir)
        config_dirs = collector._get_config_dirs()

        self.assertEqual(len(config_dirs), 1)
        self.assertEqual(config_dirs[0].name, "config_0")

    def test_load_performance(self):
        """Test loading performance data."""
        collector = DataCollector(self.results_dir)
        config_dir = self.results_dir / "results" / "config_0"

        perf_data = collector._load_performance(config_dir)

        self.assertIn("accuracy_test", perf_data)
        self.assertEqual(perf_data["accuracy_test"], 0.95)
        self.assertIn("auc_test", perf_data)
        self.assertEqual(perf_data["auc_test"], 0.98)

    def test_load_json_file_with_extension(self):
        """Test loading JSON file with .json extension."""
        collector = DataCollector(self.results_dir)
        config_dir = self.results_dir / "results" / "config_0"
        perf_file = config_dir / "performance" / "final.json"

        data = collector._load_json_file(perf_file)

        self.assertIsNotNone(data)
        self.assertIn("test", data)

    def test_load_json_file_without_extension(self):
        """Test loading JSON file without extension."""
        collector = DataCollector(self.results_dir)
        config_dir = self.results_dir / "results" / "config_0"

        # Create file without .json extension
        info_dir = config_dir / "information_analysis"
        info_dir.mkdir()
        with open(info_dir / "final", "w") as f:
            json.dump({"basic_mi": {"I(E;C)": 1.5}}, f)

        data = collector._load_json_file(info_dir / "final")

        self.assertIsNotNone(data)
        self.assertIn("basic_mi", data)

    def test_load_ablation_canonical_schema_exports_layer_and_module_columns(self):
        """Canonical 3-level ablation payload should export both views."""
        collector = DataCollector(self.results_dir)
        config_dir = self.results_dir / "results" / "config_0"

        ablation_dir = config_dir / "ablation_analysis"
        ablation_dir.mkdir()
        with open(ablation_dir / "final.json", "w") as f:
            json.dump(
                {
                    "layer": {
                        "lesion": {
                            "excitation": {
                                "core.layers.0": {
                                    "depth": 0,
                                    "accuracy_drop": 0.1,
                                },
                                "core.layers.1": {
                                    "depth": 1,
                                    "accuracy_drop": 0.2,
                                },
                            }
                        }
                    }
                },
                f,
            )

        data = collector._load_ablation(config_dir)
        self.assertIn("ablation_lesion_excitation_accuracy_drop_depth0", data)
        self.assertIn(
            "ablation_module__lesion__excitation__accuracy_drop__core_layers_0__depth0",
            data,
        )

    def test_load_ablation_legacy_schema_converts_to_canonical(self):
        """Legacy 2-level payloads should still flatten correctly."""
        collector = DataCollector(self.results_dir)
        config_dir = self.results_dir / "results" / "config_0"

        ablation_dir = config_dir / "ablation_analysis"
        ablation_dir.mkdir(exist_ok=True)
        with open(ablation_dir / "final.json", "w") as f:
            json.dump(
                {
                    "layer": {
                        "mean_clamp_upstream": {
                            "core.layers.1": {
                                "depth": 1,
                                "pred_label_mi_bits_drop": 0.2,
                            }
                        }
                    }
                },
                f,
            )

        data = collector._load_ablation(config_dir)
        self.assertIn(
            "ablation_mean_clamp_upstream_pred_label_mi_bits_drop_depth1", data
        )


if __name__ == "__main__":
    unittest.main()
