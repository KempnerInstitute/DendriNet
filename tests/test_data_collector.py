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

    def test_load_gain_enriched_schema_exports_raw_and_generalization_scores(self):
        """New multiplicative-gain payloads should expose raw, ID, and OOD scores."""
        collector = DataCollector(self.results_dir)
        config_dir = self.results_dir / "results" / "config_0"

        gain_dir = config_dir / "multiplicative_gain_analysis"
        gain_dir.mkdir()
        with open(gain_dir / "final.json", "w") as f:
            json.dump(
                {
                    "gain_factors": [0.5, 1.0, 2.0],
                    "accuracy": [0.2, 0.4, 0.6],
                    "raw_scores": {"accuracy": [0.2, 0.4, 0.6]},
                    "ID": {"accuracy": 0.4},
                    "OOD_lower": {"accuracy": 0.2},
                    "OOD_upper": {"accuracy": 0.6},
                },
                f,
            )

        data = collector._load_gain(config_dir)
        self.assertEqual(data["mult_gain_accuracy_gf0.5"], 0.2)
        self.assertEqual(data["mult_gain_accuracy_gf1.0"], 0.4)
        self.assertEqual(data["mult_gain_accuracy_gf2.0"], 0.6)
        self.assertEqual(data["mult_gain_accuracy_id"], 0.4)
        self.assertEqual(data["mult_gain_accuracy_ood_lower"], 0.2)
        self.assertEqual(data["mult_gain_accuracy_ood_upper"], 0.6)

    def test_load_compositional_ood_flat_schema_exports_numeric_metrics(self):
        """Flat compositional-OOD payloads should export prefixed numeric columns."""
        collector = DataCollector(self.results_dir)
        config_dir = self.results_dir / "results" / "config_0"

        ood_dir = config_dir / "compositional_ood_analysis"
        ood_dir.mkdir()
        with open(ood_dir / "final.json", "w") as f:
            json.dump(
                {
                    "accuracy": 0.81,
                    "auc": 0.6,
                    "nested": {"ignored": 1.0},
                },
                f,
            )

        data = collector._load_compositional_ood(config_dir)
        self.assertEqual(data["comp_ood_accuracy"], 0.81)
        self.assertEqual(data["comp_ood_auc"], 0.6)
        self.assertNotIn("comp_ood_nested", data)

    def test_load_compositional_ood_missing_file_returns_empty(self):
        """Missing compositional-OOD results should not add columns."""
        collector = DataCollector(self.results_dir)
        config_dir = self.results_dir / "results" / "config_0"

        self.assertEqual(collector._load_compositional_ood(config_dir), {})


if __name__ == "__main__":
    unittest.main()
