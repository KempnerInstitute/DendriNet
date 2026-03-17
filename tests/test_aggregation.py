"""Tests for sweep aggregation utilities."""

import unittest

import pandas as pd

from dendritic_modeling.scripts.sweeps.utils.aggregation import (
    aggregate_over_seeds,
    compute_ei_ratio,
    extract_network_category,
)


class TestAggregation(unittest.TestCase):
    """Test aggregation utilities."""

    def test_aggregate_over_seeds(self):
        """Test aggregation over seeds."""
        data = pd.DataFrame(
            {
                "ee_value": [40, 40, 40, 20, 20, 20],
                "ie_value": [10, 10, 10, 10, 10, 10],
                "seed": [0, 1, 2, 0, 1, 2],
                "test_accuracy": [0.90, 0.91, 0.89, 0.85, 0.86, 0.84],
            }
        )

        result = aggregate_over_seeds(
            data,
            group_by_cols=["ee_value", "ie_value"],
        )

        self.assertEqual(len(result), 2)
        self.assertIn("test_accuracy_mean", result.columns)
        self.assertIn("test_accuracy_std", result.columns)

        # Check aggregation values
        group1 = result[result["ee_value"] == 40].iloc[0]
        self.assertAlmostEqual(group1["test_accuracy_mean"], 0.90, places=2)

    def test_compute_ei_ratio(self):
        """Test E/I ratio computation."""
        data = pd.DataFrame(
            {
                "ee_value": [40, 20, 10],
                "ie_value": [10, 10, 5],
            }
        )

        result = compute_ei_ratio(data)

        self.assertIn("ei_ratio", result.columns)
        self.assertAlmostEqual(result.iloc[0]["ei_ratio"], 4.0)
        self.assertAlmostEqual(result.iloc[1]["ei_ratio"], 2.0)
        self.assertAlmostEqual(result.iloc[2]["ei_ratio"], 2.0)

    def test_compute_ei_ratio_zero_inhibition(self):
        """Test E/I ratio with zero inhibition."""
        data = pd.DataFrame(
            {
                "ee_value": [40],
                "ie_value": [0],
            }
        )

        result = compute_ei_ratio(data)

        # Should not compute ratio when ie_value is 0
        self.assertTrue(pd.isna(result.iloc[0]["ei_ratio"]))

    def test_extract_network_category(self):
        """Test network category extraction."""
        data = pd.DataFrame(
            {
                "use_shunting": [True, False, None],
            }
        )

        result = extract_network_category(data)

        self.assertIn("network_category", result.columns)
        self.assertEqual(result.iloc[0]["network_category"], "dendritic_shunting")
        self.assertEqual(result.iloc[1]["network_category"], "dendritic_additive")
        self.assertEqual(result.iloc[2]["network_category"], "unknown")

    def test_extract_network_category_deprecated_aliases(self):
        """Deprecated aliases should still normalize to canonical categories."""
        data = pd.DataFrame(
            {
                "network_type": [
                    "matchedtotalparammlp",
                    "matchedactiveparammlp",
                    "einet_sh",
                    "einet_ns",
                ],
                "use_shunting": [False, False, True, False],
            }
        )

        result = extract_network_category(data)

        self.assertEqual(result.iloc[0]["network_category"], "total_param_mlp")
        self.assertEqual(result.iloc[1]["network_category"], "active_param_mlp")
        self.assertEqual(result.iloc[2]["network_category"], "dendritic_shunting")
        self.assertEqual(result.iloc[3]["network_category"], "dendritic_additive")

    def test_extract_network_category_flat_mlp(self):
        """flat_mlp should be treated as a first-class canonical category."""
        data = pd.DataFrame(
            {
                "network_type": ["flat_mlp"],
                "use_shunting": [False],
            }
        )

        result = extract_network_category(data)
        self.assertEqual(result.iloc[0]["network_category"], "flat_mlp")


if __name__ == "__main__":
    unittest.main()
