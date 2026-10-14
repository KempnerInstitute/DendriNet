"""Tests for sweep result analyzers."""

import unittest

from dendritic_modeling.scripts.sweeps.analyzers import (
    BranchSweepAnalyzer,
    EISweepAnalyzer,
    GeneralSweepAnalyzer,
    LocalLearningSweepAnalyzer,
    get_analyzer,
)


class TestAnalyzerRegistry(unittest.TestCase):
    """Test analyzer registry and factory."""

    def test_get_analyzer_ei(self):
        """Test getting E/I analyzer."""
        analyzer = get_analyzer("ei")
        self.assertIsInstance(analyzer, EISweepAnalyzer)

    def test_get_analyzer_local_learning(self):
        """Test getting local learning analyzer."""
        analyzer = get_analyzer("local_learning")
        self.assertIsInstance(analyzer, LocalLearningSweepAnalyzer)

    def test_get_analyzer_branch(self):
        """Test getting branch analyzer."""
        analyzer = get_analyzer("branch")
        self.assertIsInstance(analyzer, BranchSweepAnalyzer)

    def test_get_analyzer_general(self):
        """Test getting general analyzer."""
        analyzer = get_analyzer("general")
        self.assertIsInstance(analyzer, GeneralSweepAnalyzer)

    def test_get_analyzer_invalid(self):
        """Test error on invalid analyzer type."""
        with self.assertRaises(ValueError):
            get_analyzer("invalid_type")


class TestEISweepAnalyzer(unittest.TestCase):
    """Test E/I sweep analyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = EISweepAnalyzer()

    def test_groupby_columns(self):
        """Test groupby columns."""
        cols = self.analyzer.get_groupby_columns()
        self.assertIn("ee_value", cols)
        self.assertIn("ie_value", cols)
        self.assertIn("ei_ratio", cols)
        self.assertIn("use_shunting", cols)
        self.assertIn("reactivation_init_policy", cols)

    def test_plot_types(self):
        """Test plot types."""
        plot_types = self.analyzer.get_plot_types()
        self.assertIn("performance_vs_ei_ratio", plot_types)
        self.assertIn("information_panels", plot_types)


class TestLocalLearningSweepAnalyzer(unittest.TestCase):
    """Test local learning sweep analyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = LocalLearningSweepAnalyzer()

    def test_groupby_columns(self):
        """Test groupby columns."""
        cols = self.analyzer.get_groupby_columns()
        self.assertIn("rule_variant", cols)
        self.assertIn("error_broadcast_mode", cols)
        self.assertIn("rho_mode", cols)

    def test_plot_types(self):
        """Test plot types."""
        plot_types = self.analyzer.get_plot_types()
        self.assertIn("performance_by_rule_variant", plot_types)
        self.assertIn("parameter_combination_comparison", plot_types)


class TestBranchSweepAnalyzer(unittest.TestCase):
    """Test branch sweep analyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = BranchSweepAnalyzer()

    def test_groupby_columns(self):
        """Test groupby columns."""
        cols = self.analyzer.get_groupby_columns()
        self.assertIn("nparams", cols)
        self.assertIn("branch_factors", cols)
        self.assertIn("reactivation_init_policy", cols)

    def test_plot_types(self):
        """Test plot types."""
        plot_types = self.analyzer.get_plot_types()
        self.assertIn("performance_vs_nparams", plot_types)
        self.assertIn("parameter_scaling", plot_types)


if __name__ == "__main__":
    unittest.main()
