"""
Test new parameters added for enhanced analysis and trainer configurations.
"""

# Test imports - assuming proper PYTHONPATH or pip install -e .

from dendritic_modeling.analysis.core.information import InformationAnalyzer
from dendritic_modeling.config.analysis import InformationAnalysisParams
from dendritic_modeling.config.training import TrainConfig, TrainerConfig


class TestNewAnalysisParameters:
    """Test the new information analysis parameters."""

    def test_computation_level_options(self):
        """Test that all computation_level options work."""
        valid_levels = ["single_branch", "layer_branch", "all_branch"]

        for level in valid_levels:
            params = InformationAnalysisParams(computation_level=level)
            analyzer = InformationAnalyzer(params)
            assert analyzer.computation_level == level

    def test_branch_aggregation_options(self):
        """Test that all branch_aggregation options work."""
        valid_aggregations = ["mean", "multivariate", "sample"]

        for aggregation in valid_aggregations:
            params = InformationAnalysisParams(branch_aggregation=aggregation)
            analyzer = InformationAnalyzer(params)
            assert analyzer.branch_aggregation == aggregation

    def test_enhanced_analysis_parameters(self):
        """Test that enhanced analysis parameters work."""
        params = InformationAnalysisParams(
            per_neuron_analysis=True, per_einet_analysis=True, per_layer_analysis=True
        )
        analyzer = InformationAnalyzer(params)

        assert analyzer.per_neuron_analysis is True
        assert analyzer.per_einet_analysis is True
        assert analyzer.per_layer_analysis is True

    def test_default_values(self):
        """Test that default values are set correctly."""
        params = InformationAnalysisParams()
        analyzer = InformationAnalyzer(params)

        assert analyzer.computation_level == "single_branch"
        assert analyzer.branch_aggregation == "mean"
        assert analyzer.per_neuron_analysis is False
        assert analyzer.per_einet_analysis is False
        assert analyzer.per_layer_analysis is False


class TestNewTrainerParameters:
    """Test the new trainer configuration parameters."""

    def test_training_optimization_flags(self):
        """Test that training optimization flags work."""
        trainer = TrainerConfig(
            enable_nan_checking=True, enable_adaptive_clipping=True, use_amp=True
        )

        assert trainer.enable_nan_checking is True
        assert trainer.enable_adaptive_clipping is True
        assert trainer.use_amp is True

    def test_dual_trainer_structure(self):
        """Test that the dual trainer structure works."""
        main_trainer = TrainerConfig(
            enable_nan_checking=True, enable_adaptive_clipping=True, use_amp=True
        )

        encoder_trainer = TrainerConfig(
            enable_nan_checking=False, enable_adaptive_clipping=False, use_amp=False
        )

        config = TrainConfig(
            main_trainer=main_trainer, encoder_net_trainer=encoder_trainer
        )

        assert config.main_trainer.enable_nan_checking is True
        assert config.encoder_net_trainer.enable_nan_checking is False
        assert config.main_trainer.use_amp is True
        assert config.encoder_net_trainer.use_amp is False

    def test_trainer_defaults(self):
        """Test that trainer defaults are correct."""
        trainer = TrainerConfig()

        assert trainer.enable_nan_checking is False
        assert trainer.enable_adaptive_clipping is False
        assert trainer.use_amp is False


def test_backward_compatibility():
    """Test that existing code still works with default parameters."""
    # Test that old-style InformationAnalysisParams still works
    params = InformationAnalysisParams()
    analyzer = InformationAnalyzer(params)

    # Should have sensible defaults
    assert hasattr(analyzer, "computation_level")
    assert hasattr(analyzer, "branch_aggregation")
    assert hasattr(analyzer, "per_neuron_analysis")
    assert hasattr(analyzer, "per_einet_analysis")
    assert hasattr(analyzer, "per_layer_analysis")

    # Test that old-style TrainerConfig still works
    trainer = TrainerConfig()
    assert hasattr(trainer, "enable_nan_checking")
    assert hasattr(trainer, "enable_adaptive_clipping")
