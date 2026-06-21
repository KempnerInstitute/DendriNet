"""
Test new parameters added for enhanced analysis and trainer configurations.
"""

from types import SimpleNamespace

from dendritic_modeling.analysis.core.information import InformationAnalyzer
from dendritic_modeling.config.analysis import (
    InformationAnalysisParams,
    InformationComputeParams,
)
from dendritic_modeling.config.training import (
    CommonTrainerConfig,
    EncoderTrainingConfig,
    MainTrainingConfig,
    TrainingConfig,
)


class TestNewAnalysisParameters:
    """Test the new information analysis parameters."""

    def test_computation_level_options(self):
        """Test that all computation_level options work."""
        valid_levels = ["single_branch", "layer_branch", "all_branch"]

        for level in valid_levels:
            params = InformationAnalysisParams(
                compute=InformationComputeParams(computation_level=level)
            )
            analyzer = InformationAnalyzer(params)
            assert analyzer.computation_level == level

    def test_branch_aggregation_options(self):
        """Test that all branch_aggregation options work."""
        valid_aggregations = ["mean", "multivariate", "sample"]

        for aggregation in valid_aggregations:
            params = InformationAnalysisParams(
                compute=InformationComputeParams(branch_aggregation=aggregation)
            )
            analyzer = InformationAnalyzer(params)
            assert analyzer.branch_aggregation == aggregation

    def test_enhanced_analysis_parameters(self):
        """Test that enhanced analysis parameters work."""
        params = InformationAnalysisParams(
            compute=InformationComputeParams(
                per_neuron_analysis=True,
                per_einet_analysis=True,
                per_layer_analysis=True,
            )
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
        trainer = CommonTrainerConfig(
            enable_nan_checking=True, enable_adaptive_clipping=True, use_amp=True
        )

        assert trainer.enable_nan_checking is True
        assert trainer.enable_adaptive_clipping is True
        assert trainer.use_amp is True

    def test_dual_trainer_structure(self):
        """Test that the dual trainer structure works."""
        main_trainer = CommonTrainerConfig(
            enable_nan_checking=True, enable_adaptive_clipping=True, use_amp=True
        )

        encoder_trainer = CommonTrainerConfig(
            enable_nan_checking=False, enable_adaptive_clipping=False, use_amp=False
        )

        config = TrainingConfig(
            main=MainTrainingConfig(common=main_trainer),
            encoder=EncoderTrainingConfig(common=encoder_trainer),
        )

        assert config.main.common.enable_nan_checking is True
        assert config.encoder.common.enable_nan_checking is False
        assert config.main.common.use_amp is True
        assert config.encoder.common.use_amp is False

    def test_trainer_defaults(self):
        """Test that trainer defaults are correct."""
        trainer = CommonTrainerConfig()

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

    # Test that trainer configuration still exposes optimization knobs
    trainer = CommonTrainerConfig()
    assert hasattr(trainer, "enable_nan_checking")
    assert hasattr(trainer, "enable_adaptive_clipping")


def test_synthetic_soma_entry_uses_real_soma_statistics():
    analyzer = InformationAnalyzer(InformationAnalysisParams())
    model = SimpleNamespace(core_network=SimpleNamespace(somatic_synapses=False))
    results = {
        "layer_statistics": {
            "core.branch_layers.0": {
                "has_synapses": True,
                "I(Vout;C)_mean": 0.1,
                "I(Vb;C)_mean": 0.2,
            },
            "core.branch_layers.2": {
                "has_synapses": True,
                "I(Vout;C)_mean": 1.2,
                "I(Vout;C)_std": 0.12,
                "I(Vb;C)_mean": 0.7,
                "I(Vb;C)_std": 0.07,
                "I(Vb;Vout)_mean": 0.4,
                "I(Vb;Vout|C)_mean": 0.33,
                "I(Vb_lin;Vout_lin)_mean": 0.22,
                "I(Vb_lin;Vout_lin|C)_mean": 0.11,
            },
        }
    }

    somatic_synapses = analyzer._add_synthetic_soma_entry_if_needed(results, model)

    assert somatic_synapses is False
    soma = results["layer_statistics"]["synthetic_soma"]
    assert soma["has_synapses"] is False
    assert soma["I(Vout;C)_mean"] == 1.2
    assert soma["I(Vout;C)_std"] == 0.12
    assert soma["I(Vb;C)_mean"] == 0.7
    assert soma["I(Vb;Vout)_mean"] == 0.4
    assert soma["I(Vb;Vout|C)_mean"] == 0.33
    assert soma["I(Vb_lin;Vout_lin)_mean"] == 0.22
    assert soma["I(Vb_lin;Vout_lin|C)_mean"] == 0.11
    assert "I(E;C)_mean" not in soma
    assert "I(I;C)_mean" not in soma


def test_synthetic_soma_metric_filter_removes_missing_synapse_terms():
    analyzer = InformationAnalyzer(InformationAnalysisParams())
    soma_entry = {
        "I(E;C)_mean": 1.0,
        "I(I;C)_mean": 2.0,
        "I(Vb;C)_mean": 3.0,
        "I(Vout;C)_mean": 4.0,
        "I(Vb;Vout)_mean": 5.0,
    }

    filtered = analyzer._filter_synthetic_soma_entry_metrics(soma_entry)

    assert filtered["has_synapses"] is False
    assert "I(E;C)_mean" not in filtered
    assert "I(I;C)_mean" not in filtered
    assert filtered["I(Vb;C)_mean"] == 3.0
    assert filtered["I(Vout;C)_mean"] == 4.0
    assert filtered["I(Vb;Vout)_mean"] == 5.0


def test_synthetic_soma_decision_requires_missing_somatic_synapses_and_stats():
    analyzer = InformationAnalyzer(InformationAnalysisParams())

    assert analyzer._should_add_synthetic_soma_entry(
        somatic_synapses=False,
        results={"layer_statistics": {"layer0": {}}},
        soma_already_captured=False,
    )
    assert not analyzer._should_add_synthetic_soma_entry(
        somatic_synapses=True,
        results={"layer_statistics": {"layer0": {}}},
        soma_already_captured=False,
    )
    assert not analyzer._should_add_synthetic_soma_entry(
        somatic_synapses=False,
        results={},
        soma_already_captured=False,
    )
    assert not analyzer._should_add_synthetic_soma_entry(
        somatic_synapses=False,
        results={"layer_statistics": {"layer0": {}}},
        soma_already_captured=True,
    )


def test_branch_layer_index_parser_accepts_only_valid_branch_layer_paths():
    analyzer = InformationAnalyzer(InformationAnalysisParams())

    assert analyzer._parse_branch_layer_index("core.branch_layers.0") == 0
    assert analyzer._parse_branch_layer_index("core.branch_layers.12.output") == 12
    assert analyzer._parse_branch_layer_index("core.branch_layers.bad.output") is None
    assert analyzer._parse_branch_layer_index("core.branch_layers") is None
    assert analyzer._parse_branch_layer_index("core.other_layers.1") is None


def test_find_real_soma_layer_uses_highest_valid_branch_layer_index():
    analyzer = InformationAnalyzer(InformationAnalysisParams())
    layer_statistics = {
        "core.branch_layers.1": {},
        "core.branch_layers.bad": {},
        "core.branch_layers.4.output": {},
        "core.other_layers.9": {},
        "core.branch_layers.2": {},
    }

    assert (
        analyzer._find_real_soma_layer(layer_statistics)
        == "core.branch_layers.4.output"
    )


def test_extract_soma_statistics_copies_all_mapped_metrics():
    analyzer = InformationAnalyzer(InformationAnalysisParams())
    metric_map = analyzer._soma_statistics_metric_map()
    source_stats = {
        metric_key: float(idx + 1) for idx, metric_key in enumerate(metric_map.values())
    }
    layer_statistics = {"core.branch_layers.3": source_stats}

    soma_stats = analyzer._extract_soma_statistics(
        "core.branch_layers.3", layer_statistics
    )

    for idx, soma_key in enumerate(metric_map.keys()):
        assert soma_stats[soma_key] == float(idx + 1)


def test_extract_soma_statistics_defaults_to_zero_without_real_soma_layer():
    analyzer = InformationAnalyzer(InformationAnalysisParams())

    soma_stats = analyzer._extract_soma_statistics(None, {})

    assert set(soma_stats) == set(analyzer._soma_statistics_metric_map())
    assert all(value == 0.0 for value in soma_stats.values())


def test_synthetic_soma_entry_skips_when_soma_already_captured():
    analyzer = InformationAnalyzer(InformationAnalysisParams())
    model = SimpleNamespace(core_network=SimpleNamespace(somatic_synapses=False))
    results = {
        "layer_statistics": {
            "core.branch_layers.0": {"has_synapses": True},
            "core.branch_layers.1": {"has_synapses": False},
        }
    }

    somatic_synapses = analyzer._add_synthetic_soma_entry_if_needed(results, model)

    assert somatic_synapses is False
    assert "synthetic_soma" not in results["layer_statistics"]
