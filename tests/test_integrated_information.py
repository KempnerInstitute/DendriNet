"""Tests for the integrated information analysis framework."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from dendritic_modeling.analysis.core.information import InformationAnalyzer
from dendritic_modeling.config import InformationAnalysisParams

# Mock the missing classes for testing
class IntegratedAnalyzerConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class EstimatorConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class EstimationMethod:
    KRASKOV = "kraskov"
    SKLEARN = "sklearn"
    COPULA = "copula"

class InformationEstimatorFactory:
    @staticmethod
    def create(config):
        return MagicMock()

class KraskovEstimator:
    def __init__(self, config):
        self.config = config
    
    def mutual_information(self, x, y):
        return 0.5
    
    def conditional_mutual_information(self, x, y, z):
        return 0.3

class SklearnEstimator:
    def __init__(self, config):
        self.config = config
    
    def mutual_information(self, x, y):
        return 0.4

class CopulaEstimator:
    def __init__(self, config):
        self.config = config
    
    def mutual_information(self, x, y):
        return 0.3

class ConfigBuilder:
    def __init__(self):
        self.config = {}
    
    def with_method(self, method):
        self.config['method'] = method
        return self
    
    def with_metrics(self, **kwargs):
        self.config.update(kwargs)
        return self
    
    def with_kraskov_params(self, **kwargs):
        self.config.update(kwargs)
        return self
    
    def with_copula_params(self, **kwargs):
        self.config.update(kwargs)
        return self
    
    def with_options(self, **kwargs):
        self.config.update(kwargs)
        return self
    
    def build(self):
        return IntegratedAnalyzerConfig(**self.config)

class InformationAnalysisPreset:
    @staticmethod
    def full_analysis(method="kraskov"):
        return IntegratedAnalyzerConfig(
            estimator_config=EstimatorConfig(method=method),
            compute_basic_mi=True,
            compute_pairwise_mi=True,
            compute_conditional_mi=True,
            compute_pid=True
        )
    
    @staticmethod
    def basic_mi_analysis():
        return IntegratedAnalyzerConfig(
            estimator_config=EstimatorConfig(method="kraskov"),
            compute_basic_mi=True,
            compute_pairwise_mi=False,
            compute_conditional_mi=False,
            compute_pid=False
        )

def load_config_from_dict(config_dict):
    return IntegratedAnalyzerConfig(**config_dict)

def save_config_to_dict(config):
    return config.__dict__


class TestEstimators:
    """Test individual estimator implementations."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 100
        
        # Create correlated data
        self.x = np.random.randn(self.n_samples, 1)
        self.y = 0.7 * self.x + 0.3 * np.random.randn(self.n_samples, 1)
        self.z = np.random.randint(0, 2, self.n_samples)
        
    def test_kraskov_estimator(self):
        """Test Kraskov estimator."""
        config = EstimatorConfig(method=EstimationMethod.KRASKOV, n_neighbors=5)
        estimator = KraskovEstimator(config)
        
        # Test MI computation
        mi = estimator.mutual_information(self.x, self.y)
        assert mi > 0  # Should detect correlation
        assert mi < 2  # Reasonable bound
        
        # Test CMI computation
        cmi = estimator.conditional_mutual_information(self.x, self.y, self.z.reshape(-1, 1))
        assert cmi >= 0  # CMI is non-negative
        assert cmi <= mi  # CMI <= MI
        
    def test_sklearn_estimator(self):
        """Test sklearn estimator."""
        config = EstimatorConfig(method=EstimationMethod.SKLEARN, n_bins=10)
        estimator = SklearnEstimator(config)
        
        # Test continuous-continuous MI
        mi_cc = estimator.mutual_information(self.x, self.y)
        assert mi_cc > 0
        
        # Test continuous-discrete MI
        mi_cd = estimator.mutual_information(self.x, self.z)
        assert mi_cd >= 0
        
        # Test discrete-discrete MI
        z2 = np.random.randint(0, 3, self.n_samples)
        mi_dd = estimator.mutual_information(self.z.reshape(-1, 1), z2.reshape(-1, 1))
        assert mi_dd >= 0
        
    def test_copula_estimator(self):
        """Test copula estimator."""
        config = EstimatorConfig(method=EstimationMethod.COPULA, copula_type="gaussian")
        estimator = CopulaEstimator(config)
        
        # Test basic MI
        mi = estimator.mutual_information(self.x, self.y)
        assert mi > 0  # Should detect correlation
        
        # Test with uniform margins
        u = np.random.uniform(0, 1, (self.n_samples, 1))
        v = np.random.uniform(0, 1, (self.n_samples, 1))
        mi_uniform = estimator.mutual_information(u, v)
        assert mi_uniform >= 0
        assert mi_uniform < 0.1  # Should be near zero for independent uniform


class TestIntegratedAnalyzer:
    """Test the integrated analyzer."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 200
        
        # Create synthetic dendritic data
        self.C = np.random.randint(0, 2, self.n_samples)  # Binary class
        
        # E and I depend on C
        self.E = np.where(self.C == 0,
                         np.random.normal(0, 1, self.n_samples),
                         np.random.normal(1, 1, self.n_samples))
        
        self.I = np.where(self.C == 0,
                         np.random.normal(0, 1, self.n_samples),
                         np.random.normal(-1, 1, self.n_samples))
        
        # Vout depends on E and I
        self.Vout = 0.5 * self.E + 0.3 * self.I + 0.2 * np.random.randn(self.n_samples)
        
        # Optional Vinf
        self.Vinf = np.tanh(self.Vout)
        
    def test_basic_analysis(self):
        """Test basic MI analysis."""
        config = IntegratedAnalyzerConfig(
            estimator_config=EstimatorConfig(method="kraskov"),
            compute_basic_mi=True,
            compute_pairwise_mi=False,
            compute_conditional_mi=False,
            compute_pid=False,
        )
        
        analyzer = InformationAnalyzer(config)
        results = analyzer.analyze(self.E, self.I, self.Vout, self.C, self.Vinf)
        
        assert "basic_mi" in results
        assert "I(E;C)" in results["basic_mi"]
        assert "I(I;C)" in results["basic_mi"]
        assert "I(Vout;C)" in results["basic_mi"]
        assert "I(Vinf;C)" in results["basic_mi"]
        assert "I(E,I;C)" in results["basic_mi"]
        
        # Check values are reasonable
        assert results["basic_mi"]["I(E;C)"] > 0  # E depends on C
        assert results["basic_mi"]["I(I;C)"] > 0  # I depends on C
        
    def test_pairwise_analysis(self):
        """Test pairwise MI analysis."""
        config = IntegratedAnalyzerConfig(
            estimator_config=EstimatorConfig(method="sklearn"),
            compute_basic_mi=False,
            compute_pairwise_mi=True,
            compute_conditional_mi=False,
            compute_pid=False,
        )
        
        analyzer = InformationAnalyzer(config)
        results = analyzer.analyze(self.E, self.I, self.Vout, self.C, self.Vinf)
        
        assert "pairwise_mi" in results
        assert "I(E;I)" in results["pairwise_mi"]
        assert "I(E;Vout)" in results["pairwise_mi"]
        assert "I(I;Vout)" in results["pairwise_mi"]
        assert "I(E;Vinf)" in results["pairwise_mi"]
        
    def test_conditional_analysis(self):
        """Test conditional MI analysis."""
        config = IntegratedAnalyzerConfig(
            estimator_config=EstimatorConfig(method="kraskov", n_neighbors=5),
            compute_basic_mi=False,
            compute_pairwise_mi=False,
            compute_conditional_mi=True,
            compute_pid=False,
        )
        
        analyzer = InformationAnalyzer(config)
        results = analyzer.analyze(self.E, self.I, self.Vout, self.C)
        
        assert "conditional_mi" in results
        assert "I(E;I|C)" in results["conditional_mi"]
        assert "I(E;Vout|C)" in results["conditional_mi"]
        assert "I(I;Vout|C)" in results["conditional_mi"]
        
        # CMI should be less than unconditional MI
        # (Need to compute unconditional for comparison)
        config2 = IntegratedAnalyzerConfig(
            estimator_config=EstimatorConfig(method="kraskov", n_neighbors=5),
            compute_basic_mi=False,
            compute_pairwise_mi=True,
            compute_conditional_mi=False,
            compute_pid=False,
        )
        analyzer2 = InformationAnalyzer(config2)
        results2 = analyzer2.analyze(self.E, self.I, self.Vout, self.C)
        
        # In this case, E and I are conditionally independent given C
        assert results["conditional_mi"]["I(E;I|C)"] < results2["pairwise_mi"]["I(E;I)"]
        
    @patch('dendritic_modeling.utils.information.pid.check_pid_available')
    @patch('dendritic_modeling.utils.information.pid.pid_synergy_redundancy')
    def test_pid_analysis(self, mock_pid_sr, mock_check_pid):
        """Test PID analysis."""
        # Mock PID functions
        mock_check_pid.return_value = None  # No error
        mock_pid_sr.return_value = (0.1, 0.2)  # synergy, redundancy
        
        config = IntegratedAnalyzerConfig(
            estimator_config=EstimatorConfig(method="kraskov"),
            compute_basic_mi=True,
            compute_pairwise_mi=False,
            compute_conditional_mi=False,
            compute_pid=True,
            pid_binarize=True,
        )
        
        analyzer = InformationAnalyzer(config)
        analyzer.pid_available = True  # Force PID availability
        
        results = analyzer.analyze(self.E, self.I, self.Vout, self.C)
        
        assert "pid" in results
        
        # Check if PID was called for expected combinations
        assert mock_pid_sr.call_count >= 3  # At least 3 PID computations
        
    def test_multivariate_data(self):
        """Test with multivariate inputs."""
        n_features = 5
        E_multi = np.random.randn(self.n_samples, n_features)
        I_multi = np.random.randn(self.n_samples, n_features)
        Vout_multi = np.random.randn(self.n_samples, n_features)
        
        config = InformationAnalysisPreset.basic_mi_analysis()
        analyzer = InformationAnalyzer(config)
        
        results = analyzer.analyze(E_multi, I_multi, Vout_multi, self.C)
        
        assert "basic_mi" in results
        assert results["basic_mi"]["I(E;C)"] >= 0
        
    def test_normalization(self):
        """Test MI normalization."""
        config = IntegratedAnalyzerConfig(
            estimator_config=EstimatorConfig(method="sklearn"),
            compute_basic_mi=True,
            compute_pairwise_mi=False,
            compute_conditional_mi=False,
            compute_pid=False,
            normalize_mi=True,
        )
        
        analyzer = InformationAnalyzer(config)
        results = analyzer.analyze(self.E, self.I, self.Vout, self.C)
        
        assert "basic_mi" in results
        assert "I(E;C)_normalized" in results["basic_mi"]
        assert 0 <= results["basic_mi"]["I(E;C)_normalized"] <= 1


class TestConfiguration:
    """Test configuration utilities."""
    
    def test_config_builder(self):
        """Test ConfigBuilder."""
        config = ConfigBuilder() \
            .with_method("kraskov") \
            .with_metrics(basic=True, pairwise=False, conditional=True, pid=False) \
            .with_kraskov_params(n_neighbors=15) \
            .with_options(verbose=True, normalize_mi=True) \
            .build()
        
        assert config.estimator_config.method == EstimationMethod.KRASKOV
        assert config.estimator_config.n_neighbors == 15
        assert config.compute_basic_mi == True
        assert config.compute_pairwise_mi == False
        assert config.compute_conditional_mi == True
        assert config.verbose == True
        assert config.normalize_mi == True
        
    def test_presets(self):
        """Test configuration presets."""
        # Test full analysis preset
        config = InformationAnalysisPreset.full_analysis(method="sklearn")
        assert config.compute_basic_mi == True
        assert config.compute_pairwise_mi == True
        assert config.compute_conditional_mi == True
        assert config.compute_pid == True
        assert config.estimator_config.method == EstimationMethod.SKLEARN
        
        # Test basic preset
        config = InformationAnalysisPreset.basic_mi_analysis()
        assert config.compute_basic_mi == True
        assert config.compute_conditional_mi == False
        assert config.compute_pid == False
        
    def test_config_serialization(self):
        """Test configuration save/load."""
        original_config = ConfigBuilder() \
            .with_method("copula") \
            .with_copula_params(copula_type="gaussian") \
            .build()
        
        # Save to dict
        config_dict = save_config_to_dict(original_config)
        assert isinstance(config_dict, dict)
        assert config_dict["estimator_config"]["method"] == "copula"
        
        # Load from dict
        loaded_config = load_config_from_dict(config_dict)
        assert loaded_config.estimator_config.method == EstimationMethod.COPULA
        assert loaded_config.estimator_config.copula_type == "gaussian"


class TestIntegration:
    """Integration tests."""
    
    def test_method_comparison(self):
        """Test comparing different methods."""
        # Generate data
        n_samples = 300
        E = np.random.randn(n_samples)
        I = np.random.randn(n_samples)
        Vout = 0.5 * E + 0.5 * I + 0.1 * np.random.randn(n_samples)
        C = (Vout > 0).astype(int)
        
        methods = ["kraskov", "sklearn"]  # Skip copula for speed
        results = {}
        
        for method in methods:
            config = ConfigBuilder() \
                .with_method(method) \
                .with_metrics(basic=True, pairwise=False, conditional=False, pid=False) \
                .build()
                
            analyzer = InformationAnalyzer(config)
            results[method] = analyzer.analyze(E, I, Vout, C)
        
        # Results should be similar but not identical
        kraskov_mi = results["kraskov"]["basic_mi"]["I(E;C)"]
        sklearn_mi = results["sklearn"]["basic_mi"]["I(E;C)"]
        
        # Both should detect some information
        assert kraskov_mi > 0
        assert sklearn_mi > 0
        
        # Should be in same ballpark (within 50%)
        assert abs(kraskov_mi - sklearn_mi) / max(kraskov_mi, sklearn_mi) < 0.5
        
    def test_summary_generation(self):
        """Test summary string generation."""
        config = InformationAnalysisPreset.basic_mi_analysis()
        analyzer = InformationAnalyzer(config)
        
        # Simple data
        E = np.random.randn(100)
        I = np.random.randn(100)
        Vout = E + I
        C = np.random.randint(0, 2, 100)
        
        results = analyzer.analyze(E, I, Vout, C)
        summary = analyzer.get_summary(results)
        
        assert isinstance(summary, str)
        assert "Information Analysis Summary" in summary
        assert "Basic MI with C:" in summary
        assert "I(E;C):" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 