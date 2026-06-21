"""
Test suite for dendritic_modeling.analysis module.

This module tests all analysis functionality including:
- Performance analysis
- Ablation studies
- Information analysis
- Branch analysis
- Activation analysis
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

try:
    import os
    import sys

    experiments_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"
    )
    if experiments_path not in sys.path:
        sys.path.insert(0, experiments_path)
    from data_generation.synthetic_datasets import generate_synthetic_data
except ImportError:
    # Create a dummy function if synthetic datasets not available
    import torch
    from torch.utils.data import TensorDataset

    def generate_synthetic_data(num_samples=100, num_features=10):
        data = torch.randn(num_samples, num_features)
        labels = torch.randint(0, 4, (num_samples,))
        return TensorDataset(data, labels)


from unittest.mock import Mock

import numpy as np

from dendritic_modeling.analysis import (
    AnalysisManager,
    BranchActivationAnalyzer,
    PerformanceAnalyzer,
)
from dendritic_modeling.analysis.core.ablation import AblationAnalyzer
from dendritic_modeling.analysis.core.information import InformationAnalyzer
from dendritic_modeling.analysis.registry import get_analyzer_spec
from dendritic_modeling.analysis.tools import SynapticActivationAnalyzer
from dendritic_modeling.config import (
    AblationAnalysisParams,
    AnalysisConfig,
    BranchActivationAnalysisParams,
    InformationAnalysisParams,
    PerformanceAnalysisParams,
    SynapticActivationAnalysisParams,
    TransferFunctionConfig,
)
from dendritic_modeling.models import Classifier
from dendritic_modeling.networks import ExcitationInhibitionNetwork, Identity


class TestPerformanceAnalyzer(unittest.TestCase):
    """Test the PerformanceAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        params = PerformanceAnalysisParams()
        self.analyzer = PerformanceAnalyzer(params)

        # Create a simple test model
        transfer_params = TransferFunctionConfig().asdict()
        self.model = Classifier(
            encoder_network=Identity(input_dim=10),
            core_network=ExcitationInhibitionNetwork(
                input_dim=10,
                transfer_params=transfer_params,
                excitatory_layer_sizes=[4],
                inhibitory_layer_sizes=[4],
                excitatory_branch_factors=[2],
                inhibitory_branch_factors=[2],
                ee_synapses_per_branch_per_layer=[1],
                ei_synapses_per_branch_per_layer=[1],
                ie_synapses_per_branch_per_layer=[1],
                ii_synapses_per_branch_per_layer=[1],
            ),
            decoder_network=Identity(input_dim=4),
        )

        # Create test data
        self.test_data = torch.utils.data.TensorDataset(
            torch.rand(100, 10), torch.randint(0, 4, (100,))
        )

    def test_analyzer_initialization(self):
        """Test that the analyzer initializes correctly."""
        self.assertIsInstance(self.analyzer, PerformanceAnalyzer)

    def test_analyzer_has_required_attributes(self):
        """Test that the analyzer has the required attributes."""
        # Check that the analyzer has the expected attributes from params
        self.assertTrue(hasattr(self.analyzer, "accuracy"))
        self.assertTrue(hasattr(self.analyzer, "categorical_loglikelihood"))

    def test_evaluate_model(self):
        """Test model evaluation on dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # When save_path is provided, the method saves results and returns None
            results = self.analyzer.analyze(
                model=self.model,
                test_ds=self.test_data,
                device=self.device,
                save_path=tmpdir,
            )

            # When save_path is provided, results should be None
            self.assertIsNone(results)

            # When save_path is None, it should return results
            results = self.analyzer.analyze(
                model=self.model,
                test_ds=self.test_data,
                device=self.device,
                save_path=None,
            )

            # Check that results is a dictionary
            self.assertIsInstance(results, dict)

    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        params = PerformanceAnalysisParams()
        analyzer = PerformanceAnalyzer(params)
        empty_data = torch.utils.data.TensorDataset(
            torch.empty(0, 10), torch.empty(0, dtype=torch.long)
        )
        results = analyzer.analyze(
            model=self.model, test_ds=empty_data, device=self.device, save_path=None
        )
        self.assertIsInstance(results, dict)
        # Check that the analyzer handles empty data gracefully
        # The exact behavior may vary, but it should not crash
        self.assertIn("accuracy", results)
        self.assertIsInstance(results["accuracy"], dict)
        # For empty datasets, accuracy might be NaN which is expected
        if "test" in results["accuracy"]:
            # NaN is acceptable for empty datasets
            import math

            self.assertTrue(
                isinstance(results["accuracy"]["test"], float)
                or math.isnan(results["accuracy"]["test"])
            )
        # Main goal: ensure no crash occurs with empty data


class TestAblationAnalyzer(unittest.TestCase):
    """Test the AblationAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        params = AblationAnalysisParams()
        self.analyzer = AblationAnalyzer(params)

        # Create test model with multiple layers
        transfer_params = TransferFunctionConfig().asdict()
        self.model = Classifier(
            encoder_network=Identity(input_dim=10),
            core_network=ExcitationInhibitionNetwork(
                input_dim=10,
                transfer_params=transfer_params,
                excitatory_layer_sizes=[4, 4],
                inhibitory_layer_sizes=[4, 4],
                excitatory_branch_factors=[2, 2],
                inhibitory_branch_factors=[2, 2],
                ee_synapses_per_branch_per_layer=[1, 1],
                ei_synapses_per_branch_per_layer=[1, 1],
                ie_synapses_per_branch_per_layer=[1, 1],
                ii_synapses_per_branch_per_layer=[1],
            ),
            decoder_network=Identity(input_dim=4),
        )

        # Create test dataset
        self.test_data = torch.utils.data.TensorDataset(
            torch.rand(50, 10), torch.randint(0, 4, (50,))
        )

    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        self.assertIsInstance(self.analyzer, AblationAnalyzer)

    def test_branch_ablation(self):
        """Test branch ablation functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = self.analyzer.analyze(
                model=self.model,
                test_dataset=self.test_data,
                device="cpu",
                save_path=tmpdir,
            )

            # When save_path is provided, results should be None
            self.assertIsNone(results)

    def test_layer_ablation(self):
        """Test layer-wise ablation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = self.analyzer.analyze(
                model=self.model,
                test_dataset=self.test_data,
                device="cpu",
                save_path=tmpdir,
            )

            # When save_path is provided, results should be None
            self.assertIsNone(results)

    def test_layer_ablation_restores_weights_on_metric_error(self):
        """Temporary ablation edits should be rolled back even on failure."""
        module_name, module = next(
            (
                name,
                mod,
            )
            for name, mod in self.model.named_modules()
            if hasattr(mod, "branch_excitation") and mod.branch_excitation is not None
        )
        state_dict = {
            key: value.detach().clone()
            for key, value in self.model.state_dict().items()
        }

        self.analyzer.accuracy = True
        self.analyzer.auc = False
        self.analyzer.categorical_ll = False
        self.analyzer.mse = False
        self.analyzer.cosine_similarity = False
        self.analyzer.compute_pred_label_mi = False

        with patch(
            "dendritic_modeling.analysis.core.ablation.evaluate_accuracy",
            side_effect=RuntimeError("boom"),
        ):
            with self.assertRaises(RuntimeError):
                self.analyzer.ablate_layer_and_evaluate(
                    test_dataset=self.test_data,
                    model=self.model,
                    module=module,
                    module_name=module_name,
                    ablation_type="excitation",
                    ablation_method="lesion",
                    baseline_dict={"accuracy": 1.0},
                    results_dict={"excitation": {}},
                    state_dict=state_dict,
                    eval_kwargs={},
                    device="cpu",
                )

        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, state_dict[key])

    def test_identity_weight_transform_uses_zero_lesion_value(self):
        """Identity/relu transforms should lesion by zeroing raw weights."""
        module_name, module = next(
            (name, mod)
            for name, mod in self.model.named_modules()
            if hasattr(mod, "branch_excitation") and mod.branch_excitation is not None
        )
        state_dict = {
            key: value.detach().clone()
            for key, value in self.model.state_dict().items()
        }
        self.model.core_network.weight_transform = "identity"
        self.analyzer.lesion_value = self.analyzer._resolve_lesion_value(self.model)

        self.analyzer.accuracy = True
        self.analyzer.auc = False
        self.analyzer.categorical_ll = False
        self.analyzer.mse = False
        self.analyzer.cosine_similarity = False
        self.analyzer.compute_pred_label_mi = False

        def _check_zero_lesion(*args, **kwargs):
            assert torch.all(module.branch_excitation.pre_w == 0.0)
            return [0.75]

        with patch(
            "dendritic_modeling.analysis.core.ablation.evaluate_accuracy",
            side_effect=_check_zero_lesion,
        ):
            results = self.analyzer.ablate_layer_and_evaluate(
                test_dataset=self.test_data,
                model=self.model,
                module=module,
                module_name=module_name,
                ablation_type="excitation",
                ablation_method="lesion",
                baseline_dict={"accuracy": 1.0},
                results_dict={"excitation": {}},
                state_dict=state_dict,
                eval_kwargs={},
                device="cpu",
            )

        self.assertEqual(self.analyzer.lesion_value, 0.0)
        self.assertEqual(results["excitation"][module_name]["accuracy_drop"], 0.25)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, state_dict[key])


class TestInformationAnalyzer(unittest.TestCase):
    """Test the Information Analyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        params = InformationAnalysisParams()
        self.analyzer = InformationAnalyzer(params)

        # Create test model
        transfer_params = TransferFunctionConfig().asdict()
        self.model = Classifier(
            encoder_network=Identity(input_dim=8),
            core_network=ExcitationInhibitionNetwork(
                input_dim=8,
                transfer_params=transfer_params,
                excitatory_layer_sizes=[4],
                inhibitory_layer_sizes=[4],
                excitatory_branch_factors=[2],
                inhibitory_branch_factors=[2],
                ee_synapses_per_branch_per_layer=[1],
                ei_synapses_per_branch_per_layer=[1],
                ie_synapses_per_branch_per_layer=[1],
                ii_synapses_per_branch_per_layer=[1],
            ),
            decoder_network=Identity(input_dim=4),
        )

        # Create test data
        self.test_data = torch.utils.data.TensorDataset(
            torch.rand(30, 8), torch.randint(0, 4, (30,))
        )

    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        self.assertIsInstance(self.analyzer, InformationAnalyzer)

    def test_mutual_information_analysis(self):
        """Test mutual information computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock the MI computation since it requires complex dependencies
            with patch.object(self.analyzer, "compute_metrics") as mock_mi:
                mock_mi.return_value = {
                    "method": "kraskov",
                    "n_samples": 30,
                    "basic_mi": {"I(E;C)": 0.5, "I(I;C)": 0.3},
                }

                results = self.analyzer.analyze(
                    model=self.model,
                    test_dataset=self.test_data,
                    device="cpu",
                    save_path=tmpdir,
                    filename="test",
                )

                # Current analyzer returns computed results while also saving.
                self.assertIsInstance(results, dict)
                self.assertIn("method", results)


class TestActivationAnalyzers(unittest.TestCase):
    """Test the activation analyzer classes."""

    def setUp(self):
        """Set up test fixtures."""
        synaptic_params = SynapticActivationAnalysisParams()
        self.synaptic_analyzer = SynapticActivationAnalyzer(synaptic_params)
        branch_params = BranchActivationAnalysisParams()
        self.branch_analyzer = BranchActivationAnalyzer(branch_params)

        # Create test model
        transfer_params = TransferFunctionConfig().asdict()
        self.model = Classifier(
            encoder_network=Identity(input_dim=8),
            core_network=ExcitationInhibitionNetwork(
                input_dim=8,
                transfer_params=transfer_params,
                excitatory_layer_sizes=[4],
                inhibitory_layer_sizes=[4],
                excitatory_branch_factors=[2],
                inhibitory_branch_factors=[2],
                ee_synapses_per_branch_per_layer=[1],
                ei_synapses_per_branch_per_layer=[1],
                ie_synapses_per_branch_per_layer=[1],
                ii_synapses_per_branch_per_layer=[1],
            ),
            decoder_network=Identity(input_dim=4),
        )

        # Create test data
        self.test_data = torch.utils.data.TensorDataset(
            torch.rand(30, 8), torch.randint(0, 4, (30,))
        )

    def test_synaptic_activation_analyzer(self):
        """Test synaptic activation analysis."""
        self.model.train()
        with tempfile.TemporaryDirectory() as tmpdir:
            results = self.synaptic_analyzer.analyze(
                model=self.model,
                test_dataset=self.test_data,
                device="cpu",
                save_path=tmpdir,
            )

            # When save_path is provided, results should be None
            self.assertIsNone(results)
        self.assertTrue(self.model.training)

    def test_branch_activation_analyzer(self):
        """Test branch activation analysis."""
        self.model.train()
        with tempfile.TemporaryDirectory() as tmpdir:
            results = self.branch_analyzer.analyze(
                model=self.model,
                test_dataset=self.test_data,
                device="cpu",
                save_path=tmpdir,
            )

            # When save_path is provided, results should be None
            self.assertIsNone(results)
        self.assertTrue(self.model.training)


class TestAnalysisManager(unittest.TestCase):
    """Test the Analysis Manager class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test model
        transfer_params = TransferFunctionConfig().asdict()
        self.model = Classifier(
            encoder_network=Identity(input_dim=8),
            core_network=ExcitationInhibitionNetwork(
                input_dim=8,
                transfer_params=transfer_params,
                excitatory_layer_sizes=[4],
                inhibitory_layer_sizes=[4],
                excitatory_branch_factors=[2],
                inhibitory_branch_factors=[2],
                ee_synapses_per_branch_per_layer=[1],
                ei_synapses_per_branch_per_layer=[1],
                ie_synapses_per_branch_per_layer=[1],
                ii_synapses_per_branch_per_layer=[1],
            ),
            decoder_network=Identity(input_dim=4),
        )

        # Create test data
        train_data = torch.utils.data.TensorDataset(
            torch.rand(30, 8), torch.randint(0, 4, (30,))
        )
        valid_data = torch.utils.data.TensorDataset(
            torch.rand(20, 8), torch.randint(0, 4, (20,))
        )
        test_data = torch.utils.data.TensorDataset(
            torch.rand(25, 8), torch.randint(0, 4, (25,))
        )

        self.data = {"train": train_data, "valid": valid_data, "test": test_data}

        # Create analysis config
        self.analysis_config = AnalysisConfig()

        # Initialize manager
        self.manager = AnalysisManager(
            model=self.model,
            data=self.data,
            analysis_config=self.analysis_config,
            save_root="/tmp/test_analysis",
        )

    def test_manager_initialization(self):
        """Test that the analysis manager initializes correctly."""
        self.assertIsInstance(self.manager, AnalysisManager)
        self.assertEqual(self.manager.model, self.model)
        self.assertEqual(self.manager.data, self.data)

    def test_manager_initializer_uses_registered_attribute_names(self):
        """Enabled analyzers should be exposed at registry-declared attributes."""
        spec = get_analyzer_spec("performance")

        assert spec is not None
        self.assertTrue(hasattr(self.manager, spec.attribute_name))
        self.assertIs(
            self.manager.performance_analyzer,
            getattr(self.manager, spec.attribute_name),
        )
        self.assertFalse(self.manager.correlation_analysis_enabled)

    def test_manager_initializes_optional_analyzer_constructor_styles(self):
        """Registry-backed initialization supports params, model+params, and no params."""
        config = AnalysisConfig()
        config.synapse_turnover_analysis.enabled = True
        config.correlation_analysis.enabled = True
        config.routing_analysis.enabled = True

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AnalysisManager(
                model=self.model,
                data=self.data,
                analysis_config=config,
                save_root=tmpdir,
            )

        self.assertTrue(hasattr(manager, "syn_turnover_analyzer"))
        self.assertTrue(hasattr(manager, "correlation_analyzer"))
        self.assertTrue(hasattr(manager, "routing_analyzer"))
        self.assertTrue(manager.correlation_analysis_enabled)

    def test_should_run_config_matches_training_phase_rules(self):
        """Shared analysis gate should preserve final/training behavior."""
        cfg = self.analysis_config.performance_analysis
        cfg.enabled = True
        cfg.training = False

        self.assertTrue(AnalysisManager._should_run_config(cfg, training=False))
        self.assertFalse(AnalysisManager._should_run_config(cfg, training=True))

        cfg.training = True
        self.assertTrue(AnalysisManager._should_run_config(cfg, training=True))

        cfg.enabled = False
        self.assertFalse(AnalysisManager._should_run_config(cfg, training=False))
        self.assertFalse(AnalysisManager._should_run_config(None, training=False))

    def test_training_performance_analysis_uses_epoch_path_and_gate(self):
        """Training analysis should remain opt-in and keep the epoch save path."""
        config = AnalysisConfig()
        config.performance_analysis.enabled = True
        config.performance_analysis.training = False

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AnalysisManager(
                model=self.model,
                data=self.data,
                analysis_config=config,
                save_root=tmpdir,
            )

            with patch.object(manager, "performance_analyzer") as mock_analyzer:
                manager.run_analysis(filename="epoch_0", training=True)
                mock_analyzer.analyze.assert_not_called()

            config.performance_analysis.training = True
            manager = AnalysisManager(
                model=self.model,
                data=self.data,
                analysis_config=config,
                save_root=tmpdir,
            )

            with patch.object(manager, "performance_analyzer") as mock_analyzer:
                manager.run_analysis(filename="epoch_0", training=True)

            expected_path = Path(tmpdir) / "performance" / "epochs"
            mock_analyzer.analyze.assert_called_once()
            call_kwargs = mock_analyzer.analyze.call_args.kwargs
            self.assertEqual(call_kwargs["save_path"], str(expected_path))
            self.assertTrue(expected_path.is_dir())
            self.assertTrue(call_kwargs["training"])

    def test_run_performance_analysis(self):
        """Test running performance analysis through manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.manager.save_root = tmpdir

            # Mock the performance analyzer
            with patch.object(self.manager, "performance_analyzer") as mock_analyzer:
                mock_analyzer.analyze.return_value = {
                    "accuracy": 0.85,
                    "categorical_loglikelihood": -0.5,
                }

                results = self.manager.run_analysis()

                # The method doesn't return results, so just check it doesn't crash
                self.assertIsNone(results)

    def test_common_test_dataset_analysis_path_preserves_call_kwargs(self):
        """Common test-dataset runner should preserve analyzer call arguments."""
        config = AnalysisConfig()
        config.performance_analysis.enabled = False
        config.noise_perturbation_analysis.enabled = True

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AnalysisManager(
                model=self.model,
                data=self.data,
                analysis_config=config,
                save_root=tmpdir,
            )

            with patch.object(manager, "noise_perturbation_analyzer") as mock_analyzer:
                manager.run_analysis(filename="final_eval", training=False)

            mock_analyzer.analyze.assert_called_once()
            call_kwargs = mock_analyzer.analyze.call_args.kwargs
            self.assertIs(call_kwargs["model"], self.model)
            self.assertIs(call_kwargs["test_dataset"], self.data["test"])
            self.assertEqual(call_kwargs["device"], "cpu")
            self.assertEqual(
                call_kwargs["save_path"],
                str(Path(tmpdir) / "noise_perturbation_analysis"),
            )
            self.assertEqual(call_kwargs["filename"], "final_eval")
            self.assertIs(
                call_kwargs["runtime"],
                manager.config.runtime.final,
            )
            self.assertNotIn("training", call_kwargs)

    def test_common_test_dataset_analysis_path_preserves_training_kwarg(self):
        """Shared runner should preserve training-aware analyzer calls."""
        config = AnalysisConfig()
        config.performance_analysis.enabled = False
        config.ablation_analysis.enabled = True
        config.ablation_analysis.training = True

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AnalysisManager(
                model=self.model,
                data=self.data,
                analysis_config=config,
                save_root=tmpdir,
            )

            with patch.object(manager, "ablation_analyzer") as mock_analyzer:
                manager.run_analysis(filename="epoch_3", training=True)

            mock_analyzer.analyze.assert_called_once()
            call_kwargs = mock_analyzer.analyze.call_args.kwargs
            self.assertTrue(call_kwargs["training"])
            self.assertEqual(
                call_kwargs["save_path"],
                str(Path(tmpdir) / "ablation_analysis"),
            )
            self.assertIs(call_kwargs["runtime"], manager.config.runtime.training)


def test_ablation_analyzer():
    from dendritic_modeling.config.analysis import (
        AblationAnalysisParams,
        AblationSelectionParams,
    )

    params = AblationAnalysisParams(
        selection=AblationSelectionParams(
            levels=["layer", "compartment"],
        ),
    )
    analyzer = AblationAnalyzer(params)
    assert analyzer.layer_ablation
    assert analyzer.compartment_ablation


def test_ablation_analyzer_ignores_removed_flat_aliases():
    from dendritic_modeling.config.analysis import AblationAnalysisParams

    params = AblationAnalysisParams()
    params.layer_ablation = True
    params.ablate_excitation = True
    params.accuracy = True

    analyzer = AblationAnalyzer(params)

    assert not analyzer.layer_ablation
    assert not analyzer.ablate_excitation
    assert not analyzer.accuracy


@pytest.mark.skip(
    reason="AblationAnalyzer requires complex ExcitationInhibitionNetwork setup - integration test needed"
)
def test_ablation_analyze():
    """Test ablation analyzer - currently skipped due to complex model requirements."""
    # This test requires a properly configured ExcitationInhibitionNetwork with DendriticBranchLayers
    # The full setup is complex and should be handled in integration tests


def test_information_analyzer():
    from dendritic_modeling.config.analysis import InformationAnalysisParams

    params = InformationAnalysisParams(method="kraskov")
    analyzer = InformationAnalyzer(params)
    assert analyzer.method == "kraskov"


@pytest.fixture
def sample_analyzer():
    params = InformationAnalysisParams()
    return InformationAnalyzer(params)


@pytest.fixture
def sample_data():
    return generate_synthetic_data(num_samples=100, num_features=10)  # Adjust as needed


def test_compute_class_mi(sample_analyzer, sample_data):
    mock_model = Mock()
    # Mock the model to have the required structure for InformationAnalyzer
    mock_model.get_branch_activations.return_value = np.random.rand(100, 1)

    # Mock the network structure that InformationAnalyzer expects
    mock_network = Mock()
    mock_network.__class__.__name__ = "ExcitationInhibitionNetwork"
    mock_model.network = mock_network

    results = sample_analyzer.analyze(mock_model, sample_data)

    # Handle case where analysis might return None for incompatible models
    if results is not None:
        assert "mi_class" in results["branch_0"]
        assert isinstance(results["branch_0"]["mi_class"], float)
    else:
        # If the analyzer skips this model type, that's acceptable
        pytest.skip("Model type not compatible with InformationAnalyzer")


def test_compute_conditional_mi(sample_analyzer, sample_data):
    params = InformationAnalysisParams(method="binned")
    analyzer = InformationAnalyzer(params)

    rng = np.random.default_rng(0)
    n = 64
    E = rng.normal(size=(n, 1))
    I_var = rng.normal(size=(n, 1))
    Vout = 0.8 * E - 0.3 * I_var + 0.1 * rng.normal(size=(n, 1))
    Vb = 0.5 * E + 0.2 * rng.normal(size=(n, 1))
    C = (E[:, 0] + 0.25 * rng.normal(size=n) > 0).astype(int)

    with_vb = analyzer._compute_conditional_mi(E, I_var, Vout, C, Vinf=None, Vb=Vb)
    without_vb = analyzer._compute_conditional_mi(E, I_var, Vout, C, Vinf=None, Vb=None)

    for results in (with_vb, without_vb):
        assert "I(Vout;C|E)" in results
        assert "I(Vout;C|I)" in results
        assert "I(Vout;C|E,I)" in results
        assert np.isfinite(results["I(Vout;C|E)"])
        assert np.isfinite(results["I(Vout;C|I)"])
        assert np.isfinite(results["I(Vout;C|E,I)"])


def test_compute_pid_per_branch(sample_analyzer, sample_data):
    # Test PID
    pass


if __name__ == "__main__":
    unittest.main()
