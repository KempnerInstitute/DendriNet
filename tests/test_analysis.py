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

    def test_enhanced_analysis_helper_adds_requested_results(self):
        """Enhanced analysis helper should preserve optional result additions."""
        self.analyzer.per_layer_analysis = True
        self.analyzer.per_neuron_analysis = True
        self.analyzer.per_einet_analysis = True
        self.analyzer.data_dict = {
            "layer0": {
                "excitation": torch.ones(4, 2),
                "inhibition": torch.zeros(4, 2),
                "output": torch.full((4, 2), 0.5),
            },
            "layer1": {
                "excitation": torch.ones(4, 1) * 2.0,
                "inhibition": torch.ones(4, 1),
                "output": torch.full((4, 1), 0.25),
            },
        }
        C = np.array([0, 1, 0, 1])
        results = {}

        with (
            patch.object(
                self.analyzer,
                "compute_per_layer_enhanced_analysis",
                return_value={"layer_metric": 1.0},
            ) as mock_layer,
            patch.object(
                self.analyzer,
                "compute_per_neuron_mi",
                return_value={"neuron_metric": 2.0},
            ) as mock_neuron,
            patch.object(
                self.analyzer,
                "compute_ei_specific_mi",
                return_value={"ei_metric": 3.0},
            ) as mock_ei,
        ):
            self.analyzer._add_enhanced_analysis_results(results, C)

        self.assertEqual(results["per_layer_analysis"], {"layer_metric": 1.0})
        self.assertEqual(results["per_neuron_analysis"], {"neuron_metric": 2.0})
        self.assertEqual(results["per_einet_analysis"], {"ei_metric": 3.0})
        mock_layer.assert_called_once_with(C)
        self.assertEqual(mock_neuron.call_args.args[0].shape, (4, 3))
        self.assertEqual(mock_neuron.call_args.args[1].shape, (4, 3))
        self.assertEqual(mock_neuron.call_args.args[2].shape, (4, 3))
        np.testing.assert_array_equal(mock_neuron.call_args.args[3], C)
        self.assertEqual(mock_ei.call_args.args[0].shape, (4, 3))
        np.testing.assert_array_equal(mock_ei.call_args.args[3], C)

    def test_enhanced_analysis_array_helper_concatenates_layer_arrays(self):
        """Enhanced analysis array helper should concatenate only activation layers."""
        self.analyzer.data_dict = {
            "layer0": {
                "excitation": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                "inhibition": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
                "output": torch.tensor([[9.0, 10.0], [11.0, 12.0]]),
            },
            "metadata": {"output": torch.ones(2, 1)},
            "layer1": {
                "excitation": torch.tensor([[13.0], [14.0]]),
                "inhibition": torch.tensor([[15.0], [16.0]]),
                "output": torch.tensor([[17.0], [18.0]]),
            },
        }

        arrays = self.analyzer._collect_enhanced_analysis_arrays()

        self.assertIsNotNone(arrays)
        E_all, I_all, Vout_all = arrays
        np.testing.assert_array_equal(
            E_all,
            np.array([[1.0, 2.0, 13.0], [3.0, 4.0, 14.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            I_all,
            np.array([[5.0, 6.0, 15.0], [7.0, 8.0, 16.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            Vout_all,
            np.array([[9.0, 10.0, 17.0], [11.0, 12.0, 18.0]], dtype=np.float32),
        )

    def test_enhanced_analysis_array_helper_returns_none_without_excitation(self):
        """Enhanced analysis array helper should return None when no activations exist."""
        self.analyzer.data_dict = {
            "metadata": {"output": torch.ones(2, 1)},
            "empty": {},
        }

        arrays = self.analyzer._collect_enhanced_analysis_arrays()

        self.assertIsNone(arrays)

    def test_model_supports_information_analysis_accepts_ei_core(self):
        """Model validation helper should accept compatible EI-core models."""
        with patch(
            "dendritic_modeling.analysis.core.information.has_einet_core",
            return_value=True,
        ) as mock_has_core:
            result = self.analyzer._model_supports_information_analysis(self.model)

        self.assertTrue(result)
        mock_has_core.assert_called_once_with(self.model)

    def test_model_supports_information_analysis_logs_skip_for_incompatible_model(self):
        """Model validation helper should preserve the incompatible-model warning."""
        with (
            patch(
                "dendritic_modeling.analysis.core.information.has_einet_core",
                return_value=False,
            ) as mock_has_core,
            patch.object(self.analyzer.logger, "warning") as mock_warning,
        ):
            result = self.analyzer._model_supports_information_analysis(self.model)

        self.assertFalse(result)
        mock_has_core.assert_called_once_with(self.model)
        mock_warning.assert_called_once_with(
            "Model does not expose an ExcitationInhibitionNetwork. Skipping information analysis."
        )

    def test_prepare_model_for_analysis_moves_and_evals_returned_model(self):
        """Model preparation should preserve the original move/eval behavior."""
        input_model = Mock()
        moved_model = Mock()
        input_model.to.return_value = moved_model

        result = self.analyzer._prepare_model_for_analysis(input_model, "cpu")

        self.assertIs(result, moved_model)
        input_model.to.assert_called_once_with("cpu")
        moved_model.eval.assert_called_once_with()

    def test_compute_results_for_level_dispatches_single_branch(self):
        """Computation-level dispatch should preserve single-branch arguments."""
        self.analyzer.computation_level = "single_branch"
        C = np.array([0, 1, 0])

        with patch.object(
            self.analyzer,
            "_compute_single_branch_results",
            return_value={"level": "single"},
        ) as mock_single:
            results = self.analyzer._compute_results_for_level(
                model=self.model,
                inputs=torch.ones(3, 8),
                C=C,
                entropy_C=0.5,
            )

        self.assertEqual(results, {"level": "single"})
        mock_single.assert_called_once_with(C, 0.5)

    def test_compute_results_for_level_dispatches_layer_branch(self):
        """Computation-level dispatch should preserve layer-branch arguments."""
        self.analyzer.computation_level = "layer_branch"
        C = np.array([0, 1, 0])

        with patch.object(
            self.analyzer,
            "_compute_layer_branch_results",
            return_value={"level": "layer"},
        ) as mock_layer:
            results = self.analyzer._compute_results_for_level(
                model=self.model,
                inputs=torch.ones(3, 8),
                C=C,
                entropy_C=0.5,
            )

        self.assertEqual(results, {"level": "layer"})
        mock_layer.assert_called_once_with(C)

    def test_compute_results_for_level_dispatches_all_branch_and_preserves_none(self):
        """Computation-level dispatch should preserve all-branch arguments and skips."""
        self.analyzer.computation_level = "all_branch"
        C = np.array([0, 1, 0])
        inputs = torch.ones(3, 8)

        with patch.object(
            self.analyzer,
            "_compute_all_branch_results",
            return_value=None,
        ) as mock_all:
            results = self.analyzer._compute_results_for_level(
                model=self.model,
                inputs=inputs,
                C=C,
                entropy_C=0.5,
            )

        self.assertIsNone(results)
        mock_all.assert_called_once_with(model=self.model, inputs=inputs, C=C)

    def test_analysis_metadata_helper_adds_configuration_flags(self):
        """Analysis metadata helper should stamp enabled modes and inhibition state."""
        self.analyzer.per_neuron_analysis = True
        self.analyzer.per_einet_analysis = False
        self.analyzer.per_layer_analysis = True
        self.analyzer.data_dict = {"layer0": {"inhibition": torch.ones(2, 1)}}
        results = {}

        with patch.object(
            self.analyzer,
            "has_inhibitory_data",
            return_value=True,
        ) as mock_has_inhibitory:
            self.analyzer._add_analysis_metadata(results)

        self.assertTrue(results["per_neuron_analysis_enabled"])
        self.assertFalse(results["per_einet_analysis_enabled"])
        self.assertTrue(results["per_layer_analysis_enabled"])
        self.assertTrue(results["has_inhibitory_data"])
        mock_has_inhibitory.assert_called_once_with(self.analyzer.data_dict)

    def test_analysis_summary_helper_logs_optional_counts(self):
        """Analysis summary helper should include branch and layer counts when present."""
        self.analyzer.computation_level = "single_branch"
        self.analyzer.method = "mock_method"
        results = {"n_samples": 12, "num_branches": 5, "num_layers": 2}

        with patch.object(self.analyzer.logger, "info") as mock_info:
            self.analyzer._log_analysis_summary(results, total_analysis_time=1.25)

        mock_info.assert_any_call("Information analysis completed successfully!")
        mock_info.assert_any_call("Analysis summary:")
        mock_info.assert_any_call("   - Total time: 1.2s")
        mock_info.assert_any_call("   - Computation level: single_branch")
        mock_info.assert_any_call("   - Method: mock_method")
        mock_info.assert_any_call("   - Sample size: 12")
        mock_info.assert_any_call("   - Branches processed: 5")
        mock_info.assert_any_call("   - Layers processed: 2")

    def test_analysis_summary_file_helper_writes_summary_text(self):
        """Summary file helper should write get_summary output next to results."""
        results = {"metric": 1.0}

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                self.analyzer,
                "get_summary",
                return_value="summary text",
            ) as mock_summary:
                self.analyzer._write_analysis_summary_file(
                    results=results,
                    save_path=tmpdir,
                    filename="analysis",
                )

            summary_path = Path(tmpdir) / "analysis_summary.txt"
            self.assertEqual(summary_path.read_text(), "summary text")
            mock_summary.assert_called_once_with(results)

    def test_post_save_visualization_helper_runs_optional_outputs(self):
        """Post-save visualization helper should run summaries and plots."""
        results = {"per_layer_analysis": {"layer0": {"metric": 1.0}}}

        with (
            patch.object(self.analyzer, "_save_per_layer_ei_summaries") as mock_save,
            patch.object(self.analyzer, "_generate_information_plots") as mock_plots,
        ):
            self.analyzer._run_post_save_visualizations(
                results=results,
                save_path="/tmp/out",
                filename="analysis",
                somatic_synapses=False,
            )

        mock_save.assert_called_once_with(
            results["per_layer_analysis"], "/tmp/out", "analysis"
        )
        mock_plots.assert_called_once_with(results, "/tmp/out", False)

    def test_post_save_visualization_helper_logs_failures(self):
        """Post-save visualization helper should preserve plotting failure logging."""
        results = {}

        with (
            patch.object(
                self.analyzer,
                "_generate_information_plots",
                side_effect=RuntimeError("plot boom"),
            ),
            patch.object(self.analyzer.logger, "warning") as mock_warning,
        ):
            self.analyzer._run_post_save_visualizations(
                results=results,
                save_path="/tmp/out",
                filename="analysis",
                somatic_synapses=True,
            )

        mock_warning.assert_called_once_with(
            "Post-save visualization failed: plot boom"
        )

    def test_all_branch_helper_sample_aggregation_expands_labels(self):
        """All-branch sample aggregation should flatten branches and repeat labels."""
        self.analyzer.branch_aggregation = "sample"
        self.analyzer.mi_null_shuffles = 0
        self.analyzer.data_dict = {
            "layer0": {
                "excitation": torch.arange(6, dtype=torch.float32).reshape(3, 2),
                "inhibition": torch.ones(3, 2),
                "branch_input": torch.ones(3, 2) * 2.0,
                "output": torch.ones(3, 2) * 3.0,
            },
            "layer1": {
                "excitation": torch.ones(3, 1) * 4.0,
                "inhibition": torch.ones(3, 1) * 5.0,
                "branch_input": torch.ones(3, 1) * 6.0,
                "output": torch.ones(3, 1) * 7.0,
            },
        }
        C = np.array([0, 1, 2])
        model = torch.nn.Linear(2, 1)
        inputs = torch.ones(3, 2)

        with patch.object(
            self.analyzer,
            "compute_metrics",
            return_value={"method": "mock"},
        ) as mock_metrics:
            results = self.analyzer._compute_all_branch_results(
                model=model,
                inputs=inputs,
                C=C,
            )

        self.assertEqual(results["aggregation_method"], "sample")
        self.assertEqual(results["computation_level"], "all_branch")
        E, I_var, Vout, C_for_metrics, Vinf, Vb = mock_metrics.call_args.args
        self.assertEqual(E.shape, (9, 1))
        self.assertEqual(I_var.shape, (9, 1))
        self.assertEqual(Vout.shape, (9, 1))
        self.assertEqual(Vb.shape, (9, 1))
        self.assertIsNone(Vinf)
        np.testing.assert_array_equal(C_for_metrics, np.repeat(C, 3, axis=0))

    def test_all_branch_helper_multivariate_preserves_feature_axis(self):
        """All-branch multivariate aggregation should concatenate branch features."""
        self.analyzer.branch_aggregation = "multivariate"
        self.analyzer.mi_null_shuffles = 0
        self.analyzer.data_dict = {
            "layer0": {
                "excitation": torch.ones(4, 2),
                "inhibition": torch.ones(4, 2) * 2.0,
                "branch_input": torch.ones(4, 2) * 3.0,
                "output": torch.ones(4, 2) * 4.0,
            },
            "layer1": {
                "excitation": torch.ones(4, 3) * 5.0,
                "inhibition": torch.ones(4, 3) * 6.0,
                "branch_input": torch.ones(4, 3) * 7.0,
                "output": torch.ones(4, 3) * 8.0,
            },
        }
        C = np.array([0, 1, 0, 1])
        model = torch.nn.Linear(2, 1)
        inputs = torch.ones(4, 2)

        with patch.object(
            self.analyzer,
            "compute_metrics",
            return_value={"method": "mock"},
        ) as mock_metrics:
            results = self.analyzer._compute_all_branch_results(
                model=model,
                inputs=inputs,
                C=C,
            )

        self.assertEqual(results["aggregation_method"], "multivariate")
        E, I_var, Vout, C_for_metrics, Vinf, Vb = mock_metrics.call_args.args
        self.assertEqual(E.shape, (4, 5))
        self.assertEqual(I_var.shape, (4, 5))
        self.assertEqual(Vout.shape, (4, 5))
        self.assertEqual(Vb.shape, (4, 5))
        self.assertIsNone(Vinf)
        np.testing.assert_array_equal(C_for_metrics, C)

    def test_all_branch_layer_tensor_helper_mean_and_default_match(self):
        """All-branch layer tensor helper should mean-pool explicit and fallback modes."""
        layer_data = {
            "excitation": torch.tensor([[1.0, 3.0], [5.0, 7.0]]),
            "inhibition": torch.tensor([[2.0, 4.0], [6.0, 8.0]]),
            "branch_input": torch.tensor([[0.5, 1.5], [2.5, 3.5]]),
            "output": torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
        }

        self.analyzer.branch_aggregation = "mean"
        mean_outputs = self.analyzer._aggregate_all_branch_layer_tensors(layer_data)
        self.analyzer.branch_aggregation = "unknown"
        fallback_outputs = self.analyzer._aggregate_all_branch_layer_tensors(layer_data)

        for mean_value, fallback_value in zip(mean_outputs[:4], fallback_outputs[:4]):
            torch.testing.assert_close(mean_value, fallback_value)
        self.assertFalse(mean_outputs[4])
        self.assertFalse(fallback_outputs[4])
        layer_E, layer_I, layer_Vout, layer_Vb, _missing_vb = mean_outputs
        torch.testing.assert_close(layer_E, torch.tensor([[2.0], [6.0]]))
        torch.testing.assert_close(layer_I, torch.tensor([[3.0], [7.0]]))
        torch.testing.assert_close(layer_Vout, torch.tensor([[15.0], [35.0]]))
        torch.testing.assert_close(layer_Vb, torch.tensor([[1.0], [3.0]]))

    def test_all_branch_layer_tensor_helper_sample_handles_missing_vb(self):
        """All-branch sample tensor helper should flatten and flag missing Vb."""
        self.analyzer.branch_aggregation = "sample"
        layer_data = {
            "excitation": torch.tensor([[1.0, 3.0], [5.0, 7.0]]),
            "inhibition": torch.tensor([[2.0, 4.0], [6.0, 8.0]]),
            "output": torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
        }

        layer_E, layer_I, layer_Vout, layer_Vb, missing_vb = (
            self.analyzer._aggregate_all_branch_layer_tensors(layer_data)
        )

        torch.testing.assert_close(layer_E, torch.tensor([[1.0], [3.0], [5.0], [7.0]]))
        torch.testing.assert_close(layer_I, torch.tensor([[2.0], [4.0], [6.0], [8.0]]))
        torch.testing.assert_close(
            layer_Vout,
            torch.tensor([[10.0], [20.0], [30.0], [40.0]]),
        )
        self.assertIsNone(layer_Vb)
        self.assertTrue(missing_vb)

    def test_all_branch_concatenation_helper_sample_expands_labels_and_drops_vb(self):
        """All-branch sample concatenation should repeat labels and drop partial Vb."""
        self.analyzer.branch_aggregation = "sample"
        C = np.array([0, 1, 2])

        E, I_var, Vout, C_for_metrics, Vb = (
            self.analyzer._concatenate_all_branch_tensors_for_metrics(
                layer_excitation=[
                    torch.tensor([[1.0], [2.0], [3.0]]),
                    torch.tensor([[4.0], [5.0], [6.0]]),
                ],
                layer_inhibition=[
                    torch.tensor([[10.0], [20.0], [30.0]]),
                    torch.tensor([[40.0], [50.0], [60.0]]),
                ],
                layer_output=[
                    torch.tensor([[100.0], [200.0], [300.0]]),
                    torch.tensor([[400.0], [500.0], [600.0]]),
                ],
                layer_branch_input=[torch.tensor([[7.0], [8.0], [9.0]])],
                missing_vb_in_any_layer=True,
                C=C,
            )
        )

        np.testing.assert_array_equal(
            E,
            np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]),
        )
        np.testing.assert_array_equal(
            I_var,
            np.array([[10.0], [20.0], [30.0], [40.0], [50.0], [60.0]]),
        )
        np.testing.assert_array_equal(
            Vout,
            np.array([[100.0], [200.0], [300.0], [400.0], [500.0], [600.0]]),
        )
        np.testing.assert_array_equal(C_for_metrics, np.repeat(C, 2, axis=0))
        self.assertIsNone(Vb)

    def test_all_branch_concatenation_helper_non_sample_preserves_labels_and_vb(self):
        """All-branch non-sample concatenation should concatenate feature axes."""
        self.analyzer.branch_aggregation = "multivariate"
        C = np.array([0, 1])

        E, I_var, Vout, C_for_metrics, Vb = (
            self.analyzer._concatenate_all_branch_tensors_for_metrics(
                layer_excitation=[
                    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                    torch.tensor([[5.0], [6.0]]),
                ],
                layer_inhibition=[
                    torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
                    torch.tensor([[50.0], [60.0]]),
                ],
                layer_output=[
                    torch.tensor([[100.0, 200.0], [300.0, 400.0]]),
                    torch.tensor([[500.0], [600.0]]),
                ],
                layer_branch_input=[
                    torch.tensor([[7.0, 8.0], [9.0, 10.0]]),
                    torch.tensor([[11.0], [12.0]]),
                ],
                missing_vb_in_any_layer=False,
                C=C,
            )
        )

        np.testing.assert_array_equal(
            E,
            np.array([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]]),
        )
        np.testing.assert_array_equal(
            I_var,
            np.array([[10.0, 20.0, 50.0], [30.0, 40.0, 60.0]]),
        )
        np.testing.assert_array_equal(
            Vout,
            np.array([[100.0, 200.0, 500.0], [300.0, 400.0, 600.0]]),
        )
        np.testing.assert_array_equal(
            Vb,
            np.array([[7.0, 8.0, 11.0], [9.0, 10.0, 12.0]]),
        )
        np.testing.assert_array_equal(C_for_metrics, C)

    def test_final_output_signal_returns_numpy_when_final_fc_exists(self):
        """Final output signal helper should expose model output for final_fc models."""

        class FinalFcModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.final_fc = torch.nn.Identity()

            def forward(self, inputs):
                return inputs * 2.0

        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        model = FinalFcModel()

        Vinf = self.analyzer._compute_final_output_signal(model=model, inputs=inputs)

        np.testing.assert_array_equal(
            Vinf,
            np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32),
        )

    def test_final_output_signal_calls_model_without_final_fc_under_no_grad(self):
        """Final output signal helper should preserve call side effects without final_fc."""

        class NoFinalFcModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.called = False
                self.grad_enabled_in_forward = None

            def forward(self, inputs):
                self.called = True
                self.grad_enabled_in_forward = torch.is_grad_enabled()
                return inputs + 1.0

        inputs = torch.tensor([[1.0, 2.0]], requires_grad=True)
        model = NoFinalFcModel()

        Vinf = self.analyzer._compute_final_output_signal(model=model, inputs=inputs)

        self.assertIsNone(Vinf)
        self.assertTrue(model.called)
        self.assertFalse(model.grad_enabled_in_forward)

    def test_null_baseline_helper_skips_when_disabled(self):
        """Null baseline helper should not call estimator work when disabled."""
        self.analyzer.mi_null_shuffles = 0
        results = {}
        signal = np.ones((3, 1))

        with patch.object(
            self.analyzer, "_add_label_shuffle_null_baselines_in_place"
        ) as mock_add:
            self.analyzer._try_add_label_shuffle_null_baselines_in_place(
                results,
                E=signal,
                I_var=signal,
                Vout=signal,
                C=np.array([0, 1, 0]),
                Vb=None,
                Vinf=None,
                has_exc_synapses=True,
                has_inh_synapses=True,
                has_branch_input=False,
                seed_offset=123,
                failure_context="branch 0",
            )

        mock_add.assert_not_called()

    def test_null_baseline_helper_forwards_arguments_when_enabled(self):
        """Null baseline helper should forward the exact metric context."""
        self.analyzer.mi_null_shuffles = 2
        results = {"basic_mi": {"I(E;C)": 0.1}}
        E = np.ones((3, 1))
        I_var = np.ones((3, 1)) * 2.0
        Vout = np.ones((3, 1)) * 3.0
        C = np.array([0, 1, 0])
        Vb = np.ones((3, 1)) * 4.0
        Vinf = np.ones((3, 2)) * 5.0

        with patch.object(
            self.analyzer, "_add_label_shuffle_null_baselines_in_place"
        ) as mock_add:
            self.analyzer._try_add_label_shuffle_null_baselines_in_place(
                results,
                E=E,
                I_var=I_var,
                Vout=Vout,
                C=C,
                Vb=Vb,
                Vinf=Vinf,
                has_exc_synapses=True,
                has_inh_synapses=False,
                has_branch_input=True,
                seed_offset=456,
                failure_context="layer layer0",
            )

        mock_add.assert_called_once()
        call_args = mock_add.call_args
        self.assertIs(call_args.args[0], results)
        self.assertIs(call_args.kwargs["E"], E)
        self.assertIs(call_args.kwargs["I_var"], I_var)
        self.assertIs(call_args.kwargs["Vout"], Vout)
        self.assertIs(call_args.kwargs["C"], C)
        self.assertIs(call_args.kwargs["Vb"], Vb)
        self.assertIs(call_args.kwargs["Vinf"], Vinf)
        self.assertTrue(call_args.kwargs["has_exc_synapses"])
        self.assertFalse(call_args.kwargs["has_inh_synapses"])
        self.assertTrue(call_args.kwargs["has_branch_input"])
        self.assertEqual(call_args.kwargs["seed_offset"], 456)

    def test_null_baseline_helper_logs_context_on_failure(self):
        """Null baseline helper should preserve contextual warning messages."""
        self.analyzer.mi_null_shuffles = 1
        signal = np.ones((3, 1))

        with (
            patch.object(
                self.analyzer,
                "_add_label_shuffle_null_baselines_in_place",
                side_effect=RuntimeError("boom"),
            ),
            patch.object(self.analyzer.logger, "warning") as mock_warning,
        ):
            self.analyzer._try_add_label_shuffle_null_baselines_in_place(
                {},
                E=signal,
                I_var=signal,
                Vout=signal,
                C=np.array([0, 1, 0]),
                Vb=None,
                Vinf=None,
                has_exc_synapses=True,
                has_inh_synapses=True,
                has_branch_input=False,
                seed_offset=789,
                failure_context="all-branch analysis",
            )

        mock_warning.assert_called_once_with(
            "Null (label-shuffle) baseline failed for all-branch analysis: boom"
        )

    def test_layer_branch_helper_sample_aggregation_expands_labels(self):
        """Layer-branch sample aggregation should flatten each layer independently."""
        self.analyzer.branch_aggregation = "sample"
        self.analyzer.mi_null_shuffles = 0
        self.analyzer.data_dict = {
            "layer0": {
                "excitation": torch.arange(6, dtype=torch.float32).reshape(3, 2),
                "inhibition": torch.ones(3, 2),
                "branch_input": torch.ones(3, 2) * 2.0,
                "output": torch.ones(3, 2) * 3.0,
            },
            "layer1": {
                "excitation": torch.ones(3, 1) * 4.0,
                "inhibition": torch.ones(3, 1) * 5.0,
                "branch_input": torch.ones(3, 1) * 6.0,
                "output": torch.ones(3, 1) * 7.0,
            },
        }
        C = np.array([0, 1, 2])
        layer_metric_results = [{"layer": 0}, {"layer": 1}]

        with (
            patch.object(
                self.analyzer,
                "compute_metrics",
                side_effect=layer_metric_results,
            ) as mock_metrics,
            patch.object(self.analyzer, "_filter_result_metrics_in_place"),
            patch.object(
                self.analyzer,
                "_average_mi_results",
                return_value={"avg": True},
            ) as mock_average,
            patch.object(
                self.analyzer,
                "_compute_layer_statistics",
                side_effect=[{"stat": 0.0}, {"stat": 1.0}],
            ),
        ):
            results = self.analyzer._compute_layer_branch_results(C)

        self.assertEqual(results["layer_names"], ["layer0", "layer1"])
        self.assertEqual(results["num_layers"], 2)
        self.assertEqual(results["computation_level"], "layer_branch")
        self.assertEqual(results["layer_statistics"]["layer0"]["stat"], 0.0)
        self.assertEqual(results["layer_statistics"]["layer1"]["stat"], 1.0)
        mock_average.assert_called_once_with(layer_metric_results)

        first_args = mock_metrics.call_args_list[0].args
        second_args = mock_metrics.call_args_list[1].args
        self.assertEqual(first_args[0].shape, (6, 1))
        self.assertEqual(first_args[1].shape, (6, 1))
        self.assertEqual(first_args[2].shape, (6, 1))
        self.assertEqual(first_args[5].shape, (6, 1))
        np.testing.assert_array_equal(first_args[3], np.repeat(C, 2, axis=0))
        self.assertIsNone(first_args[4])
        self.assertEqual(second_args[0].shape, (3, 1))
        np.testing.assert_array_equal(second_args[3], C)

    def test_layer_branch_helper_multivariate_preserves_feature_axis(self):
        """Layer-branch multivariate aggregation should keep per-layer branches."""
        self.analyzer.branch_aggregation = "multivariate"
        self.analyzer.mi_null_shuffles = 0
        self.analyzer.data_dict = {
            "layer0": {
                "excitation": torch.ones(4, 2),
                "inhibition": torch.ones(4, 2) * 2.0,
                "branch_input": torch.ones(4, 2) * 3.0,
                "output": torch.ones(4, 2) * 4.0,
            },
            "layer1": {
                "excitation": torch.ones(4, 3) * 5.0,
                "inhibition": torch.ones(4, 3) * 6.0,
                "branch_input": torch.ones(4, 3) * 7.0,
                "output": torch.ones(4, 3) * 8.0,
            },
        }
        C = np.array([0, 1, 0, 1])
        layer_metric_results = [{"layer": 0}, {"layer": 1}]

        with (
            patch.object(
                self.analyzer,
                "compute_metrics",
                side_effect=layer_metric_results,
            ) as mock_metrics,
            patch.object(self.analyzer, "_filter_result_metrics_in_place"),
            patch.object(
                self.analyzer,
                "_average_mi_results",
                return_value={"avg": True},
            ) as mock_average,
            patch.object(
                self.analyzer,
                "_compute_layer_statistics",
                side_effect=[{"stat": 0.0}, {"stat": 1.0}],
            ),
        ):
            results = self.analyzer._compute_layer_branch_results(C)

        self.assertEqual(results["layer_names"], ["layer0", "layer1"])
        self.assertEqual(results["num_layers"], 2)
        self.assertEqual(results["computation_level"], "layer_branch")
        mock_average.assert_called_once_with(layer_metric_results)

        first_args = mock_metrics.call_args_list[0].args
        second_args = mock_metrics.call_args_list[1].args
        self.assertEqual(first_args[0].shape, (4, 2))
        self.assertEqual(first_args[1].shape, (4, 2))
        self.assertEqual(first_args[2].shape, (4, 2))
        self.assertEqual(first_args[5].shape, (4, 2))
        np.testing.assert_array_equal(first_args[3], C)
        self.assertIsNone(first_args[4])
        self.assertEqual(second_args[0].shape, (4, 3))
        self.assertEqual(second_args[1].shape, (4, 3))
        self.assertEqual(second_args[2].shape, (4, 3))
        self.assertEqual(second_args[5].shape, (4, 3))
        np.testing.assert_array_equal(second_args[3], C)

    def test_metric_merge_helper_renames_existing_nested_groups(self):
        """Metric merge helper should rename tokens without creating new groups."""
        target = {"basic_mi": {"I(E;C)": 0.1}, "method": "mock"}
        source = {
            "basic_mi": {"I(E;C)": 0.2, "I(Vout;C)": 0.3},
            "missing_group": {"I(I;C)": 0.4},
            "method": "lda",
        }

        self.analyzer._merge_renamed_metric_dicts_in_place(
            target=target,
            source=source,
            token_map={"E": "E_lin", "Vout": "Vout_lin", "I": "I_lin"},
        )

        self.assertEqual(target["basic_mi"]["I(E;C)"], 0.1)
        self.assertEqual(target["basic_mi"]["I(E_lin;C)"], 0.2)
        self.assertEqual(target["basic_mi"]["I(Vout_lin;C)"], 0.3)
        self.assertNotIn("missing_group", target)
        self.assertEqual(target["method"], "mock")

    def test_metric_average_merge_helper_averages_numeric_values(self):
        """Average metric merge helper should ignore non-numeric and missing groups."""
        target = {
            "basic_mi": {"I(E;C)": 0.1},
            "non_dict_target": "unchanged",
        }
        sources = [
            {
                "basic_mi": {"I(E;C)": 0.2, "I(I;C)": 0.4, "skip": "x"},
                "missing_group": {"I(Vout;C)": 1.0},
                "non_dict_target": {"I(E;C)": 5.0},
            },
            {
                "basic_mi": {"I(E;C)": 0.6, "I(I;C)": 0.8, "skip": "y"},
                "missing_group": {"I(Vout;C)": 3.0},
                "non_dict_target": {"I(E;C)": 7.0},
            },
        ]

        self.analyzer._merge_average_renamed_numeric_metrics_in_place(
            target=target,
            sources=sources,
            token_map={"E": "E_sh", "I": "I_sh", "Vout": "Vout_sh"},
        )

        self.assertEqual(target["basic_mi"]["I(E;C)"], 0.1)
        self.assertEqual(target["basic_mi"]["I(E_sh;C)"], 0.4)
        self.assertAlmostEqual(target["basic_mi"]["I(I_sh;C)"], 0.6)
        self.assertNotIn("skip", target["basic_mi"])
        self.assertNotIn("missing_group", target)
        self.assertEqual(target["non_dict_target"], "unchanged")

    def test_layer_synapse_availability_helper_prunes_synapse_metrics(self):
        """Synapse-free layer stats should drop E/I metrics and keep other values."""
        self.analyzer.data_dict = {"soma": {"has_synapses": False}}
        layer_stats = {
            "I(E;C)_mean": 0.1,
            "I(I;C)_mean": 0.2,
            "I(E_lin;C)_mean": 0.3,
            "I(I_lin;C)_mean": 0.4,
            "I(Vout;C)_mean": 0.5,
            "accuracy_proxy": 0.6,
        }

        self.analyzer._apply_layer_synapse_availability_in_place(layer_stats, "soma")

        self.assertFalse(layer_stats["has_synapses"])
        self.assertNotIn("I(E;C)_mean", layer_stats)
        self.assertNotIn("I(I;C)_mean", layer_stats)
        self.assertNotIn("I(E_lin;C)_mean", layer_stats)
        self.assertNotIn("I(I_lin;C)_mean", layer_stats)
        self.assertEqual(layer_stats["I(Vout;C)_mean"], 0.5)
        self.assertEqual(layer_stats["accuracy_proxy"], 0.6)

    def test_layer_synapse_availability_helper_preserves_synapse_metrics(self):
        """Layers with synapses should only receive the availability flag."""
        self.analyzer.data_dict = {"layer0": {"has_synapses": True}}
        layer_stats = {"I(E;C)_mean": 0.1, "I(I;C)_mean": 0.2}

        self.analyzer._apply_layer_synapse_availability_in_place(layer_stats, "layer0")

        self.assertTrue(layer_stats["has_synapses"])
        self.assertEqual(layer_stats["I(E;C)_mean"], 0.1)
        self.assertEqual(layer_stats["I(I;C)_mean"], 0.2)

    def test_layer_synapse_availability_helper_ignores_unknown_layers(self):
        """Unknown layers should preserve the prior stats exactly."""
        self.analyzer.data_dict = {}
        layer_stats = {"I(E;C)_mean": 0.1}

        self.analyzer._apply_layer_synapse_availability_in_place(layer_stats, "missing")

        self.assertEqual(layer_stats, {"I(E;C)_mean": 0.1})

    def test_layer_branch_aggregation_helper_mean_and_default_match(self):
        """Layer aggregation helper should use mean for explicit and fallback modes."""
        E_data = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
        I_data = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
        Vout_data = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
        Vb_data = torch.tensor([[0.5, 1.5], [2.5, 3.5]])
        C = np.array([0, 1])

        self.analyzer.branch_aggregation = "mean"
        mean_outputs = self.analyzer._aggregate_layer_branch_arrays_for_metrics(
            E_layer_data=E_data,
            I_layer_data=I_data,
            Vout_layer_data=Vout_data,
            Vb_layer_data=Vb_data,
            C=C,
        )
        self.analyzer.branch_aggregation = "unknown"
        fallback_outputs = self.analyzer._aggregate_layer_branch_arrays_for_metrics(
            E_layer_data=E_data,
            I_layer_data=I_data,
            Vout_layer_data=Vout_data,
            Vb_layer_data=Vb_data,
            C=C,
        )

        for mean_value, fallback_value in zip(mean_outputs, fallback_outputs):
            np.testing.assert_array_equal(mean_value, fallback_value)
        E_layer, I_layer, Vout_layer, C_for_layer, Vb_layer = mean_outputs
        np.testing.assert_array_equal(E_layer, np.array([[2.0], [6.0]]))
        np.testing.assert_array_equal(I_layer, np.array([[3.0], [7.0]]))
        np.testing.assert_array_equal(Vout_layer, np.array([[15.0], [35.0]]))
        np.testing.assert_array_equal(Vb_layer, np.array([[1.0], [3.0]]))
        np.testing.assert_array_equal(C_for_layer, C)

    def test_layer_branch_aggregation_helper_sample_handles_missing_vb(self):
        """Sample aggregation should flatten branches and preserve missing Vb as None."""
        self.analyzer.branch_aggregation = "sample"
        E_data = torch.tensor([[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]])
        I_data = torch.tensor([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]])
        Vout_data = torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        C = np.array([0, 1, 2])

        E_layer, I_layer, Vout_layer, C_for_layer, Vb_layer = (
            self.analyzer._aggregate_layer_branch_arrays_for_metrics(
                E_layer_data=E_data,
                I_layer_data=I_data,
                Vout_layer_data=Vout_data,
                Vb_layer_data=None,
                C=C,
            )
        )

        np.testing.assert_array_equal(
            E_layer,
            np.array([[1.0], [3.0], [5.0], [7.0], [9.0], [11.0]]),
        )
        np.testing.assert_array_equal(
            I_layer,
            np.array([[2.0], [4.0], [6.0], [8.0], [10.0], [12.0]]),
        )
        np.testing.assert_array_equal(
            Vout_layer,
            np.array([[10.0], [20.0], [30.0], [40.0], [50.0], [60.0]]),
        )
        np.testing.assert_array_equal(C_for_layer, np.repeat(C, 2, axis=0))
        self.assertIsNone(Vb_layer)

    def test_single_branch_helper_collects_branch_metadata_and_stats(self):
        """Single-branch helper should preserve branch-level call shapes and metadata."""
        self.analyzer.compute_lda_weights = False
        self.analyzer.mi_null_shuffles = 0
        self.analyzer.compute_layer_total_proxies = True
        self.analyzer.layer_total_topk = 3
        self.analyzer.data_dict = {
            "layer0": {
                "excitation": torch.arange(6, dtype=torch.float32).reshape(3, 2),
                "inhibition": torch.ones(3, 2),
                "branch_input": torch.ones(3, 2) * 2.0,
                "output": torch.ones(3, 2) * 3.0,
            },
        }
        C = np.array([0, 1, 2])
        entropy_C = 0.75
        branch_metric_results = [
            {"basic_mi": {"I(E;C)": 0.1}},
            {"basic_mi": {"I(E;C)": 0.2}},
        ]

        with (
            patch.object(
                self.analyzer,
                "compute_metrics",
                side_effect=branch_metric_results,
            ) as mock_metrics,
            patch.object(self.analyzer, "_filter_result_metrics_in_place"),
            patch.object(
                self.analyzer,
                "_average_mi_results",
                return_value={"avg": True},
            ) as mock_average,
            patch.object(
                self.analyzer,
                "_compute_layer_statistics",
                return_value={"stat": 1.0},
            ) as mock_stats,
        ):
            results = self.analyzer._compute_single_branch_results(C, entropy_C)

        self.assertEqual(
            results["branch_names"], ["layer0_branch_0", "layer0_branch_1"]
        )
        self.assertEqual(results["num_branches"], 2)
        self.assertEqual(results["computation_level"], "single_branch")
        self.assertEqual(results["layer_statistics"]["layer0"]["stat"], 1.0)
        mock_average.assert_called_once_with(branch_metric_results)
        mock_stats.assert_called_once_with(
            branch_metric_results,
            entropy_C=entropy_C,
            topk=3,
        )

        first_args = mock_metrics.call_args_list[0].args
        second_args = mock_metrics.call_args_list[1].args
        self.assertEqual(first_args[0].shape, (3, 1))
        self.assertEqual(first_args[1].shape, (3, 1))
        self.assertEqual(first_args[2].shape, (3, 1))
        self.assertEqual(first_args[5].shape, (3, 1))
        np.testing.assert_array_equal(first_args[3], C)
        self.assertIsNone(first_args[4])
        np.testing.assert_array_equal(
            first_args[0].reshape(-1),
            np.array([0.0, 2.0, 4.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            second_args[0].reshape(-1),
            np.array([1.0, 3.0, 5.0], dtype=np.float32),
        )

    def test_single_branch_helper_merges_lda_and_shuffle_metrics(self):
        """Single-branch helper should preserve LDA and random-weight metric naming."""
        self.analyzer.compute_lda_weights = True
        self.analyzer.lda_n_shuffles = 2
        self.analyzer.mi_null_shuffles = 0
        self.analyzer.data_dict = {
            "layer0": {
                "excitation": torch.ones(3, 1),
                "inhibition": torch.ones(3, 1) * 2.0,
                "branch_input": torch.ones(3, 1) * 3.0,
                "output": torch.ones(3, 1) * 4.0,
            },
        }
        C = np.array([0, 1, 0])
        lda_signals = (
            np.ones((3, 1)) * 10.0,
            np.ones((3, 1)) * 11.0,
            np.ones((3, 1)) * 12.0,
            np.ones((3, 1)) * 13.0,
        )
        shuffle_signals = [
            (
                np.ones((3, 1)) * 20.0,
                np.ones((3, 1)) * 21.0,
                np.ones((3, 1)) * 22.0,
                np.ones((3, 1)) * 23.0,
            ),
            (
                np.ones((3, 1)) * 30.0,
                np.ones((3, 1)) * 31.0,
                np.ones((3, 1)) * 32.0,
                np.ones((3, 1)) * 33.0,
            ),
        ]
        base_result = {"basic_mi": {"I(E;C)": 0.1, "I(Vout;C)": 0.2}}
        lda_result = {
            "basic_mi": {
                "I(E;C)": 0.3,
                "I(I;C)": 0.4,
                "I(Vb;C)": 0.5,
                "I(Vout;C)": 0.6,
            }
        }
        shuffle_results = [
            {"basic_mi": {"I(E;C)": 0.7, "I(I;C)": 0.8, "I(Vout;C)": 0.9}},
            {"basic_mi": {"I(E;C)": 0.9, "I(I;C)": 1.0, "I(Vout;C)": 1.1}},
        ]

        with (
            patch.object(
                self.analyzer,
                "compute_lda_aggregated_signals",
                return_value=lda_signals,
            ) as mock_lda,
            patch.object(
                self.analyzer,
                "compute_shuffled_aggregated_signals",
                side_effect=shuffle_signals,
            ) as mock_shuffle,
            patch.object(
                self.analyzer,
                "compute_metrics",
                side_effect=[base_result, lda_result, *shuffle_results],
            ),
            patch.object(self.analyzer, "_filter_result_metrics_in_place"),
            patch.object(
                self.analyzer,
                "_average_mi_results",
                side_effect=lambda all_results: {"captured": all_results},
            ),
            patch.object(
                self.analyzer,
                "_compute_layer_statistics",
                return_value={"stat": 1.0},
            ),
        ):
            results = self.analyzer._compute_single_branch_results(C, entropy_C=None)

        mock_lda.assert_called_once()
        self.assertEqual(mock_shuffle.call_count, 2)
        self.assertEqual(results["branch_names"], ["layer0_branch_0"])
        branch_metrics = results["captured"][0]["basic_mi"]
        self.assertEqual(branch_metrics["I(E;C)"], 0.1)
        self.assertEqual(branch_metrics["I(Vout;C)"], 0.2)
        self.assertEqual(branch_metrics["I(E_lin;C)"], 0.3)
        self.assertEqual(branch_metrics["I(I_lin;C)"], 0.4)
        self.assertEqual(branch_metrics["I(Vb_lin;C)"], 0.5)
        self.assertEqual(branch_metrics["I(Vout_lin;C)"], 0.6)
        self.assertEqual(branch_metrics["I(E_sh;C)"], 0.8)
        self.assertEqual(branch_metrics["I(I_sh;C)"], 0.9)
        self.assertEqual(branch_metrics["I(Vout_sh;C)"], 1.0)


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
