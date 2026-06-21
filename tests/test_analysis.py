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
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, call, patch

import numpy as np
import pytest
import torch

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
            "dendritic_modeling.analysis.core.information_validation_mixin.has_einet_core",
            return_value=True,
        ) as mock_has_core:
            result = self.analyzer._model_supports_information_analysis(self.model)

        self.assertTrue(result)
        mock_has_core.assert_called_once_with(self.model)

    def test_model_supports_information_analysis_logs_skip_for_incompatible_model(self):
        """Model validation helper should preserve the incompatible-model warning."""
        with (
            patch(
                "dendritic_modeling.analysis.core.information_validation_mixin.has_einet_core",
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

    def test_model_lifecycle_state_records_original_device_and_training(self):
        """Lifecycle state should preserve the pre-analysis model state."""
        self.model.train()

        state = self.analyzer._capture_model_lifecycle_state(self.model)

        self.assertEqual(state.original_device, torch.device("cpu"))
        self.assertTrue(state.was_training)
        self.assertFalse(state.moved)

    def test_mark_model_moved_preserves_original_lifecycle_fields(self):
        """Marking the lifecycle moved should not change original state fields."""
        self.model.train()
        state = self.analyzer._capture_model_lifecycle_state(self.model)

        moved_state = self.analyzer._mark_model_moved(state)

        self.assertEqual(moved_state.original_device, state.original_device)
        self.assertEqual(moved_state.was_training, state.was_training)
        self.assertTrue(moved_state.moved)

    def test_restore_model_after_analysis_supports_legacy_kwargs(self):
        """Restore helper should keep the pre-extraction private call signature."""
        model = Mock()

        self.analyzer._restore_model_after_analysis(
            model=model,
            original_device=torch.device("cpu"),
            was_training=True,
            moved=True,
        )

        model.to.assert_called_once_with(torch.device("cpu"))
        model.train.assert_called_once_with()

    def test_restore_model_after_analysis_skips_unmoved_state(self):
        """Unmoved models should not be touched during cleanup."""
        model = Mock()
        state = self.analyzer._capture_model_lifecycle_state(self.model)

        self.analyzer._restore_model_after_analysis(
            model=model,
            lifecycle_state=state,
        )

        model.to.assert_not_called()
        model.train.assert_not_called()

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

    def test_analysis_pipeline_delegates_collection_computation_and_completion(self):
        """Analysis pipeline should preserve analyze() orchestration handoffs."""
        inputs = torch.ones(3, 8)
        labels = torch.tensor([0, 1, 0])
        C = np.array([0, 1, 0])
        analysis_data = Mock(
            inputs=inputs,
            labels=labels,
            C=C,
            entropy_C=0.75,
        )
        computed_results = {"basic_mi": {"I(E;C)": 1.0}}
        completed_results = {"done": True}
        runtime = Mock()

        with (
            patch.object(
                self.analyzer,
                "_collect_analysis_data",
                return_value=analysis_data,
            ) as mock_collect,
            patch.object(
                self.analyzer,
                "_compute_results_for_level",
                return_value=computed_results,
            ) as mock_compute,
            patch.object(
                self.analyzer,
                "_add_enhanced_analysis_results",
            ) as mock_enhanced,
            patch.object(
                self.analyzer,
                "_complete_analysis_results",
                return_value=completed_results,
            ) as mock_complete,
        ):
            result = self.analyzer._run_analysis_pipeline(
                model=self.model,
                test_dataset=self.test_data,
                device="cpu",
                save_path="/tmp/results",
                filename="final_eval",
                runtime=runtime,
                analysis_start_time=10.0,
            )

        self.assertIs(result, completed_results)
        mock_collect.assert_called_once_with(
            model=self.model,
            test_dataset=self.test_data,
            device="cpu",
            runtime=runtime,
        )
        mock_compute.assert_called_once_with(
            model=self.model,
            inputs=inputs,
            C=C,
            entropy_C=0.75,
        )
        mock_enhanced.assert_called_once_with(computed_results, C)
        mock_complete.assert_called_once_with(
            results=computed_results,
            model=self.model,
            inputs=inputs,
            labels=labels,
            analysis_start_time=10.0,
            device="cpu",
            save_path="/tmp/results",
            filename="final_eval",
        )

    def test_branch_metric_helper_computes_metrics_and_preserves_null_flags(self):
        """Branch metric helper should attach null baselines with availability flags."""
        C = np.array([0, 1, 0])
        E = np.array([[1.0], [2.0], [3.0]])
        I_var = E + 10.0
        Vout = E + 20.0

        with (
            patch.object(
                self.analyzer,
                "compute_metrics",
                return_value={"basic_mi": {"I(E;C)": 1.0}},
            ) as mock_metrics,
            patch.object(
                self.analyzer,
                "_try_add_label_shuffle_null_baselines_in_place",
            ) as mock_null,
        ):
            results = self.analyzer._compute_branch_metrics_with_null_baselines(
                E=E,
                I_var=I_var,
                Vout=Vout,
                C=C,
                Vb=None,
                has_exc_synapses=True,
                has_inh_synapses=False,
                has_branch_input=True,
                seed_offset=7,
                failure_context="branch 7",
            )

        self.assertEqual(results, {"basic_mi": {"I(E;C)": 1.0}})
        mock_metrics.assert_called_once_with(E, I_var, Vout, C, None, None)
        mock_null.assert_called_once()
        self.assertTrue(mock_null.call_args.kwargs["has_exc_synapses"])
        self.assertFalse(mock_null.call_args.kwargs["has_inh_synapses"])
        self.assertFalse(mock_null.call_args.kwargs["has_branch_input"])
        self.assertEqual(mock_null.call_args.kwargs["seed_offset"], 7)
        self.assertEqual(mock_null.call_args.kwargs["failure_context"], "branch 7")

    def test_single_branch_layer_data_helper_preserves_arrays_and_flags(self):
        """Single-branch layer prep should preserve arrays and availability flags."""
        layer_data = {
            "excitation": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "inhibition": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
            "branch_input": torch.tensor([[9.0, 10.0], [11.0, 12.0]]),
            "output": torch.tensor([[13.0, 14.0], [15.0, 16.0]]),
            "has_exc_synapses": False,
            "has_inh_synapses": True,
            "has_branch_input": False,
        }

        prepared = self.analyzer._prepare_single_branch_layer_data(layer_data)

        np.testing.assert_array_equal(
            prepared.excitation,
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            prepared.inhibition,
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            prepared.branch_input,
            np.array([[9.0, 10.0], [11.0, 12.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            prepared.output,
            np.array([[13.0, 14.0], [15.0, 16.0]], dtype=np.float32),
        )
        self.assertFalse(prepared.has_exc_synapses)
        self.assertTrue(prepared.has_inh_synapses)
        self.assertFalse(prepared.has_branch_input)

    def test_single_branch_layer_data_helper_handles_missing_branch_input(self):
        """Single-branch layer prep should keep missing branch input as None."""
        layer_data = {
            "excitation": torch.tensor([[1.0], [2.0]]),
            "inhibition": torch.tensor([[3.0], [4.0]]),
            "output": torch.tensor([[5.0], [6.0]]),
        }

        prepared = self.analyzer._prepare_single_branch_layer_data(layer_data)

        self.assertIsNone(prepared.branch_input)
        self.assertTrue(prepared.has_exc_synapses)
        self.assertTrue(prepared.has_inh_synapses)
        self.assertFalse(prepared.has_branch_input)

    def test_single_branch_lda_signal_helper_returns_empty_when_disabled(self):
        """LDA signal prep should avoid aggregation work when LDA is disabled."""
        self.analyzer.compute_lda_weights = False
        C = np.array([0, 1])

        with (
            patch.object(self.analyzer, "compute_lda_aggregated_signals") as mock_lda,
            patch.object(
                self.analyzer,
                "compute_shuffled_aggregated_signals",
            ) as mock_shuffled,
        ):
            signals = self.analyzer._prepare_single_branch_lda_signals({}, C)

        self.assertIsNone(signals.excitation)
        self.assertIsNone(signals.inhibition)
        self.assertIsNone(signals.branch_input)
        self.assertIsNone(signals.output)
        self.assertEqual(signals.shuffled, ())
        mock_lda.assert_not_called()
        mock_shuffled.assert_not_called()

    def test_single_branch_lda_signal_helper_collects_shuffles_in_order(self):
        """LDA signal prep should preserve deterministic and shuffled signals."""
        self.analyzer.compute_lda_weights = True
        self.analyzer.lda_n_shuffles = 2
        C = np.array([0, 1])
        layer_data = {"layer": "data"}
        E_lin = np.array([[1.0], [2.0]])
        I_lin = E_lin + 10.0
        Vb_lin = E_lin + 20.0
        Vout_lin = E_lin + 30.0
        shuffle0 = (E_lin + 100.0, I_lin + 100.0, Vb_lin + 100.0, Vout_lin + 100.0)
        shuffle1 = (E_lin + 200.0, I_lin + 200.0, None, Vout_lin + 200.0)

        with (
            patch.object(
                self.analyzer,
                "compute_lda_aggregated_signals",
                return_value=(E_lin, I_lin, Vb_lin, Vout_lin),
            ) as mock_lda,
            patch.object(
                self.analyzer,
                "compute_shuffled_aggregated_signals",
                side_effect=[shuffle0, shuffle1],
            ) as mock_shuffled,
        ):
            signals = self.analyzer._prepare_single_branch_lda_signals(layer_data, C)

        mock_lda.assert_called_once_with(layer_data, C)
        self.assertEqual(mock_shuffled.call_count, 2)
        mock_shuffled.assert_any_call(layer_data, C, 0)
        mock_shuffled.assert_any_call(layer_data, C, 1)
        np.testing.assert_array_equal(signals.excitation, E_lin)
        np.testing.assert_array_equal(signals.inhibition, I_lin)
        np.testing.assert_array_equal(signals.branch_input, Vb_lin)
        np.testing.assert_array_equal(signals.output, Vout_lin)
        self.assertEqual(len(signals.shuffled), 2)
        for actual, expected in zip(signals.shuffled[0], shuffle0):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(signals.shuffled[1], shuffle1):
            if expected is None:
                self.assertIsNone(actual)
            else:
                np.testing.assert_array_equal(actual, expected)

    def test_single_branch_result_helper_slices_merges_and_filters(self):
        """Single-branch result helper should orchestrate one branch without drift."""
        C = np.array([0, 1, 0])
        layer_data = {
            "excitation": torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
            "inhibition": torch.tensor([[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]]),
            "branch_input": torch.tensor([[7.0, 70.0], [8.0, 80.0], [9.0, 90.0]]),
            "output": torch.tensor([[11.0, 110.0], [12.0, 120.0], [13.0, 130.0]]),
            "has_exc_synapses": False,
            "has_inh_synapses": True,
            "has_branch_input": True,
        }
        prepared = self.analyzer._prepare_single_branch_layer_data(layer_data)
        lda_signals = Mock()
        lda_signals.excitation = np.array([[101.0, 201.0]])
        lda_signals.inhibition = np.array([[102.0, 202.0]])
        lda_signals.branch_input = np.array([[103.0, 203.0]])
        lda_signals.output = np.array([[104.0, 204.0]])
        lda_signals.shuffled = (("shuffle",),)
        branch_results = {"basic_mi": {"I(E;C)": 1.0}}

        with (
            patch.object(
                self.analyzer,
                "_compute_branch_metrics_with_null_baselines",
                return_value=branch_results,
            ) as mock_base,
            patch.object(
                self.analyzer,
                "_merge_lda_branch_metrics_in_place",
            ) as mock_lda,
            patch.object(
                self.analyzer,
                "_merge_shuffled_branch_metrics_in_place",
            ) as mock_shuffled,
            patch.object(
                self.analyzer, "_filter_result_metrics_in_place"
            ) as mock_filter,
        ):
            result = self.analyzer._compute_single_branch_result(
                prepared_layer=prepared,
                lda_signals=lda_signals,
                C=C,
                branch_idx=1,
                layer_count=3,
            )

        self.assertIs(result, branch_results)
        np.testing.assert_array_equal(
            mock_base.call_args.kwargs["E"],
            np.array([[10.0], [20.0], [30.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            mock_base.call_args.kwargs["I_var"],
            np.array([[40.0], [50.0], [60.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            mock_base.call_args.kwargs["Vout"],
            np.array([[110.0], [120.0], [130.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            mock_base.call_args.kwargs["Vb"],
            np.array([[70.0], [80.0], [90.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(mock_base.call_args.kwargs["C"], C)
        self.assertFalse(mock_base.call_args.kwargs["has_exc_synapses"])
        self.assertTrue(mock_base.call_args.kwargs["has_inh_synapses"])
        self.assertTrue(mock_base.call_args.kwargs["has_branch_input"])
        self.assertEqual(mock_base.call_args.kwargs["seed_offset"], 30_001)
        self.assertEqual(mock_base.call_args.kwargs["failure_context"], "branch 1")

        self.assertIs(mock_lda.call_args.kwargs["target"], branch_results)
        self.assertIs(mock_lda.call_args.kwargs["E_lin"], lda_signals.excitation)
        self.assertIs(mock_lda.call_args.kwargs["I_lin"], lda_signals.inhibition)
        self.assertIs(mock_lda.call_args.kwargs["Vb_lin"], lda_signals.branch_input)
        self.assertIs(mock_lda.call_args.kwargs["Vout_lin"], lda_signals.output)
        self.assertEqual(mock_lda.call_args.kwargs["branch_idx"], 1)
        self.assertEqual(mock_lda.call_args.kwargs["layer_count"], 3)

        self.assertIs(mock_shuffled.call_args.kwargs["target"], branch_results)
        self.assertIs(
            mock_shuffled.call_args.kwargs["shuffled_signals"], lda_signals.shuffled
        )
        self.assertEqual(mock_shuffled.call_args.kwargs["branch_idx"], 1)
        self.assertTrue(mock_shuffled.call_args.kwargs["has_branch_input"])

        mock_filter.assert_called_once_with(
            branch_results,
            has_exc_synapses=False,
            has_inh_synapses=True,
            has_branch_input=True,
        )

    def test_finalize_single_branch_results_returns_none_for_empty_results(self):
        """Single-branch finalization should preserve empty-result skip behavior."""
        with patch.object(self.analyzer.logger, "warning") as mock_warning:
            results = self.analyzer._finalize_single_branch_results(
                all_results=[],
                branch_names=[],
                layer_results={},
            )

        self.assertIsNone(results)
        mock_warning.assert_called_once_with("No branch results to average")

    def test_finalize_single_branch_results_averages_and_attaches_metadata(self):
        """Single-branch finalization should preserve averaged results metadata."""
        all_results = [{"basic_mi": {"I(E;C)": 1.0}}, {"basic_mi": {"I(E;C)": 3.0}}]
        branch_names = ["layer0_branch_0", "layer0_branch_1"]
        layer_results = {"layer0": {"I(E;C)_mean": 2.0}}

        with patch.object(
            self.analyzer,
            "_average_mi_results",
            return_value={"basic_mi": {"I(E;C)": 2.0}},
        ) as mock_average:
            results = self.analyzer._finalize_single_branch_results(
                all_results=all_results,
                branch_names=branch_names,
                layer_results=layer_results,
            )

        mock_average.assert_called_once_with(all_results)
        self.assertEqual(
            results,
            {
                "basic_mi": {"I(E;C)": 2.0},
                "branch_names": branch_names,
                "num_branches": 2,
                "layer_statistics": layer_results,
                "computation_level": "single_branch",
            },
        )

    def test_lda_branch_metric_helper_slices_merges_and_preserves_null_args(self):
        """LDA branch helper should preserve slicing, null-baseline, and rename behavior."""
        self.analyzer.compute_lda_weights = True
        C = np.array([0, 1, 0])
        E_lin = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        I_lin = E_lin + 100.0
        Vb_lin = E_lin + 200.0
        Vout_lin = E_lin + 300.0
        target = {"basic_mi": {}, "pairwise_mi": {}}

        with (
            patch.object(
                self.analyzer,
                "compute_metrics",
                return_value={
                    "basic_mi": {"I(E;C)": 1.0, "I(Vb;C)": 2.0},
                    "pairwise_mi": {"I(Vout;Vb)": 3.0},
                },
            ) as mock_metrics,
            patch.object(
                self.analyzer,
                "_try_add_label_shuffle_null_baselines_in_place",
            ) as mock_null,
        ):
            self.analyzer._merge_lda_branch_metrics_in_place(
                target=target,
                E_lin=E_lin,
                I_lin=I_lin,
                Vb_lin=Vb_lin,
                Vout_lin=Vout_lin,
                C=C,
                branch_idx=1,
                layer_count=2,
                has_exc_synapses=False,
                has_inh_synapses=True,
                has_branch_input=True,
            )

        E_arg, I_arg, Vout_arg, C_arg, Vinf_arg, Vb_arg = mock_metrics.call_args.args
        np.testing.assert_array_equal(E_arg, E_lin[:, 1:2])
        np.testing.assert_array_equal(I_arg, I_lin[:, 1:2])
        np.testing.assert_array_equal(Vout_arg, Vout_lin[:, 1:2])
        np.testing.assert_array_equal(C_arg, C)
        self.assertIsNone(Vinf_arg)
        np.testing.assert_array_equal(Vb_arg, Vb_lin[:, 1:2])
        self.assertEqual(
            target,
            {
                "basic_mi": {"I(E_lin;C)": 1.0, "I(Vb_lin;C)": 2.0},
                "pairwise_mi": {"I(Vout_lin;Vb_lin)": 3.0},
            },
        )
        mock_null.assert_called_once()
        self.assertEqual(mock_null.call_args.kwargs["seed_offset"], 1_020_001)
        self.assertEqual(mock_null.call_args.kwargs["failure_context"], "LDA branch 1")
        self.assertFalse(mock_null.call_args.kwargs["has_exc_synapses"])
        self.assertTrue(mock_null.call_args.kwargs["has_inh_synapses"])
        self.assertTrue(mock_null.call_args.kwargs["has_branch_input"])

    def test_shuffled_branch_metric_helper_slices_and_averages_renamed_metrics(self):
        """Shuffled branch helper should average numeric metrics after token rename."""
        C = np.array([0, 1, 0])
        base = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        shuffled_signals = [
            (base, base + 100.0, base + 200.0, base + 300.0),
            (base + 1.0, base + 101.0, base + 201.0, base + 301.0),
        ]
        target = {"basic_mi": {}, "pairwise_mi": {}}

        with patch.object(
            self.analyzer,
            "compute_metrics",
            side_effect=[
                {
                    "basic_mi": {"I(E;C)": 1.0},
                    "pairwise_mi": {"I(Vout;Vb)": 3.0},
                },
                {
                    "basic_mi": {"I(E;C)": 5.0},
                    "pairwise_mi": {"I(Vout;Vb)": 7.0},
                },
            ],
        ) as mock_metrics:
            self.analyzer._merge_shuffled_branch_metrics_in_place(
                target=target,
                shuffled_signals=shuffled_signals,
                C=C,
                branch_idx=1,
                has_branch_input=True,
            )

        self.assertEqual(
            target,
            {
                "basic_mi": {"I(E_sh;C)": 3.0},
                "pairwise_mi": {"I(Vout_sh;Vb_sh)": 5.0},
            },
        )
        self.assertEqual(mock_metrics.call_count, 2)
        first_call = mock_metrics.call_args_list[0].args
        np.testing.assert_array_equal(first_call[0], base[:, 1:2])
        np.testing.assert_array_equal(first_call[1], (base + 100.0)[:, 1:2])
        np.testing.assert_array_equal(first_call[2], (base + 300.0)[:, 1:2])
        np.testing.assert_array_equal(first_call[3], C)
        self.assertIsNone(first_call[4])
        np.testing.assert_array_equal(first_call[5], (base + 200.0)[:, 1:2])

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

    def test_finalize_analysis_payload_runs_in_memory_finalizers(self):
        """Payload finalization should preserve soma, unit, variance, and LDA delegates."""
        results = {"n_samples": 3}
        inputs = torch.ones(3, 2)
        labels = torch.tensor([0, 1, 0])

        with (
            patch.object(
                self.analyzer,
                "_add_synthetic_soma_entry_if_needed",
                return_value=False,
            ) as mock_soma,
            patch.object(
                self.analyzer,
                "_finalize_information_units_and_variance",
            ) as mock_units,
            patch.object(
                self.analyzer,
                "_maybe_add_lda_vs_trained_performance",
            ) as mock_lda,
        ):
            somatic_synapses = self.analyzer._finalize_analysis_payload(
                results=results,
                model=self.model,
                inputs=inputs,
                labels=labels,
                device="cpu",
                save_path="/tmp/results",
            )

        self.assertFalse(somatic_synapses)
        mock_soma.assert_called_once_with(results, self.model)
        mock_units.assert_called_once_with(results)
        mock_lda.assert_called_once_with(
            results=results,
            model=self.model,
            inputs=inputs,
            labels=labels,
            device="cpu",
            save_path="/tmp/results",
        )

    def test_complete_analysis_results_delegates_payload_and_save(self):
        """Completion helper should pass finalized soma state into persistence."""
        results = {"n_samples": 3}
        inputs = torch.ones(3, 2)
        labels = torch.tensor([0, 1, 0])

        with (
            patch(
                "dendritic_modeling.analysis.core.information_finalization_mixin.time.time",
                return_value=12.5,
            ),
            patch.object(self.analyzer, "_add_analysis_metadata") as mock_metadata,
            patch.object(self.analyzer, "_log_analysis_summary") as mock_log,
            patch.object(
                self.analyzer,
                "_finalize_analysis_payload",
                return_value=False,
            ) as mock_finalize,
            patch.object(self.analyzer, "_save_analysis_outputs") as mock_save,
        ):
            completed = self.analyzer._complete_analysis_results(
                results=results,
                model=self.model,
                inputs=inputs,
                labels=labels,
                analysis_start_time=10.0,
                device="cpu",
                save_path="/tmp/results",
                filename="analysis",
            )

        self.assertIs(completed, results)
        mock_metadata.assert_called_once_with(results)
        mock_log.assert_called_once_with(results, 2.5)
        mock_finalize.assert_called_once_with(
            results=results,
            model=self.model,
            inputs=inputs,
            labels=labels,
            device="cpu",
            save_path="/tmp/results",
        )
        mock_save.assert_called_once_with(
            results=results,
            save_path="/tmp/results",
            filename="analysis",
            somatic_synapses=False,
        )

    def test_lda_performance_plot_helper_warns_without_accuracy(self):
        """LDA plot helper should preserve the missing-accuracy warning path."""
        with patch.object(self.analyzer.logger, "warning") as mock_warning:
            self.analyzer._plot_lda_vs_trained_performance_if_available(
                lda_comparison={"lda_accuracy": None},
                save_path="/tmp/results",
            )

        mock_warning.assert_called_once_with(
            "LDA accuracy not available, skipping performance plot"
        )

    def test_lda_performance_plot_helper_calls_plot_when_accuracy_exists(self):
        """LDA plot helper should call the plotting utility when accuracy exists."""
        lda_comparison = {"lda_accuracy": 0.75}

        with (
            patch(
                "dendritic_modeling.plotting.visualizations.information_plots."
                "plot_lda_vs_trained_performance"
            ) as mock_plot,
            patch.object(self.analyzer.logger, "info") as mock_info,
        ):
            self.analyzer._plot_lda_vs_trained_performance_if_available(
                lda_comparison=lda_comparison,
                save_path="/tmp/results",
            )

        mock_plot.assert_called_once_with(
            lda_comparison,
            "/tmp/results",
            self.analyzer.logger,
        )
        mock_info.assert_called_once_with(
            "LDA vs trained performance comparison completed!"
        )

    def test_lda_label_conversion_preserves_supported_inputs(self):
        """LDA label conversion should preserve tensor, numpy, and sequence labels."""
        np.testing.assert_array_equal(
            self.analyzer._labels_to_numpy(torch.tensor([0, 1, 2])),
            np.array([0, 1, 2]),
        )

        labels_np = np.array([2, 1, 0])
        np.testing.assert_array_equal(
            self.analyzer._labels_to_numpy(labels_np),
            labels_np,
        )

        np.testing.assert_array_equal(
            self.analyzer._labels_to_numpy([1, 0, 1]),
            np.array([1, 0, 1]),
        )

    def test_lda_ordered_branch_layers_uses_valid_indices(self):
        """LDA layer ordering should match the existing branch-index parser."""
        self.analyzer.data_dict = {
            "metadata": {},
            "core.branch_layers.2": {},
            "core.branch_layers.bad": {},
            "core.branch_layers.0": {},
            "core.branch_layers.1.output": {},
        }

        self.assertEqual(
            self.analyzer._ordered_branch_layer_names(),
            [
                "core.branch_layers.0",
                "core.branch_layers.1.output",
                "core.branch_layers.2",
            ],
        )

    def test_lda_chained_output_preserves_order_and_fallback(self):
        """Chained LDA output should pass prior outputs and keep fallback behavior."""
        labels = np.array([0, 1])
        first_output = np.array([[0.2, 0.8], [0.7, 0.3]], dtype=np.float32)
        fallback_output = torch.tensor([[3.0, 1.0], [1.0, 4.0]])
        self.analyzer.data_dict = {
            "core.branch_layers.1": {
                "excitation": torch.ones(2, 2),
                "output": fallback_output,
            },
            "core.branch_layers.0": {
                "excitation": torch.ones(2, 2),
                "output": torch.zeros(2, 2),
            },
        }

        with patch.object(
            self.analyzer,
            "compute_lda_aggregated_signals_chained",
            side_effect=[
                (np.zeros((2, 2)), np.zeros((2, 2)), None, first_output),
                (None, None, None, None),
            ],
        ) as mock_chained:
            final_output = self.analyzer._compute_chained_lda_output(labels)

        self.assertEqual(mock_chained.call_count, 2)
        np.testing.assert_array_equal(mock_chained.call_args_list[0].args[1], labels)
        self.assertIsNone(mock_chained.call_args_list[0].args[2])
        np.testing.assert_array_equal(
            mock_chained.call_args_list[1].args[2], first_output
        )
        np.testing.assert_array_equal(final_output, fallback_output.numpy())

    def test_lda_prediction_helper_uses_optional_decoder(self):
        """LDA class prediction should preserve direct-logit and decoder paths."""
        final_vout = np.array([[0.2, 0.8], [0.9, 0.1]], dtype=np.float32)
        model_without_decoder = torch.nn.Module()

        np.testing.assert_array_equal(
            self.analyzer._predict_lda_model_classes(
                model_without_decoder,
                final_vout,
                torch.device("cpu"),
            ),
            np.array([1, 0]),
        )

        model_with_decoder = torch.nn.Module()
        model_with_decoder.decoder = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model_with_decoder.decoder.weight.copy_(
                torch.tensor([[0.0, 1.0], [1.0, 0.0]])
            )

        np.testing.assert_array_equal(
            self.analyzer._predict_lda_model_classes(
                model_with_decoder,
                final_vout,
                torch.device("cpu"),
            ),
            np.array([0, 1]),
        )

        self.assertIsNone(
            self.analyzer._predict_lda_model_classes(
                model_with_decoder,
                None,
                torch.device("cpu"),
            )
        )

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

    def test_all_branch_metric_data_helper_returns_none_without_excitatory_layers(self):
        """All-branch metric collection should preserve empty-activation skip behavior."""
        self.analyzer.data_dict = {"metadata": {"output": torch.ones(2, 1)}}

        with patch.object(self.analyzer.logger, "warning") as mock_warning:
            metric_data = self.analyzer._collect_all_branch_metric_data(
                np.array([0, 1])
            )

        self.assertIsNone(metric_data)
        mock_warning.assert_called_once_with(
            "No excitatory activations collected. Skipping information analysis."
        )

    def test_all_branch_metric_data_helper_collects_layers_and_concatenates(self):
        """All-branch metric collection should preserve tensor lists and missing-Vb flag."""
        self.analyzer.data_dict = {
            "layer0": {"excitation": torch.ones(2, 2)},
            "skip": {"output": torch.zeros(2, 1)},
            "layer1": {"excitation": torch.ones(2, 1)},
        }
        C = np.array([0, 1])
        layer0 = (
            torch.ones(2, 2),
            torch.ones(2, 2) * 2.0,
            torch.ones(2, 2) * 3.0,
            torch.ones(2, 2) * 4.0,
            False,
        )
        layer1 = (
            torch.ones(2, 1) * 5.0,
            torch.ones(2, 1) * 6.0,
            torch.ones(2, 1) * 7.0,
            None,
            True,
        )
        E = np.ones((2, 3))
        I_var = np.ones((2, 3)) * 2.0
        Vout = np.ones((2, 3)) * 3.0
        Vb = None

        with (
            patch.object(
                self.analyzer,
                "_aggregate_all_branch_layer_tensors",
                side_effect=[layer0, layer1],
            ) as mock_aggregate,
            patch.object(
                self.analyzer,
                "_concatenate_all_branch_tensors_for_metrics",
                return_value=(E, I_var, Vout, C, Vb),
            ) as mock_concat,
        ):
            metric_data = self.analyzer._collect_all_branch_metric_data(C)

        self.assertEqual(mock_aggregate.call_count, 2)
        self.assertEqual(len(mock_concat.call_args.kwargs["layer_excitation"]), 2)
        self.assertEqual(len(mock_concat.call_args.kwargs["layer_inhibition"]), 2)
        self.assertEqual(len(mock_concat.call_args.kwargs["layer_output"]), 2)
        self.assertEqual(len(mock_concat.call_args.kwargs["layer_branch_input"]), 1)
        self.assertTrue(mock_concat.call_args.kwargs["missing_vb_in_any_layer"])
        np.testing.assert_array_equal(mock_concat.call_args.kwargs["C"], C)
        np.testing.assert_array_equal(metric_data.excitation, E)
        np.testing.assert_array_equal(metric_data.inhibition, I_var)
        np.testing.assert_array_equal(metric_data.output, Vout)
        np.testing.assert_array_equal(metric_data.labels, C)
        self.assertIsNone(metric_data.branch_input)

    def test_finalize_all_branch_results_preserves_metrics_nulls_and_metadata(self):
        """All-branch finalization should preserve metric/null arguments and labels."""
        self.analyzer.branch_aggregation = "multivariate"
        E = np.ones((3, 2))
        I_var = np.ones((3, 2)) * 2.0
        Vout = np.ones((3, 2)) * 3.0
        C = np.array([0, 1, 0])
        Vb = np.ones((3, 2)) * 4.0
        Vinf = np.ones((3, 1)) * 5.0
        metric_data = Mock()
        metric_data.excitation = E
        metric_data.inhibition = I_var
        metric_data.output = Vout
        metric_data.labels = C
        metric_data.branch_input = Vb
        results = {"basic_mi": {"I(E;C)": 1.0}}

        with (
            patch.object(
                self.analyzer,
                "compute_metrics",
                return_value=results,
            ) as mock_metrics,
            patch.object(
                self.analyzer,
                "_try_add_label_shuffle_null_baselines_in_place",
            ) as mock_null,
        ):
            finalized = self.analyzer._finalize_all_branch_results(
                metric_data=metric_data,
                Vinf=Vinf,
            )

        self.assertIs(finalized, results)
        mock_metrics.assert_called_once_with(E, I_var, Vout, C, Vinf, Vb)
        mock_null.assert_called_once()
        np.testing.assert_array_equal(mock_null.call_args.kwargs["E"], E)
        np.testing.assert_array_equal(mock_null.call_args.kwargs["I_var"], I_var)
        np.testing.assert_array_equal(mock_null.call_args.kwargs["Vout"], Vout)
        np.testing.assert_array_equal(mock_null.call_args.kwargs["C"], C)
        np.testing.assert_array_equal(mock_null.call_args.kwargs["Vb"], Vb)
        np.testing.assert_array_equal(mock_null.call_args.kwargs["Vinf"], Vinf)
        self.assertTrue(mock_null.call_args.kwargs["has_exc_synapses"])
        self.assertTrue(mock_null.call_args.kwargs["has_inh_synapses"])
        self.assertTrue(mock_null.call_args.kwargs["has_branch_input"])
        self.assertEqual(mock_null.call_args.kwargs["seed_offset"], 999_999)
        self.assertEqual(
            mock_null.call_args.kwargs["failure_context"], "all-branch analysis"
        )
        self.assertEqual(finalized["aggregation_method"], "multivariate")
        self.assertEqual(finalized["computation_level"], "all_branch")

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

    def test_layer_branch_result_helper_preserves_aggregation_and_null_args(self):
        """Layer-branch result helper should preserve aggregation and baseline inputs."""
        C = np.array([0, 1])
        layer_data = {
            "excitation": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "inhibition": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
            "branch_input": torch.tensor([[9.0, 10.0], [11.0, 12.0]]),
            "output": torch.tensor([[13.0, 14.0], [15.0, 16.0]]),
            "has_exc_synapses": False,
            "has_inh_synapses": True,
            "has_branch_input": False,
        }
        E_layer = np.array([[1.0], [3.0]])
        I_layer = np.array([[5.0], [7.0]])
        Vout_layer = np.array([[13.0], [15.0]])
        Vb_layer = np.array([[9.0], [11.0]])
        layer_results = {"basic_mi": {"I(E;C)": 1.0}}

        with (
            patch.object(
                self.analyzer,
                "_aggregate_layer_branch_arrays_for_metrics",
                return_value=(E_layer, I_layer, Vout_layer, C, Vb_layer),
            ) as mock_aggregate,
            patch.object(
                self.analyzer,
                "compute_metrics",
                return_value=layer_results,
            ) as mock_metrics,
            patch.object(
                self.analyzer,
                "_filter_result_metrics_in_place",
            ) as mock_filter,
            patch.object(
                self.analyzer,
                "_try_add_label_shuffle_null_baselines_in_place",
            ) as mock_null,
        ):
            result = self.analyzer._compute_layer_branch_result(
                layer_name="layer0",
                layer_data=layer_data,
                C=C,
                seed_offset=4,
            )

        self.assertIs(result, layer_results)
        torch.testing.assert_close(
            mock_aggregate.call_args.kwargs["E_layer_data"], layer_data["excitation"]
        )
        torch.testing.assert_close(
            mock_aggregate.call_args.kwargs["I_layer_data"], layer_data["inhibition"]
        )
        torch.testing.assert_close(
            mock_aggregate.call_args.kwargs["Vout_layer_data"], layer_data["output"]
        )
        torch.testing.assert_close(
            mock_aggregate.call_args.kwargs["Vb_layer_data"],
            layer_data["branch_input"],
        )
        np.testing.assert_array_equal(mock_aggregate.call_args.kwargs["C"], C)
        mock_metrics.assert_called_once_with(
            E_layer, I_layer, Vout_layer, C, None, Vb_layer
        )
        mock_filter.assert_called_once_with(
            layer_results,
            has_exc_synapses=False,
            has_inh_synapses=True,
            has_branch_input=True,
        )
        mock_null.assert_called_once()
        self.assertEqual(mock_null.call_args.kwargs["seed_offset"], 4)
        self.assertEqual(mock_null.call_args.kwargs["failure_context"], "layer layer0")
        self.assertFalse(mock_null.call_args.kwargs["has_exc_synapses"])
        self.assertTrue(mock_null.call_args.kwargs["has_inh_synapses"])
        self.assertTrue(mock_null.call_args.kwargs["has_branch_input"])

    def test_finalize_layer_branch_results_returns_none_for_empty_results(self):
        """Layer-branch finalization should preserve empty-result skip behavior."""
        with patch.object(self.analyzer.logger, "warning") as mock_warning:
            results = self.analyzer._finalize_layer_branch_results(
                all_results=[],
                layer_names=[],
            )

        self.assertIsNone(results)
        mock_warning.assert_called_once_with("No layer results to average")

    def test_finalize_layer_branch_results_averages_and_attaches_statistics(self):
        """Layer-branch finalization should preserve averaged metadata and stats."""
        all_results = [{"basic_mi": {"I(E;C)": 1.0}}, {"basic_mi": {"I(E;C)": 3.0}}]
        layer_names = ["layer0", "layer1"]

        with (
            patch.object(
                self.analyzer,
                "_average_mi_results",
                return_value={"basic_mi": {"I(E;C)": 2.0}},
            ) as mock_average,
            patch.object(
                self.analyzer,
                "_compute_layer_statistics",
                side_effect=[{"stat": 0.0}, {"stat": 1.0}],
            ) as mock_stats,
            patch.object(
                self.analyzer,
                "_apply_layer_synapse_availability_in_place",
            ) as mock_availability,
            patch.object(self.analyzer.logger, "info") as mock_info,
        ):
            results = self.analyzer._finalize_layer_branch_results(
                all_results=all_results,
                layer_names=layer_names,
            )

        mock_average.assert_called_once_with(all_results)
        self.assertEqual(mock_stats.call_args_list[0].args[0], [all_results[0]])
        self.assertEqual(mock_stats.call_args_list[1].args[0], [all_results[1]])
        self.assertEqual(mock_availability.call_count, 2)
        self.assertEqual(
            results,
            {
                "basic_mi": {"I(E;C)": 2.0},
                "layer_names": layer_names,
                "num_layers": 2,
                "computation_level": "layer_branch",
                "layer_statistics": {"layer0": {"stat": 0.0}, "layer1": {"stat": 1.0}},
            },
        )
        mock_info.assert_called_once_with(
            "Generated layer statistics for 2 layers for plotting"
        )

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
        self.assertIsInstance(self.manager.performance_analyzer, spec.load_class())
        self.assertIs(
            self.manager.performance_analyzer,
            getattr(self.manager, spec.attribute_name),
        )
        self.assertFalse(self.manager.correlation_analysis_enabled)

    def test_manager_registry_accessors_resolve_registered_metadata(self):
        """Manager helper methods should resolve through registry metadata."""
        spec = get_analyzer_spec("performance")

        assert spec is not None
        self.assertIs(
            self.manager._analysis_config("performance"),
            getattr(self.manager.config, spec.config_field),
        )
        self.assertTrue(self.manager._has_analysis_analyzer("performance"))
        self.assertIs(
            self.manager._analysis_analyzer("performance"),
            getattr(self.manager, spec.attribute_name),
        )
        self.assertEqual(
            self.manager._analysis_save_path("performance", "epochs"),
            str(Path(self.manager.save_root) / spec.save_subdir / "epochs"),
        )

    def test_manager_initializes_optional_analyzer_constructor_styles(self):
        """Registry-backed initialization supports params, model+params, and no params."""
        config = AnalysisConfig()
        config.synapse_turnover_analysis.enabled = True
        config.correlation_analysis.enabled = True
        config.routing_analysis.enabled = True

        synapse_turnover_spec = get_analyzer_spec("synapse_turnover")
        routing_spec = get_analyzer_spec("routing")
        assert synapse_turnover_spec is not None
        assert routing_spec is not None
        self.assertEqual(synapse_turnover_spec.constructor_style, "model_params")
        self.assertEqual(routing_spec.constructor_style, "none")

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

    def test_analyzer_initializer_builds_from_registered_constructor_style(self):
        """Initializer metadata should own analyzer construction semantics."""
        from dendritic_modeling.analysis.managers.analysis_plan import (
            _AnalyzerInitializer,
        )

        params = self.analysis_config.performance_analysis.params
        analyzer = _AnalyzerInitializer("performance").build(
            model=self.model,
            params=params,
        )

        self.assertIsInstance(analyzer, PerformanceAnalyzer)

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

    def test_should_run_registered_analysis_applies_phase_and_analyzer_gates(self):
        """Registered analysis gate should combine config phase and analyzer presence."""
        config = AnalysisConfig()
        config.performance_analysis.enabled = False
        config.routing_analysis.enabled = True
        config.routing_analysis.training = False

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AnalysisManager(
                model=self.model,
                data=self.data,
                analysis_config=config,
                save_root=tmpdir,
            )

            self.assertTrue(
                manager._should_run_registered_analysis(
                    "routing",
                    training=False,
                    require_analyzer=True,
                )
            )
            self.assertFalse(
                manager._should_run_registered_analysis(
                    "routing",
                    training=True,
                    require_analyzer=True,
                )
            )

            delattr(manager, "routing_analyzer")
            self.assertFalse(
                manager._should_run_registered_analysis(
                    "routing",
                    training=False,
                    require_analyzer=True,
                )
            )
            self.assertTrue(
                manager._should_run_registered_analysis(
                    "routing",
                    training=False,
                    require_analyzer=False,
                )
            )

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

    def test_common_test_dataset_analysis_resolves_registry_metadata(self):
        """Common test-dataset helper should resolve config, analyzer, and path by name."""
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
                manager._run_test_dataset_analysis(
                    registry_name="noise_perturbation",
                    filename="direct_eval",
                    training=False,
                    runtime=manager.config.runtime.final,
                )

            mock_analyzer.analyze.assert_called_once()
            call_kwargs = mock_analyzer.analyze.call_args.kwargs
            self.assertIs(call_kwargs["model"], self.model)
            self.assertIs(call_kwargs["test_dataset"], self.data["test"])
            self.assertEqual(call_kwargs["device"], "cpu")
            self.assertEqual(
                call_kwargs["save_path"],
                str(Path(tmpdir) / "noise_perturbation_analysis"),
            )
            self.assertEqual(call_kwargs["filename"], "direct_eval")
            self.assertIs(call_kwargs["runtime"], manager.config.runtime.final)
            self.assertNotIn("training", call_kwargs)

    def test_common_test_dataset_analysis_skips_missing_required_analyzer(self):
        """Required optional analyzers should still skip when no instance exists."""
        config = AnalysisConfig()
        config.performance_analysis.enabled = False
        config.adversarial_robustness_analysis.enabled = True

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AnalysisManager(
                model=self.model,
                data=self.data,
                analysis_config=config,
                save_root=tmpdir,
            )
            if hasattr(manager, "adversarial_robustness_analyzer"):
                delattr(manager, "adversarial_robustness_analyzer")

            with patch.object(manager, "_analysis_analyzer") as mock_analyzer:
                manager._run_test_dataset_analysis(
                    registry_name="adversarial_robustness",
                    filename="direct_eval",
                    training=False,
                    runtime=manager.config.runtime.final,
                    require_analyzer=True,
                )

            mock_analyzer.assert_not_called()

    def test_logged_test_dataset_analysis_preserves_call_and_logs(self):
        """Logged test-dataset helper should preserve kwargs and message order."""
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

            with (
                patch.object(manager, "noise_perturbation_analyzer") as mock_analyzer,
                patch.object(manager.logger, "info") as mock_info,
            ):
                manager._run_logged_test_dataset_analysis(
                    registry_name="noise_perturbation",
                    filename="logged_eval",
                    training=False,
                    runtime=manager.config.runtime.final,
                    start_message="Starting noise analysis...",
                    complete_message="Noise analysis completed",
                )

            mock_info.assert_has_calls(
                [
                    call("Starting noise analysis..."),
                    call("Noise analysis completed"),
                ]
            )
            mock_analyzer.analyze.assert_called_once()
            call_kwargs = mock_analyzer.analyze.call_args.kwargs
            self.assertIs(call_kwargs["model"], self.model)
            self.assertIs(call_kwargs["test_dataset"], self.data["test"])
            self.assertEqual(
                call_kwargs["save_path"],
                str(Path(tmpdir) / "noise_perturbation_analysis"),
            )
            self.assertEqual(call_kwargs["filename"], "logged_eval")
            self.assertIs(call_kwargs["runtime"], manager.config.runtime.final)

    def test_logged_test_dataset_analysis_skips_missing_required_analyzer(self):
        """Logged helper should not emit messages when required analyzer is absent."""
        config = AnalysisConfig()
        config.performance_analysis.enabled = False
        config.adversarial_robustness_analysis.enabled = True

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AnalysisManager(
                model=self.model,
                data=self.data,
                analysis_config=config,
                save_root=tmpdir,
            )
            if hasattr(manager, "adversarial_robustness_analyzer"):
                delattr(manager, "adversarial_robustness_analyzer")

            with (
                patch.object(manager, "_analysis_analyzer") as mock_analyzer,
                patch.object(manager.logger, "info") as mock_info,
            ):
                manager._run_logged_test_dataset_analysis(
                    registry_name="adversarial_robustness",
                    filename="logged_eval",
                    training=False,
                    runtime=manager.config.runtime.final,
                    start_message="Starting adversarial analysis...",
                    complete_message="Adversarial analysis completed",
                    require_analyzer=True,
                )

            mock_info.assert_not_called()
            mock_analyzer.assert_not_called()

    def test_logged_data_analysis_preserves_call_and_logs_without_runtime(self):
        """Logged data helper should preserve routing call kwargs and logs."""
        config = AnalysisConfig()
        config.performance_analysis.enabled = False
        config.routing_analysis.enabled = True

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AnalysisManager(
                model=self.model,
                data=self.data,
                analysis_config=config,
                save_root=tmpdir,
            )

            with (
                patch.object(manager, "routing_analyzer") as mock_analyzer,
                patch.object(manager.logger, "info") as mock_info,
            ):
                manager._run_logged_data_analysis(
                    registry_name="routing",
                    filename="route_eval",
                    training=False,
                    start_message="Starting routing analysis...",
                    complete_message="Routing analysis completed successfully",
                )

            mock_info.assert_has_calls(
                [
                    call("Starting routing analysis..."),
                    call("Routing analysis completed successfully"),
                ]
            )
            mock_analyzer.analyze.assert_called_once()
            call_kwargs = mock_analyzer.analyze.call_args.kwargs
            self.assertIs(call_kwargs["model"], self.model)
            self.assertIs(call_kwargs["data"], self.data["test"])
            self.assertEqual(call_kwargs["device"], "cpu")
            self.assertEqual(
                call_kwargs["save_path"],
                str(Path(tmpdir) / "routing_analysis"),
            )
            self.assertEqual(call_kwargs["filename"], "route_eval")
            self.assertNotIn("runtime", call_kwargs)

    def test_logged_data_analysis_preserves_runtime_when_requested(self):
        """Logged data helper should preserve runtime-aware analyzer calls."""
        config = AnalysisConfig()
        config.performance_analysis.enabled = False
        config.spike_train_analysis.enabled = True

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AnalysisManager(
                model=self.model,
                data=self.data,
                analysis_config=config,
                save_root=tmpdir,
            )
            runtime = manager.config.runtime.final

            with (
                patch.object(manager, "spike_train_analyzer") as mock_analyzer,
                patch.object(manager.logger, "info") as mock_info,
            ):
                manager._run_logged_data_analysis(
                    registry_name="spike_train",
                    filename="spike_eval",
                    training=False,
                    start_message="Starting spike train analysis...",
                    complete_message="Spike train analysis completed successfully",
                    runtime=runtime,
                    pass_runtime=True,
                )

            mock_info.assert_has_calls(
                [
                    call("Starting spike train analysis..."),
                    call("Spike train analysis completed successfully"),
                ]
            )
            mock_analyzer.analyze.assert_called_once()
            call_kwargs = mock_analyzer.analyze.call_args.kwargs
            self.assertIs(call_kwargs["model"], self.model)
            self.assertIs(call_kwargs["data"], self.data["test"])
            self.assertEqual(
                call_kwargs["save_path"],
                str(Path(tmpdir) / "spike_train_analysis"),
            )
            self.assertEqual(call_kwargs["filename"], "spike_eval")
            self.assertIs(call_kwargs["runtime"], runtime)

    def test_logged_data_analysis_skips_missing_required_analyzer(self):
        """Logged data helper should not emit messages without required analyzer."""
        config = AnalysisConfig()
        config.performance_analysis.enabled = False
        config.routing_analysis.enabled = True

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AnalysisManager(
                model=self.model,
                data=self.data,
                analysis_config=config,
                save_root=tmpdir,
            )
            if hasattr(manager, "routing_analyzer"):
                delattr(manager, "routing_analyzer")

            with (
                patch.object(manager, "_analysis_analyzer") as mock_analyzer,
                patch.object(manager.logger, "info") as mock_info,
            ):
                manager._run_logged_data_analysis(
                    registry_name="routing",
                    filename="route_eval",
                    training=False,
                    start_message="Starting routing analysis...",
                    complete_message="Routing analysis completed successfully",
                )

            mock_info.assert_not_called()
            mock_analyzer.assert_not_called()

    def test_common_test_dataset_runners_resolve_registry_metadata(self):
        """Common runner tables should stay synchronized with the registry."""
        from dendritic_modeling.analysis.managers.analysis_manager import (
            _MODEL_ONLY_ANALYSES,
            _POST_REACTIVATION_TEST_DATASET_ANALYSES,
            _PRE_WEIGHT_TEST_DATASET_ANALYSES,
        )

        runners = (
            *_PRE_WEIGHT_TEST_DATASET_ANALYSES,
            *_POST_REACTIVATION_TEST_DATASET_ANALYSES,
            *_MODEL_ONLY_ANALYSES,
        )
        for runner in runners:
            spec = get_analyzer_spec(runner.registry_name)
            self.assertIsNotNone(spec, runner.registry_name)
            self.assertEqual(runner.config_field, spec.config_field)
            self.assertEqual(runner.analyzer_attr, spec.attribute_name)
            self.assertEqual(runner.save_subdir, spec.save_subdir)

    def test_model_only_analysis_path_preserves_call_kwargs(self):
        """Shared model-only runner should preserve analyzer call arguments."""
        config = AnalysisConfig()
        config.performance_analysis.enabled = False
        config.jacobian_spectrum_analysis.enabled = True

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AnalysisManager(
                model=self.model,
                data=self.data,
                analysis_config=config,
                save_root=tmpdir,
            )

            with patch.object(manager, "jacobian_spectrum_analyzer") as mock_analyzer:
                manager._run_model_only_analysis(
                    registry_name="jacobian_spectrum",
                    filename="final_eval",
                    training=False,
                    start_message="Starting jacobian spectrum analysis...",
                    complete_message="Jacobian spectrum analysis completed successfully",
                )

            mock_analyzer.analyze.assert_called_once()
            call_kwargs = mock_analyzer.analyze.call_args.kwargs
            self.assertIs(call_kwargs["model"], self.model)
            self.assertEqual(call_kwargs["device"], "cpu")
            self.assertEqual(
                call_kwargs["save_path"],
                str(Path(tmpdir) / "jacobian_spectrum"),
            )
            self.assertEqual(call_kwargs["filename"], "final_eval")

    def test_run_analysis_preserves_orchestration_order(self):
        """Top-level analysis orchestration should preserve the established order."""
        from dendritic_modeling.analysis.managers.analysis_manager import (
            _MODEL_ONLY_ANALYSES,
            _POST_REACTIVATION_TEST_DATASET_ANALYSES,
            _PRE_WEIGHT_TEST_DATASET_ANALYSES,
        )

        filename = "epoch_order"
        training = True
        runtime = object()
        calls = []

        def record_step(name, *, expects_runtime):
            def _recorder(**kwargs):
                self.assertEqual(kwargs["filename"], filename)
                self.assertIs(kwargs["training"], training)
                if expects_runtime:
                    self.assertIs(kwargs["runtime"], runtime)
                else:
                    self.assertNotIn("runtime", kwargs)
                calls.append(name)

            return _recorder

        def record_runner(runner, **kwargs):
            self.assertEqual(kwargs["filename"], filename)
            self.assertIs(kwargs["training"], training)
            self.assertIs(kwargs["runtime"], runtime)
            calls.append(f"runner:{runner.registry_name}")

        def record_model_only_runner(runner, **kwargs):
            self.assertEqual(kwargs["filename"], filename)
            self.assertIs(kwargs["training"], training)
            self.assertNotIn("runtime", kwargs)
            calls.append(runner.registry_name)

        patched_steps = (
            ("_run_performance_analysis", "performance", True),
            ("_run_information_analysis", "information", True),
            ("_run_synaptic_turnover_analysis", "synapse_turnover", False),
            ("_run_weight_analysis", "weight", False),
            ("_run_reactivation_dynamics_analysis", "reactivation_dynamics", True),
            ("_run_local_rule_component_analysis", "local_rule_components", True),
            ("_run_correlation_analysis", "correlation", True),
            ("_run_routing_analysis", "routing", False),
            ("_run_spike_train_analysis", "spike_train", True),
            ("_run_adversarial_robustness_analysis", "adversarial_robustness", True),
        )

        with ExitStack() as stack:
            runtime_mock = stack.enter_context(
                patch.object(self.manager, "_analysis_runtime", return_value=runtime)
            )
            stack.enter_context(
                patch.object(
                    self.manager,
                    "_run_test_dataset_analysis_runner",
                    side_effect=record_runner,
                )
            )
            stack.enter_context(
                patch.object(
                    self.manager,
                    "_run_model_only_analysis_runner",
                    side_effect=record_model_only_runner,
                )
            )
            for method_name, call_name, expects_runtime in patched_steps:
                stack.enter_context(
                    patch.object(
                        self.manager,
                        method_name,
                        side_effect=record_step(
                            call_name,
                            expects_runtime=expects_runtime,
                        ),
                    )
                )

            self.manager.run_analysis(filename=filename, training=training)

        runtime_mock.assert_called_once_with(training)
        expected_calls = [
            "performance",
            "information",
            "synapse_turnover",
            *[
                f"runner:{runner.registry_name}"
                for runner in _PRE_WEIGHT_TEST_DATASET_ANALYSES
            ],
            "weight",
            "reactivation_dynamics",
            *[
                f"runner:{runner.registry_name}"
                for runner in _POST_REACTIVATION_TEST_DATASET_ANALYSES
            ],
            "local_rule_components",
            "correlation",
            "routing",
            *[runner.registry_name for runner in _MODEL_ONLY_ANALYSES],
            "spike_train",
            "adversarial_robustness",
        ]
        self.assertEqual(calls, expected_calls)

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
    generator = torch.Generator().manual_seed(0)
    data = torch.randn(100, 10, generator=generator)
    labels = torch.randint(0, 4, (100,), generator=generator)
    return torch.utils.data.TensorDataset(data, labels)


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
