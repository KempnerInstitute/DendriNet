"""
Test suite for dendritic_modeling.training module.

This module tests all training functionality including:
- Different training strategies (MLE, two-step, voltage stabilization, etc.)
- Trainer factory functions
- Optimization and gradient handling
- Training workflows
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from dendritic_modeling.config import TransferFunctionConfig
from dendritic_modeling.config.training import PruningConfig
from dendritic_modeling.models import Classifier
from dendritic_modeling.networks import ExcitationInhibitionNetwork, Identity
from dendritic_modeling.networks.activations.parametric import (
    ParametricLinearSigmoid,
    ParametricTanh,
)
from dendritic_modeling.training import Trainer, get_trainer
from dendritic_modeling.training.strategies import (
    LocalCreditAssignment,
    TwoStepTrainer,
    TwoStepTrainerWithKL,
)
from dendritic_modeling.training.strategies.specialized import (
    _merge_warmup_results,
    _run_warmup_then_standard_training,
    _set_parametric_activation_trainability,
    _start_standard_phase,
    _start_warmup_phase,
)
from dendritic_modeling.training.strategies.two_step import (
    _forward_through_local_layers,
    _initialize_local_heads,
    _make_local_optimizers,
    _prepend_pretrain_losses,
    _resolve_layer_container,
    _restore_full_model_trainability,
)


class TestTrainerFactory(unittest.TestCase):
    """Test the trainer factory function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple optimizer for testing
        transfer_params = TransferFunctionConfig().asdict()
        model = Classifier(
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
        self.optimizer = torch.optim.Adam(model.parameters())

    def test_get_trainer_standard(self):
        """Test getting standard trainer."""
        trainer_config = {"loss_function": "cross_entropy", "epochs": 10}
        trainer = get_trainer("standard", self.optimizer, trainer_config)
        self.assertIsInstance(trainer, Trainer)

    def test_get_trainer_two_step(self):
        """Test getting two-step trainer."""
        trainer_config = {"loss_function": "cross_entropy", "epochs": 10}
        trainer = get_trainer("freeze_branches", self.optimizer, trainer_config)
        self.assertIsInstance(trainer, TwoStepTrainer)

    def test_get_trainer_two_step_kl(self):
        """Test getting two-step KL trainer."""
        trainer_config = {"loss_function": "cross_entropy", "epochs": 10}
        trainer = get_trainer("freeze_branch_kl", self.optimizer, trainer_config)
        self.assertIsInstance(trainer, TwoStepTrainerWithKL)

    def test_get_trainer_voltage_stabilization(self):
        """Test getting voltage stabilization trainer."""
        trainer_config = {"loss_function": "cross_entropy", "epochs": 10}
        with self.assertRaises(ValueError):
            get_trainer("voltage_stablization", self.optimizer, trainer_config)

    def test_get_trainer_freeze_branches(self):
        """Test getting local credit assignment trainer."""
        trainer_config = {"loss_function": "cross_entropy", "epochs": 10}
        trainer = get_trainer("freeze_branches", self.optimizer, trainer_config)
        self.assertIsInstance(trainer, TwoStepTrainer)

    def test_get_trainer_invalid_strategy(self):
        """Test handling of invalid training strategy."""
        trainer_config = {"loss_function": "cross_entropy", "epochs": 10}
        with self.assertRaises(ValueError):
            get_trainer("invalid_strategy", self.optimizer, trainer_config)

    def test_get_trainer_with_options(self):
        """Test getting trainer with additional options."""
        trainer_config = {
            "loss_function": "cross_entropy",
            "epochs": 10,
            "suppress_prints": True,
            "print_every": 20,
        }
        trainer = get_trainer("standard", self.optimizer, trainer_config)
        self.assertIsInstance(trainer, Trainer)
        self.assertTrue(trainer.suppress_prints)
        self.assertEqual(trainer.print_every, 20)

    def test_get_trainer_quantile_reactivation_options(self):
        """Quantile-update options should be passed through unchanged."""
        trainer_config = {
            "loss_function": "cross_entropy",
            "epochs": 10,
            "reactivation_update_mode": "quantile",
            "reactivation_recalibration_mode": "occupancy_quantile",
            "reactivation_recalibration_start_epoch": 2,
            "reactivation_recalibration_num_batches": 7,
            "reactivation_recalibration_ema_alpha": 0.3,
        }
        trainer = get_trainer("standard", self.optimizer, trainer_config)
        self.assertIsInstance(trainer, Trainer)
        self.assertEqual(trainer.reactivation_update_mode, "quantile")
        self.assertEqual(trainer.reactivation_recalibration_mode, "occupancy_quantile")
        self.assertEqual(trainer.reactivation_recalibration_start_epoch, 2)
        self.assertEqual(trainer.reactivation_recalibration_num_batches, 7)
        self.assertAlmostEqual(trainer.reactivation_recalibration_ema_alpha, 0.3)


class TestTrainer(unittest.TestCase):
    """Test the standard trainer class."""

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

        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        # Create trainer
        self.trainer = Trainer(
            optimizer=self.optimizer,
            loss_function="cross_entropy",
            suppress_prints=True,
            print_every=10,
            num_workers=0,
        )

        # Create test data
        self.train_data = torch.utils.data.TensorDataset(
            torch.rand(50, 8), torch.randint(0, 2, (50,))
        )
        self.valid_data = torch.utils.data.TensorDataset(
            torch.rand(20, 8), torch.randint(0, 2, (20,))
        )

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        self.assertIsInstance(self.trainer, Trainer)
        self.assertEqual(self.trainer.optimizer, self.optimizer)
        self.assertTrue(self.trainer.suppress_prints)
        self.assertEqual(self.trainer.print_every, 10)

    def test_single_epoch_training(self):
        """Test single epoch of training."""
        # Train for one epoch
        results = self.trainer.train(
            model=self.model, train_data=self.train_data, valid_data=self.valid_data
        )

        # Check that results are returned
        self.assertIsInstance(results, dict)
        self.assertIn("train_losses", results)
        self.assertIn("valid_losses", results)

        # Check that we have at least one loss value
        self.assertGreater(len(results["train_losses"]), 0)
        self.assertGreater(len(results["valid_losses"]), 0)

    def test_apply_pruning_and_evaluate_returns_pre_post_metrics(self):
        """Test separate pruning step used by train_experiments."""
        trainer = Trainer(
            optimizer=torch.optim.Adam(self.model.parameters(), lr=0.01),
            loss_function="cross_entropy",
            pruning_config=PruningConfig(enabled=True, threshold=1e9),
            batch_size=8,
            suppress_prints=True,
            num_workers=0,
        )

        results = trainer._apply_pruning_and_evaluate(
            self.model, self.train_data, self.valid_data
        )

        self.assertIn("pre_pruning_performance", results)
        self.assertIn("post_pruning_performance", results)
        self.assertIn("valid_loss", results["pre_pruning_performance"])
        self.assertIn("valid_loss", results["post_pruning_performance"])
        self.assertGreater(results["pruning_stats"]["pruned_params"], 0)


class _TinyLayeredCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 3)])
        self.output_dim = 3

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


class _TinyLayeredModel(nn.Module):
    output_dim = 2

    def __init__(self):
        super().__init__()
        self.core_network = _TinyLayeredCore()
        self.decoder = nn.Linear(3, 2)

    def forward(self, x):
        return self.decoder(self.core_network(x))


class _ActivationLayer(nn.Linear):
    def __init__(self):
        super().__init__(3, 2)
        self.activation = torch.sigmoid


class TestTwoStepLocalHeads(unittest.TestCase):
    def test_resolve_layer_container_prefers_known_wrappers(self):
        core_network = object()
        wrapped_model = SimpleNamespace(
            core_network=core_network,
            net=object(),
            network=object(),
        )
        net_model = SimpleNamespace(net=core_network, network=object())
        bare_model = SimpleNamespace(layers=[])

        self.assertIs(_resolve_layer_container(wrapped_model), core_network)
        self.assertIs(_resolve_layer_container(net_model), core_network)
        self.assertIs(_resolve_layer_container(bare_model), bare_model)

    def test_restore_full_model_trainability_enables_all_parameters(self):
        model = _TinyLayeredModel()
        for param in model.parameters():
            param.requires_grad = False

        _restore_full_model_trainability(model)

        self.assertTrue(all(param.requires_grad for param in model.parameters()))

    def test_prepend_pretrain_losses_preserves_stage_order(self):
        results = {"train_losses": [3.0], "valid_losses": [4.0]}

        merged = _prepend_pretrain_losses(results, [1.0, 2.0], [1.5, 2.5])

        self.assertIs(merged, results)
        self.assertEqual(merged["train_losses"], [1.0, 2.0, 3.0])
        self.assertEqual(merged["valid_losses"], [1.5, 2.5, 4.0])

    def test_forward_through_local_layers_matches_inline_path(self):
        net_obj = SimpleNamespace(
            layers=nn.ModuleList([nn.Linear(4, 3), _ActivationLayer()])
        )
        x_batch = torch.randn(5, 4)

        manual = net_obj.layers[0](x_batch)
        manual = net_obj.layers[1](manual)
        manual = net_obj.layers[1].activation(manual)

        actual = _forward_through_local_layers(net_obj, x_batch, layer_idx=1)

        torch.testing.assert_close(actual, manual)

    @patch("torch.distributed.is_initialized", return_value=True)
    def test_two_step_raises_explicitly_under_ddp(self, _mock_dist):
        model = _TinyLayeredModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        trainer = TwoStepTrainer(
            optimizer=optimizer,
            loss_function="cross_entropy",
            epochs=1,
            batch_size=4,
            suppress_prints=True,
            num_workers=0,
        )
        train_data = torch.utils.data.TensorDataset(
            torch.randn(4, 4), torch.randint(0, 2, (4,))
        )

        with self.assertRaisesRegex(NotImplementedError, "not supported under DDP"):
            trainer.train(model, train_data, train_data)

    def test_local_heads_are_optimizer_owned_and_updated(self):
        model = _TinyLayeredModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        trainer = TwoStepTrainer(
            optimizer=optimizer,
            loss_function="cross_entropy",
            epochs=1,
            batch_size=4,
            suppress_prints=True,
            num_workers=0,
        )
        trainer.pretrain_epochs = 1
        trainer.device = "cpu"
        trainer.local_heads = _initialize_local_heads(
            model.core_network, model, trainer.device
        )
        trainer.local_optimizers = _make_local_optimizers(
            model.core_network, trainer.local_heads, trainer.optimizer
        )
        before = trainer.local_heads[0].weight.detach().clone()

        train_data = torch.utils.data.TensorDataset(
            torch.randn(12, 4), torch.randint(0, 2, (12,))
        )
        valid_data = torch.utils.data.TensorDataset(
            torch.randn(8, 4), torch.randint(0, 2, (8,))
        )
        trainer.train(model, train_data, valid_data)

        self.assertFalse(torch.allclose(before, trainer.local_heads[0].weight))


if __name__ == "__main__":
    unittest.main()


class _ToyReactivationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_in = nn.Linear(8, 4)
        self.reactivation = ParametricTanh(4, init_m=1.5, init_b=0.5)
        self.linear_out = nn.Linear(4, 2)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.reactivation(x)
        return self.linear_out(x)


class _MixedTrainabilityReactivationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trainable = ParametricTanh(2, init_m=1.5, init_b=0.5)
        self.frozen = ParametricLinearSigmoid(2, trainable=False)

    def forward(self, x):
        return self.frozen(self.trainable(x))


class TestReactivationUpdateModes(unittest.TestCase):
    def test_specialized_activation_trainability_helper_sets_all_parametric_params(
        self,
    ):
        model = _MixedTrainabilityReactivationModel()

        _set_parametric_activation_trainability(model, False)
        self.assertFalse(model.trainable.log_m.requires_grad)
        self.assertFalse(model.trainable.b.requires_grad)
        self.assertFalse(model.frozen.log_m.requires_grad)

        _set_parametric_activation_trainability(model, True)
        self.assertTrue(model.trainable.log_m.requires_grad)
        self.assertTrue(model.trainable.b.requires_grad)
        self.assertTrue(model.frozen.log_m.requires_grad)

    def test_specialized_warmup_phase_restores_standard_trainer_state(self):
        model = _ToyReactivationModel()
        trainer = Trainer(
            optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
            loss_function="cross_entropy",
            epochs=7,
            suppress_prints=True,
            num_workers=0,
        )
        trainer.pretrain_epochs = 3
        warmup_loss = nn.MSELoss()
        original_loss_fn = trainer.loss_function

        default_loss_fn, default_epochs = _start_warmup_phase(
            trainer,
            warmup_loss,
            "warmup_",
            "[Warmup] ",
        )

        self.assertIs(default_loss_fn, original_loss_fn)
        self.assertEqual(default_epochs, 7)
        self.assertIs(trainer.loss_function, warmup_loss)
        self.assertEqual(trainer.epochs, 3)
        self.assertEqual(trainer.filename_prefix, "warmup_")
        self.assertEqual(trainer.logger_info_prefix, "[Warmup] ")

        _start_standard_phase(trainer, default_loss_fn, default_epochs)

        self.assertIs(trainer.loss_function, default_loss_fn)
        self.assertEqual(trainer.epochs, 7)
        self.assertEqual(trainer.filename_prefix, "standard_")
        self.assertEqual(trainer.logger_info_prefix, "[Standard] ")
        self.assertEqual(trainer.best_loss, float("inf"))

    def test_merge_warmup_results_preserves_stage_order(self):
        pretrain_results = {"train_losses": [1.0], "valid_losses": [1.5]}
        final_results = {
            "train_losses": [2.0],
            "valid_losses": [2.5],
            "best_loss": 2.5,
        }

        merged = _merge_warmup_results(pretrain_results, final_results)

        self.assertIs(merged, final_results)
        self.assertEqual(merged["train_losses"], [1.0, 2.0])
        self.assertEqual(merged["valid_losses"], [1.5, 2.5])
        self.assertEqual(merged["best_loss"], 2.5)

    def test_warmup_training_restores_state_after_failure(self):
        model = _MixedTrainabilityReactivationModel()
        default_loss_fn = nn.L1Loss()
        trainer = SimpleNamespace(
            loss_function=default_loss_fn,
            epochs=7,
            pretrain_epochs=3,
            freeze_reactivation=True,
            best_loss=0.25,
        )
        train_data = object()
        valid_data = object()

        class _WarmupLoss:
            def __init__(self):
                self.handles = [object()]
                self.removed_handles = None

            def attach_forward_hooks(self, attached_model):
                self.attached_model = attached_model
                return self.handles

            def remove_forward_hooks(self, handles):
                self.removed_handles = handles

        warmup_loss = _WarmupLoss()

        def _failing_train_fn(observed_model, observed_train_data, observed_valid_data):
            self.assertIs(observed_model, model)
            self.assertIs(observed_train_data, train_data)
            self.assertIs(observed_valid_data, valid_data)
            self.assertIs(trainer.loss_function, warmup_loss)
            self.assertEqual(trainer.epochs, 3)
            self.assertFalse(model.trainable.log_m.requires_grad)
            self.assertFalse(model.frozen.log_m.requires_grad)
            raise RuntimeError("warmup failed")

        with self.assertRaisesRegex(RuntimeError, "warmup failed"):
            _run_warmup_then_standard_training(
                trainer,
                _failing_train_fn,
                model,
                train_data,
                valid_data,
                warmup_loss,
                filename_prefix="warmup_",
                logger_info_prefix="[Warmup] ",
                warmup_label="Warmup",
                standard_label="Standard",
            )

        self.assertIs(warmup_loss.attached_model, model)
        self.assertIs(warmup_loss.removed_handles, warmup_loss.handles)
        self.assertIs(trainer.loss_function, default_loss_fn)
        self.assertEqual(trainer.epochs, 7)
        self.assertEqual(trainer.filename_prefix, "standard_")
        self.assertEqual(trainer.logger_info_prefix, "[Standard] ")
        self.assertEqual(trainer.best_loss, float("inf"))
        self.assertTrue(model.trainable.log_m.requires_grad)
        self.assertTrue(model.frozen.log_m.requires_grad)

    def test_quantile_mode_freezes_and_backprop_restores_original_trainability(self):
        model = _MixedTrainabilityReactivationModel()
        trainer = Trainer(
            optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
            loss_function="cross_entropy",
            suppress_prints=True,
            num_workers=0,
        )

        trainer._capture_reactivation_param_states(model)
        trainer.reactivation_update_mode = "quantile"
        trainer._configure_reactivation_update_mode(model)

        self.assertFalse(model.trainable.log_m.requires_grad)
        self.assertFalse(model.trainable.b.requires_grad)
        self.assertFalse(model.frozen.log_m.requires_grad)

        trainer.reactivation_update_mode = "backprop"
        trainer._configure_reactivation_update_mode(model)

        self.assertTrue(model.trainable.log_m.requires_grad)
        self.assertTrue(model.trainable.b.requires_grad)
        self.assertFalse(model.frozen.log_m.requires_grad)

    def test_quantile_mode_training_recalibrates_without_backprop_updates(self):
        model = _ToyReactivationModel()
        trainer = Trainer(
            optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
            loss_function="cross_entropy",
            epochs=1,
            batch_size=8,
            suppress_prints=True,
            reactivation_update_mode="quantile",
            recalibrate_reactivation_every=1,
            num_workers=0,
        )
        train_data = torch.utils.data.TensorDataset(
            torch.rand(32, 8), torch.randint(0, 2, (32,))
        )
        valid_data = torch.utils.data.TensorDataset(
            torch.rand(16, 8), torch.randint(0, 2, (16,))
        )

        with patch.object(trainer, "_recalibrate_reactivation") as mock_recalibrate:
            trainer.train(model=model, train_data=train_data, valid_data=valid_data)

        self.assertEqual(mock_recalibrate.call_count, 1)
        self.assertFalse(model.reactivation.log_m.requires_grad)
        self.assertFalse(model.reactivation.b.requires_grad)
        self.assertIsNone(model.reactivation.log_m.grad)
        self.assertIsNone(model.reactivation.b.grad)

    def test_quantile_update_defaults_to_quantile_rule_even_with_analytical_init(self):
        model = _ToyReactivationModel()
        model.reactivation_init_policy = "analytical"
        trainer = Trainer(
            optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
            loss_function="cross_entropy",
            suppress_prints=True,
            reactivation_update_mode="quantile",
            num_workers=0,
        )
        trainer.device = torch.device("cpu")
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.rand(16, 8), torch.randint(0, 2, (16,))
            ),
            batch_size=8,
            shuffle=False,
        )

        with patch(
            "dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize.calibrate_reactivation_from_data"
        ) as mock_calibrate:
            mock_calibrate.return_value = {}
            trainer._recalibrate_reactivation(model, train_loader)

        self.assertEqual(mock_calibrate.call_args.kwargs["mode"], "occupancy_quantile")

    def test_from_init_policy_recalibration_uses_per_layer_mode(self):
        model = _ToyReactivationModel()
        model.reactivation_init_policy = "quantile"
        trainer = Trainer(
            optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
            loss_function="cross_entropy",
            suppress_prints=True,
            reactivation_update_mode="backprop",
            reactivation_recalibration_mode="from_init_policy",
            num_workers=0,
        )
        trainer.device = torch.device("cpu")
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.rand(16, 8), torch.randint(0, 2, (16,))
            ),
            batch_size=8,
            shuffle=False,
        )

        with patch(
            "dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize.calibrate_reactivation_from_data"
        ) as mock_calibrate:
            mock_calibrate.return_value = {}
            trainer._recalibrate_reactivation(model, train_loader)

        self.assertIsNone(mock_calibrate.call_args.kwargs["mode"])

    def test_quantile_recalibration_uses_configured_batch_count_and_ema(self):
        model = _ToyReactivationModel()
        trainer = Trainer(
            optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
            loss_function="cross_entropy",
            suppress_prints=True,
            reactivation_update_mode="quantile",
            reactivation_recalibration_num_batches=2,
            reactivation_recalibration_ema_alpha=0.25,
            num_workers=0,
        )
        trainer.device = torch.device("cpu")
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.rand(20, 8), torch.randint(0, 2, (20,))
            ),
            batch_size=4,
            shuffle=False,
        )

        with patch(
            "dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize.calibrate_reactivation_from_data"
        ) as mock_calibrate:
            mock_calibrate.return_value = {}
            trainer._recalibrate_reactivation(model, train_loader)

        self.assertEqual(len(mock_calibrate.call_args.args[1]), 2)
        self.assertAlmostEqual(mock_calibrate.call_args.kwargs["ema_alpha"], 0.25)


class TestLocalReactivationUpdateModes(unittest.TestCase):
    def test_localca_quantile_mode_disables_local_reactivation_updates(self):
        model = _ToyReactivationModel()
        trainer = LocalCreditAssignment(
            optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
            loss_function="cross_entropy",
            suppress_prints=True,
            num_workers=0,
            reactivation_update_mode="quantile",
            local_rule_config={"update_reactivation": True},
        )

        self.assertFalse(trainer._resolve_local_reactivation_update_flag(True))

    def test_localca_backprop_mode_preserves_local_reactivation_updates(self):
        model = _ToyReactivationModel()
        trainer = LocalCreditAssignment(
            optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
            loss_function="cross_entropy",
            suppress_prints=True,
            num_workers=0,
            reactivation_update_mode="backprop",
            local_rule_config={"update_reactivation": True},
        )

        self.assertTrue(trainer._resolve_local_reactivation_update_flag(True))
