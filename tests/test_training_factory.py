"""
Tests for training factory module to improve coverage.
"""

from unittest.mock import Mock, patch

import pytest
import torch

from dendritic_modeling.config.training import (
    HomeostaticControlStrategyConfig,
    OptimizerConfig,
    ParamGroupsConfig,
    PruningConfig,
    RegularizationConfig,
    VoltageStabilizationStrategyConfig,
)
from dendritic_modeling.training.factory import (
    TRAINER_STRATEGY_REGISTRY,
    _extract_enhanced_configs,
    get_available_trainer_strategies,
    get_trainer,
    register_trainer_strategy,
)


class TestExtractEnhancedConfigs:
    """Test _extract_enhanced_configs function."""

    def test_extract_optimizer_config(self):
        """Test extracting optimizer configuration."""
        trainer_config = {
            "optimizer": {
                "name": "adam",
                "lr": 0.001,
                "weight_decay": 1e-5,
                "betas": (0.9, 0.999),
            }
        }

        result = _extract_enhanced_configs(trainer_config)

        assert "optimizer_config" in result
        assert isinstance(result["optimizer_config"], OptimizerConfig)
        assert result["optimizer_config"].name == "adam"
        assert result["optimizer_config"].lr == 0.001

        # Original config should be cleaned
        assert "optimizer" not in trainer_config

    def test_register_trainer_strategy_adds_config_only_extension_point(self):
        class DummyTrainer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        strategy_name = "dummy_registry_strategy"
        optimizer = Mock()
        try:
            register_trainer_strategy(
                strategy_name,
                DummyTrainer,
                fixed_kwargs={"custom_mode": "probe"},
            )

            trainer = get_trainer(
                strategy_name,
                optimizer,
                {"device": "cpu"},
                analysis_manager="analysis",
            )

            assert strategy_name in get_available_trainer_strategies()
            assert isinstance(trainer, DummyTrainer)
            assert trainer.kwargs["optimizer"] is optimizer
            assert trainer.kwargs["analysis_manager"] == "analysis"
            assert trainer.kwargs["custom_mode"] == "probe"
        finally:
            TRAINER_STRATEGY_REGISTRY.pop(strategy_name, None)

    def test_extract_pruning_config(self):
        """Test extracting pruning configuration."""
        trainer_config = {
            "pruning": {
                "enabled": True,
                "threshold": 0.01,
            }
        }

        result = _extract_enhanced_configs(trainer_config)

        assert "pruning_config" in result
        assert isinstance(result["pruning_config"], PruningConfig)
        assert result["pruning_config"].enabled is True
        assert result["pruning_config"].threshold == 0.01

    def test_extract_regularization_config(self):
        """Test extracting regularization configuration."""
        trainer_config = {
            "regularization": {
                "l1_weight": 0.001,
                "l2_weight": 0.01,
                "split_params": True,
            }
        }

        result = _extract_enhanced_configs(trainer_config)

        assert "regularization_config" in result
        assert isinstance(result["regularization_config"], RegularizationConfig)
        assert result["regularization_config"].l1_weight == 0.001
        assert result["regularization_config"].l2_weight == 0.01

    def test_removed_reporting_option_raises(self):
        """The trainer option 'reporting' was never read and is rejected."""
        with pytest.raises(ValueError, match="'reporting' was removed"):
            _extract_enhanced_configs({"reporting": {"report_non_pruned": True}})

    def test_extract_all_configs(self):
        """Test extracting all configuration types together."""
        trainer_config = {
            "optimizer": {"name": "sgd", "lr": 0.1},
            "pruning": {"enabled": False},
            "regularization": {"l1_weight": 0.001},
        }

        result = _extract_enhanced_configs(trainer_config)

        assert "optimizer_config" in result
        assert "pruning_config" in result
        assert "regularization_config" in result

        # All original configs should be cleaned
        assert "optimizer" not in trainer_config
        assert "pruning" not in trainer_config
        assert "regularization" not in trainer_config

    def test_extract_no_configs(self):
        """Test extracting when no configs are present."""
        trainer_config = {"some_other_param": "value"}

        result = _extract_enhanced_configs(trainer_config)

        assert result == {}
        assert trainer_config["some_other_param"] == "value"

    def test_extract_param_groups_config_normalized_dict(self):
        """Test extracting and normalizing param groups from dict config."""
        trainer_config = {"param_groups": {"lr": 0.02, "split_params": False}}

        result = _extract_enhanced_configs(trainer_config)

        assert "param_groups_config" in result
        assert isinstance(result["param_groups_config"], ParamGroupsConfig)
        assert result["param_groups_config"].lr == 0.02
        assert result["param_groups_config"].split_params is False
        assert "param_groups" not in trainer_config

    def test_unknown_param_groups_keys_raise(self):
        from dendritic_modeling.config.keys import UnknownConfigKeyError

        with pytest.raises(UnknownConfigKeyError, match="did you mean 'topk_lr'"):
            _extract_enhanced_configs({"param_groups": {"topk_lrr": 0.1}})

    def test_extract_param_groups_config_passthrough_typed(self):
        """Test preserving typed param groups config objects."""
        typed_config = ParamGroupsConfig(lr=0.03, split_params=False)
        trainer_config = {"param_groups": typed_config}

        result = _extract_enhanced_configs(trainer_config)

        assert result["param_groups_config"] is typed_config
        assert "param_groups" not in trainer_config

    def test_extract_non_dict_configs(self):
        """Test that non-dict configs are handled properly."""
        trainer_config = {
            "optimizer": "not_a_dict",
            "pruning": None,
            "regularization": 42,
        }

        result = _extract_enhanced_configs(trainer_config)

        # Non-dict configs are still extracted but kept as-is
        assert "pruning_config" in result
        assert "regularization_config" in result
        assert result["pruning_config"] is None
        assert result["regularization_config"] == 42

        # Dict optimizer should be ignored, others should be popped
        assert trainer_config["optimizer"] == "not_a_dict"
        assert "pruning" not in trainer_config
        assert "regularization" not in trainer_config

    def test_extract_voltage_stabilization_config_preserves_target_saturation(self):
        """Nested voltage-stabilization config should flatten target_saturation."""
        trainer_config = {
            "voltage_stabilization_config": {
                "stabilize_mode": "gate_occupancy",
                "target_saturation": 0.35,
            }
        }

        result = _extract_enhanced_configs(trainer_config)

        assert result["stabilize_mode"] == "gate_occupancy"
        assert result["target_saturation"] == 0.35
        assert "voltage_stabilization_config" not in trainer_config

    def test_extract_homeostatic_control_config_preserves_target_saturation(self):
        """Nested homeostatic-control config should flatten target_saturation."""
        trainer_config = {
            "homeostatic_control_config": {
                "stabilize_mode": "gate_occupancy",
                "target_saturation": 0.15,
            }
        }

        result = _extract_enhanced_configs(trainer_config)

        assert result["stabilize_mode"] == "gate_occupancy"
        assert result["target_saturation"] == 0.15
        assert "homeostatic_control_config" not in trainer_config

    def test_extract_flat_strategy_configs_preserves_known_keys(self):
        """Strategy-specific nested dicts should flatten into trainer kwargs."""
        trainer_config = {
            "voltage_stabilization_config": {"voltage_mode": "gate"},
            "homeostatic_control_config": {"homeostasis_mode": "occupancy"},
            "freeze_layers_config": {"freeze_until": 2},
            "freeze_branches_config": {"branch_warmup": 3},
            "freeze_branch_kl_config": {"kl_weight": 0.7},
        }

        result = _extract_enhanced_configs(trainer_config)

        assert result == {
            "voltage_mode": "gate",
            "homeostasis_mode": "occupancy",
            "freeze_until": 2,
            "branch_warmup": 3,
            "kl_weight": 0.7,
        }
        assert trainer_config == {}

    def test_extract_flat_strategy_config_rejects_non_dict_value(self):
        """A strategy config that is not a mapping raises instead of vanishing."""
        with pytest.raises(TypeError, match="freeze_layers_config must be a mapping"):
            _extract_enhanced_configs({"freeze_layers_config": "not-a-mapping"})


class TestGetTrainer:
    """Test get_trainer factory function."""

    @pytest.fixture
    def mock_optimizer(self):
        """Create a mock optimizer."""
        return Mock(spec=torch.optim.Adam)

    @pytest.fixture
    def basic_trainer_config(self):
        """Create basic trainer configuration."""
        return {
            "device": "cpu",
            "max_epochs": 10,
            "patience": 5,
        }

    def test_get_standard_trainer(self, mock_optimizer, basic_trainer_config):
        """Test creating standard trainer."""
        with patch("dendritic_modeling.training.factory.Trainer") as MockTrainer:
            get_trainer("standard", mock_optimizer, basic_trainer_config)

            MockTrainer.assert_called_once()
            call_args = MockTrainer.call_args[1]
            assert call_args["optimizer"] == mock_optimizer
            assert call_args["device"] == "cpu"
            assert call_args["max_epochs"] == 10

    def test_get_enhanced_trainer_is_not_supported_legacy_alias(
        self, mock_optimizer, basic_trainer_config
    ):
        """Legacy enhanced alias should now be rejected."""
        with pytest.raises(ValueError, match="Unknown training strategy"):
            get_trainer("enhanced", mock_optimizer, basic_trainer_config)

    def test_get_layerwise_trainer(self, mock_optimizer, basic_trainer_config):
        """Test creating layer-wise trainer."""
        with patch(
            "dendritic_modeling.training.factory.LayerWiseTrainer"
        ) as MockTrainer:
            get_trainer("freeze_layers", mock_optimizer, basic_trainer_config)

            MockTrainer.assert_called_once()
            call_args = MockTrainer.call_args[1]
            assert call_args["optimizer"] == mock_optimizer

    def test_get_twostep_trainer(self, mock_optimizer, basic_trainer_config):
        """Test creating two-step trainer."""
        with patch("dendritic_modeling.training.factory.TwoStepTrainer") as MockTrainer:
            get_trainer("freeze_branches", mock_optimizer, basic_trainer_config)

            MockTrainer.assert_called_once()

    def test_get_local_ca_trainer(self, mock_optimizer, basic_trainer_config):
        """Test creating local credit assignment trainer."""
        with patch(
            "dendritic_modeling.training.factory.LocalCreditAssignment"
        ) as MockTrainer:
            get_trainer("local_ca", mock_optimizer, basic_trainer_config)

            MockTrainer.assert_called_once()

    def test_get_twostep_kl_trainer(self, mock_optimizer, basic_trainer_config):
        """Test creating two-step trainer with KL divergence."""
        with patch(
            "dendritic_modeling.training.factory.TwoStepTrainerWithKL"
        ) as MockTrainer:
            get_trainer("freeze_branch_kl", mock_optimizer, basic_trainer_config)

            MockTrainer.assert_called_once()

    def test_get_voltage_stabilization_trainer(
        self, mock_optimizer, basic_trainer_config
    ):
        """Test creating voltage stabilization trainer."""
        trainer_config = dict(basic_trainer_config)
        trainer_config["voltage_stabilization_config"] = {
            "stabilize_mode": "gate_occupancy",
            "target_saturation": 0.3,
        }
        with patch(
            "dendritic_modeling.training.factory.VoltageStabilizationTrainer"
        ) as MockTrainer:
            get_trainer("voltage_stabilization", mock_optimizer, trainer_config)

            MockTrainer.assert_called_once()
            call_args = MockTrainer.call_args[1]
            assert call_args["stabilize_mode"] == "gate_occupancy"
            assert call_args["target_saturation"] == 0.3

    def test_get_homeostatic_control_trainer(
        self, mock_optimizer, basic_trainer_config
    ):
        """Test creating homeostatic control trainer."""
        trainer_config = dict(basic_trainer_config)
        trainer_config["homeostatic_control_config"] = {
            "stabilize_mode": "gate_occupancy",
            "target_saturation": 0.25,
        }
        with patch(
            "dendritic_modeling.training.factory.HomeostaticControlTrainer"
        ) as MockTrainer:
            get_trainer("homeostatic_control", mock_optimizer, trainer_config)

            MockTrainer.assert_called_once()
            call_args = MockTrainer.call_args[1]
            assert call_args["stabilize_mode"] == "gate_occupancy"
            assert call_args["target_saturation"] == 0.25

    @pytest.mark.parametrize(
        ("strategy", "mode"),
        [
            ("shunting_fa", "fa"),
            ("shunting_dfa", "dfa"),
        ],
    )
    def test_shunting_feedback_alignment_names_build_fa_and_dfa(
        self,
        strategy,
        mode,
        mock_optimizer,
        basic_trainer_config,
    ):
        """The retired shunting variants build the trainer they name."""
        with patch(
            "dendritic_modeling.training.factory.FeedbackAlignmentTrainer"
        ) as mock_trainer:
            get_trainer(strategy, mock_optimizer, dict(basic_trainer_config))

            mock_trainer.assert_called_once()
            call_args = mock_trainer.call_args[1]
            assert call_args["optimizer"] == mock_optimizer
            assert call_args["mode"] == mode

    def test_get_fsdp_trainer_is_not_supported(
        self, mock_optimizer, basic_trainer_config
    ):
        """FSDP should be selected via distributed.mode, not trainer strategy."""
        with pytest.raises(ValueError, match="Unknown training strategy"):
            get_trainer("fsdp", mock_optimizer, basic_trainer_config)

    def test_get_fsdp_standard_trainer_is_not_supported(
        self, mock_optimizer, basic_trainer_config
    ):
        """Legacy fsdp_standard strategy should also be rejected."""
        with pytest.raises(ValueError, match="Unknown training strategy"):
            get_trainer("fsdp_standard", mock_optimizer, basic_trainer_config)

    def test_get_multi_stage_trainer(self, mock_optimizer, basic_trainer_config):
        """Test creating multi-stage trainer."""
        trainer_config = dict(basic_trainer_config)
        trainer_config["multi_stage_config"] = {
            "stages": [{"learning_strategy": "standard", "epochs": 1}]
        }

        with patch(
            "dendritic_modeling.training.factory.MultiStageTrainer"
        ) as mock_trainer:
            get_trainer("multi_stage", mock_optimizer, trainer_config)
            mock_trainer.assert_called_once()
            call_args = mock_trainer.call_args[1]
            assert call_args["base_optimizer"] == mock_optimizer

    def test_get_recurrent_trainer_uses_recurrent_config(self, mock_optimizer):
        """Recurrent strategy should consume nested recurrent_config fields."""
        trainer_config = {
            "task": "classification",
            "grad_clip_value": 5.0,
            "use_amp": False,
            "recurrent_config": {
                "grad_clip_value": 0.25,
                "use_amp": True,
                "lr_schedule": "cosine",
                "lr_warmup_epochs": 3,
            },
        }

        with patch(
            "dendritic_modeling.training.factory.RecurrentTrainer"
        ) as mock_trainer:
            get_trainer("recurrent", mock_optimizer, trainer_config)
            mock_trainer.assert_called_once()
            call_args = mock_trainer.call_args[1]
            assert call_args["optimizer"] == mock_optimizer
            assert call_args["task"] == "classification"
            assert call_args["grad_clip_value"] == 0.25
            assert call_args["use_amp"] is True
            assert call_args["lr_schedule"] == "cosine"
            assert call_args["lr_warmup_epochs"] == 3

    def test_get_enhanced_trainer_is_not_supported(
        self, mock_optimizer, basic_trainer_config
    ):
        """Legacy enhanced alias should be rejected."""
        with pytest.raises(ValueError, match="Unknown training strategy"):
            get_trainer("enhanced", mock_optimizer, basic_trainer_config)

    def test_get_unknown_trainer_raises(self, mock_optimizer, basic_trainer_config):
        """Test that unknown strategy raises a ValueError."""
        with pytest.raises(ValueError, match="Unknown training strategy"):
            get_trainer("unknown_strategy", mock_optimizer, basic_trainer_config)

    def test_case_insensitive_strategy(self, mock_optimizer, basic_trainer_config):
        """Test that strategy names are case insensitive."""
        with patch("dendritic_modeling.training.factory.Trainer") as MockTrainer:
            get_trainer("STANDARD", mock_optimizer, basic_trainer_config)

            MockTrainer.assert_called_once()

    def test_with_analysis_manager(self, mock_optimizer, basic_trainer_config):
        """Test passing analysis manager to trainer."""
        analysis_manager = Mock()

        with patch("dendritic_modeling.training.factory.Trainer") as MockTrainer:
            get_trainer(
                "standard", mock_optimizer, basic_trainer_config, analysis_manager
            )

            call_args = MockTrainer.call_args[1]
            assert call_args["analysis_manager"] == analysis_manager

    def test_with_enhanced_configs(self, mock_optimizer):
        """Test trainer creation with enhanced configurations."""
        trainer_config = {
            "device": "cpu",
            "optimizer": {"name": "adam", "lr": 0.001},
            "pruning": {"enabled": True},
        }

        with patch("dendritic_modeling.training.factory.Trainer") as MockTrainer:
            get_trainer("standard", mock_optimizer, trainer_config)

            call_args = MockTrainer.call_args[1]
            assert "pruning_config" in call_args
            assert call_args["optimizer"] == mock_optimizer
            assert isinstance(call_args["pruning_config"], PruningConfig)


class TestStrategyConfigDefaults:
    """Test specialized strategy config defaults."""

    def test_voltage_stabilization_default_target_saturation(self):
        cfg = VoltageStabilizationStrategyConfig()
        assert cfg.target_saturation == 0.2

    def test_homeostatic_control_default_target_saturation(self):
        cfg = HomeostaticControlStrategyConfig()
        assert cfg.target_saturation == 0.2


class TestTrainingFactoryEdgeCases:
    """Test edge cases for training factory functions."""

    def test_trainer_config_modification(self):
        """Test that trainer config is properly modified."""
        original_config = {
            "device": "cpu",
            "optimizer": {"name": "adam", "lr": 0.001},
        }
        config_copy = original_config.copy()

        enhanced_configs = _extract_enhanced_configs(config_copy)

        # Original config should be unchanged
        assert "optimizer" in original_config
        # Copy should be modified
        assert "optimizer" not in config_copy
        assert "optimizer_config" in enhanced_configs

    def test_get_trainer_preserves_other_params(self):
        """Test that get_trainer preserves other parameters."""
        mock_optimizer = Mock()
        trainer_config = {
            "device": "cuda",
            "custom_param": "custom_value",
            "max_epochs": 100,
        }

        with patch("dendritic_modeling.training.factory.Trainer") as MockTrainer:
            get_trainer("standard", mock_optimizer, trainer_config)

            call_args = MockTrainer.call_args[1]
            assert call_args["device"] == "cuda"
            assert call_args["custom_param"] == "custom_value"
            assert call_args["max_epochs"] == 100
            assert call_args["optimizer"] == mock_optimizer

    def test_fsdp_trainer_creation_flow_is_rejected(self):
        """Legacy FSDP trainer strategy path should be rejected."""
        mock_optimizer = Mock()
        trainer_config = {"device": "cuda"}

        with pytest.raises(ValueError, match="Unknown training strategy"):
            get_trainer("fsdp", mock_optimizer, trainer_config)

    def test_logger_usage(self):
        """Test that logger is used appropriately."""
        mock_optimizer = Mock()
        trainer_config = {"device": "cpu"}

        with patch("dendritic_modeling.training.factory.Trainer"):
            with patch("dendritic_modeling.training.factory.logger") as mock_logger:
                get_trainer("standard", mock_optimizer, trainer_config)
                mock_logger.warning.assert_not_called()

                # Removed legacy aliases and unknown strategies should raise.
                with pytest.raises(ValueError, match="Unknown training strategy"):
                    get_trainer("enhanced", mock_optimizer, trainer_config)
                with pytest.raises(ValueError, match="Unknown training strategy"):
                    get_trainer("unknown", mock_optimizer, trainer_config)
