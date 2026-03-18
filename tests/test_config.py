"""Tests for the current dendritic_modeling configuration schema."""

import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from dendritic_modeling.config import (
    AnalysisConfig,
    ArchitectureConfig,
    Config,
    ConnectivityConfig,
    CoreConfig,
    DataConfig,
    ModelConfig,
    OutputsConfig,
    TrainingConfig,
    TransferFunctionConfig,
    WandbConfig,
    load_config,
)


def _get_field(obj, key):
    """Return a field from either a dict-like or attribute-like object."""
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)


class TestConfigDataclasses(unittest.TestCase):
    """Test current dataclass structures."""

    def test_core_config_creation(self):
        architecture = ArchitectureConfig(
            excitatory_layer_sizes=[64, 32],
            inhibitory_layer_sizes=[16, 8],
            excitatory_branch_factors=[2, 2],
            inhibitory_branch_factors=[1, 1],
        )
        connectivity = ConnectivityConfig(
            ee_synapses_per_branch_per_layer=[24, 24],
            ei_synapses_per_branch_per_layer=[6, 6],
            ie_synapses_per_branch_per_layer=[2, 2],
            ii_synapses_per_branch_per_layer=[1, 1],
        )
        transfer = TransferFunctionConfig(
            input_mode=1,
            independent_pathways=False,
            excitatory_dim=128,
            inhibitory_dim=128,
        )
        core = CoreConfig(
            type="EINet",
            architecture=architecture,
            connectivity=connectivity,
            transfer=transfer,
        )

        self.assertEqual(core.type, "EINet")
        self.assertEqual(core.architecture.excitatory_layer_sizes, [64, 32])
        self.assertEqual(core.connectivity.ee_synapses_per_branch_per_layer, [24, 24])
        self.assertEqual(core.transfer.input_mode, 1)

    def test_model_config_creation(self):
        model_config = ModelConfig(
            task="classification",
            core=CoreConfig(
                architecture=ArchitectureConfig(excitatory_layer_sizes=[32]),
                connectivity=ConnectivityConfig(
                    ee_synapses_per_branch_per_layer=[8],
                    ei_synapses_per_branch_per_layer=[2],
                    ie_synapses_per_branch_per_layer=[2],
                    ii_synapses_per_branch_per_layer=[1],
                ),
            ),
        )
        self.assertEqual(model_config.task, "classification")
        self.assertEqual(model_config.core.architecture.excitatory_layer_sizes, [32])

    def test_complete_config_creation(self):
        config = Config(
            model=ModelConfig(task="classification"),
            training=TrainingConfig(),
            data=DataConfig(dataset_name="mnist"),
            wandb=WandbConfig(use_wandb=False),
            outputs=OutputsConfig(results_dir="/tmp/outputs"),
            analysis=AnalysisConfig(),
        )

        self.assertIsInstance(config, Config)
        self.assertIsInstance(config.model, ModelConfig)
        self.assertIsInstance(config.training, TrainingConfig)
        self.assertEqual(config.data.dataset_name, "mnist")


class TestConfigLoading(unittest.TestCase):
    """Test loading/parsing configuration files."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_config = {
            "experiment": {"seed": 42, "train_valid_split": 0.8},
            "data": {
                "dataset_name": "mnist",
                "base_dir": "",
                "processing": {"flatten": True, "normalize": False},
            },
            "model": {
                "task": "classification",
                "core": {
                    "type": "EINet",
                    "architecture": {
                        "excitatory_layer_sizes": [64, 32],
                        "inhibitory_layer_sizes": [32, 16],
                        "excitatory_branch_factors": [2, 2],
                        "inhibitory_branch_factors": [1, 1],
                    },
                    "connectivity": {
                        "ee_synapses_per_branch_per_layer": [24, 24],
                        "ei_synapses_per_branch_per_layer": [6, 6],
                        "ie_synapses_per_branch_per_layer": [2, 2],
                        "ii_synapses_per_branch_per_layer": [1, 1],
                    },
                    "transfer": {"input_mode": 1, "independent_pathways": False},
                },
            },
            "training": {
                "main": {
                    "strategy": "standard",
                    "common": {
                        "epochs": 100,
                        "batch_size": 64,
                        "param_groups": {"lr": 0.005, "topk_lr": 0.001},
                    },
                }
            },
            "wandb": {"use_wandb": False, "project": "test_project"},
            "outputs": {"run_name": "test_run", "results_dir": "/tmp/outputs"},
            "analysis": {},
        }
        self.config_file = os.path.join(self.temp_dir, "test_config.yaml")
        with open(self.config_file, "w") as f:
            yaml.safe_dump(self.sample_config, f)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_config_from_file(self):
        load_config.cache_clear()
        config = load_config(self.config_file)

        self.assertIsInstance(config, Config)
        self.assertIsInstance(config.model.core, CoreConfig)
        self.assertEqual(config.model.task, "classification")
        self.assertEqual(config.experiment.seed, 42)
        self.assertEqual(config.data.dataset_name, "mnist")
        self.assertEqual(_get_field(config.training.main.common, "epochs"), 100)
        self.assertEqual(
            _get_field(
                _get_field(config.training.main.common, "param_groups"),
                "lr",
            ),
            0.005,
        )
        self.assertEqual(
            config.model.core.architecture.excitatory_layer_sizes,
            [64, 32],
        )
        self.assertTrue(hasattr(config.model.core, "baseline_rnn"))

    def test_load_config_preserves_recurrent_strategy_config(self):
        load_config.cache_clear()
        recurrent_config = {
            **self.sample_config,
            "training": {
                "main": {
                    "strategy": "recurrent",
                    "common": {
                        "epochs": 10,
                        "batch_size": 16,
                        "param_groups": {"lr": 0.002},
                    },
                    "strategies": {
                        "recurrent": {
                            "grad_clip": 0.25,
                            "use_amp": True,
                            "lr_schedule": "cosine",
                            "lr_warmup_epochs": 3,
                        }
                    },
                }
            },
        }
        recurrent_file = os.path.join(self.temp_dir, "recurrent_config.yaml")
        with open(recurrent_file, "w") as f:
            yaml.safe_dump(recurrent_config, f)

        config = load_config(recurrent_file)

        self.assertEqual(config.training.main.strategy, "recurrent")
        self.assertEqual(config.training.main.strategies.recurrent.grad_clip, 0.25)
        self.assertTrue(config.training.main.strategies.recurrent.use_amp)
        self.assertEqual(
            config.training.main.strategies.recurrent.lr_schedule, "cosine"
        )
        self.assertEqual(config.training.main.strategies.recurrent.lr_warmup_epochs, 3)

    def test_load_config_invalid_file(self):
        load_config.cache_clear()
        with self.assertRaises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_load_config_invalid_yaml(self):
        load_config.cache_clear()
        invalid_yaml_file = os.path.join(self.temp_dir, "invalid.yaml")
        with open(invalid_yaml_file, "w") as f:
            f.write("invalid: yaml: content: [")

        with self.assertRaises(yaml.YAMLError):
            load_config(invalid_yaml_file)

    def test_create_default_config(self):
        default_config = Config()
        self.assertIsInstance(default_config, Config)
        self.assertEqual(default_config.model.task, "classification")
        self.assertEqual(default_config.training.main.strategy, "standard")


class TestConfigUpdating(unittest.TestCase):
    """Test mutability/updates on nested config objects."""

    def setUp(self):
        self.config = Config()

    def test_update_training_fields(self):
        self.config.training.main.common.epochs = 200
        self.config.training.main.common.batch_size = 128

        self.assertEqual(self.config.training.main.common.epochs, 200)
        self.assertEqual(self.config.training.main.common.batch_size, 128)

    def test_update_model_fields(self):
        self.config.model.core.architecture.excitatory_layer_sizes = [128, 64]
        self.assertEqual(
            self.config.model.core.architecture.excitatory_layer_sizes,
            [128, 64],
        )

    def test_param_groups_updates(self):
        self.config.training.main.common.param_groups.lr = 0.005
        self.config.training.main.common.param_groups.topk_lr = 0.001

        self.assertEqual(self.config.training.main.common.param_groups.lr, 0.005)
        self.assertEqual(self.config.training.main.common.param_groups.topk_lr, 0.001)


class TestConfigSerialization(unittest.TestCase):
    """Test file-based round-trip serialization via load_config."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_round_trip(self):
        payload = {
            "experiment": {"seed": 7},
            "data": {"dataset_name": "cifar10"},
            "model": {
                "task": "classification",
                "core": {
                    "architecture": {"excitatory_layer_sizes": [16, 8]},
                    "connectivity": {"ee_synapses_per_branch_per_layer": [4, 4]},
                },
            },
            "training": {"main": {"common": {"epochs": 150}}},
        }
        config_path = os.path.join(self.temp_dir, "round_trip_config.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(payload, f)

        load_config.cache_clear()
        restored = load_config(config_path)
        self.assertEqual(restored.experiment.seed, 7)
        self.assertEqual(_get_field(restored.training.main.common, "epochs"), 150)
        self.assertEqual(
            restored.model.core.architecture.excitatory_layer_sizes,
            [16, 8],
        )
        self.assertEqual(restored.data.dataset_name, "cifar10")


class TestStrategyPrecedence(unittest.TestCase):
    """Verify that strategy config merging follows the correct precedence.

    Rules:
    - Omitted strategy fields must NOT override common values.
    - Explicit non-default strategy fields MUST override common values.
    - Explicit default-valued strategy fields MUST still override common values
      (user intentionally chose the default).
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _make_config_and_prepare(self, common_overrides, strategy_overrides=None):
        """Write a YAML, load it, and return the prepared trainer config dict."""
        from dendritic_modeling.scripts.script_utils.config_utils import (
            prepare_trainer_config,
        )

        common = {
            "epochs": 10,
            "batch_size": 16,
            "param_groups": {"lr": 0.001},
        }
        common.update(common_overrides)
        training_main = {
            "strategy": "recurrent",
            "common": common,
        }
        if strategy_overrides is not None:
            training_main["strategies"] = {"recurrent": strategy_overrides}

        payload = {
            "experiment": {"seed": 1},
            "data": {"dataset_name": "lorenz"},
            "model": {"task": "regression"},
            "training": {"main": training_main},
        }
        config_path = os.path.join(self.temp_dir, "precedence_test.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(payload, f)

        load_config.cache_clear()
        config = load_config(config_path)
        return prepare_trainer_config(
            config.training.main, config.experiment, "/tmp/test"
        )

    def _make_recurrent_trainer(self, common_overrides, strategy_overrides=None):
        """Construct a recurrent trainer from the loaded YAML config."""
        from dendritic_modeling.training.factory import get_trainer

        trainer_cfg = self._make_config_and_prepare(
            common_overrides=common_overrides,
            strategy_overrides=strategy_overrides,
        )
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(()))], lr=1e-3)
        with patch("torch.cuda.is_available", return_value=True):
            trainer = get_trainer("recurrent", optimizer, dict(trainer_cfg))
        return trainer

    def test_omitted_strategy_does_not_override_common(self):
        """Common sets use_amp=True; strategy block omitted entirely."""
        trainer_cfg = self._make_config_and_prepare(
            common_overrides={"use_amp": True, "lr_schedule": "cosine"},
            strategy_overrides=None,
        )
        recurrent_cfg = trainer_cfg.get("recurrent_config", {})
        # Recurrent defaults (use_amp=False) must not be present
        self.assertNotIn("use_amp", recurrent_cfg)
        # Common values must survive into the trainer config
        self.assertTrue(trainer_cfg.get("use_amp", False))
        self.assertEqual(trainer_cfg.get("lr_schedule"), "cosine")

    def test_explicit_nondefault_override(self):
        """Common sets use_amp=False; strategy explicitly sets use_amp=True."""
        trainer_cfg = self._make_config_and_prepare(
            common_overrides={"use_amp": False},
            strategy_overrides={"use_amp": True, "lr_schedule": "cosine"},
        )
        recurrent_cfg = trainer_cfg.get("recurrent_config", {})
        # Explicit strategy overrides must be forwarded
        self.assertIn("use_amp", recurrent_cfg)
        self.assertTrue(recurrent_cfg["use_amp"])
        self.assertEqual(recurrent_cfg.get("lr_schedule"), "cosine")

    def test_explicit_default_valued_override(self):
        """Common sets grad_clip=5.0; strategy explicitly sets grad_clip=1.0 (the default).

        The user intentionally chose 1.0 in the strategy block.  Even though
        1.0 equals the dataclass default, the explicit override path must
        still forward it so it can override the common value.
        """
        trainer_cfg = self._make_config_and_prepare(
            common_overrides={"grad_clip_value": 5.0},
            strategy_overrides={"grad_clip": 1.0},
        )
        recurrent_cfg = trainer_cfg.get("recurrent_config", {})
        # The explicit-override path reads the raw YAML, so grad_clip=1.0
        # should be forwarded even though it matches the dataclass default.
        self.assertIn("grad_clip", recurrent_cfg)
        self.assertEqual(recurrent_cfg["grad_clip"], 1.0)

    def test_explicit_default_runtime_fields_are_preserved(self):
        """Explicit default-valued recurrent fields must override common values.

        This covers the exact fields from the recent regression bug:
        use_amp=False, lr_schedule='none', and lr_warmup_epochs=0.
        """
        trainer_cfg = self._make_config_and_prepare(
            common_overrides={
                "use_amp": True,
                "lr_schedule": "cosine",
                "lr_warmup_epochs": 5,
            },
            strategy_overrides={
                "use_amp": False,
                "lr_schedule": "none",
                "lr_warmup_epochs": 0,
            },
        )
        recurrent_cfg = trainer_cfg.get("recurrent_config", {})
        self.assertEqual(
            recurrent_cfg,
            {
                "use_amp": False,
                "lr_schedule": "none",
                "lr_warmup_epochs": 0,
            },
        )

    def test_end_to_end_recurrent_trainer_respects_explicit_defaults(self):
        """End-to-end check through get_trainer() for explicit default-valued overrides."""
        trainer = self._make_recurrent_trainer(
            common_overrides={
                "use_amp": True,
                "lr_schedule": "cosine",
                "lr_warmup_epochs": 5,
                "grad_clip_value": 5.0,
            },
            strategy_overrides={
                "use_amp": False,
                "lr_schedule": "none",
                "lr_warmup_epochs": 0,
                "grad_clip": 1.0,
            },
        )
        self.assertEqual(type(trainer).__name__, "RecurrentTrainer")
        self.assertFalse(trainer.recurrent_amp)
        self.assertEqual(trainer.lr_schedule, "none")
        self.assertEqual(trainer.lr_warmup_epochs, 0)
        self.assertEqual(trainer.grad_clip, 1.0)


class TestConfigIntegration(unittest.TestCase):
    """Smoke checks for expected fields consumed by other modules."""

    def setUp(self):
        self.config = Config()

    def test_model_compatibility_fields(self):
        core = self.config.model.core
        self.assertTrue(hasattr(core, "architecture"))
        self.assertTrue(hasattr(core, "connectivity"))
        self.assertTrue(hasattr(core, "transfer"))

    def test_training_compatibility_fields(self):
        train = self.config.training
        self.assertTrue(hasattr(train, "main"))
        self.assertTrue(hasattr(train.main, "common"))
        self.assertTrue(hasattr(train.main.common, "param_groups"))
        self.assertTrue(hasattr(train.main.common.param_groups, "lr"))

    def test_analysis_compatibility_fields(self):
        analysis = self.config.analysis
        self.assertTrue(hasattr(analysis, "performance_analysis"))
        self.assertTrue(hasattr(analysis, "information_analysis"))
        self.assertTrue(hasattr(analysis, "ablation_analysis"))


if __name__ == "__main__":
    unittest.main()
