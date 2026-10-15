"""Tests for the current dendritic_modeling configuration schema."""

import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml
from omegaconf import OmegaConf

from dendritic_modeling.analysis.registry import get_registered_analyzers
from dendritic_modeling.config import (
    AdversarialRobustnessAnalysisConfig,
    AnalysisConfig,
    ArchitectureConfig,
    CommonTrainerConfig,
    CompartmentSNRAnalysisConfig,
    CompartmentSNRAnalysisParams,
    Config,
    ConnectivityConfig,
    CoreConfig,
    DataConfig,
    FSDPConfig,
    InformationAnalysisConfig,
    LocalRuleComponentAnalysisConfig,
    LocalRuleComponentAnalysisParams,
    LocalRuleConfig,
    ModelConfig,
    OptimizerConfig,
    OutputsConfig,
    ReceptiveFieldAnalysisConfig,
    ReceptiveFieldAnalysisParams,
    SpikeTrainAnalysisConfig,
    SpikeTrainAnalysisParams,
    StructuredConnectivityConfig,
    TrainingConfig,
    TransferFunctionConfig,
    load_config,
)
from dendritic_modeling.config.config import (
    _analysis_field_mapping,
    _build_data_config,
    _map_analysis_config,
    _map_training_config,
)
from dendritic_modeling.config.keys import UnknownConfigKeyError, build_section


def _get_field(obj, key):
    """Return a field from either a dict-like or attribute-like object."""
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)


def test_both_launchers_record_canonical_rnn_alias_flags():
    from dendritic_modeling.scripts.training import (
        train_experiments,
        train_experiments_fsdp,
    )

    # Both entry points write config.json with the same function.
    assert train_experiments_fsdp._save_run_config is train_experiments._save_run_config
    payload = {
        "model": {
            "core": {
                "biological_neuron": False,
                "type": "rnn_dendritic_additive",
                "unified_ei": {"layers": [{"excitatory": {}, "inhibitory": {}}]},
            }
        }
    }

    canonical = train_experiments.canonicalize_rnn_core_flags(payload)
    layer = canonical["model"]["core"]["unified_ei"]["layers"][0]
    exc = layer["excitatory"]
    inh = layer["inhibitory"]

    assert exc["use_shunting"] is False
    assert exc["use_additive_normalization"] is False
    assert inh["use_shunting"] is False
    assert inh["use_additive_normalization"] is False


def test_dendritic_activation_none_preserves_reactivation_type():
    from dendritic_modeling.scripts.script_utils.config_utils import (
        prepare_ei_network_params,
    )

    params = prepare_ei_network_params(
        {
            "reactivation": {
                "enabled": True,
                "type": "param_tanh",
                "dendritic_activation": None,
            }
        },
        input_dim=4,
    )

    assert params["reactivation_type"] == "param_tanh"


def test_dendritic_activation_alias_overrides_reactivation_type():
    from dendritic_modeling.scripts.script_utils.config_utils import (
        prepare_ei_network_params,
    )

    params = prepare_ei_network_params(
        {
            "reactivation": {
                "enabled": True,
                "type": "param_tanh",
                "dendritic_activation": "tanh",
            }
        },
        input_dim=4,
    )

    assert params["reactivation_type"] == "tanh"


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
            independent_pathways=True,
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
        self.assertTrue(core.transfer.independent_pathways)

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
            outputs=OutputsConfig(results_dir="/tmp/outputs"),
            analysis=AnalysisConfig(),
        )

        self.assertIsInstance(config, Config)
        self.assertIsInstance(config.model, ModelConfig)
        self.assertIsInstance(config.training, TrainingConfig)
        self.assertEqual(config.data.dataset_name, "mnist")

    def test_analysis_config_any_enabled_matches_final_analysis_gate(self):
        analysis = AnalysisConfig()

        final_gate_fields = tuple(
            spec.config_field for spec in get_registered_analyzers().values()
        )
        for field_name in final_gate_fields:
            self.assertFalse(
                getattr(analysis, field_name).enabled,
                f"{field_name} should be disabled by default",
            )
        self.assertFalse(analysis.any_enabled())

        analysis.information.enabled = True
        self.assertTrue(analysis.any_enabled())

        analysis.information.enabled = False
        analysis.spike_train.enabled = True
        self.assertTrue(analysis.any_enabled())

        analysis.spike_train.enabled = False
        analysis.compartment_snr.enabled = True
        self.assertTrue(analysis.any_enabled())

        analysis.compartment_snr.enabled = False
        self.assertFalse(analysis.any_enabled())

    def test_analysis_field_mapping_tracks_registered_analyzers(self):
        """YAML analysis aliases should be derived from analyzer registry metadata."""
        mapping = _analysis_field_mapping()

        for name, spec in get_registered_analyzers().items():
            self.assertEqual(mapping[name], spec.config_field)
        self.assertEqual(mapping["spike_trains"], "spike_train")

    def test_map_analysis_config_preserves_registry_and_legacy_aliases(self):
        """Config mapping should preserve registered names and legacy aliases."""
        mapped = _map_analysis_config(
            {
                "information": {"enabled": False},
                "spike_trains": {"enabled": True, "training": True},
                "adversarial_robustness": {"enabled": True},
            }
        )

        self.assertIsInstance(mapped["information"], InformationAnalysisConfig)
        self.assertFalse(mapped["information"].enabled)
        self.assertTrue(mapped["spike_train"].enabled)
        self.assertTrue(mapped["spike_train"].training)
        self.assertIsInstance(
            mapped["adversarial_robustness"],
            AdversarialRobustnessAnalysisConfig,
        )
        self.assertTrue(mapped["adversarial_robustness"].enabled)

    def test_map_analysis_config_builds_registered_snr_and_rf_params(self):
        mapped = _map_analysis_config(
            {
                "compartment_snr": {
                    "enabled": True,
                    "params": {
                        "input_capture_mode": "network",
                        "branch_aggregation": False,
                    },
                },
                "receptive_field": {
                    "enabled": True,
                    "params": {
                        "rf_shape": [1, 3],
                        "compute_component_rfs": True,
                        "plot_component_rfs": False,
                        "compute_rf_tuning": True,
                        "plot_rf_tuning": False,
                    },
                },
            }
        )

        snr_config = mapped["compartment_snr"]
        self.assertIsInstance(snr_config, CompartmentSNRAnalysisConfig)
        self.assertIsInstance(snr_config.params, CompartmentSNRAnalysisParams)
        self.assertTrue(snr_config.enabled)
        self.assertEqual(snr_config.params.input_capture_mode, "network")
        self.assertFalse(snr_config.params.branch_aggregation)

        rf_config = mapped["receptive_field"]
        self.assertIsInstance(rf_config, ReceptiveFieldAnalysisConfig)
        self.assertIsInstance(rf_config.params, ReceptiveFieldAnalysisParams)
        self.assertTrue(rf_config.enabled)
        self.assertEqual(rf_config.params.rf_shape, [1, 3])
        self.assertTrue(rf_config.params.compute_component_rfs)
        self.assertFalse(rf_config.params.plot_component_rfs)
        self.assertTrue(rf_config.params.compute_rf_tuning)
        self.assertFalse(rf_config.params.plot_rf_tuning)

    def test_map_analysis_config_builds_local_rule_component_params(self):
        mapped = _map_analysis_config(
            {
                "local_rule_components": {
                    "enabled": True,
                    "training": True,
                    "params": {
                        "max_samples": 33,
                        "broadcast_mode": "per_soma",
                        "error_mode": "ce",
                    },
                }
            }
        )

        config = mapped["local_rule_components"]
        self.assertIsInstance(config, LocalRuleComponentAnalysisConfig)
        self.assertIsInstance(config.params, LocalRuleComponentAnalysisParams)
        self.assertTrue(config.enabled)
        self.assertTrue(config.training)
        self.assertEqual(config.params.max_samples, 33)
        self.assertEqual(config.params.broadcast_mode, "per_soma")
        self.assertEqual(config.params.error_mode, "ce")

    def test_map_analysis_config_builds_spike_train_params(self):
        mapped = _map_analysis_config(
            {
                "spike_trains": {
                    "enabled": True,
                    "params": {
                        "max_samples": 17,
                        "include_raster": False,
                        "dt": 0.25,
                    },
                }
            }
        )

        config = mapped["spike_train"]
        self.assertIsInstance(config, SpikeTrainAnalysisConfig)
        self.assertIsInstance(config.params, SpikeTrainAnalysisParams)
        self.assertTrue(config.enabled)
        self.assertEqual(config.params.max_samples, 17)
        self.assertFalse(config.params.include_raster)
        self.assertEqual(config.params.dt, 0.25)

    def test_local_rule_explicit_inhibitory_update_modes(self):
        local_cfg = LocalRuleConfig(explicit_inhibitory_update_mode="local_ca")
        self.assertEqual(local_cfg.explicit_inhibitory_update_mode, "local_ca")
        self.assertTrue(local_cfg.update_explicit_inhibitory_cells)

        freeze_cfg = LocalRuleConfig(explicit_inhibitory_update_mode="freeze")
        self.assertEqual(freeze_cfg.explicit_inhibitory_update_mode, "freeze")
        self.assertFalse(freeze_cfg.update_explicit_inhibitory_cells)

        alias_cfg = LocalRuleConfig(
            explicit_inhibitory_update_mode="same_as_excitatory"
        )
        self.assertEqual(alias_cfg.explicit_inhibitory_update_mode, "local_ca")

    def test_local_rule_legacy_explicit_inhibitory_update_bool(self):
        legacy_false = LocalRuleConfig(update_explicit_inhibitory_cells=False)
        self.assertEqual(legacy_false.explicit_inhibitory_update_mode, "freeze")
        self.assertFalse(legacy_false.update_explicit_inhibitory_cells)

        legacy_true = LocalRuleConfig(update_explicit_inhibitory_cells=True)
        self.assertEqual(legacy_true.explicit_inhibitory_update_mode, "local_ca")
        self.assertTrue(legacy_true.update_explicit_inhibitory_cells)

    def test_local_rule_rejects_invalid_explicit_inhibitory_update_mode(self):
        with self.assertRaises(ValueError):
            LocalRuleConfig(explicit_inhibitory_update_mode="graph_local")

    def test_build_section_rejects_unknown_mapping_keys(self):
        with self.assertRaisesRegex(
            UnknownConfigKeyError,
            r"Unknown distributed.fsdp option\(s\): cpu_ofload "
            r"\(did you mean 'cpu_offload'\?\)",
        ):
            build_section(
                FSDPConfig,
                {"mixed_precision": False, "cpu_ofload": True},
                section="distributed.fsdp",
            )

    def test_build_section_accepts_omegaconf_payload(self):
        payload = OmegaConf.create({"epochs": 7, "batch_size": 32})

        config = build_section(
            CommonTrainerConfig, payload, section="training.main.common"
        )

        self.assertEqual(config.epochs, 7)
        self.assertEqual(config.batch_size, 32)

    def test_build_section_rejects_non_mapping_payload(self):
        with self.assertRaisesRegex(ValueError, "outputs must be a mapping"):
            build_section(OutputsConfig, ["not", "a", "mapping"], section="outputs")

    def test_build_data_config_defaults_dataset_params_to_an_empty_mapping(self):
        config = _build_data_config({"dataset_name": "mnist"})

        self.assertEqual(config.dataset_name, "mnist")
        self.assertEqual(config.base_dir, "")
        self.assertIsNone(config.processing.flatten)  # each loader's default
        self.assertEqual(config.dataset_params, {})

    def test_build_data_config_rejects_a_non_mapping_dataset_block(self):
        with self.assertRaisesRegex(
            ValueError, r"data\.dataset_params\.mnist must be a mapping"
        ):
            _build_data_config({"dataset_params": {"mnist": [1, 2]}})

    def test_build_data_config_preserves_custom_dataset_param_sections(self):
        config = _build_data_config(
            {
                "dataset_name": "custom_stream",
                "base_dir": "/tmp/data",
                "processing": {"flatten": False, "normalize": True},
                "dataset_params": {
                    "custom_stream": {"window": 5, "stride": 2},
                    "imagenet": {"loader_backend": "imagefolder"},
                },
            }
        )

        self.assertEqual(config.dataset_name, "custom_stream")
        self.assertEqual(config.base_dir, "/tmp/data")
        self.assertFalse(config.processing.flatten)
        self.assertTrue(config.processing.normalize)
        self.assertEqual(config.dataset_params["custom_stream"]["window"], 5)
        self.assertEqual(
            config.dataset_params["imagenet"]["loader_backend"], "imagefolder"
        )

    def test_build_data_config_resolves_omegaconf_dataset_params(self):
        payload = OmegaConf.create(
            {
                "dataset_name": "imagenet",
                "dataset_params": {
                    "imagenet": {
                        "loader_backend": "torchvision",
                        "train_samples_per_class": 2,
                    }
                },
            }
        )

        config = _build_data_config(payload)

        self.assertIsInstance(config.dataset_params, dict)
        self.assertEqual(
            config.dataset_params["imagenet"]["loader_backend"],
            "torchvision",
        )
        self.assertEqual(
            config.dataset_params["imagenet"]["train_samples_per_class"], 2
        )

    def test_map_training_config_hoists_legacy_learning_strategy_config(self):
        training = {
            "main": {
                "strategy": "local_ca",
                "common": {
                    "epochs": 2,
                    "learning_strategy_config": {
                        "rule_variant": "4f",
                        "error_mode": "mse",
                    },
                },
            }
        }

        with self.assertLogs(
            "dendritic_modeling.config.config", level="WARNING"
        ) as logs:
            mapped = _map_training_config(training)

        main = mapped["main"]
        self.assertIn("hoisting to training.main", "\n".join(logs.output))
        self.assertNotIn("learning_strategy_config", main["common"])
        self.assertIsInstance(main["learning_strategy_config"], LocalRuleConfig)
        self.assertEqual(main["learning_strategy_config"].rule_variant, "4f")
        self.assertEqual(main["learning_strategy_config"].error_mode, "mse")

    def test_map_training_config_rejects_learning_strategy_config_twice(self):
        training = {
            "main": {
                "learning_strategy_config": {"rule_variant": "5f"},
                "common": {
                    "learning_strategy_config": {"rule_variant": "stdp"},
                },
            }
        }

        with self.assertRaisesRegex(ValueError, "set under both training.main"):
            _map_training_config(training)

    def test_map_training_config_builds_main_optimizer_config(self):
        mapped = _map_training_config(
            {
                "main": {
                    "optimizer": {
                        "name": "adamw",
                        "lr": 0.002,
                        "weight_decay": 0.01,
                    }
                }
            }
        )

        optimizer = mapped["main"]["optimizer"]
        self.assertIsInstance(optimizer, OptimizerConfig)
        self.assertEqual(optimizer.name, "adamw")
        self.assertEqual(optimizer.lr, 0.002)
        self.assertEqual(optimizer.weight_decay, 0.01)


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
                    "biological_neuron": False,
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
                    "transfer": {"independent_pathways": False},
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
        self.assertIsInstance(
            config.model.core.connectivity.structured,
            StructuredConnectivityConfig,
        )

    def test_load_config_preserves_sparse_point_parameter_budget(self):
        configured = {
            **self.sample_config,
            "model": {
                **self.sample_config["model"],
                "core": {
                    **self.sample_config["model"]["core"],
                    "type": "sparse_active_matched_point",
                    "target_active_parameters": 86_016,
                },
            },
        }
        with open(self.config_file, "w") as f:
            yaml.safe_dump(configured, f)

        config = load_config(self.config_file)

        self.assertEqual(config.model.core.target_active_parameters, 86_016)

    def test_strategy_settings_of_an_uninstalled_plugin_are_unknown(self):
        configured = {
            **self.sample_config,
            "training": {
                "main": {
                    **self.sample_config["training"]["main"],
                    "strategies": {"vision_distillation": {"setting": 1}},
                }
            },
        }
        with open(self.config_file, "w") as f:
            yaml.safe_dump(configured, f)

        with self.assertRaisesRegex(
            ValueError, "Unknown training.main.strategies block"
        ):
            load_config(self.config_file)

    def test_load_config_preserves_initial_checkpoint_contract(self):
        configured = {
            **self.sample_config,
            "model": {
                **self.sample_config["model"],
                "initial_checkpoint": "/tmp/model.pt",
                "initial_checkpoint_sha256": "a" * 64,
                "initial_checkpoint_strict": False,
            },
        }
        with open(self.config_file, "w") as f:
            yaml.safe_dump(configured, f)

        config = load_config(self.config_file)

        self.assertEqual(config.model.initial_checkpoint, "/tmp/model.pt")
        self.assertEqual(config.model.initial_checkpoint_sha256, "a" * 64)
        self.assertFalse(config.model.initial_checkpoint_strict)

    def test_load_config_returns_independent_objects(self):
        """Cached config parsing must not leak runtime mutations."""
        config1 = load_config(self.config_file)
        config1.training.main.common.batch_size = 12345

        config2 = load_config(self.config_file)

        self.assertIsNot(config1, config2)
        self.assertEqual(config2.training.main.common.batch_size, 64)

    def test_training_dataloader_fields_are_preserved(self):
        """Large-data loader controls should survive typed config loading."""
        dataloader_config = {
            **self.sample_config,
            "training": {
                "main": {
                    "strategy": "standard",
                    "common": {
                        "epochs": 1,
                        "batch_size": 16,
                        "num_workers": 6,
                        "pin_memory": False,
                        "persistent_workers": True,
                        "prefetch_factor": 3,
                        "in_order": False,
                        "profile_dataloader": True,
                    },
                }
            },
        }
        config_path = os.path.join(self.temp_dir, "dataloader_config.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(dataloader_config, f)

        config = load_config(config_path)
        common = config.training.main.common

        self.assertEqual(common.num_workers, 6)
        self.assertFalse(common.pin_memory)
        self.assertTrue(common.persistent_workers)
        self.assertEqual(common.prefetch_factor, 3)
        self.assertFalse(common.in_order)
        self.assertTrue(common.profile_dataloader)

    def test_additive_alias_defaults_to_occupancy_quantile_when_policy_omitted(self):
        """Additive configs should not silently use weak analytical gates."""
        additive_config = {
            **self.sample_config,
            "model": {
                **self.sample_config["model"],
                "core": {
                    **self.sample_config["model"]["core"],
                    "type": "dendritic_additive",
                },
            },
        }
        config_path = os.path.join(self.temp_dir, "additive_config.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(additive_config, f)

        config = load_config(config_path)

        self.assertEqual(
            config.model.core.reactivation.init_policy,
            "occupancy_quantile",
        )

    def test_explicit_additive_init_policy_is_preserved(self):
        """Users can still reproduce analytical/fixed additive experiments."""
        additive_config = {
            **self.sample_config,
            "model": {
                **self.sample_config["model"],
                "core": {
                    **self.sample_config["model"]["core"],
                    "type": "einet",
                    "morphology": {"use_shunting": False},
                    "reactivation": {"init_policy": "analytical"},
                },
            },
        }
        config_path = os.path.join(self.temp_dir, "additive_analytical.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(additive_config, f)

        config = load_config(config_path)

        self.assertEqual(config.model.core.reactivation.init_policy, "analytical")

    def test_retired_transfer_routing_keys_load_without_fields(self):
        alias_config = {
            **self.sample_config,
            "model": {
                **self.sample_config["model"],
                "core": {
                    **self.sample_config["model"]["core"],
                    "transfer": {
                        "input_mode": 1,
                        "allow_direct_inhibitory_input": True,
                        "build_inhibitory_population_for_input_mode1": True,
                    },
                },
            },
        }
        alias_file = os.path.join(self.temp_dir, "transfer_alias_config.yaml")
        with open(alias_file, "w") as f:
            yaml.safe_dump(alias_config, f)

        with self.assertLogs("dendritic_modeling.config.keys", "WARNING") as logs:
            config = load_config(alias_file)
        transfer = config.model.core.transfer

        self.assertIn("model.core.transfer.input_mode", "\n".join(logs.output))
        for retired in (
            "input_mode",
            "allow_direct_inhibitory_stream",
            "allow_direct_inhibitory_input",
            "input_mode1_build_inhibitory_population",
            "build_inhibitory_population_for_input_mode1",
        ):
            self.assertFalse(hasattr(transfer, retired))

    def test_enabled_plugin_model_section_without_plugin_is_rejected(self):
        legacy_config = {
            **self.sample_config,
            "model": {
                **self.sample_config["model"],
                "core": {
                    **self.sample_config["model"]["core"],
                    "transfer": {
                        "force_projection": False,
                        "zero_pad_output": True,
                    },
                },
                "pretrained_replacement": {"enabled": True},
            },
        }
        legacy_file = os.path.join(self.temp_dir, "legacy_vision_zero_pad.yaml")
        with open(legacy_file, "w") as f:
            yaml.safe_dump(legacy_config, f)

        with self.assertRaisesRegex(ValueError, "DendriNet plugin"):
            load_config(legacy_file)

    def test_legacy_forced_projection_fails_closed(self):
        legacy_config = {
            **self.sample_config,
            "model": {
                **self.sample_config["model"],
                "core": {
                    **self.sample_config["model"]["core"],
                    "transfer": {
                        "force_projection": True,
                    },
                },
                "pretrained_replacement": {
                    "enabled": True,
                    "backbone": "alexnet",
                },
            },
        }
        legacy_file = os.path.join(self.temp_dir, "legacy_forced_projection.yaml")
        with open(legacy_file, "w") as f:
            yaml.safe_dump(legacy_config, f)

        with self.assertRaisesRegex(ValueError, "cannot be migrated losslessly"):
            load_config(legacy_file)

    def test_load_config_with_structured_connectivity(self):
        structured_config = {
            **self.sample_config,
            "model": {
                **self.sample_config["model"],
                "core": {
                    **self.sample_config["model"]["core"],
                    "connectivity": {
                        **self.sample_config["model"]["core"]["connectivity"],
                        "structured": {
                            "enabled": True,
                            "seed": 123,
                            "pathways": {
                                "ee": {
                                    "method": "fixed_indegree",
                                    "indegree": 4,
                                }
                            },
                        },
                    },
                },
            },
        }
        structured_file = os.path.join(self.temp_dir, "structured_config.yaml")
        with open(structured_file, "w") as f:
            yaml.safe_dump(structured_config, f)

        config = load_config(structured_file)

        structured = config.model.core.connectivity.structured
        self.assertTrue(structured.enabled)
        self.assertEqual(structured.seed, 123)
        self.assertEqual(structured.pathways["ee"]["method"], "fixed_indegree")

    def test_load_config_with_structured_connectivity_alias_warns(self):
        structured_config = {
            **self.sample_config,
            "model": {
                **self.sample_config["model"],
                "core": {
                    **self.sample_config["model"]["core"],
                    "connectivity": {
                        **self.sample_config["model"]["core"]["connectivity"],
                        "structured_connectivity": {
                            "enabled": True,
                            "seed": 123,
                            "pathways": {
                                "ee": {
                                    "method": "fixed_indegree",
                                    "indegree": 4,
                                }
                            },
                        },
                    },
                },
            },
        }
        structured_file = os.path.join(self.temp_dir, "structured_alias_config.yaml")
        with open(structured_file, "w") as f:
            yaml.safe_dump(structured_config, f)

        with self.assertLogs(
            "dendritic_modeling.config.config", level="WARNING"
        ) as logs:
            config = load_config(structured_file)

        self.assertIn("structured_connectivity is deprecated", "\n".join(logs.output))
        structured = config.model.core.connectivity.structured
        self.assertTrue(structured.enabled)
        self.assertEqual(structured.seed, 123)

    def test_load_config_with_annealed_topk_block_alias_warns(self):
        sparse_config = {
            **self.sample_config,
            "model": {
                **self.sample_config["model"],
                "core": {
                    **self.sample_config["model"]["core"],
                    "sparsity": {
                        "type": "dense_to_sparse",
                        "annealed_topk": {
                            "initial_density": 0.75,
                            "end_step": 5,
                        },
                    },
                },
            },
        }
        sparse_file = os.path.join(self.temp_dir, "annealed_topk_alias_config.yaml")
        with open(sparse_file, "w") as f:
            yaml.safe_dump(sparse_config, f)

        with self.assertLogs(
            "dendritic_modeling.config.config", level="WARNING"
        ) as logs:
            config = load_config(sparse_file)

        self.assertIn("sparsity.annealed_topk is deprecated", "\n".join(logs.output))
        self.assertEqual(config.model.core.sparsity.dense_to_sparse["end_step"], 5)
        self.assertEqual(
            config.model.core.sparsity.dense_to_sparse["initial_density"], 0.75
        )

    def test_load_config_preserves_synapse_types(self):
        synapse_config = {
            **self.sample_config,
            "model": {
                **self.sample_config["model"],
                "core": {
                    **self.sample_config["model"]["core"],
                    "synapse_types": {
                        "enabled": True,
                        "excitatory": [
                            {"name": "AMPA", "fraction": 0.8, "tau": 2.0},
                            {"name": "NMDA", "fraction": 0.2, "tau": 70.0},
                        ],
                        "inhibitory": [{"name": "GABA_A", "fraction": 1.0, "tau": 7.0}],
                    },
                },
            },
        }
        synapse_file = os.path.join(self.temp_dir, "synapse_types_config.yaml")
        with open(synapse_file, "w") as f:
            yaml.safe_dump(synapse_config, f)

        config = load_config(synapse_file)

        synapse_types = config.model.core.synapse_types
        self.assertTrue(synapse_types["enabled"])
        self.assertEqual(synapse_types["excitatory"][1]["name"], "NMDA")
        self.assertEqual(synapse_types["inhibitory"][0]["tau"], 7.0)

    def test_load_config_preserves_soma_feedback(self):
        feedback_config = {
            **self.sample_config,
            "model": {
                **self.sample_config["model"],
                "core": {
                    **self.sample_config["model"]["core"],
                    "soma_feedback": {
                        "enabled": True,
                        "mode": "additive",
                        "source": "output",
                        "strength": 0.2,
                    },
                },
            },
        }
        feedback_file = os.path.join(self.temp_dir, "soma_feedback_config.yaml")
        with open(feedback_file, "w") as f:
            yaml.safe_dump(feedback_config, f)

        config = load_config(feedback_file)

        feedback = config.model.core.soma_feedback
        self.assertTrue(feedback["enabled"])
        self.assertEqual(feedback["mode"], "additive")
        self.assertAlmostEqual(feedback["strength"], 0.2)

    def test_load_config_preserves_dendritic_spikes(self):
        spikes_config = {
            **self.sample_config,
            "model": {
                **self.sample_config["model"],
                "core": {
                    **self.sample_config["model"]["core"],
                    "dendritic_spikes": {
                        "enabled": True,
                        "mode": "plateau",
                        "levels": "distal",
                        "threshold": 0.6,
                    },
                },
            },
        }
        spikes_file = os.path.join(self.temp_dir, "dendritic_spikes_config.yaml")
        with open(spikes_file, "w") as f:
            yaml.safe_dump(spikes_config, f)

        config = load_config(spikes_file)

        dendritic_spikes = config.model.core.dendritic_spikes
        self.assertTrue(dendritic_spikes["enabled"])
        self.assertEqual(dendritic_spikes["levels"], "distal")
        self.assertAlmostEqual(dendritic_spikes["threshold"], 0.6)

    def test_load_config_preserves_sparsity_ultrafast(self):
        ultrafast_config = {
            **self.sample_config,
            "model": {
                **self.sample_config["model"],
                "core": {
                    **self.sample_config["model"]["core"],
                    "sparsity": {
                        **self.sample_config["model"]["core"].get("sparsity", {}),
                        "ultrafast": True,
                    },
                },
            },
        }
        ultrafast_file = os.path.join(self.temp_dir, "ultrafast_config.yaml")
        with open(ultrafast_file, "w") as f:
            yaml.safe_dump(ultrafast_config, f)

        config = load_config(ultrafast_file)

        self.assertTrue(config.model.core.sparsity.ultrafast)

    def test_load_config_preserves_spiking_dynamics(self):
        dynamics_config = {
            **self.sample_config,
            "model": {
                **self.sample_config["model"],
                "core": {
                    **self.sample_config["model"]["core"],
                    "dynamics": {
                        "mode": "spike",
                        "threshold": 0.6,
                        "readout": "rate",
                        "tau": 20.0,
                        "refractory_steps": 2,
                    },
                },
            },
        }
        dynamics_file = os.path.join(self.temp_dir, "spiking_dynamics_config.yaml")
        with open(dynamics_file, "w") as f:
            yaml.safe_dump(dynamics_config, f)

        config = load_config(dynamics_file)

        dynamics = config.model.core.dynamics
        self.assertEqual(dynamics["mode"], "spike")
        self.assertEqual(dynamics["readout"], "rate")
        self.assertEqual(dynamics["refractory_steps"], 2)

    def test_load_config_preserves_transformer_replacement(self):
        transformer_config = {
            **self.sample_config,
            "model": {
                **self.sample_config["model"],
                "transformer_replacement": {
                    "enabled": True,
                    "model_name": "my-org/my-causal-lm",
                    "layers": [0, 1, 2],
                    "target_module": "mlp",
                    "layers_attr": "model.layers",
                    "input_transform": "signed_split",
                    "model_kwargs": {"torch_dtype": "auto"},
                },
            },
        }
        transformer_file = os.path.join(
            self.temp_dir, "transformer_replacement_config.yaml"
        )
        with open(transformer_file, "w") as f:
            yaml.safe_dump(transformer_config, f)

        config = load_config(transformer_file)

        replacement = config.model.transformer_replacement
        self.assertTrue(replacement.enabled)
        self.assertEqual(replacement.model_name, "my-org/my-causal-lm")
        self.assertEqual(replacement.layers, [0, 1, 2])
        self.assertEqual(replacement.layers_attr, "model.layers")
        self.assertEqual(replacement.model_kwargs["torch_dtype"], "auto")

    def test_enabled_plugin_transformer_training_without_plugin_is_rejected(self):
        transformer_training_config = {
            **self.sample_config,
            "training": {"transformer_replacement": {"enabled": True, "max_steps": 25}},
        }
        transformer_training_file = os.path.join(
            self.temp_dir, "transformer_replacement_training_config.yaml"
        )
        with open(transformer_training_file, "w") as f:
            yaml.safe_dump(transformer_training_config, f)

        with self.assertRaisesRegex(ValueError, "DendriNet plugin"):
            load_config(transformer_training_file)

    def test_enabled_plugin_vision_training_without_plugin_is_rejected(self):
        vision_training_config = {
            **self.sample_config,
            "training": {"vision_replacement": {"enabled": True, "max_steps": 25}},
        }
        vision_training_file = os.path.join(
            self.temp_dir, "vision_replacement_training_config.yaml"
        )
        with open(vision_training_file, "w") as f:
            yaml.safe_dump(vision_training_config, f)

        with self.assertRaisesRegex(ValueError, "DendriNet plugin"):
            load_config(vision_training_file)

    def test_load_config_maps_spike_train_analysis(self):
        spike_config = {
            **self.sample_config,
            "analysis": {
                "spike_train": {
                    "enabled": True,
                    "training": False,
                    "params": {
                        "max_samples": 32,
                        "include_raster": False,
                        "dt": 0.001,
                    },
                }
            },
        }
        spike_file = os.path.join(self.temp_dir, "spike_analysis_config.yaml")
        with open(spike_file, "w") as f:
            yaml.safe_dump(spike_config, f)

        config = load_config(spike_file)

        self.assertTrue(config.analysis.spike_train.enabled)
        self.assertEqual(config.analysis.spike_train.params.max_samples, 32)
        self.assertFalse(config.analysis.spike_train.params.include_raster)
        self.assertEqual(config.analysis.spike_train.params.dt, 0.001)

    def test_load_config_preserves_recurrent_strategy_config(self):
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
        # The pre-2026-10-10 name loads under grad_clip_value.
        self.assertEqual(
            config.training.main.strategies.recurrent.grad_clip_value, 0.25
        )
        self.assertTrue(config.training.main.strategies.recurrent.use_amp)
        self.assertEqual(
            config.training.main.strategies.recurrent.lr_schedule, "cosine"
        )
        self.assertEqual(config.training.main.strategies.recurrent.lr_warmup_epochs, 3)

    def test_load_config_invalid_file(self):
        with self.assertRaises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_load_config_invalid_yaml(self):
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
                    "biological_neuron": False,
                    "architecture": {"excitatory_layer_sizes": [16, 8]},
                    "connectivity": {"ee_synapses_per_branch_per_layer": [4, 4]},
                },
            },
            "training": {"main": {"common": {"epochs": 150}}},
        }
        config_path = os.path.join(self.temp_dir, "round_trip_config.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(payload, f)

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
        """Common sets grad_clip_value=5.0; the strategy explicitly sets 1.0.

        The user intentionally chose 1.0 in the strategy block.  Even though
        1.0 equals the dataclass default, the explicit override path must
        still forward it so it can override the common value.
        """
        trainer_cfg = self._make_config_and_prepare(
            common_overrides={"grad_clip_value": 5.0},
            strategy_overrides={"grad_clip_value": 1.0},
        )
        recurrent_cfg = trainer_cfg.get("recurrent_config", {})
        # The explicit-override path reads the raw YAML, so grad_clip_value=1.0
        # should be forwarded even though it matches the dataclass default.
        self.assertIn("grad_clip_value", recurrent_cfg)
        self.assertEqual(recurrent_cfg["grad_clip_value"], 1.0)

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
                "grad_clip_value": 1.0,
            },
        )
        self.assertEqual(type(trainer).__name__, "RecurrentTrainer")
        self.assertFalse(trainer.recurrent_amp)
        self.assertEqual(trainer.lr_schedule, "none")
        self.assertEqual(trainer.lr_warmup_epochs, 0)
        self.assertEqual(trainer.grad_clip_value, 1.0)


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
        self.assertTrue(hasattr(analysis, "performance"))
        self.assertTrue(hasattr(analysis, "information"))
        self.assertTrue(hasattr(analysis, "ablation"))


if __name__ == "__main__":
    unittest.main()
