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
    AnalysisConfig,
    ArchitectureConfig,
    CommonTrainerConfig,
    Config,
    ConnectivityConfig,
    CoreConfig,
    DataConfig,
    FSDPConfig,
    LocalRuleComponentAnalysisConfig,
    LocalRuleComponentAnalysisParams,
    LocalRuleConfig,
    ModelConfig,
    OptimizerConfig,
    OutputsConfig,
    SpikeTrainAnalysisConfig,
    SpikeTrainAnalysisParams,
    StructuredConnectivityConfig,
    TrainingConfig,
    TransferFunctionConfig,
    WandbConfig,
    load_config,
)
from dendritic_modeling.config.config import (
    _analysis_field_mapping,
    _build_filtered_config,
    _map_analysis_config,
    _map_training_config,
)


def _get_field(obj, key):
    """Return a field from either a dict-like or attribute-like object."""
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)


def test_fsdp_launcher_exposes_rnn_alias_canonicalizer():
    from dendritic_modeling.scripts.training import train_experiments_fsdp

    payload = {
        "model": {
            "core": {
                "type": "rnn_dendritic_additive",
                "unified_ei": {"layers": [{"excitatory": {}, "inhibitory": {}}]},
            }
        }
    }

    canonical = train_experiments_fsdp.canonicalize_rnn_core_flags(payload)
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

    def test_analysis_config_any_enabled_matches_final_analysis_gate(self):
        analysis = AnalysisConfig()
        self.assertTrue(analysis.any_enabled())

        final_gate_fields = tuple(
            spec.config_field for spec in get_registered_analyzers().values()
        )
        for field_name in final_gate_fields:
            getattr(analysis, field_name).enabled = False
        self.assertFalse(analysis.any_enabled())

        analysis.information_analysis.enabled = True
        self.assertTrue(analysis.any_enabled())

        analysis.information_analysis.enabled = False
        analysis.spike_train_analysis.enabled = True
        self.assertTrue(analysis.any_enabled())

        analysis.spike_train_analysis.enabled = False
        analysis.compartment_snr_analysis.enabled = True
        self.assertTrue(analysis.any_enabled())

        analysis.compartment_snr_analysis.enabled = False
        self.assertFalse(analysis.any_enabled())

    def test_analysis_field_mapping_tracks_registered_analyzers(self):
        """YAML analysis aliases should be derived from analyzer registry metadata."""
        mapping = _analysis_field_mapping()

        for name, spec in get_registered_analyzers().items():
            self.assertEqual(mapping[name], spec.config_field)
        self.assertEqual(mapping["spike_trains"], "spike_train_analysis")

    def test_map_analysis_config_preserves_registry_and_legacy_aliases(self):
        """Config mapping should preserve registered names and legacy aliases."""
        mapped = _map_analysis_config(
            {
                "information": {"enabled": False},
                "spike_trains": {"enabled": True, "training": True},
                "adversarial_robustness": {"enabled": True},
            }
        )

        self.assertEqual(mapped["information_analysis"], {"enabled": False})
        self.assertTrue(mapped["spike_train_analysis"].enabled)
        self.assertTrue(mapped["spike_train_analysis"].training)
        self.assertEqual(mapped["adversarial_robustness_analysis"], {"enabled": True})

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

        config = mapped["local_rule_component_analysis"]
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

        config = mapped["spike_train_analysis"]
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

    def test_build_filtered_config_drops_unknown_mapping_keys(self):
        config = _build_filtered_config(
            FSDPConfig,
            {
                "mixed_precision": False,
                "limit_all_gathers": False,
                "unknown_field": "ignored",
            },
        )

        self.assertFalse(config.mixed_precision)
        self.assertFalse(config.limit_all_gathers)
        self.assertEqual(config.sharding_strategy, "FULL_SHARD")
        self.assertFalse(hasattr(config, "unknown_field"))

    def test_build_filtered_config_accepts_omegaconf_payload(self):
        payload = OmegaConf.create(
            {
                "epochs": 7,
                "batch_size": 32,
                "not_a_trainer_field": "ignored",
            }
        )

        config = _build_filtered_config(CommonTrainerConfig, payload)

        self.assertEqual(config.epochs, 7)
        self.assertEqual(config.batch_size, 32)
        self.assertFalse(hasattr(config, "not_a_trainer_field"))

    def test_build_filtered_config_defaults_for_non_mapping_payload(self):
        config = _build_filtered_config(WandbConfig, object())

        self.assertFalse(config.use_wandb)
        self.assertEqual(config.project, "dendritic-sweeps")

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

    def test_map_training_config_prefers_explicit_main_learning_strategy_config(self):
        training = {
            "main": {
                "learning_strategy_config": {"rule_variant": "5f"},
                "common": {
                    "learning_strategy_config": {"rule_variant": "stdp"},
                },
            }
        }

        mapped = _map_training_config(training)

        main = mapped["main"]
        self.assertNotIn("learning_strategy_config", main["common"])
        self.assertIsInstance(main["learning_strategy_config"], LocalRuleConfig)
        self.assertEqual(main["learning_strategy_config"].rule_variant, "5f")

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
        self.assertIsInstance(
            config.model.core.connectivity.structured,
            StructuredConnectivityConfig,
        )

    def test_load_config_returns_independent_objects(self):
        """Cached config parsing must not leak runtime mutations."""
        load_config.cache_clear()
        config1 = load_config(self.config_file)
        config1.training.main.common.batch_size = 12345

        config2 = load_config(self.config_file)

        self.assertIsNot(config1, config2)
        self.assertEqual(config2.training.main.common.batch_size, 64)

    def test_training_dataloader_fields_are_preserved(self):
        """Large-data loader controls should survive typed config loading."""
        load_config.cache_clear()
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

    def test_additive_alias_defaults_to_occupancy_quantile_when_policy_omitted(self):
        """Additive configs should not silently use weak analytical gates."""
        load_config.cache_clear()
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
        load_config.cache_clear()
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

    def test_transfer_legacy_aliases_load_as_canonical_fields(self):
        load_config.cache_clear()
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

        config = load_config(alias_file)
        transfer = config.model.core.transfer

        self.assertTrue(transfer.allow_direct_inhibitory_stream)
        self.assertTrue(transfer.input_mode1_build_inhibitory_population)
        self.assertFalse(hasattr(transfer, "allow_direct_inhibitory_input"))
        self.assertFalse(
            hasattr(transfer, "build_inhibitory_population_for_input_mode1")
        )

    def test_load_config_with_structured_connectivity(self):
        load_config.cache_clear()
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
        load_config.cache_clear()
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
        load_config.cache_clear()
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
        load_config.cache_clear()
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
        load_config.cache_clear()
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
        load_config.cache_clear()
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
        load_config.cache_clear()
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
        load_config.cache_clear()
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
        load_config.cache_clear()
        transformer_config = {
            **self.sample_config,
            "model": {
                **self.sample_config["model"],
                "transformer_replacement": {
                    "enabled": True,
                    "model_name": "deepseek-ai/DeepSeek-V3",
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
        self.assertEqual(replacement.model_name, "deepseek-ai/DeepSeek-V3")
        self.assertEqual(replacement.layers, [0, 1, 2])
        self.assertEqual(replacement.layers_attr, "model.layers")
        self.assertEqual(replacement.model_kwargs["torch_dtype"], "auto")

    def test_load_config_preserves_transformer_replacement_training(self):
        load_config.cache_clear()
        transformer_training_config = {
            **self.sample_config,
            "training": {
                "transformer_replacement": {
                    "enabled": True,
                    "mode": "layerwise_distillation",
                    "teacher_source": "synthetic_mlp",
                    "train_target": "replacement_only",
                    "layers": [0],
                    "hidden_size": 8,
                    "intermediate_size": 32,
                    "max_steps": 25,
                    "num_workers": 3,
                    "pin_memory": False,
                    "persistent_workers": True,
                    "prefetch_factor": 2,
                    "synthetic_teacher": {"kind": "linear"},
                }
            },
        }
        transformer_training_file = os.path.join(
            self.temp_dir, "transformer_replacement_training_config.yaml"
        )
        with open(transformer_training_file, "w") as f:
            yaml.safe_dump(transformer_training_config, f)

        config = load_config(transformer_training_file)

        training = config.training.transformer_replacement
        self.assertTrue(training.enabled)
        self.assertEqual(training.teacher_source, "synthetic_mlp")
        self.assertEqual(training.train_target, "replacement_only")
        self.assertEqual(training.layers, [0])
        self.assertEqual(training.hidden_size, 8)
        self.assertEqual(training.num_workers, 3)
        self.assertFalse(training.pin_memory)
        self.assertTrue(training.persistent_workers)
        self.assertEqual(training.prefetch_factor, 2)
        self.assertEqual(training.synthetic_teacher["kind"], "linear")

    def test_load_config_preserves_vision_replacement_training(self):
        load_config.cache_clear()
        vision_training_config = {
            **self.sample_config,
            "training": {
                "vision_replacement": {
                    "enabled": True,
                    "mode": "blockwise_distillation",
                    "teacher_source": "tensor_cache",
                    "train_target": "replacement_only",
                    "max_steps": 25,
                    "num_workers": 4,
                    "pin_memory": False,
                    "persistent_workers": True,
                    "prefetch_factor": 3,
                    "tensor_cache": {"train_inputs": "train_x.pt"},
                }
            },
        }
        vision_training_file = os.path.join(
            self.temp_dir, "vision_replacement_training_config.yaml"
        )
        with open(vision_training_file, "w") as f:
            yaml.safe_dump(vision_training_config, f)

        config = load_config(vision_training_file)

        training = config.training.vision_replacement
        self.assertTrue(training.enabled)
        self.assertEqual(training.teacher_source, "tensor_cache")
        self.assertEqual(training.train_target, "replacement_only")
        self.assertEqual(training.max_steps, 25)
        self.assertEqual(training.num_workers, 4)
        self.assertFalse(training.pin_memory)
        self.assertTrue(training.persistent_workers)
        self.assertEqual(training.prefetch_factor, 3)
        self.assertEqual(training.tensor_cache["train_inputs"], "train_x.pt")

    def test_load_config_maps_spike_train_analysis(self):
        load_config.cache_clear()
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

        self.assertTrue(config.analysis.spike_train_analysis.enabled)
        self.assertEqual(config.analysis.spike_train_analysis.params.max_samples, 32)
        self.assertFalse(config.analysis.spike_train_analysis.params.include_raster)
        self.assertEqual(config.analysis.spike_train_analysis.params.dt, 0.001)

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
