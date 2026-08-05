"""Tests for sweep ConfigExtractor."""

import unittest

from omegaconf import OmegaConf

from dendritic_modeling.scripts.sweeps.collectors.config_extractor import (
    ConfigExtractor,
)


class TestConfigExtractor(unittest.TestCase):
    """Test ConfigExtractor functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.extractor = ConfigExtractor()

    def test_extract_model_params(self):
        """Test extraction of model parameters."""
        config = OmegaConf.create(
            {
                "model": {
                    "encoder": {
                        "type": "pathway_router",
                        "params": {
                            "router_mode": "learned",
                            "pathway_dim": 10,
                        },
                    },
                    "core": {
                        "connectivity": {
                            "ee_synapses_per_branch_per_layer": [40],
                            "ie_synapses_per_branch_per_layer": [10],
                        },
                        "architecture": {
                            "excitatory_layer_sizes": [20],
                            "inhibitory_layer_sizes": [5],
                            "excitatory_branch_factors": [3, 3, 3],
                        },
                        "morphology": {
                            "use_shunting": True,
                        },
                    },
                }
            }
        )

        params = self.extractor._extract_model_params(config)

        self.assertEqual(params["ee_value"], 40)
        self.assertEqual(params["ie_value"], 10)
        self.assertEqual(params["use_shunting"], True)
        self.assertEqual(params["excitatory_layers"], [20])
        self.assertEqual(params["branch_factors"], "3-3-3")
        self.assertEqual(params["encoder_type"], "pathway_router")
        self.assertEqual(params["router_mode"], "learned")
        self.assertEqual(params["pathway_dim"], 10)

    def test_extract_training_params(self):
        """Test extraction of training parameters."""
        config = OmegaConf.create(
            {
                "training": {
                    "main": {
                        "strategy": "local_ca",
                        "common": {
                            "param_groups": {"lr": 0.001},
                            "batch_size": 128,
                        },
                    }
                }
            }
        )

        params = self.extractor._extract_training_params(config)

        self.assertEqual(params["strategy"], "local_ca")
        self.assertEqual(params["learning_rate"], 0.001)
        self.assertEqual(params["batch_size"], 128)

    def test_extract_local_learning_params_new_structure(self):
        """Test extraction of local learning params from new structure."""
        config = OmegaConf.create(
            {
                "training": {
                    "main": {
                        "learning_strategy_config": {
                            "rule_variant": "5f",
                            "error_broadcast_mode": "pathway_vector",
                            "broadcast_rank": 3,
                            "pathway_broadcast_residual": 0.2,
                            "pathway_activity_gate_strength": 0.8,
                            "encoder_update_mode": "backprop",
                            "decoder_update_mode": "local",
                            "four_factor": {
                                "rho_mode": "dot",
                                "rho_estimator": "ema",
                            },
                            "morphology_aware": {
                                "use_dendritic_normalization": True,
                                "path_factor_mode": "per_branch",
                                "use_branch_role_rules": True,
                                "specialized_branch_scale": 1.3,
                                "mixed_branch_scale": 0.8,
                                "branch_role_power": 1.5,
                                "branch_role_alignment_weight": 0.4,
                            },
                            "inhibitory_homeostasis": {
                                "enabled": True,
                                "mode": "r_tot",
                                "weight": 0.05,
                            },
                        }
                    }
                }
            }
        )

        params = self.extractor._extract_training_params(config)

        self.assertEqual(params["rule_variant"], "5f")
        self.assertEqual(params["error_broadcast_mode"], "pathway_vector")
        self.assertEqual(params["broadcast_rank"], 3)
        self.assertEqual(params["pathway_broadcast_residual"], 0.2)
        self.assertEqual(params["pathway_activity_gate_strength"], 0.8)
        self.assertEqual(params["encoder_update_mode"], "backprop")
        self.assertEqual(params["decoder_update_mode"], "local")
        self.assertEqual(params["rho_mode"], "dot")
        self.assertEqual(params["use_dendritic_normalization"], True)
        self.assertEqual(params["path_factor_mode"], "per_branch")
        self.assertEqual(params["use_branch_role_rules"], True)
        self.assertEqual(params["specialized_branch_scale"], 1.3)
        self.assertEqual(params["mixed_branch_scale"], 0.8)
        self.assertEqual(params["branch_role_power"], 1.5)
        self.assertEqual(params["branch_role_alignment_weight"], 0.4)
        self.assertEqual(params["inhibitory_homeostasis_enabled"], True)
        self.assertEqual(params["inhibitory_homeostasis_mode"], "r_tot")
        self.assertEqual(params["inhibitory_homeostasis_weight"], 0.05)

    def test_extract_local_learning_params_old_structure_is_ignored(self):
        """Deprecated strategies.local_ca should no longer drive extraction."""
        config = OmegaConf.create(
            {
                "training": {
                    "main": {
                        "strategies": {
                            "local_ca": {
                                "rule_variant": "3f",
                                "four_factor": {
                                    "rho_mode": "pearson",
                                },
                            }
                        }
                    }
                }
            }
        )

        params = self.extractor._extract_training_params(config)

        self.assertNotIn("rule_variant", params)
        self.assertNotIn("rho_mode", params)

    def test_extract_experiment_params(self):
        """Test extraction of experiment parameters."""
        config = OmegaConf.create(
            {"experiment": {"seed": 42}, "data": {"dataset_name": "cifar10"}}
        )

        params = self.extractor._extract_experiment_params(config)

        self.assertEqual(params["seed"], 42)
        self.assertEqual(params["dataset"], "cifar10")

    def test_extract_model_params_normalizes_deprecated_network_type(self):
        """Deprecated network type aliases should extract as canonical names."""
        config = OmegaConf.create(
            {
                "model": {
                    "core": {
                        "type": "matchedtotalparammlp",
                        "morphology": {"use_shunting": False},
                    }
                }
            }
        )

        params = self.extractor._extract_model_params(config)
        self.assertEqual(params["network_type"], "total_param_mlp")
        self.assertEqual(params["use_shunting"], False)

    def test_extract_model_params_flat_mlp(self):
        """flat_mlp should be extracted as canonical non-shunting network."""
        config = OmegaConf.create(
            {
                "model": {
                    "core": {
                        "type": "flat_mlp",
                        "morphology": {"use_shunting": True},
                    }
                }
            }
        )

        params = self.extractor._extract_model_params(config)
        self.assertEqual(params["network_type"], "flat_mlp")
        self.assertEqual(params["use_shunting"], False)

    def test_extract_model_params_normalized_additive_alias(self):
        """Normalized additive aliases should extract as non-shunting networks."""
        config = OmegaConf.create(
            {
                "model": {
                    "core": {
                        "type": "normadd",
                        "morphology": {"use_shunting": True},
                    }
                }
            }
        )

        params = self.extractor._extract_model_params(config)
        self.assertEqual(params["network_type"], "dendritic_normalized_additive")
        self.assertEqual(params["use_shunting"], False)

    def test_extract_population_network_sweep_axes(self):
        """Population-network configs should expose standard E/I sweep fields."""
        config = OmegaConf.create(
            {
                "model": {
                    "core": {
                        "type": "population_network",
                        "population_network": {
                            "use_transfer": True,
                            "transfer_params": {
                                "input_mode": 1,
                                "independent_pathways": False,
                            },
                            "layers": [
                                {
                                    "name": "layer0",
                                    "readout_population": "e0",
                                    "population_defaults": {
                                        "ff_excitatory_synapses": 25,
                                        "ff_inhibitory_synapses": 5,
                                        "use_shunting": False,
                                        "use_additive_normalization": True,
                                        "blocklinear_strategy": "conductance_dynamic",
                                        "topk_type": "indexed",
                                        "topk_strategy": "none",
                                        "reactivation_strategy": "none",
                                        "reactivation_init_policy": "occupancy_quantile",
                                        "reactivate": True,
                                        "dbl_init_method": "analytical_expectation",
                                        "init_method": "xavier_normal",
                                    },
                                    "populations": [
                                        {
                                            "name": "i0",
                                            "polarity": "inhibitory",
                                            "n_neurons": 8,
                                            "branch_factors": [1],
                                        },
                                        {
                                            "name": "e0",
                                            "polarity": "excitatory",
                                            "n_neurons": 20,
                                            "branch_factors": [3, 3, 3],
                                        },
                                    ],
                                }
                            ],
                        },
                    }
                }
            }
        )

        params = self.extractor._extract_model_params(config)

        self.assertEqual(params["network_type"], "population_network")
        self.assertEqual(params["ee_value"], 25)
        self.assertEqual(params["ie_value"], 5)
        self.assertEqual(params["use_shunting"], False)
        self.assertEqual(params["use_additive_normalization"], True)
        self.assertEqual(params["branch_factors"], "3-3-3")
        self.assertEqual(params["readout_n_neurons"], 20)
        self.assertEqual(params["excitatory_layers"], [20])
        self.assertEqual(params["inhibitory_layers"], [8])
        self.assertEqual(params["inhibitory_population_count"], 1)
        self.assertEqual(params["inhibitory_neuron_count"], 8)
        self.assertEqual(params["inhibitory_routing"], "none")
        self.assertEqual(params["input_mode"], 1)
        self.assertEqual(params["independent_pathways"], False)
        self.assertEqual(params["network_layer_count"], 1)
        self.assertEqual(params["blocklinear_strategy"], "conductance_dynamic")
        self.assertEqual(params["topk_type"], "indexed")
        self.assertEqual(params["topk_strategy"], "none")
        self.assertEqual(params["reactivation_strategy"], "none")
        self.assertEqual(params["reactivation_init_policy"], "occupancy_quantile")
        self.assertEqual(params["reactivate"], True)
        self.assertEqual(params["dbl_init_method"], "analytical_expectation")
        self.assertEqual(params["init_method"], "xavier_normal")
        self.assertEqual(params["structured_connectivity_enabled"], False)
        self.assertEqual(params["structured_connectivity_method"], "none")
        self.assertEqual(params["synaptic_routing"], "global_fixed_index")

    def test_extract_population_network_spatial_routing(self):
        """Spatial routing must be an explicit standard sweep axis."""
        config = OmegaConf.create(
            {
                "model": {
                    "core": {
                        "type": "population_network",
                        "population_network": {
                            "layers": [
                                {
                                    "readout_population": "e0",
                                    "population_defaults": {
                                        "topk_type": "indexed",
                                        "structured_connectivity": {
                                            "enabled": True,
                                            "method": "spatial_morphology",
                                            "seed": 42,
                                            "spatial": {"input_shape": [3, 32, 32]},
                                        },
                                    },
                                    "populations": [
                                        {
                                            "name": "e0",
                                            "polarity": "excitatory",
                                            "n_neurons": 64,
                                            "branch_factors": [2, 2, 2, 2],
                                        }
                                    ],
                                }
                            ]
                        },
                    }
                }
            }
        )

        params = self.extractor._extract_model_params(config)

        self.assertEqual(params["structured_connectivity_enabled"], True)
        self.assertEqual(params["structured_connectivity_method"], "spatial_morphology")
        self.assertEqual(params["spatial_region_mode"], "partition")
        self.assertEqual(params["synaptic_routing"], "spatial_morphology")

    def test_extract_population_network_overlapping_spatial_routing(self):
        """Overlapping spatial windows must remain a distinct sweep axis."""
        config = OmegaConf.create(
            {
                "model": {
                    "core": {
                        "type": "population_network",
                        "population_network": {
                            "layers": [
                                {
                                    "readout_population": "e0",
                                    "population_defaults": {
                                        "topk_type": "indexed",
                                        "structured_connectivity": {
                                            "enabled": True,
                                            "method": "spatial_morphology",
                                            "seed": 42,
                                            "spatial": {
                                                "input_shape": [3, 32, 32],
                                                "region_mode": "overlapping_grid",
                                                "minimum_window_shape": [4, 4],
                                            },
                                        },
                                    },
                                    "populations": [
                                        {
                                            "name": "e0",
                                            "polarity": "excitatory",
                                            "n_neurons": 64,
                                            "branch_factors": [120],
                                        }
                                    ],
                                }
                            ]
                        },
                    }
                }
            }
        )

        params = self.extractor._extract_model_params(config)

        self.assertEqual(params["spatial_region_mode"], "overlapping_grid")
        self.assertEqual(
            params["synaptic_routing"],
            "spatial_morphology_overlapping_grid",
        )

    def test_extract_population_network_distinguishes_inhibitory_routes(self):
        def _config(source, populations):
            return OmegaConf.create(
                {
                    "model": {
                        "core": {
                            "type": "population_network",
                            "population_network": {
                                "layers": [
                                    {
                                        "readout_population": "e0",
                                        "population_defaults": {
                                            "ff_excitatory_synapses": 25,
                                            "ff_inhibitory_synapses": 10,
                                        },
                                        "populations": populations,
                                        "connections": [
                                            {
                                                "source": source,
                                                "target": "e0",
                                                "pathway": "ff_inhibitory",
                                                "timing": "same_step",
                                                "enabled": True,
                                            }
                                        ],
                                    }
                                ]
                            },
                        }
                    }
                }
            )

        excitatory = {
            "name": "e0",
            "polarity": "excitatory",
            "n_neurons": 20,
            "branch_factors": [3, 3, 3],
        }
        direct = self.extractor._extract_model_params(_config("input_i", [excitatory]))
        population = self.extractor._extract_model_params(
            _config(
                "i0",
                [
                    {
                        "name": "i0",
                        "polarity": "inhibitory",
                        "n_neurons": 50,
                        "branch_factors": [1],
                    },
                    excitatory,
                ],
            )
        )

        self.assertEqual(direct["inhibitory_routing"], "direct_input")
        self.assertEqual(population["inhibitory_routing"], "population_mediated")

    def test_extract_population_network_reports_mixed_effective_strategy(self):
        """Per-population overrides must not be merged under the first default."""
        config = OmegaConf.create(
            {
                "model": {
                    "core": {
                        "type": "population_network",
                        "population_network": {
                            "layers": [
                                {
                                    "population_defaults": {
                                        "blocklinear_strategy": "none",
                                    },
                                    "populations": [
                                        {"name": "e0", "polarity": "excitatory"},
                                        {
                                            "name": "i0",
                                            "polarity": "inhibitory",
                                            "population": {
                                                "blocklinear_strategy": "conductance_dynamic"
                                            },
                                        },
                                    ],
                                },
                                {
                                    "population_defaults": {
                                        "blocklinear_strategy": "none",
                                    },
                                    "populations": [
                                        {"name": "e1", "polarity": "excitatory"}
                                    ],
                                },
                            ]
                        },
                    }
                }
            }
        )

        params = self.extractor._extract_model_params(config)
        self.assertEqual(params["network_layer_count"], 2)
        self.assertEqual(
            params["blocklinear_strategy"], "mixed:conductance_dynamic|none"
        )

    def test_extract_training_reproducibility_parameters(self):
        """Optimizer decay and sparse maintenance must remain distinguishable."""
        config = OmegaConf.create(
            {
                "training": {
                    "main": {
                        "strategy": "standard",
                        "common": {
                            "epochs": 400,
                            "batch_size": 512,
                            "early_stopping": True,
                            "patience": 80,
                            "grad_clip_value": 5.0,
                            "weight_decay_rate": 0.01,
                            "weight_boosting": True,
                            "param_groups": {"lr": 0.001},
                        },
                        "optimizer": {"name": "adam", "lr": 0.001},
                    }
                }
            }
        )

        params = self.extractor._extract_training_params(config)
        self.assertEqual(params["epochs"], 400)
        self.assertEqual(params["early_stopping"], True)
        self.assertEqual(params["early_stopping_patience"], 80)
        self.assertEqual(params["grad_clip_value"], 5.0)
        self.assertEqual(params["sparse_weight_decay_rate"], 0.01)
        self.assertEqual(params["sparse_weight_boosting"], True)
        self.assertEqual(params["optimizer_name"], "adam")
        self.assertEqual(params["optimizer_weight_decay"], 0.0)


if __name__ == "__main__":
    unittest.main()
