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


if __name__ == "__main__":
    unittest.main()
