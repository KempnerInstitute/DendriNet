"""
Unit tests for the unified sweep framework.

This module tests configuration generation logic for all sweep types to ensure
correctness and comprehensive coverage of the sweep functionality.
"""

import pytest
from omegaconf import OmegaConf, DictConfig
# Adjusted imports for unified sweeps
from scripts.sweeps.unified.sweep_types.ei_sweep import EISweepGenerator
from scripts.sweeps.unified.sweep_types.branch_sweep import BranchSweepGenerator
from scripts.sweeps.unified.sweep_types.noise_sweep import NoiseSweepGenerator
from scripts.sweeps.unified.sweep_types.general_sweep import GeneralSweepGenerator


@pytest.fixture
def base_config():
    """Fixture for a minimal base config."""
    return OmegaConf.create({
        "sweep_config": {},
        "base_config": {
            "model": {
                "network": {
                    "parameters": {}
                }
            },
            "train": {
                "main": {
                    "trainer": {
                        "epochs": 10,
                        "batch_size": 32
                    }
                }
            }
        },
        "experiment_settings": {
            "seeds_per_condition": 1,
            "base_seed": 42
        }
    })


def test_ei_generate_configs(base_config):
    """Test EI sweep configuration generation."""
    generator = EISweepGenerator()
    
    # Minimal EI sweep config
    base_config.sweep_config = {
        "model.network.parameters.ee_synapses_per_branch_per_layer": [10, 20],
        "model.network.parameters.ie_synapses_per_branch_per_layer": [5, 10],
        "model.network.parameters.use_shunting": [True, False]
    }
    
    configs = generator.generate_configs(base_config)
    
    # Should have 2 * 2 * 2 * 1 = 8 configurations (plus seeds)
    assert len(configs) == 8
    
    # Check first config
    assert configs[0].model.network.parameters.ee_synapses_per_branch_per_layer == 10
    assert configs[0].model.network.parameters.ie_synapses_per_branch_per_layer == 5
    assert configs[0].model.network.parameters.use_shunting == True
    
    # Check metadata is added
    assert hasattr(configs[0], "_sweep_metadata")
    assert "ei_ratio" in configs[0]._sweep_metadata
    assert configs[0]._sweep_metadata["ei_ratio"] == 2.0  # 10/5
    
    # Check that different combinations exist
    ee_values = [cfg.model.network.parameters.ee_synapses_per_branch_per_layer for cfg in configs]
    ie_values = [cfg.model.network.parameters.ie_synapses_per_branch_per_layer for cfg in configs]
    shunting_values = [cfg.model.network.parameters.use_shunting for cfg in configs]
    
    assert 10 in ee_values and 20 in ee_values
    assert 5 in ie_values and 10 in ie_values
    assert True in shunting_values and False in shunting_values


def test_ei_generate_configs_with_multiple_seeds(base_config):
    """Test EI sweep with multiple seeds."""
    generator = EISweepGenerator()
    
    base_config.sweep_config = {
        "model.network.parameters.ee_synapses_per_branch_per_layer": [10, 20],
        "model.network.parameters.ie_synapses_per_branch_per_layer": [10],
        "model.network.parameters.use_shunting": [True]
    }
    base_config.experiment_settings.seeds_per_condition = 3
    
    configs = generator.generate_configs(base_config)
    
    # Should have 2 * 1 * 1 * 3 = 6 configurations
    assert len(configs) == 6
    
    # Check seeds are different
    seeds = [cfg.train.seed for cfg in configs]
    assert len(set(seeds)) == 6  # All seeds should be unique


def test_branch_generate_configs(base_config):
    """Test branch sweep configuration generation."""
    generator = BranchSweepGenerator()
    
    base_config.sweep_config = {
        "model.network.parameters.num_branches_per_layer": [2, 4, 8],
        "model.network.parameters.num_layers": [2, 3],
        "model.network.parameters.branch_complexity": [0.5, 1.0]
    }
    
    configs = generator.generate_configs(base_config)
    
    # Should have 3 * 2 * 2 * 1 = 12 configurations
    assert len(configs) == 12
    
    # Check first config
    assert configs[0].model.network.parameters.num_branches_per_layer == 2
    assert configs[0].model.network.parameters.num_layers == 2
    assert configs[0].model.network.parameters.branch_complexity == 0.5
    
    # Check metadata
    assert hasattr(configs[0], "_sweep_metadata")
    assert "branch_config" in configs[0]._sweep_metadata
    assert configs[0]._sweep_metadata["branch_config"]["num_layers"] == 2
    assert configs[0]._sweep_metadata["branch_config"]["num_branches_per_layer"] == 2
    
    # Check parameter ranges are covered
    branch_counts = [cfg.model.network.parameters.num_branches_per_layer for cfg in configs]
    layer_counts = [cfg.model.network.parameters.num_layers for cfg in configs]
    complexities = [cfg.model.network.parameters.branch_complexity for cfg in configs]
    
    assert set(branch_counts) == {2, 4, 8}
    assert set(layer_counts) == {2, 3}
    assert set(complexities) == {0.5, 1.0}


def test_noise_generate_configs(base_config):
    """Test noise sweep configuration generation."""
    generator = NoiseSweepGenerator()
    
    base_config.sweep_config = {
        "noise.type": ["gaussian", "uniform"],
        "noise.level": [0.0, 0.1, 0.2],
        "noise.apply_to": ["input", "hidden"]
    }
    
    configs = generator.generate_configs(base_config)
    
    # Should have 2 * 3 * 2 * 1 = 12 configurations
    assert len(configs) == 12
    
    # Check first config
    assert configs[0].noise.type == "gaussian"
    assert configs[0].noise.level == 0.0
    assert configs[0].noise.apply_to == "input"
    
    # Check noise enabling logic
    # Configs with noise.level == 0.0 should have enabled=False
    zero_noise_configs = [cfg for cfg in configs if cfg.noise.level == 0.0]
    for cfg in zero_noise_configs:
        assert cfg.noise.enabled == False
    
    # Configs with noise.level > 0.0 should have enabled=True
    nonzero_noise_configs = [cfg for cfg in configs if cfg.noise.level > 0.0]
    for cfg in nonzero_noise_configs:
        assert cfg.noise.enabled == True
    
    # Check metadata
    assert hasattr(configs[0], "_sweep_metadata")
    assert "noise_config" in configs[0]._sweep_metadata


def test_general_generate_configs(base_config):
    """Test general sweep configuration generation."""
    generator = GeneralSweepGenerator()
    
    base_config.sweep_config = {
        "model.network.parameters.hidden_dim": [64, 128, 256],
        "train.main.trainer.learning_rate": [0.001, 0.01],
        "train.main.trainer.batch_size": [32, 64],
        "model.task": ["classification"]
    }
    
    configs = generator.generate_configs(base_config)
    
    # Should have 3 * 2 * 2 * 1 * 1 = 12 configurations
    assert len(configs) == 12
    
    # Check first config
    assert configs[0].model.network.parameters.hidden_dim == 64
    assert configs[0].train.main.trainer.learning_rate == 0.001
    assert configs[0].train.main.trainer.batch_size == 32
    assert configs[0].model.task == "classification"
    
    # Check metadata
    assert hasattr(configs[0], "_sweep_metadata")
    assert "parameters" in configs[0]._sweep_metadata
    assert "model.network.parameters.hidden_dim" in configs[0]._sweep_metadata["parameters"]
    
    # Check all parameter combinations are covered
    hidden_dims = [cfg.model.network.parameters.hidden_dim for cfg in configs]
    learning_rates = [cfg.train.main.trainer.learning_rate for cfg in configs]
    batch_sizes = [cfg.train.main.trainer.batch_size for cfg in configs]
    
    assert set(hidden_dims) == {64, 128, 256}
    assert set(learning_rates) == {0.001, 0.01}
    assert set(batch_sizes) == {32, 64}


def test_ei_sweep_validation():
    """Test EI sweep configuration validation."""
    generator = EISweepGenerator()
    
    # Test valid config
    valid_config = OmegaConf.create({
        "sweep_config": {
            "model.network.parameters.ee_synapses_per_branch_per_layer": [10, 20],
            "model.network.parameters.ie_synapses_per_branch_per_layer": [10]
        }
    })
    
    assert generator.validate_config(valid_config) == True
    
    # Test invalid config (missing required parameters)
    invalid_config = OmegaConf.create({
        "sweep_config": {
            "model.network.parameters.some_other_param": [1, 2, 3]
        }
    })
    
    assert generator.validate_config(invalid_config) == False


def test_branch_sweep_validation():
    """Test branch sweep configuration validation."""
    generator = BranchSweepGenerator()
    
    # Test valid config
    valid_config = OmegaConf.create({
        "sweep_config": {
            "model.network.parameters.num_branches_per_layer": [2, 4],
            "model.network.parameters.num_layers": [3]
        }
    })
    
    assert generator.validate_config(valid_config) == True
    
    # Test invalid config
    invalid_config = OmegaConf.create({
        "sweep_config": {
            "model.network.parameters.some_random_param": [1, 2]
        }
    })
    
    assert generator.validate_config(invalid_config) == False


def test_noise_sweep_validation():
    """Test noise sweep configuration validation."""
    generator = NoiseSweepGenerator()
    
    # Test valid config
    valid_config = OmegaConf.create({
        "sweep_config": {
            "noise.type": ["gaussian"],
            "noise.level": [0.1, 0.2]
        }
    })
    
    assert generator.validate_config(valid_config) == True
    
    # Test invalid config
    invalid_config = OmegaConf.create({
        "sweep_config": {
            "train.some_param": [1, 2]
        }
    })
    
    assert generator.validate_config(invalid_config) == False


def test_general_sweep_validation():
    """Test general sweep configuration validation."""
    generator = GeneralSweepGenerator()
    
    # General sweep should accept any valid parameter path
    valid_config = OmegaConf.create({
        "sweep_config": {
            "model.any.parameter": [1, 2],
            "train.any.other.parameter": ["a", "b"]
        }
    })
    
    assert generator.validate_config(valid_config) == True
    
    # Should still reject obviously invalid configs
    invalid_config = OmegaConf.create({
        "sweep_config": {}
    })
    
    assert generator.validate_config(invalid_config) == False


def test_config_generation_preserves_base_config():
    """Test that config generation preserves non-swept parameters from base config."""
    generator = GeneralSweepGenerator()
    
    base_config = OmegaConf.create({
        "sweep_config": {
            "train.main.trainer.learning_rate": [0.001, 0.01]
        },
        "base_config": {
            "model": {
                "task": "classification",
                "network": {
                    "type": "ExcitationInhibitionNetwork"
                }
            },
            "train": {
                "run_name": "test_run",
                "main": {
                    "trainer": {
                        "epochs": 100,
                        "batch_size": 32
                    }
                }
            }
        },
        "experiment_settings": {
            "seeds_per_condition": 1,
            "base_seed": 42
        }
    })
    
    configs = generator.generate_configs(base_config)
    
    assert len(configs) == 2
    
    # Check that base config parameters are preserved
    for cfg in configs:
        assert cfg.model.task == "classification"
        assert cfg.model.network.type == "ExcitationInhibitionNetwork"
        assert cfg.train.run_name == "test_run"
        assert cfg.train.main.trainer.epochs == 100
        assert cfg.train.main.trainer.batch_size == 32
    
    # Check that swept parameters vary
    learning_rates = [cfg.train.main.trainer.learning_rate for cfg in configs]
    assert set(learning_rates) == {0.001, 0.01}


def test_metadata_consistency():
    """Test that sweep metadata is consistent and contains expected fields."""
    generators = [
        EISweepGenerator(),
        BranchSweepGenerator(), 
        NoiseSweepGenerator(),
        GeneralSweepGenerator()
    ]
    
    base_config = OmegaConf.create({
        "base_config": {"model": {"network": {"parameters": {}}}, "train": {}},
        "experiment_settings": {"seeds_per_condition": 1, "base_seed": 42}
    })
    
    # Test each generator
    test_configs = [
        {"model.network.parameters.ee_synapses_per_branch_per_layer": [10], "model.network.parameters.ie_synapses_per_branch_per_layer": [5]},
        {"model.network.parameters.num_branches_per_layer": [2], "model.network.parameters.num_layers": [3]},
        {"noise.type": ["gaussian"], "noise.level": [0.1]},
        {"model.network.parameters.hidden_dim": [64]}
    ]
    
    for generator, sweep_config in zip(generators, test_configs):
        base_config.sweep_config = sweep_config
        configs = generator.generate_configs(base_config)
        
        assert len(configs) >= 1
        for cfg in configs:
            # All configs should have metadata
            assert hasattr(cfg, "_sweep_metadata")
            assert "sweep_type" in cfg._sweep_metadata
            assert "config_id" in cfg._sweep_metadata
            assert "seed" in cfg._sweep_metadata
            
            # Config ID should be unique within the sweep
            assert isinstance(cfg._sweep_metadata["config_id"], str)
            assert len(cfg._sweep_metadata["config_id"]) > 0 