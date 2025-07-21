#!/usr/bin/env python
"""
Legacy Configuration Migration Tool
==================================

Helps migrate legacy sweep configurations to the unified framework format.

Usage:
    python migrate_legacy_config.py <legacy_config.yaml> <output_config.yaml> --type ei
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from omegaconf import OmegaConf


def migrate_ei_config(legacy_config):
    """Migrate legacy EI sweep configuration."""
    unified_config = {
        "output_dir": legacy_config.get("run_config", {}).get("dir", "output"),
        "slurm_config": {},
        "experiment_settings": {},
        "sweep_config": {},
        "base_config": {}
    }
    
    # Migrate SLURM configuration
    run_config = legacy_config.get("run_config", {})
    unified_config["slurm_config"] = {
        "job_name": run_config.get("run_name", "ei_sweep"),
        "account": run_config.get("account", "kempner_dev"),
        "partition": run_config.get("partition", "kempner"),
        "time": run_config.get("time", "02:00:00"),
        "nodes": run_config.get("nodes", 1),
        "ntasks_per_node": run_config.get("ntasks_per_node", 1),
        "gpus_per_node": run_config.get("gpus_per_node", 1),
        "cpus_per_task": run_config.get("cpus_per_task", 4),
        "mem": run_config.get("mem", "32G"),
        "modules_to_load": run_config.get("modules_to_load", ["cuda/12.1.0-fasrc01"]),
        "conda_env_path": run_config.get("conda_env_path", "/path/to/conda/env"),
        "training_script": run_config.get("training_script", "src/dendritic_modeling/scripts/train_experiments.py")
    }
    
    # Migrate experiment settings
    exp_settings = legacy_config.get("experiment_settings", {})
    unified_config["experiment_settings"] = {
        "seeds_per_condition": exp_settings.get("seeds_per_condition", 1),
        "base_seed": exp_settings.get("base_seed", 42)
    }
    
    # Migrate sweep configuration
    sweep_config = legacy_config.get("sweep_config", {})
    unified_config["sweep_config"] = sweep_config.copy()
    
    # Migrate base training configuration
    unified_config["base_config"] = legacy_config.get("train_config", {}).copy()
    
    return unified_config


def migrate_general_config(legacy_config):
    """Migrate legacy general sweep configuration."""
    unified_config = {
        "output_dir": "output",
        "slurm_config": {},
        "experiment_settings": {},
        "sweep_config": {},
        "base_config": {}
    }
    
    # Extract SLURM config if available
    slurm_config = legacy_config.get("slurm_config", {})
    unified_config["slurm_config"] = {
        "job_name": slurm_config.get("run_name", "general_sweep"),
        "account": slurm_config.get("account", "kempner_dev"),
        "partition": slurm_config.get("partition", "kempner"),
        "time": slurm_config.get("time", "02:00:00"),
        "nodes": slurm_config.get("nodes", 1),
        "ntasks_per_node": slurm_config.get("ntasks_per_node", 1),
        "gpus_per_node": slurm_config.get("gpus_per_node", 1),
        "cpus_per_task": slurm_config.get("cpus_per_task", 4),
        "mem": slurm_config.get("mem", "32G"),
        "modules_to_load": slurm_config.get("modules_to_load", ["cuda/12.1.0-fasrc01"]),
        "conda_env_path": slurm_config.get("conda_env_path", "/path/to/conda/env"),
        "training_script": slurm_config.get("training_script", "src/dendritic_modeling/scripts/train_experiments.py")
    }
    
    # Migrate experiment settings
    exp_settings = legacy_config.get("experiment_settings", {})
    unified_config["experiment_settings"] = {
        "seeds_per_condition": exp_settings.get("seeds_per_condition", 1),
        "base_seed": exp_settings.get("base_seed", 42)
    }
    
    # Migrate sweep configuration
    unified_config["sweep_config"] = legacy_config.get("sweep_config", {}).copy()
    
    # Migrate base training configuration
    unified_config["base_config"] = legacy_config.get("train_config", {}).copy()
    
    return unified_config


def migrate_config(legacy_config_path, output_config_path, sweep_type):
    """Migrate a legacy configuration to unified format."""
    print(f" Migrating {sweep_type} configuration...")
    print(f"   Input: {legacy_config_path}")
    print(f"   Output: {output_config_path}")
    
    # Load legacy configuration
    try:
        legacy_config = OmegaConf.load(legacy_config_path)
        print(" Legacy configuration loaded successfully")
    except Exception as e:
        print(f" Error loading legacy configuration: {e}")
        return False
    
    # Migrate based on sweep type
    try:
        if sweep_type == "ei":
            unified_config = migrate_ei_config(legacy_config)
        elif sweep_type == "general":
            unified_config = migrate_general_config(legacy_config)
        else:
            # For other types, use general migration as fallback
            unified_config = migrate_general_config(legacy_config)
            print(f"  Using general migration for sweep type: {sweep_type}")
        
        print(" Configuration migrated successfully")
    except Exception as e:
        print(f" Error during migration: {e}")
        return False
    
    # Save unified configuration
    try:
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_config_path), exist_ok=True)
        
        # Save with comments
        unified_omegaconf = OmegaConf.create(unified_config)
        OmegaConf.save(unified_omegaconf, output_config_path)
        
        print(f" Unified configuration saved to: {output_config_path}")
    except Exception as e:
        print(f" Error saving unified configuration: {e}")
        return False
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate legacy sweep configurations to unified format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate EI sweep configuration
  python migrate_legacy_config.py configs/old_ei_sweep.yaml configs/new_ei_sweep.yaml --type ei
  
  # Migrate general sweep configuration
  python migrate_legacy_config.py configs/old_general.yaml configs/new_general.yaml --type general
"""
    )
    
    parser.add_argument(
        "legacy_config",
        help="Path to legacy configuration file"
    )
    parser.add_argument(
        "output_config", 
        help="Path for migrated configuration file"
    )
    parser.add_argument(
        "--type",
        choices=["ei", "branch", "noise", "general"],
        default="general",
        help="Type of sweep configuration (default: general)"
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.legacy_config):
        print(f" Error: Legacy configuration file not found: {args.legacy_config}")
        sys.exit(1)
    
    # Perform migration
    success = migrate_config(args.legacy_config, args.output_config, args.type)
    
    if success:
        print("\nMigration complete!")
        print("\nNext steps:")
        print(f"   1. Review the migrated configuration: {args.output_config}")
        print(f"   2. Test with: python scripts/sweeps/unified/sweep_manager.py {args.type} --config {args.output_config} --generate-only")
        print(f"   3. Run with: python scripts/sweeps/unified/sweep_manager.py {args.type} --config {args.output_config} --run")
    else:
        print("\n Migration failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 