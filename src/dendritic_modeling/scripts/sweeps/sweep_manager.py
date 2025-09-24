#!/usr/bin/env python
"""
Unified Sweep Manager for Dendritic Modeling Framework
=====================================================

This is the main entry point for all parameter sweeps. It coordinates
the generation of configurations, job submission, and result analysis
for ALL sweep types through YAML configuration alone.

The sweep type (EI, branch, noise, general, mixed) is determined by
the sweep_config and filter_config in the YAML file.

Usage:
    # Generate and run any sweep (default behavior)
    python sweep_manager.py --config configs/my_sweep.yaml

    # Generate configs only (no job submission)
    python sweep_manager.py --config configs/my_sweep.yaml --generate-only

    # Analyze existing results
    python sweep_manager.py --analyze --results-dir output/sweep_20240101120000

    # Generate with job splitting
    python sweep_manager.py --config configs/my_sweep.yaml --split-jobs 10

    # Dry run (show what would be submitted)
    python sweep_manager.py --config configs/my_sweep.yaml --dry-run
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Use absolute imports to avoid relative import issues
try:
    from dendritic_modeling.scripts.sweeps.unified.base_sweep import SweepJobManager
    from dendritic_modeling.scripts.sweeps.unified.sweep_types.unified_sweep import (
        UnifiedSweepGenerator,
    )
except ImportError:
    # Fallback for when run as script
    sys.path.insert(0, str(Path(__file__).parent))
    from unified.base_sweep import SweepJobManager
    from unified.sweep_types.unified_sweep import UnifiedSweepGenerator


class SweepManager:
    """
    Main orchestrator for all sweep operations.

    Uses a unified generator that handles all sweep types (EI, branch, noise, general, mixed)
    through YAML configuration. No need to specify sweep type - it's determined by the config.
    """

    def __init__(self):
        # Use unified generator for all sweep types
        self.generator = UnifiedSweepGenerator()
        # Note: Analysis is handled by separate result_analyzer.py to avoid circular imports
        self.sweep_type = "unified"  # For backward compatibility in output

    def generate_sweep(
        self,
        config_path: str,
        output_dir: Optional[str] = None,
        split_jobs: Optional[int] = None,
        use_array_jobs: bool = True,
    ) -> dict:
        """
        Generate sweep configurations and job scripts.

        Args:
            config_path: Path to sweep configuration file
            output_dir: Output directory (None to auto-generate)
            split_jobs: Number of jobs to split into (None for single job)

        Returns:
            Dictionary with sweep information
        """
        print(f"Generating sweep from {config_path}")

        # Load configuration
        config = OmegaConf.load(config_path)

        # Validate configuration
        if not self.generator.validate_config(config):
            raise ValueError("Configuration validation failed")

        # Create output directory - check multiple possible locations
        if output_dir:
            base_dir = output_dir
        else:
            # Try different locations for output directory in priority order
            base_dir = (
                config.get("output_dir")  # Top-level output_dir (old format)
                or config.get("outputs", {}).get(
                    "dir"
                )  # Top-level outputs.dir (new format)
                or config.get("base_config", {})
                .get("outputs", {})
                .get("results_dir")  # base_config.outputs.results_dir
                or "output"  # Default fallback
            )

        # Get run_name for directory naming
        run_name = (
            config.get("outputs", {}).get("run_name")
            or config.get("base_config", {}).get("outputs", {}).get("run_name")
            or "unified_sweep"
        )

        sweep_output_dir = self.generator.create_output_directory(
            base_dir, prefix=run_name
        )
        print(f"Created output directory: {sweep_output_dir}")

        # Generate configurations
        print("Generating sweep configurations...")
        sweep_configs = self.generator.generate_configs(config)
        print(f"Generated {len(sweep_configs)} configurations")

        # Save configurations
        config_paths = self.generator.save_configs(sweep_configs, sweep_output_dir)
        print(f"Saved configurations to: {os.path.join(sweep_output_dir, 'configs')}")

        # Create job scripts
        print("Creating job scripts...")
        slurm_config = config.get("slurm_config", {})
        # Add run_name to slurm_config so it can be used for job naming
        slurm_config["run_name"] = run_name
        job_manager = SweepJobManager(slurm_config)
        job_scripts = job_manager.create_job_scripts(
            config_paths, sweep_output_dir, split_jobs, use_array_jobs
        )
        print(f"Created {len(job_scripts)} job scripts")

        # Save original config for reference
        original_config_path = os.path.join(sweep_output_dir, "original_config.yaml")
        OmegaConf.save(config, original_config_path)

        sweep_info = {
            "sweep_type": self.sweep_type,
            "output_dir": sweep_output_dir,
            "num_configs": len(sweep_configs),
            "config_paths": config_paths,
            "job_scripts": job_scripts,
            "original_config": config_path,
        }

        print("Sweep generation complete!")
        print(f"   Output directory: {sweep_output_dir}")
        print(f"   Configurations: {len(sweep_configs)}")
        print(f"   Job scripts: {len(job_scripts)}")

        return sweep_info

    def submit_jobs(self, sweep_info: dict, dry_run: bool = False) -> bool:
        """
        Submit SLURM jobs for the sweep.

        Args:
            sweep_info: Sweep information from generate_sweep
            dry_run: If True, show what would be submitted without submitting

        Returns:
            True if successful
        """
        job_scripts = sweep_info["job_scripts"]
        sweep_info["output_dir"]

        if dry_run:
            print("DRY RUN - Would submit the following jobs:")
            for script in job_scripts:
                if not script.endswith("submit_all_jobs.sh"):
                    print(f"   sbatch {script}")
            return True

        print(f"Submitting {len(job_scripts)} jobs...")

        # Find the main submission script
        main_script = None
        for script in job_scripts:
            if script.endswith("submit_all_jobs.sh"):
                main_script = script
                break

        if main_script:
            print(f"Running main submission script: {main_script}")
            os.system(f"bash {main_script}")
        else:
            print("No main submission script found, submitting individual jobs...")
            for script in job_scripts:
                if script.endswith(".sh") and not script.endswith("submit_all_jobs.sh"):
                    print(f"Submitting {script}")
                    os.system(f"sbatch {script}")

        print("Jobs submitted successfully!")
        print("Monitor job progress with: squeue -u $USER")

        return True

    def analyze_results(
        self, results_dir: str, output_dir: Optional[str] = None
    ) -> dict:
        """
        Analyze results from a completed sweep.

        Args:
            results_dir: Directory containing sweep results
            output_dir: Directory to save analysis (defaults to results_dir/plots)

        Returns:
            Dictionary with analysis results
        """
        print(f"Analyzing sweep results from {results_dir}")

        if not os.path.exists(results_dir):
            raise FileNotFoundError(f"Results directory not found: {results_dir}")

        # For unified sweeps, we recommend using the separate result_analyzer.py
        # to avoid circular dependencies and provide full analysis capabilities
        print("Note: For comprehensive analysis, use:")
        print(
            f"python -m src.dendritic_modeling.scripts.sweeps.result_analyzer --results-dir {results_dir} --auto-detect"
        )

        plot_output_dir = output_dir or results_dir

        # Basic analysis info
        analysis_info = {
            "sweep_type": self.sweep_type,
            "results_dir": results_dir,
            "output_dir": plot_output_dir,
            "plot_paths": [],
            "report_path": None,
            "results": {},
        }

        print(
            "Basic analysis complete! Use result_analyzer.py for detailed plots and reports."
        )
        return analysis_info

    def run_full_sweep(
        self, config_path: str, split_jobs: Optional[int] = None, dry_run: bool = False
    ) -> dict:
        """
        Run a complete sweep: generate configs, submit jobs.

        Args:
            config_path: Path to sweep configuration file
            split_jobs: Number of jobs to split into
            dry_run: If True, generate configs and show jobs without submitting

        Returns:
            Dictionary with sweep information
        """
        print("Running full sweep")

        # Generate sweep
        sweep_info = self.generate_sweep(config_path, split_jobs=split_jobs)

        # Submit jobs
        if not dry_run:
            self.submit_jobs(sweep_info, dry_run=dry_run)
        else:
            print("DRY RUN - Skipping job submission")
            self.submit_jobs(sweep_info, dry_run=True)

        return sweep_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Sweep Manager for Dendritic Modeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate and run any sweep (default behavior)
  python sweep_manager.py --config configs/my_sweep.yaml

  # Generate configs only (no job submission)
  python sweep_manager.py --config configs/my_sweep.yaml --generate-only

  # Analyze existing results
  python sweep_manager.py --analyze --results-dir output/sweep_20240101120000

  # Generate with job splitting
  python sweep_manager.py --config configs/my_sweep.yaml --split-jobs 10

  # Dry run (show what would be submitted)
  python sweep_manager.py --config configs/my_sweep.yaml --dry-run
""",
    )

    # Action group - mutually exclusive
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--config", help="Path to sweep configuration file (for generation)"
    )
    action_group.add_argument(
        "--analyze", action="store_true", help="Analyze existing results"
    )

    # Options for generation
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate configs only, don't submit jobs",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Generate configs and submit jobs (default behavior - kept for backward compatibility)",
    )
    parser.add_argument(
        "--split-jobs", type=int, help="Number of jobs to split sweep into"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be submitted without actually submitting",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for sweep (auto-generated if not specified)",
    )

    # Options for analysis
    parser.add_argument(
        "--results-dir", help="Directory containing sweep results to analyze"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    try:
        # Create sweep manager
        manager = SweepManager()

        if args.analyze:
            # Analysis mode
            if not args.results_dir:
                print("Error: --results-dir required for analysis mode")
                sys.exit(1)

            manager.analyze_results(args.results_dir, args.output_dir)

        else:
            # Generation mode
            if not args.config:
                print("Error: --config required for generation mode")
                sys.exit(1)

            if args.generate_only:
                # Generate only
                sweep_info = manager.generate_sweep(
                    args.config, output_dir=args.output_dir, split_jobs=args.split_jobs
                )

                print("\nTo submit jobs, run:")
                print(f"   bash {sweep_info['output_dir']}/submit_all_jobs.sh")
                print("   OR run without --generate-only flag to auto-submit")
            else:
                # Default behavior: generate and submit (--run is now default)
                manager.run_full_sweep(
                    args.config, split_jobs=args.split_jobs, dry_run=args.dry_run
                )

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
