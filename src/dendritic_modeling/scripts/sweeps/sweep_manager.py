#!/usr/bin/env python
"""
Unified Sweep Manager for Dendritic Modeling Framework
=====================================================

This is the main entry point for all parameter sweeps. It coordinates
the generation of configurations, job submission, and result analysis
for different sweep types (EI, branch, noise, general).

Usage:
    # Generate and run an EI sweep
    python sweep_manager.py ei --config configs/ei_sweep.yaml --run

    # Generate configs only (no job submission)
    python sweep_manager.py branch --config configs/branch_sweep.yaml --generate-only

    # Analyze existing results
    python sweep_manager.py ei --analyze --results-dir output/ei_20240101120000

    # Generate with job splitting
    python sweep_manager.py general --config configs/general_sweep.yaml --split-jobs 10
"""

import argparse
import sys
import os
import importlib
from pathlib import Path
from typing import Optional, Dict, Type

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from omegaconf import OmegaConf

# Use absolute imports to avoid relative import issues
try:
    from dendritic_modeling.scripts.sweeps.unified.base_sweep import BaseSweepGenerator, BaseSweepAnalyzer, SweepJobManager
except ImportError:
    # Fallback for when run as script
    sys.path.insert(0, str(Path(__file__).parent))
    from unified.base_sweep import BaseSweepGenerator, BaseSweepAnalyzer, SweepJobManager


class SweepManager:
    """
    Main orchestrator for all sweep operations.
    
    Coordinates config generation, job submission, and result analysis
    for different sweep types. Automatically discovers available sweep types
    from the sweep_types directory.
    """
    
    def __init__(self, sweep_type: str):
        # Auto-discover available sweep types
        self.sweep_types = self._discover_sweep_types()
        
        if sweep_type not in self.sweep_types:
            available_types = list(self.sweep_types.keys())
            raise ValueError(f"Unknown sweep type '{sweep_type}'. "
                           f"Available types: {available_types}")
        
        self.sweep_type = sweep_type
        self.generator = self._get_generator()
        self.analyzer = self._get_analyzer()
    
    def _discover_sweep_types(self) -> Dict[str, Dict[str, Type]]:
        """
        Auto-discover sweep types from the sweep_types directory.
        
        Returns:
            Dict mapping sweep type names to their generator and analyzer classes.
        """
        sweep_types = {}
        sweep_types_dir = Path(__file__).parent / 'sweep_types'
        
        for filename in os.listdir(sweep_types_dir):
            if filename.endswith('.py') and not filename.startswith('_'):
                module_name = filename[:-3]  # Remove .py extension
                
                try:
                    # Try multiple import strategies
                    module = None
                    
                    # Strategy 1: Try absolute import
                    try:
                        module = importlib.import_module(f'dendritic_modeling.scripts.sweeps.unified.sweep_types.{module_name}')
                    except ImportError:
                        pass
                    
                    # Strategy 2: Try relative import
                    if module is None:
                        try:
                            module = importlib.import_module(f'.sweep_types.{module_name}', 
                                                           package='dendritic_modeling.scripts.sweeps.unified')
                        except (ImportError, ValueError):
                            pass
                    
                    # Strategy 3: Try direct import after adding to path
                    if module is None:
                        sweep_types_path = str(Path(__file__).parent / 'sweep_types')
                        if sweep_types_path not in sys.path:
                            sys.path.insert(0, sweep_types_path)
                        try:
                            module = importlib.import_module(module_name)
                        except ImportError:
                            pass
                    
                    if module is None:
                        print(f"Warning: Failed to import sweep module {module_name}")
                        continue
                    
                    # Find generator and analyzer classes
                    generator_class = None
                    analyzer_class = None
                    
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type):
                            if (issubclass(attr, BaseSweepGenerator) and 
                                attr != BaseSweepGenerator):
                                generator_class = attr
                            elif (issubclass(attr, BaseSweepAnalyzer) and 
                                  attr != BaseSweepAnalyzer):
                                analyzer_class = attr
                    
                    if generator_class:
                        # Extract sweep type name (e.g., 'ei_sweep' -> 'ei')
                        sweep_name = module_name.lower().replace('_sweep', '')
                        sweep_types[sweep_name] = {
                            'generator': generator_class,
                            'analyzer': analyzer_class
                        }
                        
                except ImportError as e:
                    print(f"Warning: Failed to import sweep module {module_name}: {e}")
                    continue
        
        return sweep_types
    
    def _get_generator(self) -> BaseSweepGenerator:
        """Get the appropriate generator for the sweep type."""
        generator_class = self.sweep_types[self.sweep_type]['generator']
        return generator_class()
    
    def _get_analyzer(self) -> BaseSweepAnalyzer:
        """Get the appropriate analyzer for the sweep type."""
        analyzer_class = self.sweep_types[self.sweep_type]['analyzer']
        if analyzer_class:
            return analyzer_class()
        else:
            # Fallback to base analyzer if specific one not found
            return BaseSweepAnalyzer()
    
    def generate_sweep(self, config_path: str, output_dir: Optional[str] = None,
                      split_jobs: Optional[int] = None) -> dict:
        """
        Generate sweep configurations and job scripts.
        
        Args:
            config_path: Path to sweep configuration file
            output_dir: Output directory (None to auto-generate)
            split_jobs: Number of jobs to split into (None for single job)
            
        Returns:
            Dictionary with sweep information
        """
        print(f"Generating {self.sweep_type} sweep from {config_path}")
        
        # Load configuration
        config = OmegaConf.load(config_path)
        
        # Validate configuration
        if not self.generator.validate_config(config):
            raise ValueError("Configuration validation failed")
        
        # Create output directory
        base_dir = output_dir or config.get("output_dir", "output")
        sweep_output_dir = self.generator.create_output_directory(base_dir)
        print(f"📁 Created output directory: {sweep_output_dir}")
        
        # Generate configurations
        print("Generating sweep configurations...")
        sweep_configs = self.generator.generate_configs(config)
        print(f"Generated {len(sweep_configs)} configurations")
        
        # Save configurations
        config_paths = self.generator.save_configs(sweep_configs, sweep_output_dir)
        print(f"💾 Saved configurations to: {os.path.join(sweep_output_dir, 'configs')}")
        
        # Create job scripts
        print("Creating job scripts...")
        slurm_config = config.get("slurm_config", {})
        job_manager = SweepJobManager(slurm_config)
        job_scripts = job_manager.create_job_scripts(
            config_paths, sweep_output_dir, split_jobs
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
            "original_config": config_path
        }
        
        print(f"Sweep generation complete!")
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
        output_dir = sweep_info["output_dir"]
        
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
    
    def analyze_results(self, results_dir: str, output_dir: Optional[str] = None) -> dict:
        """
        Analyze results from a completed sweep.
        
        Args:
            results_dir: Directory containing sweep results
            output_dir: Directory to save analysis (defaults to results_dir/plots)
            
        Returns:
            Dictionary with analysis results
        """
        print(f"Analyzing {self.sweep_type} sweep results from {results_dir}")
        
        if not os.path.exists(results_dir):
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
        
        # Collect results
        print("Collecting results...")
        results = self.analyzer.collect_results(results_dir)
        print(f"Collected results from {len(results.get('configs', []))} configurations")
        
        # Generate plots
        plot_output_dir = output_dir or results_dir
        print("Generating plots...")
        plot_paths = self.analyzer.generate_plots(results, plot_output_dir)
        print(f"Generated {len(plot_paths)} plots")
        
        # Generate summary report
        print("Generating summary report...")
        report_path = self.analyzer.generate_summary_report(results, plot_output_dir)
        print(f"Summary report saved to: {report_path}")
        
        # Find best configurations
        try:
            best_configs = self.analyzer.find_best_configs(results)
            if best_configs:
                print(f"Top 3 configurations:")
                for i, config_info in enumerate(best_configs[:3]):
                    print(f"   {i+1}. Score: {config_info['score']:.4f} "
                         f"(Config ID: {config_info['config_id']})")
        except Exception as e:
            print(f"Could not determine best configurations: {e}")
        
        analysis_info = {
            "sweep_type": self.sweep_type,
            "results_dir": results_dir,
            "output_dir": plot_output_dir,
            "plot_paths": plot_paths,
            "report_path": report_path,
            "results": results
        }
        
        print("Analysis complete!")
        return analysis_info
    
    def run_full_sweep(self, config_path: str, split_jobs: Optional[int] = None,
                      dry_run: bool = False) -> dict:
        """
        Run a complete sweep: generate configs, submit jobs.
        
        Args:
            config_path: Path to sweep configuration file
            split_jobs: Number of jobs to split into
            dry_run: If True, generate configs and show jobs without submitting
            
        Returns:
            Dictionary with sweep information
        """
        print(f"Running full {self.sweep_type} sweep")
        
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
  # Generate and run an EI sweep
  python sweep_manager.py ei --config configs/ei_sweep.yaml --run

  # Generate configs only (no job submission)  
  python sweep_manager.py branch --config configs/branch_sweep.yaml --generate-only

  # Analyze existing results
  python sweep_manager.py ei --analyze --results-dir output/ei_20240101120000

  # Generate with job splitting
  python sweep_manager.py general --config configs/general_sweep.yaml --split-jobs 10

  # Dry run (show what would be submitted)
  python sweep_manager.py ei --config configs/ei_sweep.yaml --run --dry-run
"""
    )
    
    # Sweep type (required)
    parser.add_argument(
        "sweep_type",
        choices=["ei", "branch", "noise", "general"],
        help="Type of sweep to perform"
    )
    
    # Action group - mutually exclusive
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--config",
        help="Path to sweep configuration file (for generation)"
    )
    action_group.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze existing results"
    )
    
    # Options for generation
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate configs only, don't submit jobs"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Generate configs and submit jobs"
    )
    parser.add_argument(
        "--split-jobs",
        type=int,
        help="Number of jobs to split sweep into"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be submitted without actually submitting"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for sweep (auto-generated if not specified)"
    )
    
    # Options for analysis
    parser.add_argument(
        "--results-dir",
        help="Directory containing sweep results to analyze"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        # Create sweep manager
        manager = SweepManager(args.sweep_type)
        
        if args.analyze:
            # Analysis mode
            if not args.results_dir:
                print("❌ Error: --results-dir required for analysis mode")
                sys.exit(1)
            
            manager.analyze_results(args.results_dir, args.output_dir)
            
        else:
            # Generation mode
            if not args.config:
                print("❌ Error: --config required for generation mode")
                sys.exit(1)
            
            if args.run and not args.generate_only:
                # Full sweep: generate and submit
                manager.run_full_sweep(
                    args.config,
                    split_jobs=args.split_jobs,
                    dry_run=args.dry_run
                )
            else:
                # Generate only
                sweep_info = manager.generate_sweep(
                    args.config,
                    output_dir=args.output_dir,
                    split_jobs=args.split_jobs
                )
                
                if not args.generate_only:
                    print("\nTo submit jobs, run:")
                    print(f"   bash {sweep_info['output_dir']}/submit_all_jobs.sh")
                    print("   OR use --run flag to auto-submit")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 