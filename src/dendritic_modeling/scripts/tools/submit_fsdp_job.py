#!/usr/bin/env python
"""
FSDP Job Submission Manager
===========================

Automated workflow for FSDP training job submission. This script:
1. Estimates GPU requirements for the model
2. Generates optimized SLURM script
3. Optionally submits the job to SLURM

Similar to sweep_manager.py but for single FSDP training jobs.

Usage:
    # Estimate resources and generate SLURM script
    python src/dendritic_modeling/scripts/tools/submit_fsdp_job.py configs/config_exp_fsdp.yaml --generate-only
    
    # Generate and submit job
    python src/dendritic_modeling/scripts/tools/submit_fsdp_job.py configs/config_exp_fsdp.yaml --submit
    
    # Dry run (show what would be submitted)
    python src/dendritic_modeling/scripts/tools/submit_fsdp_job.py configs/config_exp_fsdp.yaml --submit --dry-run
    
    # Override SLURM settings
    python src/dendritic_modeling/scripts/tools/submit_fsdp_job.py configs/config_exp_fsdp.yaml --submit --partition kempner_h100 --account my_account
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from omegaconf import OmegaConf
from dendritic_modeling import logger
from dendritic_modeling.utils.gpu_estimation import estimate_from_config
from dendritic_modeling.scripts.tools.generate_slurm_job import SlurmJobGenerator


class FSDPJobManager:
    """
    Manages FSDP job estimation, SLURM script generation, and submission.
    """
    
    def __init__(self, config_path: str, output_dir: Optional[str] = None):
        """
        Initialize the FSDP Job Manager.
        
        Args:
            config_path: Path to FSDP configuration file
            output_dir: Output directory for SLURM scripts (defaults to results/slurm_jobs/)
        """
        self.config_path = Path(config_path)
        
        # Load config first to get results directory
        self.config = OmegaConf.load(config_path)
        
        # Set output directory - prefer results directory from config
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Get results directory from config
            results_dir = self.config.get('outputs', {}).get('dir', 'results')
            self.output_dir = Path(results_dir) / 'slurm_jobs'
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get run name from config
        self.run_name = self.config.outputs.get('run_name', self.config_path.stem)
        
        # Default SLURM script path
        self.slurm_script_path = self.output_dir / f"{self.run_name}.sbatch"
        
    def estimate_resources(self) -> Dict[str, Any]:
        """
        Estimate GPU and memory requirements for the model.
        
        Returns:
            Dictionary with estimation results and recommendations
        """
        logger.info("Estimating GPU requirements...")
        
        try:
            estimator, model = estimate_from_config(str(self.config_path))
            
            # Get batch size from config
            batch_size = 32  # default
            if hasattr(self.config, 'train') and hasattr(self.config.train, 'main'):
                if hasattr(self.config.train.main, 'trainer'):
                    batch_size = getattr(self.config.train.main.trainer, 'batch_size', 32)
            
            # Get input shape for dataset
            dataset = self.config.task.dataset
            input_shapes = {
                'mnist': (784,),
                'cifar10': (3072,), 
                'cifar100': (3072,),
                'imagenet': (150528,)
            }
            input_shape = input_shapes.get(dataset, (784,))
            
            # Get optimal GPU configuration
            optimal_config = estimator.print_estimation_report(model, batch_size, input_shape)
            
            # Memory breakdown for single GPU
            memory_breakdown = estimator.estimate_memory_breakdown(
                model, batch_size, input_shape, use_fsdp=False, num_gpus=1
            )
            
            # Memory breakdown for optimal FSDP configuration
            fsdp_memory_breakdown = estimator.estimate_memory_breakdown(
                model, batch_size, input_shape, use_fsdp=True, 
                num_gpus=optimal_config['total_gpus']
            )
            
            estimation_results = {
                'model_params': sum(p.numel() for p in model.parameters()),
                'batch_size': batch_size,
                'memory_single_gpu': memory_breakdown.total,
                'memory_per_gpu_fsdp': fsdp_memory_breakdown.total,
                'optimal_config': optimal_config,
                'memory_breakdown': memory_breakdown,
                'fsdp_memory_breakdown': fsdp_memory_breakdown
            }
            
            # Log summary
            logger.info(f"Model: {estimation_results['model_params']:,} parameters")
            logger.info(f"Single GPU memory: {estimation_results['memory_single_gpu']:.2f} GB")
            logger.info(f"Optimal config: {optimal_config['nodes']} nodes × {optimal_config['gpus_per_node']} GPUs")
            
            return estimation_results
            
        except Exception as e:
            logger.error(f"Failed to estimate resources: {e}")
            raise
    
    def generate_slurm_script(self, estimation_results: Optional[Dict[str, Any]] = None,
                             overrides: Optional[Dict[str, Any]] = None) -> Path:
        """
        Generate SLURM submission script.
        
        Args:
            estimation_results: Results from estimate_resources()
            overrides: Override SLURM settings
            
        Returns:
            Path to generated SLURM script
        """
        logger.info("Generating SLURM script...")
        
        try:
            # Apply optimal GPU configuration from estimator
            if estimation_results and 'optimal_config' in estimation_results:
                optimal_config = estimation_results['optimal_config']
                
                if 'slurm_config' not in self.config:
                    self.config.slurm_config = {}
                
                # Set optimal nodes and GPU configuration
                self.config.slurm_config['nodes'] = optimal_config['nodes']
                self.config.slurm_config['gpu_per_node'] = optimal_config['gpus_per_node']
                
                logger.info(f"Using optimal GPU config: {optimal_config['nodes']} nodes × {optimal_config['gpus_per_node']} GPUs")
            
            # Apply additional overrides if provided
            if overrides:
                if 'slurm_config' not in self.config:
                    self.config.slurm_config = {}
                
                for key, value in overrides.items():
                    if key in ['partition', 'account', 'nodes', 'gpu_per_node', 'time']:
                        self.config.slurm_config[key] = value
            
            # Generate script using existing generator
            generator = SlurmJobGenerator(str(self.config_path), str(self.output_dir))
            script_path = generator.save_script(f"{self.run_name}.sbatch")
            
            self.slurm_script_path = script_path
            logger.info(f"SLURM script generated: {script_path}")
            
            return script_path
            
        except Exception as e:
            logger.error(f"Failed to generate SLURM script: {e}")
            raise
    
    def submit_job(self, dry_run: bool = False) -> bool:
        """
        Submit the SLURM job.
        
        Args:
            dry_run: If True, show what would be submitted without submitting
            
        Returns:
            True if successful (or dry run)
        """
        if not self.slurm_script_path.exists():
            logger.error(f"SLURM script not found: {self.slurm_script_path}")
            return False
        
        if dry_run:
            logger.info("DRY RUN - Would submit the following job:")
            logger.info(f"   sbatch {self.slurm_script_path}")
            
            # Show script content summary
            with open(self.slurm_script_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('#SBATCH'):
                        logger.info(f"   {line.strip()}")
            return True
        
        logger.info(f"Submitting job: {self.slurm_script_path}")
        
        try:
            result = subprocess.run(['sbatch', str(self.slurm_script_path)], 
                                  capture_output=True, text=True, check=True)
            
            logger.info("Job submitted successfully!")
            logger.info(f"   {result.stdout.strip()}")
            logger.info("Monitor job progress with: squeue -u $USER")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit job: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
    
    def run_workflow(self, submit: bool = False, dry_run: bool = False, 
                    overrides: Optional[Dict[str, Any]] = None) -> bool:
        """
        Run the complete FSDP job workflow.
        
        Args:
            submit: Whether to submit the job after generation
            dry_run: If True, show what would be submitted without submitting
            overrides: Override SLURM settings
            
        Returns:
            True if successful
        """
        try:
            # Step 1: Estimate resources
            estimation_results = self.estimate_resources()
            
            # Step 2: Generate SLURM script
            script_path = self.generate_slurm_script(estimation_results, overrides)
            
            # Step 3: Submit job (if requested)
            if submit:
                return self.submit_job(dry_run)
            else:
                logger.info("Workflow complete. To submit job, run:")
                logger.info(f"   sbatch {script_path}")
                return True
                
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            return False


def main():
    """Main entry point for the FSDP job submission manager."""
    parser = argparse.ArgumentParser(
        description='FSDP Job Submission Manager - Automate resource estimation and job submission',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate SLURM script only
    python src/dendritic_modeling/scripts/tools/submit_fsdp_job.py configs/config_exp_fsdp.yaml --generate-only
    
    # Generate and submit job
    python src/dendritic_modeling/scripts/tools/submit_fsdp_job.py configs/config_exp_fsdp.yaml --submit
    
    # Dry run to see what would be submitted
    python src/dendritic_modeling/scripts/tools/submit_fsdp_job.py configs/config_exp_fsdp.yaml --submit --dry-run
    
    # Override SLURM settings
    python src/dendritic_modeling/scripts/tools/submit_fsdp_job.py configs/config_exp_fsdp.yaml --submit \\
        --partition kempner_h100 --account my_account --gpu-per-node 4
"""
    )
    
    parser.add_argument(
        'config',
        help='Path to FSDP configuration file'
    )
    
    # Workflow options
    workflow_group = parser.add_mutually_exclusive_group()
    workflow_group.add_argument(
        '--generate-only',
        action='store_true',
        help='Generate SLURM script without submitting'
    )
    workflow_group.add_argument(
        '--submit',
        action='store_true', 
        help='Generate and submit job'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be submitted without actually submitting (requires --submit)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for SLURM scripts (default: scripts/tools/slurm/)'
    )
    
    # SLURM overrides
    slurm_group = parser.add_argument_group('SLURM Overrides')
    slurm_group.add_argument(
        '--partition',
        type=str,
        help='Override SLURM partition'
    )
    slurm_group.add_argument(
        '--account',
        type=str,
        help='Override SLURM account'
    )
    slurm_group.add_argument(
        '--nodes',
        type=int,
        help='Override number of nodes'
    )
    slurm_group.add_argument(
        '--gpu-per-node',
        type=int,
        help='Override GPUs per node'
    )
    slurm_group.add_argument(
        '--time',
        type=str,
        help='Override job time limit (e.g., "02:00:00")'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.dry_run and not args.submit:
        parser.error("--dry-run requires --submit")
    
    if not args.generate_only and not args.submit:
        # Default behavior: generate only
        args.generate_only = True
    
    # Check if config file exists
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Collect SLURM overrides
    overrides = {}
    for override_key in ['partition', 'account', 'nodes', 'time']:
        value = getattr(args, override_key)
        if value is not None:
            overrides[override_key] = value
    
    # Handle gpu-per-node with underscore conversion
    if args.gpu_per_node is not None:
        overrides['gpu_per_node'] = args.gpu_per_node
    
    # Initialize manager and run workflow
    try:
        manager = FSDPJobManager(args.config, args.output_dir)
        
        success = manager.run_workflow(
            submit=args.submit,
            dry_run=args.dry_run,
            overrides=overrides if overrides else None
        )
        
        if not success:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Failed to run FSDP job workflow: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()