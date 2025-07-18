#!/usr/bin/env python3
"""
Generate SLURM job submission scripts from dendritic model config files.

This script analyzes a config file to determine resource requirements and creates
an appropriate SLURM submission script.
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import math

# Add parent directory to path to import dendritic_modeling
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from dendritic_modeling.utils.gpu_estimation import GPUMemoryEstimator, estimate_from_config


# Cluster specifications (limits - what's available, not what to request)
CLUSTER_LIMITS = {
    'max_cpus_per_node': 120,
    'max_gpus_per_node': 4,
    'max_memory_per_gpu_gb': 350,
    'max_memory_per_node_gb': 512,
    'max_cpus_per_gpu': 25,  # Reasonable ratio
}


class SlurmJobGenerator:
    """Generate SLURM job scripts from config files."""
    
    def __init__(self, config_path: str, output_dir: str = None):
        """Initialize the generator.
        
        Args:
            config_path: Path to the configuration file
            output_dir: Directory to save the generated SLURM script
        """
        self.config_path = Path(config_path)
        # Load config YAML directly
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Determine output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(__file__).parent / 'slurm'
        self.output_dir.mkdir(exist_ok=True)
        
        # Extract common parameters
        self.run_config = self.config.get('run_config', {})
        
        # Get task info for estimation
        task_config = self.config.get('task', {})
        self.dataset = task_config.get('name', 'cifar10')
        
        # Get training info
        train_config = self.config.get('train', {})
        main_config = train_config.get('main', {})
        trainer_config = main_config.get('trainer', {})
        self.batch_size = trainer_config.get('batch_size', 32)
        self.epochs = trainer_config.get('epochs', 100)
        
        # Generate run name from config path if not specified
        if not self.run_config.get('run_name'):
            self.run_name = self.config_path.stem
        else:
            self.run_name = self.run_config.get('run_name')
        
        # Extract SLURM configuration if present
        # This is now handled by self.run_config above
        
        # Initialize GPU estimator and model (will be set in estimate_resources)
        self.estimator = None
        self.model = None 

    def estimate_resources(self) -> Dict[str, Any]:
        """Estimate computational resources needed.
        
        Returns:
            Dictionary with estimated resource requirements
        """
        # Get proper GPU memory estimation from estimator
        from dendritic_modeling.config import load_config
        config = load_config(str(self.config_path))
        
        # Build and analyze the model for precise estimation  
        estimator, model = estimate_from_config(str(self.config_path))
        
        # Get input shape for the model
        if self.dataset == 'mnist':
            input_shape = (784,)
        elif self.dataset in ['cifar10', 'cifar100']:
            input_shape = (3072,)  # Flattened
        elif self.dataset == 'imagenet':
            input_shape = (150528,)  # Flattened
        else:
            input_shape = (784,)  # Default
        
        # Get batch size from config
        trainer_params = config.train.main.trainer.__dict__ if hasattr(config.train.main, 'trainer') else {}
        batch_size = trainer_params.get('batch_size', self.batch_size)
        
        # Get memory breakdown and GPU recommendations
        memory_breakdown = estimator.estimate_memory_breakdown(
            model, batch_size, input_shape, use_fsdp=True, num_gpus=1
        )
        gpu_recs_list = estimator.get_gpu_recommendations(memory_breakdown)
        
        # Convert to dict for easier access
        gpu_recs = {rec.gpu_model: rec for rec in gpu_recs_list}
        
        print(f"Detailed Memory Analysis:")
        print(memory_breakdown)
        print(f"\nGPU Recommendations:")
        for gpu_type, rec in gpu_recs.items():
            print(f"  {gpu_type}: {rec.min_gpus_needed} GPUs (fits: {rec.fits})")
        
        # Determine GPU requirements
        gpu_memory_options = {
            'H100': 80,
            'H200': 141,
            'A100-80GB': 80,
            'A100-40GB': 40,
            'V100': 32,
        }
        
        # Always use estimation-based approach for GPU count
        # Find the best GPU option from recommendations
        best_gpu_rec = None
        for gpu_type in ['H100', 'H200', 'A100-80GB', 'A100-40GB']:
            if gpu_type in gpu_recs and gpu_recs[gpu_type].fits:
                best_gpu_rec = gpu_recs[gpu_type]
                break
        
        if not best_gpu_rec:
            # If no GPU fits, use the one that needs least GPUs
            best_gpu_rec = min(gpu_recs.values(), key=lambda x: x.min_gpus_needed)
        
        best_gpu_type = best_gpu_rec.gpu_model
        min_gpus_needed = best_gpu_rec.min_gpus_needed
        
        # Determine optimal number of GPUs (round up to power of 2 for efficiency)
        num_gpus = 1
        while num_gpus < min_gpus_needed:
            num_gpus *= 2
        
        # Calculate how many nodes we need and GPUs per node
        max_gpus_per_node = CLUSTER_LIMITS['max_gpus_per_node']
        num_nodes = max(1, (num_gpus + max_gpus_per_node - 1) // max_gpus_per_node)  # Ceiling division
        gpus_per_node = min(max_gpus_per_node, num_gpus)  # Don't exceed what we need
        ntasks_per_node = gpus_per_node  # One task per GPU
        
        # Calculate CPUs per task based on what we need (not max available)
        cpus_per_task = self._calculate_cpus_per_task(ntasks_per_node)
        
        # Estimate time based on dataset and epochs (automatic estimation)
        time_per_epoch = self._estimate_time_per_epoch()
        total_hours = (self.epochs * time_per_epoch) / 3600
        
        # Add buffer and round up
        total_hours = math.ceil(total_hours * 1.5)  # 50% buffer
        
        # Format time string
        if total_hours > 72:
            time_str = "72:00:00"  # Max typical allocation
        else:
            hours = int(total_hours)
            minutes = int((total_hours - hours) * 60)
            time_str = f"{hours:02d}:{minutes:02d}:00"
        
        # Determine memory per node (request what we need, within cluster limits)
        gpu_memory_gb = gpu_memory_options[best_gpu_type]
        estimated_mem_gb = int(gpu_memory_gb * gpus_per_node * 1.2)  # 20% buffer for actual GPUs used
        
        # Apply cluster limits (don't exceed what's available)
        max_mem_per_node = CLUSTER_LIMITS['max_memory_per_node_gb']
        max_mem_based_on_gpus = CLUSTER_LIMITS['max_memory_per_gpu_gb'] * gpus_per_node
        
        # Request what we need, but cap at cluster limits
        mem_gb = min(estimated_mem_gb, max_mem_per_node, max_mem_based_on_gpus)
        mem_per_node = f"{mem_gb}GB"
        
        # Create GPU recommendation for display
        # Use the proper GPU recommendation with updated values
        relevant_rec = gpu_recs.get(best_gpu_type)
        if relevant_rec:
            gpu_recommendation = type('GPURecommendation', (), {
                'gpu_model': best_gpu_type,
                'min_gpus_needed': relevant_rec.min_gpus_needed,
                'recommended_gpus': num_gpus,
                'memory_per_gpu_gb': memory_breakdown.total / num_gpus,
                'utilization_percent': (memory_breakdown.total / (num_gpus * gpu_memory_options[best_gpu_type])) * 100
            })()
        else:
            gpu_recommendation = type('GPURecommendation', (), {
                'gpu_model': best_gpu_type,
                'min_gpus_needed': min_gpus_needed,
                'recommended_gpus': num_gpus,
                'memory_per_gpu_gb': memory_breakdown.total / num_gpus,
                'utilization_percent': (memory_breakdown.total / (num_gpus * gpu_memory_options[best_gpu_type])) * 100
            })()
        
        return {
            'gpu_type': best_gpu_type,
            'num_nodes': num_nodes,
            'num_gpus': num_gpus,
            'gpus_per_node': min(gpus_per_node, num_gpus),
            'ntasks_per_node': ntasks_per_node,
            'cpus_per_task': cpus_per_task,
            'mem_per_node': mem_per_node,
            'time_str': time_str,
            'memory_breakdown': memory_breakdown,
            'gpu_recommendation': gpu_recommendation
        }
    
    def _estimate_time_per_epoch(self) -> float:
        """Estimate time per epoch in seconds.
        
        Returns:
            Estimated seconds per epoch
        """
        # Base estimates per dataset
        time_estimates = {
            'cifar10': 60,      # 1 minute base
            'cifar100': 60,
            'imagenet': 1800,   # 30 minutes base
            'tiny_imagenet': 300,
            'mnist': 30,
            'fashion_mnist': 30,
        }
        
        base_time = time_estimates.get(self.dataset, 300)  # Default 5 min
        
        # Scale by model complexity (simplified estimation)
        # For dendritic models, time scales with layer sizes and branch factors
        model_config = self.config.get('model', {})
        core_network = model_config.get('core_network', {})
        
        if core_network.get('type') == 'EINet':
            params = core_network.get('parameters', {})
            e_layers = params.get('excitatory_layer_sizes', [])
            i_layers = params.get('inhibitory_layer_sizes', [])
            e_branches = params.get('excitatory_branch_factors', [])
            
            # Rough complexity factor based on layers and branches
            total_neurons = sum(e_layers) + sum(i_layers)
            branch_complexity = 1
            for b in e_branches:
                branch_complexity *= b
            
            complexity_factor = (total_neurons / 1000) * (branch_complexity / 100)
            complexity_factor = max(1, complexity_factor)
        else:
            complexity_factor = 1
        
        # Scale by batch size (smaller batches = more iterations)
        batch_factor = 128 / self.batch_size  # Baseline 128
        
        return base_time * complexity_factor * batch_factor
    
    def _determine_partition(self, gpu_type: str) -> str:
        """Determine the SLURM partition based on GPU type.
        
        Args:
            gpu_type: Type of GPU (H100, A100, etc.)
            
        Returns:
            Partition name
        """
        # This mapping may need to be customized for your cluster
        partition_map = {
            'H100': 'kempner_h100',
            'H200': 'kempner_h100',  # Assuming H200s are in same partition
            'A100-80GB': 'kempner_a100',
            'A100-40GB': 'kempner_a100',
            'V100': 'gpu',
            'T4': 'gpu_test',
        }
        
        return partition_map.get(gpu_type, 'gpu')
    
    def _calculate_cpus_per_task(self, ntasks_per_node: int) -> int:
        """Calculate optimal CPUs per task (request what we need, not max available).
        
        Args:
            ntasks_per_node: Number of tasks (GPUs) per node
            
        Returns:
            Reasonable CPUs per task
        """
        max_cpus_per_gpu = CLUSTER_LIMITS['max_cpus_per_gpu']
        max_cpus_per_node = CLUSTER_LIMITS['max_cpus_per_node']
        
        # Use reasonable ratio: max 25 CPUs per GPU, but don't exceed node limits
        desired_cpus_per_task = max_cpus_per_gpu
        
        # If we're using multiple GPUs per node, make sure we don't exceed node limits
        total_requested = desired_cpus_per_task * ntasks_per_node
        if total_requested > max_cpus_per_node:
            # Scale down proportionally, leaving some headroom for system
            available_cpus = max_cpus_per_node - 8  # Reserve 8 CPUs for system
            desired_cpus_per_task = max(1, available_cpus // ntasks_per_node)
        
        return min(desired_cpus_per_task, max_cpus_per_gpu)
    
    def _generate_module_loads(self) -> str:
        """Generate module load commands from SLURM config.
        
        Returns:
            Module load commands as string
        """
        modules = self.run_config.get('modules_to_load', [])
        if not modules:
            return ""
        
        module_commands = []
        for module in modules:
            module_commands.append(f"module load {module}")
        
        return "\n".join(module_commands)
    
    def _generate_conda_activation(self, conda_env: str) -> str:
        """Generate conda activation commands.
        
        Args:
            conda_env: Conda environment name
            
        Returns:
            Conda activation commands as string
        """
        conda_env_path = self.run_config.get('conda_env_path')
        
        if conda_env_path:
            return f"""source {conda_env_path}
conda activate {conda_env}"""
        else:
            return f"""source ~/.bashrc
conda activate {conda_env}"""
    
    def generate_slurm_script(self, conda_env: str = 'dendritic_env') -> str:
        """Generate the SLURM script content.
        
        Args:
            conda_env: Default conda environment (can be overridden by slurm_config)
        
        Returns:
            SLURM script as a string
        """
        # Estimate resources
        resources = self.estimate_resources()
        
        # Get data path from config or use default
        data_path = self.config.get('task', {}).get('data_path', '/n/holylfs06/LABS/kempner_shared/Lab/data')
        
        # Determine if we need FSDP (multi-node or large model)
        use_fsdp = resources['num_nodes'] > 1 or resources['num_gpus'] > 1
        
        # Get SLURM parameters (system/person-dependent from config, estimated ones computed)
        job_name = self.run_config.get('run_name', self.run_name)
        output_dir = self.run_config.get('output_dir', '/n/holylabs/LABS/kempner_dev/Users/hsafaai/logs')
        account = self.run_config.get('account', 'kempner_dev')
        partition = self.run_config.get('partition', self._determine_partition(resources['gpu_type']))
        cpus_per_task = resources['cpus_per_task']  # Use estimated value
        
        # Get conda environment from SLURM config (overrides command line argument)
        conda_env_name = self.run_config.get('conda_env', conda_env)
        
        # Build the script
        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_dir}/{job_name}_%j.out
#SBATCH --error={output_dir}/{job_name}_%j.err
#SBATCH --nodes={resources['num_nodes']}
#SBATCH --ntasks-per-node={resources['ntasks_per_node']}
#SBATCH --gres=gpu:{resources['gpus_per_node']}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={resources['mem_per_node']}
#SBATCH --time={resources['time_str']}
#SBATCH --partition={partition}
#SBATCH --account={account}

# Load modules if specified
{self._generate_module_loads()}

# Activate environment
{self._generate_conda_activation(conda_env_name)}

# Set environment variables for distributed training
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export RANK=$SLURM_PROCID

# Set master address
master_node=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR=$master_node

# NCCL settings for {resources['gpu_type']}
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=2
export NCCL_P2P_LEVEL=NVL
export NCCL_TREE_THRESHOLD=0

# PyTorch settings
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Print environment info
echo "Master node: $master_node"
echo "Master port: $MASTER_PORT"
echo "World size: $WORLD_SIZE"
echo "Number of nodes: $SLURM_NNODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Current node: $SLURMD_NODENAME"
echo "Current rank: $RANK"

# Change to project directory
cd /n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/dendritic-modeling

# Create log directory if it doesn't exist
mkdir -p /n/holylabs/LABS/kempner_dev/Users/hsafaai/logs

# Pre-download dataset on rank 0 only
if [ $RANK -eq 0 ]; then
    echo "Rank 0: Pre-downloading {self.dataset} dataset..."
    python -c "
import torchvision
import torch
data_path = '{data_path}/{self.dataset}'
"""
        
        # Add dataset-specific download code
        if self.dataset in ['cifar10', 'cifar100']:
            dataset_class = f"torchvision.datasets.{self.dataset.upper()}"
            script += f"""# Download both train and test sets
_ = {dataset_class}(root=data_path, train=True, download=True)
_ = {dataset_class}(root=data_path, train=False, download=True)
print('Dataset download complete.')
"
fi
"""
        elif self.dataset == 'imagenet':
            script += """# ImageNet should already be downloaded
print('Using pre-downloaded ImageNet dataset.')
"
fi
"""
        else:
            script += f"""print('Dataset handling for {self.dataset} - ensure it is available.')
"
fi
"""
        
        script += """
# Synchronization barrier using a file-based approach
SYNC_FILE="/n/holylabs/LABS/kempner_dev/Users/hsafaai/logs/dataset_ready_${SLURM_JOBID}.flag"
if [ $RANK -eq 0 ]; then
    touch $SYNC_FILE
else
    # Wait for the sync file to appear (max 5 minutes)
    COUNTER=0
    while [ ! -f $SYNC_FILE ] && [ $COUNTER -lt 300 ]; do
        sleep 1
        COUNTER=$((COUNTER + 1))
    done
    if [ $COUNTER -eq 300 ]; then
        echo "Timeout waiting for dataset download"
        exit 1
    fi
fi

# Small delay to ensure all ranks have seen the sync file
sleep 5
"""
        
        # Determine training script path
        if use_fsdp:
            default_script = "scripts/train_large_model_fsdp.py"
        else:
            default_script = "scripts/train_experiments.py"
        
        training_script = self.run_config.get('training_script', default_script)
        
        # Add the training command
        if use_fsdp:
            script += f"""
# Run the training script with torchrun for FSDP
srun --ntasks-per-node=$SLURM_NTASKS_PER_NODE \\
    torchrun \\
    --nproc_per_node={resources['gpus_per_node']} \\
    --nnodes=$SLURM_NNODES \\
    --node_rank=$SLURM_NODEID \\
    --master_addr=$MASTER_ADDR \\
    --master_port=$MASTER_PORT \\
    {training_script} \\
    --config {self.config_path} \\
    --distributed
"""
        else:
            # Single GPU training
            script += f"""
# Run the training script (single GPU)
python {training_script} \\
    --config {self.config_path}
"""
        
        script += f"""
# Cleanup sync file on rank 0
if [ $RANK -eq 0 ]; then
    rm -f $SYNC_FILE
fi

# Print job summary
echo "Job completed at $(date)"
echo "Resource usage summary:"
echo "  Config: {self.config_path}"
echo "  GPU type: {resources['gpu_type']}"
echo "  Total GPUs: {resources['num_gpus']}"
echo "  Memory per GPU: {resources['gpu_recommendation'].memory_per_gpu_gb:.1f} GB"
"""
        
        return script
    
    def save_script(self, filename: str = None, conda_env: str = 'dendritic_env') -> Path:
        """Save the generated SLURM script to a file.
        
        Args:
            filename: Optional filename. If not provided, generates from config name.
            
        Returns:
            Path to the saved script
        """
        if filename is None:
            filename = f"{self.run_name}.sbatch"
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate the script
        script_content = self.generate_slurm_script(conda_env=conda_env)
        
        # Save to file
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        output_path.chmod(0o755)
        
        return output_path
    
    def print_summary(self) -> None:
        """Print a summary of the job configuration and resources."""
        resources = self.estimate_resources()
        
        print("\n" + "="*60)
        print(f"SLURM Job Generation Summary")
        print("="*60)
        print(f"Config file: {self.config_path}")
        print(f"Job name: {self.run_name}")
        print(f"Dataset: {self.dataset}")
        print(f"Batch size: {self.batch_size} per GPU")
        print(f"Epochs: {self.epochs}")
        
        # Show cluster limits
        print()
        print("Cluster Limits:")
        print(f"  Max CPUs per node: {CLUSTER_LIMITS['max_cpus_per_node']}")
        print(f"  Max GPUs per node: {CLUSTER_LIMITS['max_gpus_per_node']}")
        print(f"  Max CPUs per GPU: {CLUSTER_LIMITS['max_cpus_per_gpu']}")
        print(f"  Max memory per GPU: {CLUSTER_LIMITS['max_memory_per_gpu_gb']} GB")
        print(f"  Max memory per node: {CLUSTER_LIMITS['max_memory_per_node_gb']} GB")
        
        # Show SLURM config overrides if present
        if self.run_config:
            print()
            print("SLURM Config (from YAML):")
            for key, value in self.run_config.items():
                if key == 'modules_to_load' and isinstance(value, list):
                    print(f"  {key}:")
                    for module in value:
                        print(f"    - {module}")
                else:
                    print(f"  {key}: {value}")
        print()
        print("Resource Requirements:")
        print(f"  GPU type: {resources['gpu_type']}")
        print(f"  Number of nodes: {resources['num_nodes']}")
        print(f"  Total GPUs: {resources['num_gpus']}")
        print(f"  GPUs per node: {resources['gpus_per_node']}")
        print(f"  CPUs per task: {resources['cpus_per_task']}")
        print(f"  Memory per node: {resources['mem_per_node']}")
        print(f"  Estimated runtime: {resources['time_str']}")
        print(f"  Partition: {self._determine_partition(resources['gpu_type'])}")
        print()
        print("Memory Breakdown:")
        print(resources['memory_breakdown'])
        print()
        print("GPU Recommendation:")
        gpu_rec = resources['gpu_recommendation']
        print(f"  {gpu_rec.gpu_model}: {gpu_rec.min_gpus_needed} GPUs minimum")
        print(f"  Recommended: {gpu_rec.recommended_gpus} GPUs (power of 2)")
        print(f"  Memory per GPU: {gpu_rec.memory_per_gpu_gb:.1f} GB")
        print(f"  Utilization: {gpu_rec.utilization_percent:.1f}%")
        print("="*60)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Generate SLURM job scripts from dendritic model config files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate SLURM script for a config file
  python generate_slurm_job.py configs/my_experiment.yaml
  
  # Generate with custom output directory
  python generate_slurm_job.py configs/my_experiment.yaml --output-dir ./my_jobs/
  
  # Generate with custom filename
  python generate_slurm_job.py configs/my_experiment.yaml --filename custom_job.sbatch
  
  # Just show summary without saving
  python generate_slurm_job.py configs/my_experiment.yaml --dry-run
"""
    )
    
    parser.add_argument(
        'config',
        type=str,
        help='Path to the configuration file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save the SLURM script (default: scripts/slurm/)'
    )
    parser.add_argument(
        '--filename',
        type=str,
        default=None,
        help='Filename for the SLURM script (default: <run_name>.sbatch)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Just print summary without generating the script'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing SLURM script if it exists'
    )
    parser.add_argument(
        '--conda-env', 
        default='dendritic_env',
        help='Default conda environment name (can be overridden by run_config.conda_env in YAML)'
    )
    
    args = parser.parse_args()
    
    try:
        # Create the generator
        generator = SlurmJobGenerator(args.config, args.output_dir)
        
        # Print summary
        generator.print_summary()
        
        if not args.dry_run:
            # Check if file exists
            if args.filename:
                output_path = generator.output_dir / args.filename
            else:
                output_path = generator.output_dir / f"{generator.run_name}.sbatch"
            
            if output_path.exists() and not args.force:
                response = input(f"\nFile {output_path} already exists. Overwrite? [y/N]: ")
                if response.lower() != 'y':
                    print("Aborted.")
                    return
            
            # Save the script
            saved_path = generator.save_script(args.filename, conda_env=args.conda_env)
            print(f"\nSLURM script saved to: {saved_path}")
            print(f"\nTo submit the job, run:")
            print(f"  sbatch {saved_path}")
        else:
            print("\nDry run - no script generated.")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main() 