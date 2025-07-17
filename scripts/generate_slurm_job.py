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

from dendritic_modeling.utils.gpu_estimation import GPUMemoryEstimator
from dendritic_modeling.config.config import load_config


class SlurmJobGenerator:
    """Generate SLURM job scripts from config files."""
    
    def __init__(self, config_path: str, output_dir: str = None):
        """Initialize the generator.
        
        Args:
            config_path: Path to the configuration file
            output_dir: Directory to save the generated SLURM script
        """
        self.config_path = Path(config_path)
        self.config = load_config(config_path)
        self.output_dir = Path(output_dir) if output_dir else self.config_path.parent.parent / "scripts" / "slurm"
        
        # Extract key configuration parameters
        self.dataset = self.config.get('task', {}).get('dataset', 'unknown')
        self.batch_size = self.config.get('train', {}).get('main', {}).get('trainer', {}).get('batch_size', 32)
        self.epochs = self.config.get('train', {}).get('main', {}).get('trainer', {}).get('epochs', 100)
        self.run_name = self.config.get('outputs', {}).get('run_name', self.config_path.stem)
        
        # Initialize GPU estimator
        self.estimator = GPUMemoryEstimator(self.config) 

    def estimate_resources(self) -> Dict[str, Any]:
        """Estimate computational resources needed.
        
        Returns:
            Dictionary with resource requirements
        """
        # Get memory estimation
        memory_breakdown = self.estimator.estimate_memory(
            batch_size=self.batch_size,
            distributed=True
        )
        
        # Get GPU recommendations for different GPU types
        gpu_recs = self.estimator.get_gpu_recommendations(
            memory_breakdown=memory_breakdown,
            distributed=True
        )
        
        # Find the best GPU option (prefer H100)
        best_gpu = None
        for gpu_type in ['H100', 'H200', 'A100-80GB', 'A100-40GB']:
            if gpu_type in gpu_recs and gpu_recs[gpu_type].fits:
                best_gpu = gpu_recs[gpu_type]
                break
        
        if not best_gpu:
            # If no GPU fits, use the one that needs least GPUs
            best_gpu = min(gpu_recs.values(), key=lambda x: x.min_gpus_needed)
        
        # Determine nodes and tasks
        gpus_per_node = 4  # Standard for most clusters
        num_gpus = best_gpu.recommended_gpus
        num_nodes = max(1, num_gpus // gpus_per_node)
        
        # Estimate time based on dataset and epochs
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
        
        # Determine memory per node
        mem_per_node = int(best_gpu.memory_per_gpu_gb * gpus_per_node * 1.2)  # 20% buffer
        mem_per_node = min(mem_per_node, 512)  # Cap at typical max
        
        return {
            'gpu_type': best_gpu.gpu_model,
            'num_nodes': num_nodes,
            'num_gpus': num_gpus,
            'gpus_per_node': min(gpus_per_node, num_gpus),
            'mem_per_node': mem_per_node,
            'time_str': time_str,
            'memory_breakdown': memory_breakdown,
            'gpu_recommendation': best_gpu
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
        
        # Scale by model size
        param_count = self.estimator.count_parameters()
        active_params = param_count.get('active_total', 0)
        
        # Rough scaling: double time for every 10x parameters
        size_factor = math.log10(max(active_params / 1e6, 1)) + 1
        
        # Scale by batch size (smaller batches = more iterations)
        batch_factor = 128 / self.batch_size  # Baseline 128
        
        return base_time * size_factor * batch_factor
    
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
    
    def generate_slurm_script(self) -> str:
        """Generate the SLURM script content.
        
        Returns:
            SLURM script as a string
        """
        # Estimate resources
        resources = self.estimate_resources()
        
        # Get data path from config or use default
        data_path = self.config.get('task', {}).get('data_path', '/n/holylfs06/LABS/kempner_shared/Lab/data')
        
        # Determine if we need FSDP (multi-node or large model)
        use_fsdp = resources['num_nodes'] > 1 or resources['num_gpus'] > 1
        
        # Build the script
        script = f"""#!/bin/bash
#SBATCH --job-name={self.run_name}
#SBATCH --output=/n/holylabs/LABS/kempner_dev/Users/hsafaai/logs/{self.run_name}_%j.out
#SBATCH --error=/n/holylabs/LABS/kempner_dev/Users/hsafaai/logs/{self.run_name}_%j.err
#SBATCH --nodes={resources['num_nodes']}
#SBATCH --ntasks-per-node={resources['gpus_per_node']}
#SBATCH --gres=gpu:{resources['gpus_per_node']}
#SBATCH --cpus-per-task=16
#SBATCH --mem={resources['mem_per_node']}GB
#SBATCH --time={resources['time_str']}
#SBATCH --partition={self._determine_partition(resources['gpu_type'])}
#SBATCH --account=kempner_dev

# Activate environment
source ~/.bashrc
conda activate dendritic

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
    scripts/train_large_model_fsdp.py \\
    --config {self.config_path} \\
    --distributed
"""
        else:
            # Single GPU training
            script += f"""
# Run the training script (single GPU)
python scripts/train_experiments.py \\
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
    
    def save_script(self, filename: str = None) -> Path:
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
        script_content = self.generate_slurm_script()
        
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
        print()
        print("Resource Requirements:")
        print(f"  GPU type: {resources['gpu_type']}")
        print(f"  Number of nodes: {resources['num_nodes']}")
        print(f"  Total GPUs: {resources['num_gpus']}")
        print(f"  GPUs per node: {resources['gpus_per_node']}")
        print(f"  Memory per node: {resources['mem_per_node']} GB")
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
            saved_path = generator.save_script(args.filename)
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