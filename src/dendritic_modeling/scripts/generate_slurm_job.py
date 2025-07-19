#!/usr/bin/env python3
"""
Generate SLURM job submission scripts from dendritic model config files.

This script analyzes a config file to determine resource requirements and creates
an appropriate SLURM submission script.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any
import math
from omegaconf import OmegaConf

# Add parent directory to path to import dendritic_modeling
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from dendritic_modeling.utils.gpu_estimation import estimate_from_config


# Cluster specifications (limits - what's available, not what to request)
CLUSTER_LIMITS = {
    'max_cpus_per_node': 120,
    'max_gpus_per_node': 4,
    'max_memory_per_gpu_gb': 300,  # Request up to 300GB per GPU
    'max_memory_per_node_gb': 1200,  # Increased to support 4 GPUs * 300GB
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
        self.config = OmegaConf.load(config_path)
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent / 'slurm'
        self.output_dir.mkdir(exist_ok=True)

        self.run_config = self.config.get('run_config', {})
        self.project_dir = self.run_config.get('project_dir', '/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling')  # Configurable
        self.log_dir = self.run_config.get('log_dir', self.run_config.get('dir', 'results/') + '/logs')  # Configurable, default to dir/logs
        self.data_path = self.config.task.get('data_path', '/n/netscratch/kempner_dev/hsafaai/data')  # From task config

        task_config = self.config.get('task', {})
        self.dataset = task_config.get('name', 'cifar10')

        train_config = self.config.get('train', {})
        trainer_config = train_config.main.trainer
        self.batch_size = trainer_config.get('batch_size', 32)
        self.epochs = trainer_config.get('epochs', 100)
        self.use_mixed_precision = trainer_config.get('use_mixed_precision', False)

        self.run_name = self.run_config.get('run_name', self.config_path.stem)

    def estimate_resources(self) -> Dict[str, Any]:
        """Estimate computational resources needed.
        
        Returns:
            Dictionary with estimated resource requirements
        """
        estimator, model = estimate_from_config(str(self.config_path))
        input_shape = self._get_input_shape()
        memory_breakdown = estimator.estimate_memory_breakdown(model, self.batch_size, input_shape, use_fsdp=True, num_gpus=1)
        gpu_recs = {rec.gpu_model: rec for rec in estimator.get_gpu_recommendations(memory_breakdown)}

        best_gpu_type, min_gpus_needed = self._select_best_gpu(gpu_recs)
        num_gpus = self._calculate_num_gpus(min_gpus_needed)
        gpus_per_node = self.run_config.get('gpu_per_node', min(CLUSTER_LIMITS['max_gpus_per_node'], num_gpus))
        num_nodes = max(1, math.ceil(num_gpus / gpus_per_node))
        ntasks_per_node = 1
        cpus_per_task = self._calculate_cpus_per_task(ntasks_per_node)
        time_str = self._estimate_time()
        mem_per_node = self._estimate_memory(gpus_per_node, best_gpu_type, memory_breakdown)

        return {
            'gpu_type': best_gpu_type,
            'num_nodes': num_nodes,
            'num_gpus': num_gpus,
            'gpus_per_node': gpus_per_node,
            'ntasks_per_node': ntasks_per_node,
            'cpus_per_task': cpus_per_task,
            'mem_per_node': mem_per_node,
            'time_str': time_str,
            'memory_breakdown': memory_breakdown,
        }

    def _get_input_shape(self):
        shapes = {'mnist': (784,), 'cifar10': (3072,), 'cifar100': (3072,), 'imagenet': (150528,)}
        return shapes.get(self.dataset, (784,))

    def _select_best_gpu(self, gpu_recs):
        priority = ['H100', 'H200', 'A100-80GB', 'A100-40GB']
        for gpu_type in priority:
            if gpu_type in gpu_recs and gpu_recs[gpu_type].fits:
                return gpu_type, gpu_recs[gpu_type].min_gpus_needed
        best = min(gpu_recs.values(), key=lambda x: x.min_gpus_needed)
        return best.gpu_model, best.min_gpus_needed

    def _calculate_num_gpus(self, min_gpus_needed):
        num_gpus = 1
        while num_gpus < min_gpus_needed:
            num_gpus *= 2
        return num_gpus

    def _calculate_cpus_per_task(self, ntasks_per_node):
        desired = CLUSTER_LIMITS['max_cpus_per_gpu']
        total_requested = desired * ntasks_per_node
        if total_requested > CLUSTER_LIMITS['max_cpus_per_node']:
            available = CLUSTER_LIMITS['max_cpus_per_node'] - 8
            desired = max(4, available // ntasks_per_node)
        return desired

    def _estimate_time(self):
        time_per_epoch = self._estimate_time_per_epoch()
        total_hours = math.ceil((self.epochs * time_per_epoch / 3600) * 1.5)
        return '72:00:00' if total_hours > 72 else f'{total_hours:02d}:00:00'

    def _estimate_time_per_epoch(self):
        base_times = {'cifar10': 60, 'cifar100': 60, 'imagenet': 1800, 'mnist': 30}
        base_time = base_times.get(self.dataset, 300)
        complexity_factor = self._get_complexity_factor()
        batch_factor = 128 / self.batch_size
        return base_time * complexity_factor * batch_factor

    def _get_complexity_factor(self):
        model_config = self.config.get('model', {})
        core = model_config.core_network.parameters
        total_neurons = sum(core.excitatory_layer_sizes) + sum(core.inhibitory_layer_sizes)
        branch_complexity = math.prod(core.excitatory_branch_factors)
        return max(1, (total_neurons / 1000) * (branch_complexity / 100))

    def _estimate_memory(self, gpus_per_node, gpu_type, memory_breakdown):
        gpu_memory_options = {'H100': 80, 'H200': 141, 'A100-80GB': 80, 'A100-40GB': 40}
        mem_per_gpu_request = 250 if gpus_per_node > 1 else int(gpu_memory_options[gpu_type] * 1.5)
        estimated_mem_gb = mem_per_gpu_request * gpus_per_node
        max_mem = min(estimated_mem_gb, CLUSTER_LIMITS['max_memory_per_node_gb'], CLUSTER_LIMITS['max_memory_per_gpu_gb'] * gpus_per_node)
        return f'{max_mem}GB'

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

    def _generate_conda_activation(self) -> str:
        """Generate conda activation commands.
            
        Returns:
            Conda activation commands as string
        """
        conda_env_path = self.run_config.get('conda_env_path')

        if conda_env_path:
            # Extract environment name from path if it contains 'envs'
            if '/envs/' in conda_env_path:
                # Extract from path like: /path/to/conda/envs/env_name/etc/profile.d/conda.sh
                env_name = conda_env_path.split('/envs/')[1].split('/')[0]
            else:
                # Use the provided default name
                env_name = 'dendritic_env' # Default name

            # Use the conda_env_path if it exists, otherwise use conda info
            return f"""# Initialize conda - try provided path first, fallback to conda info
if [ -f "{conda_env_path}" ]; then
    source {conda_env_path}
else
    source $(conda info --base)/etc/profile.d/conda.sh
fi
conda activate {env_name}"""
        else:
            return """# Initialize conda using conda info
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dendritic_env"""

    def generate_slurm_script(self) -> str:
        """Generate the SLURM script content.
        
        Returns:
            SLURM script as a string
        """
        resources = self.estimate_resources()
        use_fsdp = resources['num_nodes'] > 1 or resources['num_gpus'] > 1
        default_script = 'src/dendritic_modeling/scripts/train_experiments_fstp.py' if use_fsdp else 'src/dendritic_modeling/scripts/train_experiments.py'
        training_script = self.run_config.get('training_script', default_script)
        partition = self.run_config.get('partition', 'kempner_h100')
        account = self.run_config.get('account', 'kempner_dev')
        job_name = self.run_name
        output_dir = self.run_config.get('dir', 'results/')

        template = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={self.log_dir}/{job_name}_%j.out
#SBATCH --error={self.log_dir}/{job_name}_%j.err
#SBATCH --nodes={resources['num_nodes']}
#SBATCH --ntasks-per-node={resources['ntasks_per_node']}
#SBATCH --gres=gpu:{resources['gpus_per_node']}
#SBATCH --cpus-per-task={resources['cpus_per_task']}
#SBATCH --mem={resources['mem_per_node']}
#SBATCH --time={resources['time_str']}
#SBATCH --partition={partition}
#SBATCH --account={account}

{self._generate_module_loads()}

{self._generate_conda_activation()}

export MASTER_PORT=$(python -c "import random; print(random.randint(20000, 30000))")
export WORLD_SIZE=$((SLURM_NNODES * {resources['gpus_per_node']})) 
export RANK=$SLURM_PROCID
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

    export NCCL_DEBUG=INFO
    export NCCL_SOCKET_FAMILY=AF_INET

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd {self.project_dir}
mkdir -p {self.log_dir}

# Training command
'''
        if use_fsdp:
            template += f'''if [ $SLURM_NNODES -eq 1 ]; then
    torchrun --standalone --nproc_per_node={resources['gpus_per_node']} {training_script} {self.config_path}
else
    srun --ntasks-per-node=1 torchrun --nnodes=$SLURM_NNODES --nproc_per_node={resources['gpus_per_node']} --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT {training_script} {self.config_path}
fi'''
        else:
            template += f'python {training_script} {self.config_path}'
        return template

    def save_script(self, filename: str = None) -> Path:
        """Save the generated SLURM script to a file.
        
        Args:
            filename: Optional filename. If not provided, generates from config name.
            
        Returns:
            Path to the saved script
        """
        filename = filename or f'{self.run_name}.sbatch'
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            f.write(self.generate_slurm_script())
        output_path.chmod(0o755)
        return output_path

    def print_summary(self) -> None:
        """Print a summary of the job configuration and resources."""
        resources = self.estimate_resources()

        print("\n" + "="*60)
        print("SLURM Job Generation Summary")
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

        # Show run config overrides if present
        if self.run_config:
            print()
            print("Run Config (from YAML):")
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
        # The original code had a bug here, trying to access gpu_recommendation from resources
        # which is no longer returned by estimate_resources.
        # Assuming the intent was to print the best_gpu_type and min_gpus_needed from the estimator.
        # However, the original code had a custom GPURecommendation class, which is removed.
        # For now, we'll print the best_gpu_type and min_gpus_needed directly.
        print(f"  {resources['gpu_type']}: {resources['num_gpus']} GPUs minimum")
        print(f"  Recommended: {resources['num_gpus']} GPUs (power of 2)")
        print(f"  Memory per GPU: {resources['memory_breakdown'].total / resources['num_gpus']:.1f} GB")
        print(f"  Utilization: {(resources['memory_breakdown'].total / (resources['num_gpus'] * 80)) * 100:.1f}%") # Assuming 80GB per GPU for utilization calculation
        print("="*60)


def main():
    """Main entry point for the script."""
    import argparse
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
        '--conda-env',
        default='dendritic_env',
        help='Default conda environment name (conda_env_path in run_config specifies the full path)'
    )
    parser.add_argument('--simple-network', action='store_true',
                       help='Use simple network configuration (minimal NCCL vars)')

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

            # Automatically overwrite existing files without prompting
            if output_path.exists():
                print(f"File {output_path} already exists. Overwriting...")

            # Save the script
            saved_path = generator.save_script(args.filename)
            print(f"\nSLURM script saved to: {saved_path}")
            print("\nTo submit the job, run:")
            print(f"  sbatch {saved_path}")
        else:
            print("\nDry run - no script generated.")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main() 