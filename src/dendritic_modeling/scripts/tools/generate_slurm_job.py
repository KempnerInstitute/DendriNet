#!/usr/bin/env python3
"""
Generate SLURM job submission scripts from dendritic model config files.

This script analyzes a config file to determine resource requirements and creates
an appropriate SLURM submission script.
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf

# Add parent directory to path to import dendritic_modeling
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))  # to reach root
# But since in package, imports should work relatively
from dendritic_modeling.utils.gpu_estimation import estimate_from_config

# Cluster specifications (limits - what's available, not what to request)
CLUSTER_LIMITS = {
    "max_cpus_per_node": 120,
    "max_gpus_per_node": 4,
    "max_memory_per_gpu_gb": 300,  # Request up to 300GB per GPU
    "max_memory_per_node_gb": 1200,  # Support 4 GPUs * 300GB
    "max_cpus_per_gpu": 25,  # Reasonable ratio
}


class SlurmJobGenerator:
    """Generate SLURM job scripts from config files."""

    def __init__(self, config_path: str, output_dir: Optional[str] = None):
        """Initialize the generator.

        Args:
            config_path: Path to the configuration file
            output_dir: Directory to save the generated SLURM script
        """
        self.config_path = Path(config_path)
        self.config = OmegaConf.load(config_path)

        # Set output directory - prefer results directory from config, fallback to project root
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Get results directory from config
            results_dir = self.config.get("outputs", {}).get("dir", "results")
            self.output_dir = Path(results_dir) / "slurm_jobs"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check for both slurm_config and run_config for backwards compatibility
        self.run_config = self.config.get(
            "slurm_config", self.config.get("run_config", {})
        )

        # Use current working directory as default project directory (more portable)
        default_project_dir = os.getcwd()
        self.project_dir = self.run_config.get(
            "project_dir", default_project_dir
        )  # Configurable
        self.log_dir = self.run_config.get(
            "log_dir", self.run_config.get("dir", "results/") + "/logs"
        )  # Configurable, default to dir/logs

        # Use relative data path by default for better portability
        default_data_path = os.path.join(default_project_dir, "data")
        self.data_path = self.config.task.get(
            "data_path", default_data_path
        )  # From task config

        task_config = self.config.get("task", {})
        self.dataset = task_config.get("name", "cifar10")

        train_config = self.config.get("train", {})
        trainer_config = train_config.main.trainer
        self.batch_size = trainer_config.get("batch_size", 32)
        self.epochs = trainer_config.get("epochs", 100)
        self.use_mixed_precision = trainer_config.get("use_mixed_precision", False)

        # Get run_name from outputs section or slurm_config
        outputs_config = self.config.get("outputs", {})
        self.run_name = outputs_config.get(
            "run_name", self.run_config.get("run_name", self.config_path.stem)
        )

    def estimate_resources(self) -> dict[str, Any]:
        """Estimate computational resources needed.

        Returns:
            Dictionary with estimated resource requirements
        """
        estimator, model = estimate_from_config(str(self.config_path))
        input_shape = self._get_input_shape()
        memory_breakdown = estimator.estimate_memory_breakdown(
            model, self.batch_size, input_shape, use_fsdp=True, num_gpus=1
        )

        # GPU calculation based on memory requirements
        total_memory_gb = memory_breakdown.total * 2.0  # Memory safety factor
        max_memory_per_gpu = 70  # Safe limit for H100

        # Calculate minimum GPUs needed
        min_gpus = math.ceil(total_memory_gb / max_memory_per_gpu)

        # Get gpus_per_node from config
        gpus_per_node = self.run_config.get(
            "gpu_per_node", CLUSTER_LIMITS["max_gpus_per_node"]
        )
        gpus_per_node = min(CLUSTER_LIMITS["max_gpus_per_node"], gpus_per_node)

        # Force more GPUs if memory is too high
        if total_memory_gb > 80:  # Model is large
            gpus_per_node = min(4, gpus_per_node)  # Use up to 4 GPUs per node
            min_gpus = max(min_gpus, 2)  # Force at least 2 GPUs

        # Calculate nodes and total GPUs
        num_nodes = math.ceil(min_gpus / gpus_per_node)
        num_gpus = num_nodes * gpus_per_node

        # Default settings
        best_gpu_type = "H100"  # Force H100 for large models
        ntasks_per_node = 1  # For FSDP
        mem_per_node = "300GB"  # User's max per GPU
        cpus_per_task = self._calculate_cpus_per_task(gpus_per_node)
        time_str = self._estimate_time()

        return {
            "gpu_type": best_gpu_type,
            "num_nodes": num_nodes,
            "num_gpus": num_gpus,
            "gpus_per_node": gpus_per_node,
            "ntasks_per_node": ntasks_per_node,
            "cpus_per_task": cpus_per_task,
            "mem_per_node": mem_per_node,
            "time_str": time_str,
            "memory_breakdown": memory_breakdown,
        }

    def _get_input_shape(self):
        shapes = {
            "mnist": (784,),
            "cifar10": (3072,),
            "cifar100": (3072,),
            "imagenet": (150528,),
        }
        return shapes.get(self.dataset, (784,))

    def _select_best_gpu(self, gpu_recs):
        # Select the GPU type that requires the fewest GPUs (most efficient)
        best = min(gpu_recs.values(), key=lambda x: x.min_gpus_needed)
        return best.gpu_model, best.min_gpus_needed

    def _calculate_num_gpus(self, min_gpus_needed):
        num_gpus = 1
        while num_gpus < min_gpus_needed:
            num_gpus *= 2
        return num_gpus

    def _calculate_cpus_per_task(self, gpus_per_node):
        """Calculate CPUs per task based on GPU count per node.

        For FSDP, we want enough CPUs to support all GPUs on the node.
        """
        desired_cpus_per_gpu = CLUSTER_LIMITS["max_cpus_per_gpu"]
        total_desired = desired_cpus_per_gpu * gpus_per_node

        # Ensure we don't exceed node limits, leaving some margin for system processes
        available = CLUSTER_LIMITS["max_cpus_per_node"] - 8
        if total_desired > available:
            cpus_per_task = max(4, available)
        else:
            cpus_per_task = total_desired

        return cpus_per_task

    def _estimate_time(self):
        time_per_epoch = self._estimate_time_per_epoch()
        total_hours = math.ceil((self.epochs * time_per_epoch / 3600) * 1.5)
        return "72:00:00" if total_hours > 72 else f"{total_hours:02d}:00:00"

    def _estimate_time_per_epoch(self):
        base_times = {"cifar10": 60, "cifar100": 60, "imagenet": 1800, "mnist": 30}
        base_time = base_times.get(self.dataset, 300)
        complexity_factor = self._get_complexity_factor()
        batch_factor = 128 / self.batch_size
        return base_time * complexity_factor * batch_factor

    def _get_complexity_factor(self):
        model_config = self.config.get("model", {})
        core = model_config.core_network.parameters
        total_neurons = sum(core.excitatory_layer_sizes) + sum(
            core.inhibitory_layer_sizes
        )
        branch_complexity = math.prod(core.excitatory_branch_factors)
        return max(1, (total_neurons / 1000) * (branch_complexity / 100))

    def _estimate_memory(self, gpus_per_node, gpu_type, memory_breakdown):
        gpu_memory_options = {"A100-40GB": 40, "H100": 80}

        gpu_memory_gb = gpu_memory_options.get(gpu_type, 40)

        # Use actual memory breakdown with safety factor
        # For FSDP, memory is distributed across GPUs
        if gpus_per_node > 1:
            # Multi-GPU: memory is sharded across GPUs, add some overhead for communication
            actual_memory_per_gpu = (
                memory_breakdown.total / gpus_per_node
            ) * 1.2  # 20% overhead
        else:
            # Single GPU: need full model memory
            actual_memory_per_gpu = memory_breakdown.total * 1.1  # 10% overhead

        # Ensure we don't exceed GPU capacity, but be more aggressive than before
        safety_factor = 0.95  # Use 95% of GPU memory
        max_gpu_memory = gpu_memory_gb * safety_factor

        # Use the higher of: actual calculated memory or minimum conservative amount
        conservative_min = gpu_memory_gb * 0.6  # At least 60% of GPU memory
        mem_per_gpu_request = max(actual_memory_per_gpu, conservative_min)

        # If memory per GPU is too high, force more GPUs
        if mem_per_gpu_request > gpu_memory_gb * 0.85:  # >85% utilization
            # Force the system to use more GPUs by making memory request very high
            mem_per_gpu_request = gpu_memory_gb * 0.95

        # Cap at GPU capacity
        mem_per_gpu_request = min(mem_per_gpu_request, max_gpu_memory)

        estimated_mem_gb = mem_per_gpu_request * gpus_per_node
        max_mem = min(estimated_mem_gb, CLUSTER_LIMITS["max_memory_per_node_gb"])
        max_mem_int = round(max_mem)
        return f"{max_mem_int}GB"

    def _determine_partition(self, gpu_type: str) -> str:
        """Determine the SLURM partition based on GPU type.

        The partition must be specified in run_config. Falls back to
        gpu_type-based lookup only if an explicit mapping is provided.
        """
        partition = self.run_config.get("partition")
        if partition:
            return partition
        raise ValueError("SLURM partition not set. Provide 'partition' in run_config.")

    def _generate_module_loads(self) -> str:
        """Generate module load commands from SLURM config.

        Returns:
            Module load commands as string
        """
        modules = self.run_config.get("modules_to_load", [])
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
        conda_env_path = self.run_config.get("conda_env_path")

        if conda_env_path:
            # Extract environment name from path if it contains 'envs'
            if "/envs/" in conda_env_path:
                # Extract from path like: /path/to/conda/envs/env_name/etc/profile.d/conda.sh
                env_name = conda_env_path.split("/envs/")[1].split("/")[0]
            else:
                # Use the provided default name
                env_name = "dendritic_env"  # Default name

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
        # Check if FSDP is enabled in config or if multi-GPU is needed
        config_uses_fsdp = self.config.get("train", {}).get("use_fsdp", False)
        use_fsdp = (
            config_uses_fsdp or resources["num_nodes"] > 1 or resources["num_gpus"] > 1
        )
        default_script = (
            "src/dendritic_modeling/scripts/training/train_experiments_fsdp.py"
            if use_fsdp
            else "src/dendritic_modeling/scripts/training/train_experiments.py"
        )
        training_script = self.run_config.get("training_script", default_script)
        partition = self._determine_partition(
            resources["gpu_type"]
        )  # Use estimated partition
        account = self.run_config.get("account")
        if not account:
            raise ValueError("SLURM account not set. Provide 'account' in run_config.")
        job_name = self.run_name
        self.run_config.get("dir", "results/")

        template = f"""#!/bin/bash
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

# Set MASTER_ADDR to first IPv4 from hostname -I (all local IPs)
export MASTER_ADDR=$(hostname -I | tr ' ' '\\n' | grep -E '^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+$' | head -n1)
echo "Primary IPv4 from hostname -I: $MASTER_ADDR"

# Fallback if empty: use scontrol for NodeAddr and resolve to first IPv4
if [ -z "$MASTER_ADDR" ]; then
    NODE_NAME=$(hostname -s)
    NODE_ADDR=$(scontrol show node $NODE_NAME | grep NodeAddr | cut -d= -f2)
    MASTER_ADDR=$(getent hosts $NODE_ADDR | tr ' ' '\\n' | grep -E '^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+$' | head -n1)
    echo "Fallback IPv4 from scontrol/getent: $MASTER_ADDR"
fi

# Final fallback: use Python to get IP from hostname resolution
if [ -z "$MASTER_ADDR" ]; then
    MASTER_ADDR=$(python -c "import socket; print(socket.gethostbyname(socket.gethostname()))" 2>/dev/null)
    echo "Fallback IPv4 from Python socket: $MASTER_ADDR"
fi

# Fail if still empty
if [ -z "$MASTER_ADDR" ]; then
    echo "ERROR: Could not resolve MASTER_ADDR" >&2
    exit 1
fi

export MASTER_PORT=$(python -c "import random; print(random.randint(29500, 29999))")
export WORLD_SIZE=$((SLURM_NNODES * {resources['gpus_per_node']}))

echo "=== Network Configuration Debug ==="
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "World size: $WORLD_SIZE"

# Force IPv4 for sockets and NCCL to avoid IPv6 issues
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IGNORE_DISABLED_P2P=1  # Suppress harmless IPv6 warnings

# Check if IPv6 is disabled
IPV6_DISABLED=$(sysctl -n net.ipv6.conf.all.disable_ipv6 2>/dev/null || echo "0")
if [ "$IPV6_DISABLED" = "1" ]; then
    echo "IPv6 is disabled on this system"
else
    echo "IPv6 is enabled - forcing IPv4 for PyTorch distributed"
fi

export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# PyTorch memory management - avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set CUDA device visibility based on GPU allocation
if [ {resources['gpus_per_node']} -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=0
else
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($(echo $SLURM_GPUS_PER_NODE | cut -d'(' -f2 | cut -d')' -f1) - 1)))
fi

export CUDA_LAUNCH_BLOCKING=1

# Detect Infiniband interface dynamically
IB_IFNAME=$(ip -o link show | grep -i 'link/infiniband' | cut -d: -f1 | head -n1 | tr -d ' ')
if [ -n "$IB_IFNAME" ]; then
    export NCCL_SOCKET_IFNAME=$IB_IFNAME
    export GLOO_SOCKET_IFNAME=$IB_IFNAME
    export NCCL_IB_DISABLE=0
    echo "Using Infiniband interface: $IB_IFNAME"
else
    ETH_IFNAME=$(ip -o link show | grep -i 'link/ether' | cut -d: -f1 | head -n1 | tr -d ' ')
    export NCCL_SOCKET_IFNAME=$ETH_IFNAME
    export GLOO_SOCKET_IFNAME=$ETH_IFNAME
    export NCCL_IB_DISABLE=1
    echo "Using Ethernet interface: $ETH_IFNAME"
fi

echo "Available network interfaces:"
ip addr show | grep "^[0-9]" | cut -d: -f2 | tr -d ' '

# Optional: Enable additional NCCL debugging if environment variable is set
if [ "$TORCH_DISTRIBUTED_DEBUG" = "DETAIL" ]; then
    export NCCL_DEBUG=TRACE
    export TORCH_DISTRIBUTED_DEBUG=DETAIL
    echo "Enhanced debugging enabled"
fi

export TORCHELASTIC_RENDEZVOUS_TIMEOUT=3600
export TORCHELASTIC_RDV_BACKEND=c10d
export TORCHELASTIC_STORE_TIMEOUT=3600

cd {self.project_dir}
mkdir -p {self.log_dir}

"""
        if use_fsdp:
            if resources["num_nodes"] == 1 and resources["gpus_per_node"] == 1:
                template += f"""# Single GPU training - no distributed setup needed
echo "Starting single GPU training..."
echo "CUDA device: $CUDA_VISIBLE_DEVICES"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"

python {training_script} {self.config_path}"""
            else:
                template += f"""# Multi-GPU training
if [ $SLURM_NNODES -eq 1 ]; then
    echo "Starting single-node multi-GPU training..."
    torchrun --standalone --nproc_per_node={resources['gpus_per_node']} {training_script} {self.config_path}
else
    echo "Starting multi-node FSDP training on $SLURM_NNODES nodes..."
    srun --ntasks-per-node=1 --exclusive bash -c '
        if [ $SLURM_PROCID -ne 0 ]; then
            echo "Node $SLURM_PROCID: Testing connectivity to master..."
            if ping -c 1 -W 5 $MASTER_ADDR >/dev/null 2>&1; then
                echo "OK: Can reach master at $MASTER_ADDR"
            else
                echo "WARNING: Cannot reach master at $MASTER_ADDR - may fail distributed init"
            fi
        fi
        sleep $(($SLURM_PROCID * 10))
        echo "Node $SLURM_PROCID ready. CUDA devices: $CUDA_VISIBLE_DEVICES"
        echo "Available GPUs: $(nvidia-smi -L | wc -l)"

        torchrun \\
            --nnodes=$SLURM_NNODES \\
            --nproc_per_node={resources['gpus_per_node']} \\
            --node_rank=$SLURM_PROCID \\
            --master_addr=$MASTER_ADDR \\
            --master_port=$MASTER_PORT \\
            --rdzv-backend=c10d \\
            --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \\
            {training_script} {self.config_path}
    '
fi"""
        else:
            template += f"python {training_script} {self.config_path}"
        return template

    def save_script(self, filename: Optional[str] = None) -> Path:
        """Save the generated SLURM script to a file.

        Args:
            filename: Optional filename. If not provided, generates from config name.

        Returns:
            Path to the saved script
        """
        filename = filename or f"{self.run_name}.sbatch"
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            f.write(self.generate_slurm_script())
        output_path.chmod(0o755)
        return output_path

    def print_summary(self) -> None:
        """Print a summary of the job configuration and resources."""
        resources = self.estimate_resources()

        print("\n" + "=" * 60)
        print("SLURM Job Generation Summary")
        print("=" * 60)
        print(f"Config file: {self.config_path}")
        print(f"Job name: {self.run_name}")
        print(f"Dataset: {self.dataset}")
        print(f"Batch size: {self.batch_size} per GPU")
        print(f"Epochs: {self.epochs}")

        # Show cluster limits
        print()
        print("Cluster Limits:")
        print(f" Max CPUs per node: {CLUSTER_LIMITS['max_cpus_per_node']}")
        print(f" Max GPUs per node: {CLUSTER_LIMITS['max_gpus_per_node']}")
        print(f" Max CPUs per GPU: {CLUSTER_LIMITS['max_cpus_per_gpu']}")
        print(f" Max memory per GPU: {CLUSTER_LIMITS['max_memory_per_gpu_gb']} GB")
        print(f" Max memory per node: {CLUSTER_LIMITS['max_memory_per_node_gb']} GB")

        # Show run config overrides if present
        if self.run_config:
            print()
            print("Run Config (from YAML):")
            for key, value in self.run_config.items():
                if key == "modules_to_load" and isinstance(value, list):
                    print(f" {key}:")
                    for module in value:
                        print(f" - {module}")
                else:
                    print(f" {key}: {value}")
        print()
        print("Resource Requirements:")
        print(f" GPU type: {resources['gpu_type']}")
        print(f" Number of nodes: {resources['num_nodes']}")
        print(f" Total GPUs: {resources['num_gpus']}")
        print(f" GPUs per node: {resources['gpus_per_node']}")
        print(f" CPUs per task: {resources['cpus_per_task']}")
        print(f" Memory per node: {resources['mem_per_node']}")
        print(f" Estimated runtime: {resources['time_str']}")
        print(f" Partition: {self._determine_partition(resources['gpu_type'])}")
        print()
        print("Memory Breakdown:")
        print(resources["memory_breakdown"])
        print()
        print("GPU Recommendation:")
        print(f" {resources['gpu_type']}: {resources['num_gpus']} GPUs minimum")
        print(f" Recommended: {resources['num_gpus']} GPUs (power of 2)")

        # More detailed memory info
        gpu_memory_gb = 80 if resources["gpu_type"] == "H100" else 40
        memory_per_gpu = resources["memory_breakdown"].total / resources["num_gpus"]
        utilization = (memory_per_gpu / gpu_memory_gb) * 100

        print(f" Memory per GPU: {memory_per_gpu:.1f} GB")
        print(f" Utilization: {utilization:.1f}%")
        print(f" SLURM memory request: {resources['mem_per_node']}")

        # Add warning if utilization is very high
        if utilization > 95:
            print(
                f"  CRITICAL: Memory utilization ({utilization:.1f}%) WILL cause OOM - MUST increase GPUs"
            )
            print("  EMERGENCY: Try doubling GPU count or reducing batch size")
        elif utilization > 85:
            print(
                f" WARNING: High memory utilization ({utilization:.1f}%) - consider more GPUs"
            )
        elif utilization > 75:
            print(f"  CAUTION: Memory utilization ({utilization:.1f}%) is getting high")
        print("=" * 60)


def main():
    """Main entry point for the script."""

    parser = argparse.ArgumentParser(
        description="Generate SLURM job scripts from dendritic model config files",
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
""",
    )

    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the SLURM script (default: scripts/slurm/)",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Filename for the SLURM script (default: <run_name>.sbatch)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just print summary without generating the script",
    )
    parser.add_argument(
        "--conda-env",
        default="dendritic_env",
        help="Default conda environment name (conda_env_path in run_config specifies the full path)",
    )
    parser.add_argument(
        "--simple-network",
        action="store_true",
        help="Use simple network configuration (minimal NCCL vars)",
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

            # Automatically overwrite existing files without prompting
            if output_path.exists():
                print(f"File {output_path} already exists. Overwriting...")

            # Save the script
            saved_path = generator.save_script(args.filename)
            print(f"\nSLURM script saved to: {saved_path}")
            print("\nTo submit the job, run:")
            print(f" sbatch {saved_path}")
        else:
            print("\nDry run - no script generated.")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
