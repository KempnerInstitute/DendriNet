#!/bin/bash
#SBATCH --job-name=dendritic_fsdp_no_amp
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner_eng
#SBATCH --nodes=2                # Number of nodes
#SBATCH --ntasks-per-node=4      # Number of tasks per node (one per GPU)
#SBATCH --gpus-per-node=4        # Request 4 GPUs per node (total 8 GPUs)
#SBATCH --cpus-per-task=10       # CPUs per task
#SBATCH --mem=1200G              # Memory per node
#SBATCH --time=2:00:00          # Time limit
#SBATCH --output=logs/fsdp_no_amp_%j.out
#SBATCH --error=logs/fsdp_no_amp_%j.err
#SBATCH --exclusive              # Exclusive node access for large models

# Load required modules
module load python/3.12.5-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01

# Activate conda environment
source activate /n/netscratch/kempner_dev/hsafaai/conda_envs/dendritic_env

# Set up environment
export OMP_NUM_THREADS=1

# Dynamic port assignment to avoid conflicts
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# Get master address - convert hostname to IP
MASTER_HOSTNAME=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR=$(getent hosts $MASTER_HOSTNAME | awk '{ print $1 }' | head -n 1)

# Force IPv4 to avoid address family issues
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET

# NCCL settings for better performance
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800  # 30 minutes
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

# PyTorch settings
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=OFF

# Get config file from command line
CONFIG_FILE=${1:-configs/ei_fsdp_2nodes_no_amp.yaml}

# Calculate total number of processes
NNODES=$SLURM_NNODES
NPROC_PER_NODE=4
NTASKS=$((NNODES * NPROC_PER_NODE))

echo "Using IP address: $MASTER_ADDR"
echo "=== FSDP Multi-node Training (No AMP) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $NNODES"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Config file: $CONFIG_FILE"
echo "Total GPUs: $NTASKS"
echo "Mixed Precision: DISABLED"
echo "================================"

# Show available network interfaces
echo "Available network interfaces:"
ip -o link show | awk -F': ' '{print $2}' | grep -v lo

# Change to project directory
cd /n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling

# Run training on all nodes with mixed precision disabled
echo "Starting training on all nodes..."
srun --ntasks=$NNODES --ntasks-per-node=1 bash -c "
    echo \"Node \$SLURM_NODEID (\$(hostname)) starting...\"
    torchrun \
        --nproc_per_node=$NPROC_PER_NODE \
        --nnodes=$NNODES \
        --node_rank=\$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        src/dendritic_modeling/scripts/train_large_model_fsdp.py \
        $CONFIG_FILE \
        --checkpoint_dir checkpoints/fsdp_no_amp
"

echo "Training completed" 