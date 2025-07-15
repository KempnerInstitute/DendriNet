#!/bin/bash
#SBATCH --job-name=dendritic_fsdp_multinode
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner_eng
#SBATCH --nodes=2                # Number of nodes
#SBATCH --ntasks-per-node=4      # Number of tasks per node (one per GPU)
#SBATCH --gpus-per-node=4        # Request 4 GPUs per node (total 8 GPUs)
#SBATCH --cpus-per-task=10       # CPUs per task
#SBATCH --mem=1200G              # Memory per node
#SBATCH --time=2:00:00          # Time limit
#SBATCH --output=logs/fsdp_multinode_v2_%j.out
#SBATCH --error=logs/fsdp_multinode_v2_%j.err
#SBATCH --exclusive              # Exclusive node access for large models

# Load required modules
module load python/3.12.5-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01

# Activate conda environment
source activate /n/netscratch/kempner_dev/hsafaai/conda_envs/dendritic_env

# Set up environment variables
export OMP_NUM_THREADS=1

# Use a unique port based on job ID to avoid conflicts
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# Get the master address - use IP address instead of hostname to avoid DNS issues
# This approach is more robust for clusters with IPv6/IPv4 issues
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
# Try to resolve to IPv4 address
MASTER_IP=$(getent hosts $MASTER_ADDR | awk '{ print $1 }' | grep -E '^[0-9]+\.' | head -n 1)
if [ -n "$MASTER_IP" ]; then
    export MASTER_ADDR=$MASTER_IP
    echo "Using IP address: $MASTER_ADDR"
else
    echo "Warning: Could not resolve to IPv4, using hostname: $MASTER_ADDR"
fi

# Force IPv4 and disable IPv6
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET

# Network interface configuration - let NCCL auto-detect
# Comment out to let NCCL choose the best interface
# export NCCL_SOCKET_IFNAME=^lo  # Exclude loopback, use all others

# NCCL settings for stability
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800  # 30 minutes
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available

# Additional settings for large models
export NCCL_BUFFSIZE=2097152  # 2MB buffer size
export NCCL_P2P_LEVEL=NVL  # Use NVLink when available

# PyTorch settings
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Disable CUDA memory fraction to let FSDP manage memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create log directory
mkdir -p logs

# Configuration file
CONFIG_FILE="${1:-configs/large_model_example.yaml}"

# Log job information
echo "=== FSDP Multi-node Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Node list: $SLURM_NODELIST"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Config file: $CONFIG_FILE"
echo "Total GPUs: $((SLURM_NNODES * SLURM_GPUS_PER_NODE))"
echo "================================"

# Show network interfaces on master node
if [ "$SLURM_NODEID" -eq 0 ]; then
    echo "Available network interfaces:"
    ip -o link show | awk -F': ' '{print $2}' | grep -v lo
fi

# Calculate processes per node
GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
NNODES=$SLURM_NNODES

# Alternative approach: Use pdsh for multi-node execution
# This avoids some issues with srun + torchrun combination
echo "Starting training on all nodes..."

# Export variables for all nodes
export GPUS_PER_NODE NNODES CONFIG_FILE SLURM_JOB_ID

# Run on all nodes
srun --ntasks=$NNODES --ntasks-per-node=1 bash -c '
    echo "Node $SLURM_NODEID ($HOSTNAME) starting..."
    
    # Set CUDA devices visibility
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE-1)))
    
    # Run torchrun with proper node rank
    torchrun \
        --nproc_per_node=$GPUS_PER_NODE \
        --nnodes=$NNODES \
        --node_rank=$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        src/dendritic_modeling/scripts/train_large_model_fsdp.py \
        $CONFIG_FILE \
        --checkpoint_dir checkpoints/fsdp_multinode \
        --mixed_precision
'

echo "FSDP training completed" 