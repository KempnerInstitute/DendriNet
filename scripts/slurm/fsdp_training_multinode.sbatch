#!/bin/bash
#SBATCH --job-name=dendritic_fsdp_multinode
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner_eng
#SBATCH --nodes=2                # Number of nodes (changed from 1 to 2)
#SBATCH --ntasks-per-node=4      # Number of tasks per node (one per GPU)
#SBATCH --gpus-per-node=4        # Request 4 GPUs per node (total 8 GPUs)
#SBATCH --cpus-per-task=10       # CPUs per task
#SBATCH --mem=1200G              # Memory per node
#SBATCH --time=2:00:00          # Time limit
#SBATCH --output=logs/fsdp_multinode_%j.out
#SBATCH --error=logs/fsdp_multinode_%j.err
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

# Get the master address (first node in the allocation)
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)

# Force IPv4 to avoid IPv6 issues
export NCCL_SOCKET_FAMILY=AF_INET

# Network interface configuration
# Try to auto-detect the correct network interface
# Common interfaces: eth0, ib0, eno1, etc.
if ip link show ib0 &>/dev/null; then
    export NCCL_SOCKET_IFNAME=ib0
    export GLOO_SOCKET_IFNAME=ib0
elif ip link show eth0 &>/dev/null; then
    export NCCL_SOCKET_IFNAME=eth0
    export GLOO_SOCKET_IFNAME=eth0
else
    # Let NCCL auto-select
    echo "Warning: Could not detect network interface, letting NCCL auto-select"
fi

# NCCL performance and timeout settings
export NCCL_DEBUG=INFO  # For debugging, set to WARN in production
export NCCL_TIMEOUT=1800  # 30 minutes timeout (default is 10 minutes)
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_P2P_DISABLE=0  # Enable P2P communication

# Additional NCCL settings for better performance
export NCCL_TREE_THRESHOLD=0  # Always use tree algorithm
export NCCL_LL_THRESHOLD=0    # Disable LL algorithm for large messages

# PyTorch settings
export TORCH_NCCL_BLOCKING_WAIT=1  # Better error messages
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # More debug info

# Create log directory
mkdir -p logs

# Configuration file
CONFIG_FILE="${1:-configs/large_model_example.yaml}"

# Log job information
echo "Starting FSDP training on $SLURM_NNODES nodes with $SLURM_NTASKS total GPUs"
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Config file: $CONFIG_FILE"
echo "Network interface: ${NCCL_SOCKET_IFNAME:-auto}"

# Calculate processes per node
GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
NNODES=$SLURM_NNODES

# Get node rank for this node
NODE_RANK=$SLURM_NODEID

# Run training script directly with torchrun
# Each srun task will run torchrun which spawns the GPU processes
srun --ntasks=$NNODES --ntasks-per-node=1 bash -c "
    echo \"Node \$SLURM_NODEID starting torchrun...\"
    torchrun \
        --nproc_per_node=$GPUS_PER_NODE \
        --nnodes=$NNODES \
        --node_rank=\$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        src/dendritic_modeling/scripts/train_large_model_fsdp.py \
        $CONFIG_FILE \
        --mixed_precision \
        --checkpoint_dir checkpoints/fsdp_${SLURM_JOB_ID} \
        --model_size xlarge
"

echo "FSDP training completed" 