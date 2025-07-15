#!/bin/bash
#SBATCH --job-name=dendritic_fsdp
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner_eng
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks-per-node=4      # Number of tasks per node (one per GPU)
#SBATCH --gpus-per-node=4        # Request 4 GPUs per node (total 8 GPUs)
#SBATCH --cpus-per-task=10       # CPUs per task
#SBATCH --mem=1200G              # Memory per node
#SBATCH --time=2:00:00          # Time limit
#SBATCH --output=logs/fsdp_%j.out
#SBATCH --error=logs/fsdp_%j.err
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
export GLOO_SOCKET_IFNAME=ib0  # Use InfiniBand interface

# Debug information
export NCCL_DEBUG=INFO  # For debugging, set to WARN in production

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

# Calculate processes per node
GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
NNODES=$SLURM_NNODES

# Get node rank for this node
NODE_RANK=$SLURM_NODEID

# Run training script directly with torchrun
# Each srun task will run torchrun which spawns the GPU processes
srun --ntasks=$NNODES --ntasks-per-node=1 bash -c "
    torchrun \
        --nproc_per_node=$GPUS_PER_NODE \
        --nnodes=$NNODES \
        --node_rank=\$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        src/dendritic_modeling/scripts/train_large_model_fsdp.py \
        $CONFIG_FILE \
        --mixed_precision \
        --checkpoint_dir checkpoints/fsdp_${SLURM_JOB_ID} \
        --model_size xlarge
"

echo "FSDP training completed" 