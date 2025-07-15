#!/bin/bash
#SBATCH --job-name=dendritic_fsdp_4nodes
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner_h100
#SBATCH --nodes=4                # 4 nodes for very large model
#SBATCH --ntasks-per-node=4      # 4 tasks per node (one per GPU)
#SBATCH --gpus-per-node=4        # 4 GPUs per node (total 16 GPUs)
#SBATCH --cpus-per-task=12       # More CPUs for data loading
#SBATCH --mem=1200G              # Memory per node
#SBATCH --time=12:00:00          # Longer time for ImageNet training
#SBATCH --output=logs/fsdp_4nodes_%j.out
#SBATCH --error=logs/fsdp_4nodes_%j.err
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

# Calculate total number of GPUs
NNODES=$SLURM_NNODES
GPUS_PER_NODE=4
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

# Set NCCL environment variables for better performance
export NCCL_DEBUG=INFO
export NCCL_SOCKET_FAMILY=AF_INET  # Force IPv4
export NCCL_IB_DISABLE=0            # Enable InfiniBand
export NCCL_IB_HCA=mlx5             # Use Mellanox cards
export NCCL_NET_GDR_LEVEL=5         # GPU Direct RDMA
export NCCL_P2P_LEVEL=NVL           # Use NVLink when available

# PyTorch distributed settings
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=0

# Get config file from command line argument
CONFIG_FILE=${1:-configs/ei_fsdp_4nodes.yaml}

echo "Starting FSDP training on $NNODES nodes with $WORLD_SIZE total GPUs"
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Config file: $CONFIG_FILE"

# Create checkpoint directory with job ID
CHECKPOINT_DIR="checkpoints/fsdp_4nodes_${SLURM_JOB_ID}"
mkdir -p $CHECKPOINT_DIR

# Run the training script using torchrun on each node
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
        --checkpoint_dir $CHECKPOINT_DIR \
        --model_size xlarge \
        --save_every_n_epochs 5
"

echo "Training completed" 