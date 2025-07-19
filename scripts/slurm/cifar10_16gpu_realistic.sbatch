#!/bin/bash
#SBATCH --job-name=cifar10_16gpu_realistic
#SBATCH --output=results//cifar10_16gpu_realistic_%j.out
#SBATCH --error=results//cifar10_16gpu_realistic_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=14
#SBATCH --mem=384GB
#SBATCH --time=50:00:00
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev

# Load modules if specified
module load python/3.12.5-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01

# Activate environment
# Initialize conda - try provided path first, fallback to conda info
if [ -f "/n/netscratch/kempner_dev/hsafaai/conda_envs/dendritic_env/etc/profile.d/conda.sh" ]; then
    source /n/netscratch/kempner_dev/hsafaai/conda_envs/dendritic_env/etc/profile.d/conda.sh
else
    source $(conda info --base)/etc/profile.d/conda.sh
fi
conda activate dendritic_env

# Set environment variables for distributed training
export MASTER_PORT=$(expr 20000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export RANK=$SLURM_PROCID

# Set master address
master_node=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR=$master_node

# NCCL settings for H100
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
    echo "Rank 0: Pre-downloading cifar10 dataset..."
    python -c "
import torchvision
import torch
data_path = 'None/cifar10'
# Download both train and test sets
_ = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True)
_ = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True)
print('Dataset download complete.')
"
fi

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

# Run the training script with torchrun for FSDP
srun --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
    torchrun \
    --nproc_per_node=4 \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/dendritic_modeling/scripts/train_experiment_fstp.py \
    --config configs/distributed/cifar10_fstp.yaml \
    --distributed

# Cleanup sync file on rank 0
if [ $RANK -eq 0 ]; then
    rm -f $SYNC_FILE
fi

# Print job summary
echo "Job completed at $(date)"
echo "Resource usage summary:"
echo "  Config: configs/distributed/cifar10_fstp.yaml"
echo "  GPU type: H100"
echo "  Total GPUs: 4"
echo "  Memory per GPU: 44.1 GB"
