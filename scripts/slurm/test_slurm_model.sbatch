#!/bin/bash
#SBATCH --job-name=custom_slurm_job_name
#SBATCH --output=/n/holylabs/LABS/kempner_dev/Users/hsafaai/custom_logs/custom_slurm_job_name_%j.out
#SBATCH --error=/n/holylabs/LABS/kempner_dev/Users/hsafaai/custom_logs/custom_slurm_job_name_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=25
#SBATCH --mem=96GB
#SBATCH --time=07:00:00
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev

# Load modules if specified
module load python/3.12.5-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01

# Activate environment
source /n/netscratch/kempner_dev/hsafaai/conda_envs/dendritic_env/etc/profile.d/conda.sh
conda activate dendritic_env

# Set environment variables for distributed training
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
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
data_path = '/n/holylfs06/LABS/kempner_shared/Lab/data/cifar10'
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

# Run the training script (single GPU)
python scripts/train_experiments.py \
    --config configs/test_slurm_config.yaml

# Cleanup sync file on rank 0
if [ $RANK -eq 0 ]; then
    rm -f $SYNC_FILE
fi

# Print job summary
echo "Job completed at $(date)"
echo "Resource usage summary:"
echo "  Config: configs/test_slurm_config.yaml"
echo "  GPU type: H100"
echo "  Total GPUs: 1"
echo "  Memory per GPU: 0.0 GB"
