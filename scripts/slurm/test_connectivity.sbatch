#!/bin/bash
#SBATCH --job-name=test_connectivity
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner_eng
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2      # 2 processes per node for testing
#SBATCH --gpus-per-node=2        
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=0:10:00           # 10 minutes
#SBATCH --output=logs/test_connectivity_%j.out
#SBATCH --error=logs/test_connectivity_%j.err

# Load required modules
module load python/3.12.5-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01

# Activate conda environment
source activate /n/netscratch/kempner_dev/hsafaai/conda_envs/dendritic_env

# Set up environment
export OMP_NUM_THREADS=1
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# Get master address - try multiple methods
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
echo "Initial MASTER_ADDR: $MASTER_ADDR"

# Try to get IPv4 address
MASTER_IP=$(getent hosts $MASTER_ADDR | awk '{ print $1 }' | grep -E '^[0-9]+\.' | head -n 1)
if [ -n "$MASTER_IP" ]; then
    export MASTER_ADDR=$MASTER_IP
    echo "Resolved to IPv4: $MASTER_ADDR"
fi

# Force IPv4
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET
export NCCL_DEBUG=INFO

# Create logs directory
mkdir -p logs

echo "=== Connectivity Test ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Node list: $SLURM_NODELIST"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "========================"

# First, test basic network connectivity
echo -e "\n1. Testing basic network connectivity..."
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c '
    echo "Node $SLURM_NODEID ($HOSTNAME):"
    echo "  - IP addresses: $(hostname -I)"
    echo "  - Can reach master: $(ping -c 1 -W 1 '$MASTER_ADDR' &>/dev/null && echo "YES" || echo "NO")"
'

# Second, test PyTorch distributed
echo -e "\n2. Testing PyTorch distributed..."
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c '
    torchrun \
        --nproc_per_node=2 \
        --nnodes=$SLURM_NNODES \
        --node_rank=$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        scripts/test_multinode_connectivity.py
'

echo -e "\nConnectivity test completed" 