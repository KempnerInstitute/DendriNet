#!/bin/bash
#SBATCH --job-name=cifar10_fstp_kempner_eng
#SBATCH --output=/n/holylabs/kempner_dev/Users/hsafaai/logs/cifar10_fstp_kempner_eng_%j.out
#SBATCH --error=/n/holylabs/kempner_dev/Users/hsafaai/logs/cifar10_fstp_kempner_eng_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=25
#SBATCH --mem=1000GB
#SBATCH --time=04:00:00
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev

module load cuda/12.4.1-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01

# Initialize conda - try provided path first, fallback to conda info
if [ -f "/n/netscratch/kempner_dev/hsafaai/conda_envs/dendritic_env/etc/profile.d/conda.sh" ]; then
    source /n/netscratch/kempner_dev/hsafaai/conda_envs/dendritic_env/etc/profile.d/conda.sh
else
    source $(conda info --base)/etc/profile.d/conda.sh
fi
conda activate dendritic_env

export MASTER_PORT=$(python -c "import random; print(random.randint(20000, 30000))")
export WORLD_SIZE=$((SLURM_NNODES * 4)) 
export RANK=$SLURM_PROCID
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

export NCCL_DEBUG=INFO
export NCCL_SOCKET_FAMILY=AF_INET

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling
mkdir -p /n/holylabs/kempner_dev/Users/hsafaai/logs

# Training command
python src/dendritic_modeling/scripts/train_experiments_fstp.py configs/distributed/cifar10_fstp.yaml