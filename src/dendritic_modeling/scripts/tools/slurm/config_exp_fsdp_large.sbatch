#!/bin/bash
#SBATCH --job-name=config_exp_fsdp
#SBATCH --output=results//logs/config_exp_fsdp_%j.out
#SBATCH --error=results//logs/config_exp_fsdp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=25
#SBATCH --mem=120GB
#SBATCH --time=11:00:00
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev



# Initialize conda using conda info
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dendritic_env

export MASTER_PORT=$(python -c "import random; print(random.randint(20000, 30000))")
export WORLD_SIZE=$((SLURM_NNODES * 1)) 
export RANK=$SLURM_PROCID
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

    export NCCL_DEBUG=INFO
    export NCCL_SOCKET_FAMILY=AF_INET

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling
mkdir -p results//logs

# Training command
python src/dendritic_modeling/scripts/train_experiments.py configs/config_exp_fsdp.yaml