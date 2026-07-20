#!/bin/bash
# torchrun/DDP launcher for replicated transformer replacement distillation.
# Usage: sbatch scripts/train_transformer_replacement_ddp.sh CONFIG [GPUS_PER_NODE]

#SBATCH --job-name=dendrillm_ddp
#SBATCH --partition=kempner_h100_priority
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dendrillm_ddp_%j.out
#SBATCH --error=logs/dendrillm_ddp_%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling}
PYTHON_BIN=${PYTHON_BIN:-/n/sw/Mambaforge-23.11.0-0/bin/python}
CONFIG=${1:?Pass a transformer replacement YAML config}
GPUS_PER_NODE=${2:-4}
NNODES=${SLURM_NNODES:-1}

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/n/netscratch/kempner_dev/Everyone/hf_cache}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0,eth0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "DendriLLM transformer replacement DDP"
echo "config=$CONFIG"
echo "nodes=$NNODES gpus_per_node=$GPUS_PER_NODE"
echo "master=$MASTER_ADDR:$MASTER_PORT job_id=$SLURM_JOB_ID"
echo "git_commit=$(git rev-parse HEAD)"
"$PYTHON_BIN" -c "import datasets, torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'datasets', datasets.__version__)"

# One torchrun coordinator per node; each coordinator creates one worker per GPU.
srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 \
    "$PYTHON_BIN" -m torch.distributed.run \
    --nnodes="$NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    -m dendritic_modeling.scripts.training.train_transformer_replacement \
    "$CONFIG"
