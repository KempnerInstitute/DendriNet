#!/usr/bin/env bash
#SBATCH --job-name=deepseek_v3_dendrinet_cache
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner_eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --chdir=/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling
#SBATCH --output=results/deepseek_v3_replacement/slurm/%A_%a_hidden_cache_train.out
#SBATCH --error=results/deepseek_v3_replacement/slurm/%A_%a_hidden_cache_train.err

set -euo pipefail

REPO_ROOT="/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export HF_HOME="${HF_HOME:-/n/holylabs/kempner_dev/Users/hsafaai/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
unset TRANSFORMERS_CACHE

CONFIGS=(
  "configs/benchmarks/transformer_replacement/deepseek_v3_layers0_2_dendrinet_4096_hidden_cache.yaml"
  "configs/benchmarks/transformer_replacement/deepseek_v3_layers0_2_dendrinet_8192_hidden_cache.yaml"
)
CONFIG="${CONFIGS[${SLURM_ARRAY_TASK_ID}]}"
PYTHON_BIN="${PYTHON:-$(command -v python)}"

mkdir -p results/deepseek_v3_replacement/slurm

echo "[$(date)] host=$(hostname)"
echo "[$(date)] config=${CONFIG}"
timeout 180s "${PYTHON_BIN}" - <<'PY'
import os, torch
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
print("cuda_device_count", torch.cuda.device_count(), flush=True)
print("cuda_available", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for hidden-cache replacement training")
torch.empty(1, device="cuda")
print("cuda_device", torch.cuda.get_device_name(0), flush=True)
PY

echo "[$(date)] training DeepSeek-V3 dendritic replacement from hidden cache"
"${PYTHON_BIN}" -m dendritic_modeling.scripts.training.train_transformer_replacement "${CONFIG}"

echo "[$(date)] done"
