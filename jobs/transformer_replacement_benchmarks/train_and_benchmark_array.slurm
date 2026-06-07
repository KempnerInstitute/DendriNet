#!/usr/bin/env bash
#SBATCH --job-name=dendrinet_tr_bench
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner_eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --chdir=/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling
#SBATCH --output=results/transformer_replacement_benchmarks/slurm/%A_%a.out
#SBATCH --error=results/transformer_replacement_benchmarks/slurm/%A_%a.err

set -euo pipefail

REPO_ROOT="/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

CONFIGS=(
  "configs/benchmarks/transformer_replacement/synthetic_linear_projection.yaml"
  "configs/benchmarks/transformer_replacement/synthetic_gelu_mlp.yaml"
  "configs/benchmarks/transformer_replacement/synthetic_gated_silu_mlp.yaml"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
PYTHON_BIN="${PYTHON:-$(command -v python)}"

echo "[$(date)] host=$(hostname)"
echo "[$(date)] job=${SLURM_JOB_ID:-manual} task=${SLURM_ARRAY_TASK_ID:-0}"
echo "[$(date)] config=${CONFIG}"
"${PYTHON_BIN}" - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available(), flush=True)
print("cuda_device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", flush=True)
PY

echo "[$(date)] training dendritic transformer replacement"
"${PYTHON_BIN}" -m dendritic_modeling.scripts.training.train_transformer_replacement "${CONFIG}"

echo "[$(date)] benchmarking trained replacement against teacher block"
"${PYTHON_BIN}" -m dendritic_modeling.scripts.training.benchmark_transformer_replacement "${CONFIG}"

echo "[$(date)] done"
