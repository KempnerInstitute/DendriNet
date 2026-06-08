#!/usr/bin/env bash
# Jointly fine-tune all OLMo-2 1B dendritic MLP replacements through the full
# language-modeling objective. Each array task runs one morphology/warm-start
# condition from scripts/transformer_replacement/generate_olmo_joint_configs.py.
#SBATCH --job-name=olmo2_1b_joint
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%3
#SBATCH --chdir=/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling
#SBATCH --output=results/transformer_replacement_benchmarks/slurm/%A_%a_olmo2_1b_joint.out
#SBATCH --error=results/transformer_replacement_benchmarks/slurm/%A_%a_olmo2_1b_joint.err

set -euo pipefail

REPO_ROOT="/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HOME="${HF_HOME:-/n/netscratch/kempner_dev/hsafaai/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
unset TRANSFORMERS_CACHE

PYTHON_BIN="${PYTHON:-$(command -v python)}"
CONFIG_ROOT="configs/benchmarks/transformer_replacement/generated"
RESULT_ROOT="results/transformer_replacement_benchmarks"

mkdir -p results/transformer_replacement_benchmarks/slurm

echo "[$(date)] host=$(hostname)"
echo "[$(date)] generating OLMo-2 1B joint all-layer configs"
"${PYTHON_BIN}" scripts/transformer_replacement/generate_olmo_joint_configs.py \
  --variant 1b \
  --config-root "${CONFIG_ROOT}" \
  --results-root "${RESULT_ROOT}"

mapfile -t CONFIGS < <(
  "${PYTHON_BIN}" scripts/transformer_replacement/generate_olmo_joint_configs.py \
    --variant 1b \
    --config-root "${CONFIG_ROOT}" \
    --results-root "${RESULT_ROOT}" \
    --print-configs
)

CONFIG="${CONFIGS[${SLURM_ARRAY_TASK_ID}]}"
echo "[$(date)] config=${CONFIG}"

"${PYTHON_BIN}" - <<'PY'
import os, torch, transformers
print("transformers", transformers.__version__, flush=True)
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
print("cuda_device_count", torch.cuda.device_count(), flush=True)
print("cuda_available", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for this benchmark job")
print("cuda_device", torch.cuda.get_device_name(0), flush=True)
PY

echo "[$(date)] joint fine-tuning ${CONFIG}"
"${PYTHON_BIN}" -m dendritic_modeling.scripts.training.train_transformer_replacement "${CONFIG}"

echo "[$(date)] benchmarking ${CONFIG}"
"${PYTHON_BIN}" -m dendritic_modeling.scripts.training.benchmark_transformer_replacement "${CONFIG}"

echo "[$(date)] done"
