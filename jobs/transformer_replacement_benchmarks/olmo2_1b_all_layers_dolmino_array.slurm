#!/usr/bin/env bash
# Distill one OLMo-2 1B MLP layer per array task into a dendritic replacement,
# then benchmark original-vs-dendritic LM loss for that single patched layer.
# Submit the companion aggregate job with an afterok dependency to evaluate the
# model with all trained layer replacements patched together.
#SBATCH --job-name=olmo2_1b_all_layers
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-15%4
#SBATCH --chdir=/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling
#SBATCH --output=results/transformer_replacement_benchmarks/slurm/%A_%a_olmo2_1b_all_layers.out
#SBATCH --error=results/transformer_replacement_benchmarks/slurm/%A_%a_olmo2_1b_all_layers.err

set -euo pipefail

REPO_ROOT="/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HOME="${HF_HOME:-/n/netscratch/kempner_dev/hsafaai/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
unset TRANSFORMERS_CACHE

PYTHON_BIN="${PYTHON:-$(command -v python)}"
PREFLIGHT_TIMEOUT="${PREFLIGHT_TIMEOUT:-600s}"
CONFIG_DIR="jobs/transformer_replacement_benchmarks/configs/generated/olmo2_1b_dolmino_all_layers"
CONFIG="${CONFIG_DIR}/olmo2_1b_dolmino_layer$(printf '%02d' "${SLURM_ARRAY_TASK_ID}").yaml"
RESULT_ROOT="results/transformer_replacement_benchmarks/olmo2_1b_dolmino_all_layers"

mkdir -p results/transformer_replacement_benchmarks/slurm "${RESULT_ROOT}"

echo "[$(date)] host=$(hostname)"
echo "[$(date)] preflight_timeout=${PREFLIGHT_TIMEOUT}"
echo "[$(date)] generating OLMo-2 1B all-layer configs"
"${PYTHON_BIN}" -m dendritic_modeling.scripts.transformer_replacement.generate_olmo_all_layer_configs \
  --variant 1b \
  --config-root jobs/transformer_replacement_benchmarks/configs/generated \
  --results-root results/transformer_replacement_benchmarks

echo "[$(date)] config=${CONFIG}"
timeout "${PREFLIGHT_TIMEOUT}" "${PYTHON_BIN}" - <<'PY'
import datasets, os, torch, transformers

print("transformers", transformers.__version__, flush=True)
print("datasets", datasets.__version__, flush=True)
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
print("cuda_device_count", torch.cuda.device_count(), flush=True)
print("cuda_available", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for this benchmark job")
torch.empty(1, device="cuda")
print("cuda_device", torch.cuda.get_device_name(0), flush=True)
PY

echo "[$(date)] distilling OLMo-2 1B layer ${SLURM_ARRAY_TASK_ID}"
"${PYTHON_BIN}" -m dendritic_modeling.scripts.training.train_transformer_replacement "${CONFIG}"

echo "[$(date)] benchmarking OLMo-2 1B layer ${SLURM_ARRAY_TASK_ID}"
"${PYTHON_BIN}" -m dendritic_modeling.scripts.training.benchmark_transformer_replacement "${CONFIG}"

echo "[$(date)] done layer ${SLURM_ARRAY_TASK_ID}"
