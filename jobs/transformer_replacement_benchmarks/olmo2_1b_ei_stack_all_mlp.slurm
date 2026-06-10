#!/usr/bin/env bash
# Replace every OLMo-2 1B dense MLP with one layer of a shared 16-layer EI stack
# and jointly train that stack through LM/KL/hidden-state losses.
#SBATCH --job-name=olmo2_1b_eistack
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --chdir=/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling
#SBATCH --output=results/transformer_replacement_benchmarks/slurm/%j_olmo2_1b_eistack.out
#SBATCH --error=results/transformer_replacement_benchmarks/slurm/%j_olmo2_1b_eistack.err

set -euo pipefail

REPO_ROOT="/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HOME="${HF_HOME:-/n/netscratch/kempner_dev/hsafaai/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
unset TRANSFORMERS_CACHE

PYTHON_BIN="${PYTHON:-$(command -v python)}"
CONFIG_ROOT="jobs/transformer_replacement_benchmarks/configs/generated"
RESULT_ROOT="results/transformer_replacement_benchmarks"

mkdir -p results/transformer_replacement_benchmarks/slurm

echo "[$(date)] host=$(hostname)"
echo "[$(date)] generating OLMo-2 1B EI-stack config"
"${PYTHON_BIN}" -m dendritic_modeling.scripts.transformer_replacement.generate_olmo_ei_stack_configs \
  --config-root "${CONFIG_ROOT}" \
  --results-root "${RESULT_ROOT}"

CONFIG="$("${PYTHON_BIN}" -m dendritic_modeling.scripts.transformer_replacement.generate_olmo_ei_stack_configs \
  --config-root "${CONFIG_ROOT}" \
  --results-root "${RESULT_ROOT}" \
  --print-config)"
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

echo "[$(date)] joint EI-stack fine-tuning"
"${PYTHON_BIN}" -m dendritic_modeling.scripts.training.train_transformer_replacement "${CONFIG}"

echo "[$(date)] benchmarking EI-stack replacement"
"${PYTHON_BIN}" -m dendritic_modeling.scripts.training.benchmark_transformer_replacement "${CONFIG}"

echo "[$(date)] done"
