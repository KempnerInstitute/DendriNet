#!/usr/bin/env bash
#SBATCH --job-name=olmo2_7b_dendrinet_l0
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --chdir=/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling
#SBATCH --output=results/transformer_replacement_benchmarks/slurm/%j_olmo2_7b_dolmino.out
#SBATCH --error=results/transformer_replacement_benchmarks/slurm/%j_olmo2_7b_dolmino.err

set -euo pipefail

REPO_ROOT="/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HOME="${HF_HOME:-/n/holylabs/kempner_dev/Users/hsafaai/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
unset TRANSFORMERS_CACHE

CONFIG="jobs/transformer_replacement_benchmarks/configs/olmo2_7b_dolmino_layer0_2048.yaml"
PYTHON_BIN="${PYTHON:-$(command -v python)}"

mkdir -p results/transformer_replacement_benchmarks/slurm

echo "[$(date)] host=$(hostname)"
echo "[$(date)] config=${CONFIG}"
timeout 180s "${PYTHON_BIN}" - <<'PY'
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

echo "[$(date)] distilling OLMo-2 7B layer-0 dendritic replacement"
"${PYTHON_BIN}" -m dendritic_modeling.scripts.training.train_transformer_replacement "${CONFIG}"

echo "[$(date)] evaluating original vs dendritic-replaced OLMo-2 7B causal LM"
"${PYTHON_BIN}" -m dendritic_modeling.scripts.training.benchmark_transformer_replacement "${CONFIG}"

echo "[$(date)] updating aggregate summary"
"${PYTHON_BIN}" -m dendritic_modeling.scripts.training.aggregate_transformer_replacement_benchmarks \
  results/transformer_replacement_benchmarks

echo "[$(date)] done"
