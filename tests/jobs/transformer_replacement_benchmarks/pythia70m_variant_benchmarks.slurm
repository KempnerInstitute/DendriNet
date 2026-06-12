#!/usr/bin/env bash
#SBATCH --job-name=dendrinet_pythia_variants
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner_eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --chdir=/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling
#SBATCH --output=results/transformer_replacement_benchmarks/slurm/%A_%a_pythia_variant.out
#SBATCH --error=results/transformer_replacement_benchmarks/slurm/%A_%a_pythia_variant.err

set -euo pipefail

REPO_ROOT="/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HOME="${HF_HOME:-/n/holylabs/kempner_dev/Users/hsafaai/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"

CONFIGS=(
  "jobs/transformer_replacement_benchmarks/configs/pythia70m_wikitext_layer0_standard_topk.yaml"
  "jobs/transformer_replacement_benchmarks/configs/pythia70m_wikitext_layer0_large_deep.yaml"
)
CONFIG="${CONFIGS[${SLURM_ARRAY_TASK_ID}]}"
PYTHON_BIN="${PYTHON:-$(command -v python)}"

echo "[$(date)] host=$(hostname)"
echo "[$(date)] array_task=${SLURM_ARRAY_TASK_ID}"
echo "[$(date)] config=${CONFIG}"
timeout 180s "${PYTHON_BIN}" - <<'PY'
import accelerate, datasets, torch, transformers
print("transformers", transformers.__version__, flush=True)
print("datasets", datasets.__version__, flush=True)
print("accelerate", accelerate.__version__, flush=True)
print("cuda_available", torch.cuda.is_available(), flush=True)
print("cuda_device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", flush=True)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for this benchmark job")
torch.empty(1, device="cuda")
PY

echo "[$(date)] distilling dendritic replacement"
"${PYTHON_BIN}" -m dendritic_modeling.scripts.training.train_transformer_replacement "${CONFIG}"

echo "[$(date)] evaluating original vs dendritic-replaced causal LM"
"${PYTHON_BIN}" -m dendritic_modeling.scripts.training.benchmark_transformer_replacement "${CONFIG}"

echo "[$(date)] updating aggregate summary"
"${PYTHON_BIN}" -m dendritic_modeling.scripts.training.aggregate_transformer_replacement_benchmarks \
  results/transformer_replacement_benchmarks

echo "[$(date)] done"
