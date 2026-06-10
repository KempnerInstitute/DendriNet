#!/usr/bin/env bash
# Aggregate the OLMo-2 1B all-layer sweep and benchmark the single model where
# every distilled MLP replacement is patched in at once.
#SBATCH --job-name=olmo2_1b_all_layers_agg
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --chdir=/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling
#SBATCH --output=results/transformer_replacement_benchmarks/slurm/%j_olmo2_1b_all_layers_agg.out
#SBATCH --error=results/transformer_replacement_benchmarks/slurm/%j_olmo2_1b_all_layers_agg.err

set -euo pipefail

REPO_ROOT="/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HOME="${HF_HOME:-/n/netscratch/kempner_dev/hsafaai/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
unset TRANSFORMERS_CACHE

PYTHON_BIN="${PYTHON:-$(command -v python)}"
CONFIG_DIR="jobs/transformer_replacement_benchmarks/configs/generated/olmo2_1b_dolmino_all_layers"
RESULT_ROOT="results/transformer_replacement_benchmarks/olmo2_1b_dolmino_all_layers"
COMBINED_CKPT_DIR="${RESULT_ROOT}/all_layer_checkpoints"
ALL_LAYERS_CONFIG="${CONFIG_DIR}/olmo2_1b_dolmino_all_layers.yaml"

mkdir -p results/transformer_replacement_benchmarks/slurm "${RESULT_ROOT}" "${COMBINED_CKPT_DIR}"

echo "[$(date)] generating OLMo-2 1B all-layer configs"
"${PYTHON_BIN}" -m dendritic_modeling.scripts.transformer_replacement.generate_olmo_all_layer_configs \
  --variant 1b \
  --config-root jobs/transformer_replacement_benchmarks/configs/generated \
  --results-root results/transformer_replacement_benchmarks

echo "[$(date)] collecting trained layer checkpoints"
for LAYER in $(seq 0 15); do
  SRC="${RESULT_ROOT}/layer_$(printf '%02d' "${LAYER}")/layer_${LAYER}_replacement.pt"
  DST="${COMBINED_CKPT_DIR}/layer_${LAYER}_replacement.pt"
  if [[ ! -s "${SRC}" ]]; then
    echo "Missing trained checkpoint: ${SRC}" >&2
    exit 2
  fi
  ln -sf "${REPO_ROOT}/${SRC}" "${DST}"
done

echo "[$(date)] benchmarking all OLMo-2 1B MLP layers patched together"
"${PYTHON_BIN}" -m dendritic_modeling.scripts.training.benchmark_transformer_replacement \
  "${ALL_LAYERS_CONFIG}" \
  --checkpoint-dir "${COMBINED_CKPT_DIR}"

echo "[$(date)] aggregating per-layer and all-layer benchmark metrics"
"${PYTHON_BIN}" -m dendritic_modeling.scripts.training.aggregate_transformer_replacement_benchmarks \
  "${RESULT_ROOT}" \
  --csv "${RESULT_ROOT}/summary.csv" \
  --markdown "${RESULT_ROOT}/summary.md"

echo "[$(date)] done"
