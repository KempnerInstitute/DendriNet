#!/usr/bin/env bash
#SBATCH --job-name=dendrinet_tr_agg
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner_eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --chdir=/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling
#SBATCH --output=results/transformer_replacement_benchmarks/slurm/%j_aggregate.out
#SBATCH --error=results/transformer_replacement_benchmarks/slurm/%j_aggregate.err

set -euo pipefail

REPO_ROOT="/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON:-$(command -v python)}"

"${PYTHON_BIN}" -m dendritic_modeling.scripts.training.aggregate_transformer_replacement_benchmarks \
  results/transformer_replacement_benchmarks
