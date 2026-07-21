#!/bin/bash
# Run one transformer replacement manifest cell per Slurm array task.
# Usage: sbatch --array=0-N%MAX scripts/train_transformer_replacement_array.sh MANIFEST

#SBATCH --job-name=dendrillm_array
#SBATCH --partition=kempner_h100_priority
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/dendrillm_array_%A_%a.out
#SBATCH --error=logs/dendrillm_array_%A_%a.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling}
PYTHON_BIN=${PYTHON_BIN:-/n/sw/Mambaforge-23.11.0-0/bin/python}
MANIFEST=${1:?Pass a JSON experiment manifest}
ARRAY_INDEX=${SLURM_ARRAY_TASK_ID:?Launch this script with sbatch --array}

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/n/netscratch/kempner_dev/Everyone/hf_cache}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

CONFIG=$(jq -er ".runs[$ARRAY_INDEX].config" "$MANIFEST")
RUN_NAME=$(jq -er ".runs[$ARRAY_INDEX].name" "$MANIFEST")
SAVE_DIR=$(jq -er ".runs[$ARRAY_INDEX].save_dir" "$MANIFEST")
STARTED_SECONDS=$(date +%s)

echo "DendriLLM transformer replacement array cell"
echo "manifest=$MANIFEST index=$ARRAY_INDEX run=$RUN_NAME"
echo "config=$CONFIG job_id=$SLURM_JOB_ID"
echo "git_commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$PYTHON_BIN" -c "import datasets, torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'datasets', datasets.__version__)"

"$PYTHON_BIN" -m \
    dendritic_modeling.scripts.training.train_transformer_replacement \
    "$CONFIG"
"$PYTHON_BIN" -m \
    dendritic_modeling.scripts.training.benchmark_transformer_replacement \
    "$CONFIG"

FINISHED_SECONDS=$(date +%s)
mkdir -p "$SAVE_DIR"
jq -n \
    --arg manifest "$MANIFEST" \
    --arg config "$CONFIG" \
    --arg run_name "$RUN_NAME" \
    --arg git_commit "$(git rev-parse HEAD)" \
    --arg slurm_job_id "$SLURM_JOB_ID" \
    --argjson array_index "$ARRAY_INDEX" \
    --argjson wall_time_seconds "$((FINISHED_SECONDS - STARTED_SECONDS))" \
    '{
      manifest: $manifest,
      config: $config,
      run_name: $run_name,
      git_commit: $git_commit,
      slurm_job_id: $slurm_job_id,
      array_index: $array_index,
      wall_time_seconds: $wall_time_seconds
    }' > "$SAVE_DIR/run_metadata.json"
