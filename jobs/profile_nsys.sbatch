#!/bin/bash
# Nsight Systems (Rung 2) capture as a batch job.
#
# Why batch instead of interactive: on a preemptible partition
# (e.g. kempner_requeue) interactive sessions get killed mid-run. A batch job
# with --requeue is resubmitted automatically on preemption and re-runs from
# scratch until one attempt completes — unattended. The capture is idempotent
# (--force-overwrite), so re-running is safe.
#
# Submit (from the repo root on the cluster):
#   sbatch jobs/profile_nsys.sbatch
# If `nsys` is not on PATH by default, pass the module that provides it:
#   sbatch --export=ALL,NSYS_MODULE=cuda/12.4.1-fasrc01 jobs/profile_nsys.sbatch
#
# Override partition/account without editing this file:
#   sbatch -p kempner_requeue -A kempner_undergrads jobs/profile_nsys.sbatch
#
# Output: profilings/NSights/mnist_timeline.nsys-rep  (+ a job log alongside)

#SBATCH --job-name=dendrinet-nsys
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_undergrads
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=profilings/NSights/nsys_job_%j.log

echo "[job] node=$(hostname) attempt-start=$(date) restarts=${SLURM_RESTART_COUNT:-0}"

module load python
if [ -n "${NSYS_MODULE:-}" ]; then
  module load "${NSYS_MODULE}"
fi

# Activate the conda env (batch shells are non-interactive → init the hook).
eval "$(conda shell.bash hook)"
conda activate dendritic_env

cd "${SLURM_SUBMIT_DIR}"

if ! command -v nsys >/dev/null 2>&1; then
  echo "[job] ERROR: nsys not on PATH."
  echo "[job] Re-submit with the CUDA/Nsight module, e.g.:"
  echo "[job]   sbatch --export=ALL,NSYS_MODULE=cuda/12.4.1-fasrc01 jobs/profile_nsys.sbatch"
  echo "[job] (find it on a login node with: module load python && which nsys, or: module avail cuda)"
  exit 1
fi

python -c "import torch; print('[job] torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"

# CONFIG: which YAML to profile (override via --export=...,CONFIG=...).
# Output report is named after the config so multiple configs don't collide.
CONFIG="${CONFIG:-configs/examples/standard_mnist.yaml}"
RUN_NAME="$(basename "${CONFIG}" .yaml)"

# Capture ~20 training STEPS (not epochs) so the repeating per-step pattern is
# clearly visible in the timeline. The run still early-exits after these steps.
# Override the count with --export=...,PROFILE_ACTIVE=N.
export DENDRINET_PROFILE_ACTIVE="${PROFILE_ACTIVE:-20}"

echo "[job] profiling config: ${CONFIG} -> profilings/NSights/${RUN_NAME}.nsys-rep"
echo "[job] capturing ~$((DENDRINET_PROFILE_ACTIVE + 3)) steps (PROFILE_ACTIVE=${DENDRINET_PROFILE_ACTIVE})"

nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas \
  --output="profilings/NSights/${RUN_NAME}" --force-overwrite=true \
  python src/dendritic_modeling/scripts/training/train_experiments.py \
      "${CONFIG}" --profile --profile_backend nsys

echo "[job] DONE attempt-end=$(date)"
ls -l profilings/NSights/mnist_timeline.nsys-rep
