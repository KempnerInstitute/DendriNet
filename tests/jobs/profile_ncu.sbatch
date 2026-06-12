#!/bin/bash
# Nsight Compute (Rung 3) — kernel-level profile of the dominant GPU kernel.
#
# WHAT THIS IS FOR (read first):
# DendriNet has NO custom CUDA/Triton kernels — every kernel is stock
# NVIDIA/PyTorch code you cannot rewrite. So this is a LEARNING / characterization
# run, not an optimization one. We point ncu at the top-k kernel
# (sbtopk::gatherTopK, ~39% of GPU time per Rung 1/2) to see *why* it costs what
# it does: compute- vs memory-bound, achieved occupancy, warp efficiency.
#
# WHY IT'S SET UP THIS WAY:
# ncu is slow — it replays each profiled kernel many times to read hardware
# counters. So we (a) reuse --profile_backend nsys so the training app runs only a
# few steps then exits, and (b) restrict ncu to the first few top-k launches via
# --kernel-name + --launch-count. Profiling every kernel would take ages.
#
# PERMISSIONS (the likely blocker on a shared cluster):
# ncu needs access to NVIDIA GPU performance counters. If the log shows
#   ERR_NVGPUCTRPERM: The user does not have permission to access NVIDIA GPU
#   Performance Counters on the target device
# that is a driver-level setting only RC admins can change. Email
# rchelp@rc.fas.harvard.edu and ask them to enable GPU performance counters for
# user profiling (NVreg_RestrictProfilingToAdminUsers=0). Some sections may still
# collect without it.
#
# SUBMIT (ncu ships with the cuda module, same as nsys):
#   sbatch --export=ALL,NCU_MODULE=cuda/12.9.1-fasrc01 jobs/profile_ncu.sbatch
#
# OUTPUT: profilings/NSights/topk_ncu.ncu-rep  (open in the Nsight Compute GUI)

#SBATCH --job-name=dendrinet-ncu
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_undergrads
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=profilings/NSights/ncu_job_%j.log

echo "[job] node=$(hostname) attempt-start=$(date) restarts=${SLURM_RESTART_COUNT:-0}"

module load python
if [ -n "${NCU_MODULE:-}" ]; then
  module load "${NCU_MODULE}"
fi

eval "$(conda shell.bash hook)"
conda activate dendritic_env
cd "${SLURM_SUBMIT_DIR}"

if ! command -v ncu >/dev/null 2>&1; then
  echo "[job] ERROR: ncu not on PATH. Re-submit with the CUDA module, e.g.:"
  echo "[job]   sbatch --export=ALL,NCU_MODULE=cuda/12.9.1-fasrc01 jobs/profile_ncu.sbatch"
  exit 1
fi

python -c "import torch; print('[job] cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"

# --set basic: a light metric set for a first look. Swap to 'detailed' or 'full'
# for more (much slower). --kernel-name regex matches sbtopk::gatherTopK.
ncu --target-processes all \
    --kernel-name "regex:gatherTopK" \
    --launch-count 3 \
    --set basic \
    --force-overwrite \
    -o profilings/NSights/topk_ncu \
    python src/dendritic_modeling/scripts/training/train_experiments.py \
        configs/examples/standard_mnist.yaml --profile --profile_backend nsys

echo "[job] DONE attempt-end=$(date)"
ls -l profilings/NSights/topk_ncu.ncu-rep
