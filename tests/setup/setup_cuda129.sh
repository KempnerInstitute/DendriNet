#!/usr/bin/env bash
#
# setup_cuda129.sh — create a fresh conda env for `dendritic_modeling` with a
# PyTorch build compatible with a CUDA 12.9 cluster (e.g. the Kempner cluster).
#
# WHY cu128 AND NOT cu129
# -----------------------
# PyTorch's stable pip wheels top out at the cu128 (CUDA 12.8) build — there is
# no cu129 index for stable PyTorch. That is fine: PyTorch wheels BUNDLE their
# own CUDA runtime (cudart/cuBLAS/cuDNN) and do NOT use the system CUDA toolkit.
# The only thing that must be compatible is the NVIDIA *driver*, and within CUDA
# major version 12 a newer driver runs any older 12.x runtime. So a cu128 wheel
# runs natively on a CUDA 12.9 driver. cu128 IS the "CUDA 12.9-compatible" build.
#
# This script intentionally does NOT use setup/environment.yml, because that file
# pins `pytorch-cuda=12.1` via the (deprecated) pytorch conda channel, which
# cannot produce a 12.8/12.9-compatible build. Instead we install PyTorch from
# the official pip wheel index and then `pip install -e .` (the supported
# workflow per INSTALL.md) for the package and its declared dependencies.
#
# USAGE
# -----
#   bash setup/setup_cuda129.sh
#
# Configurable via environment variables:
#   ENV_NAME     conda env name                 (default: dendritic_env)
#   ENV_PREFIX   create at a path instead of a  (default: unset; use for HPC
#                named env, e.g. scratch space             home-quota limits)
#   PY_VERSION   Python version                 (default: 3.11, matches repo)
#   CUDA_WHL     PyTorch wheel index tag        (default: cu128; set cu129 only
#                                                if your torch version ships it)
#   EXTRAS       optional pip extras            (default: empty; e.g. "[dev]",
#                                                "[all]")
#   FORCE        set to 1 to remove an existing (default: unset)
#                env of the same name first
#
# Examples:
#   ENV_PREFIX=/scratch/$USER/envs/dendritic_env bash setup/setup_cuda129.sh
#   EXTRAS="[dev]" bash setup/setup_cuda129.sh
#
set -euo pipefail

ENV_NAME="${ENV_NAME:-dendritic_env}"
ENV_PREFIX="${ENV_PREFIX:-}"
PY_VERSION="${PY_VERSION:-3.11}"
CUDA_WHL="${CUDA_WHL:-cu128}"
EXTRAS="${EXTRAS:-}"
FORCE="${FORCE:-}"

# Repo root = parent of this script's directory (setup/..), resolved absolutely.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Prefer mamba (faster solver) if available, else conda.
if command -v mamba >/dev/null 2>&1; then
  CONDA="mamba"
elif command -v conda >/dev/null 2>&1; then
  CONDA="conda"
else
  echo "ERROR: neither 'mamba' nor 'conda' found on PATH." >&2
  echo "On the cluster you may need: module load <miniforge/anaconda>" >&2
  exit 1
fi

# A named env (-n NAME) or a path-based env (--prefix PATH).
if [[ -n "${ENV_PREFIX}" ]]; then
  ENV_TARGET=(--prefix "${ENV_PREFIX}")
  ACTIVATE_REF="${ENV_PREFIX}"
else
  ENV_TARGET=(-n "${ENV_NAME}")
  ACTIVATE_REF="${ENV_NAME}"
fi

echo "==> Using ${CONDA}; repo root: ${REPO_ROOT}"
echo "==> Env target: ${ACTIVATE_REF}  (Python ${PY_VERSION}, PyTorch ${CUDA_WHL})"

# Optionally remove a pre-existing env of the same name/prefix.
if [[ "${FORCE}" == "1" ]]; then
  echo "==> FORCE=1: removing any existing env at ${ACTIVATE_REF}"
  "${CONDA}" env remove "${ENV_TARGET[@]}" -y || true
fi

echo "==> Creating conda env"
"${CONDA}" create "${ENV_TARGET[@]}" "python=${PY_VERSION}" pip -y

# Activate within this non-interactive shell (conda activate needs the hook).
echo "==> Activating env"
eval "$(conda shell.bash hook)"
conda activate "${ACTIVATE_REF}"

echo "==> Installing PyTorch + torchvision (${CUDA_WHL} wheels)"
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_WHL}"

echo "==> Installing dendritic_modeling (editable) + dependencies"
pip install -e "${REPO_ROOT}${EXTRAS}"

echo "==> Verifying package import + CLI (CPU-level checks)"
python -c "import dendritic_modeling; print('dendritic_modeling import: ok')"
python "${REPO_ROOT}/src/dendritic_modeling/scripts/training/train_experiments.py" --help >/dev/null \
  && echo "train_experiments --help: ok"

echo "==> Reporting torch build (cuda availability is False on login nodes)"
python - <<'PY'
import torch
print(f"torch {torch.__version__}  bundled-cuda {torch.version.cuda}  "
      f"cuda_available {torch.cuda.is_available()}")
PY

cat <<EOF

============================================================================
Done. Activate the environment before profiling with:

    conda activate ${ACTIVATE_REF}

Confirm the GPU build from inside a GPU allocation (not a login node):

    srun -p <gpu_partition> -A <account> --gres=gpu:1 --time=00:05:00 --pty \\
      python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
    # expect:  12.8  True   (cu128 runs on the cluster's CUDA 12.9 driver)

Then run the Rung 1 profile:

    python src/dendritic_modeling/scripts/training/train_experiments.py \\
        configs/examples/standard_mnist.yaml \\
        --profile --profile_dir profilings/PyTorch-General
============================================================================
EOF
