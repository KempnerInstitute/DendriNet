# CUDA + dendritic_env setup, in one go. SOURCE this on a GPU node:
#
#     source profilings/weight_transform_opt/setup_cuda_env.sh
#
# To first GET a GPU node:
#     salloc -p gpu --gres=gpu:1 -t 4:00:00 --mem=32G
#
# Must be SOURCED, not executed -- module/conda changes only persist in the current
# shell. The guard below stops you if you run it as ./setup_cuda_env.sh by mistake.

(return 0 2>/dev/null) && _SOURCED=1 || _SOURCED=0
if [ "$_SOURCED" -eq 0 ]; then
  echo "ERROR: source this, don't execute it  ->  source setup_cuda_env.sh"
  exit 1
fi

# 1. CUDA toolkit (nvcc, for building extensions). 12.9 matches torch's cu128 build.
module load cuda/12.9.1-fasrc01

# 2. Host compiler: torch 2.11 needs GCC >= 9, but Rocky8's default c++ is GCC 8.
#    Load a modern GCC and make it the host compiler for both c++ and nvcc (-ccbin).
#    (CUDA 12.9 supports GCC up to ~14, so 12.x is safe. Swap to a `module avail gcc` version.)
module load gcc/12.2.0-fasrc01
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"

# 3. Point CUDA_HOME at the real toolkit (torch's default guess /usr/local/cuda is wrong here).
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"

# 4. Activate the Python env (torch etc.). Bootstrap conda if it isn't on PATH yet.
if ! command -v conda >/dev/null 2>&1; then
  source /n/sw/Miniforge3-26.1.0-0/etc/profile.d/conda.sh
fi
conda activate dendritic_env

# --- verify everything resolved ---
echo "---- CUDA + dendritic_env ----"
echo "nvcc      : $(command -v nvcc || echo MISSING)"
nvcc --version 2>/dev/null | tail -1
echo "CUDA_HOME : ${CUDA_HOME:-MISSING}"
echo "g++       : $(command -v g++ || echo MISSING)  ($(g++ -dumpversion 2>/dev/null || echo '?'))"
echo "python    : $(command -v python)"
python - <<'PY' 2>/dev/null
import torch
ok = torch.cuda.is_available()
dev = torch.cuda.get_device_name(0) if ok else "none"
print(f"torch     : {torch.__version__} | cuda avail: {ok} | device: {dev}")
PY
echo "ready -> build/run CUDA scripts in profilings/weight_transform_opt/"
