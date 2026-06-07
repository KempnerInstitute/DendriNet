#!/usr/bin/env bash
# Local DeepSeek-R1-0528 vLLM server using the Kempner shared model cache.
#
# This follows the KempnerInstitute/distributed-inference-vllm DeepSeek-R1-0528
# multinode recipe, using a project-local vLLM 0.8.5 / torch 2.6 CUDA 12.4
# environment built from envs/conda/c250609_vllm085.
#
# The shared model config reports model_type=deepseek_v3 and first_k_dense_replace=3,
# so it is architecture-compatible with our first-three-dense-layer replacement
# analysis, while avoiding Hugging Face API calls and rate limits.
#SBATCH --job-name=deepseek_r1_0528_vllm_tp8
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner_eng
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=1440G
#SBATCH --exclusive
#SBATCH --constraint=h200
#SBATCH --time=03-00:00:00
#SBATCH --chdir=/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling
#SBATCH --output=results/deepseek_v3_replacement/slurm/%j_deepseek_r1_0528_vllm.out
#SBATCH --error=results/deepseek_v3_replacement/slurm/%j_deepseek_r1_0528_vllm.err

set -euo pipefail

REPO_ROOT="/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling"
cd "${REPO_ROOT}"
mkdir -p results/deepseek_v3_replacement/slurm

MODEL_PATH="${MODEL_PATH:-/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/DeepSeek-R1-0528}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-DeepSeek-R1-0528}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.93}"

VLLM_ENV="${VLLM_ENV:-/n/holylabs/kempner_dev/Users/hsafaai/.venvs/vllm-deepseek-v085}"
VLLM_INSTALL_MISSING_ENV="${VLLM_INSTALL_MISSING_ENV:-0}"
export UV_BIN="${UV:-/n/home13/hsafaai/.local/bin/uv}"
export UV_LINK_MODE=copy
export UV_CACHE_DIR="${UV_CACHE_DIR:-/n/netscratch/kempner_dev/hsafaai/cache/uv}"
export HF_HOME="${HF_HOME:-/n/netscratch/kempner_dev/hsafaai/cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset TRANSFORMERS_CACHE

echo "[$(date)] model=${MODEL_PATH}"
echo "[$(date)] vLLM env=${VLLM_ENV}"

if [[ ! -x "${VLLM_ENV}/bin/vllm" ]]; then
  if [[ "${VLLM_INSTALL_MISSING_ENV}" != "1" ]]; then
    echo "Missing vLLM executable at ${VLLM_ENV}/bin/vllm" >&2
    echo "Build the pinned env first, or set VLLM_INSTALL_MISSING_ENV=1." >&2
    exit 3
  fi
  "${UV_BIN}" venv "${VLLM_ENV}"
  "${UV_BIN}" pip install --python "${VLLM_ENV}/bin/python" --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    vllm==0.8.5.post1 'ray[cgraph]==2.43.0' transformers==4.51.3
fi
source "${VLLM_ENV}/bin/activate"

VLLM_LIB_PATH="$("${VLLM_ENV}/bin/python" - <<'PY'
import glob
import os
import sys

paths = []
paths.extend(glob.glob(os.path.join(sys.prefix, "lib/python*/site-packages/torch/lib")))
paths.extend(glob.glob(os.path.join(sys.prefix, "lib/python*/site-packages/nvidia/*/lib")))
paths.extend(glob.glob(os.path.join(sys.prefix, "lib/python*/site-packages/nvidia/cu13/lib")))
print(":".join(paths))
PY
)"
export LD_LIBRARY_PATH="${VLLM_LIB_PATH}:${LD_LIBRARY_PATH:-}"
export PATH="${VLLM_ENV}/bin:${PATH}"
export RAY_DEDUP_LOGS=0
export LOGLEVEL="${LOGLEVEL:-DEBUG}"

# Multinode settings used by the Kempner vLLM workflows. The previous failure
# timed out inside vLLM CustomAllreduce during torch.distributed setup, so we
# force IPv4/IB communication and disable the custom all-reduce path.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ib0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2,mlx5_3,mlx5_4,mlx5_5}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"
export VLLM_DISTRIBUTED_EXECUTOR_CONFIG="${VLLM_DISTRIBUTED_EXECUTOR_CONFIG:-{\"placement_group_options\":{\"strategy\":\"SPREAD\"}}}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export VLLM_USE_DEEP_GEMM_E8M0="${VLLM_USE_DEEP_GEMM_E8M0:-0}"
export VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES="${VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES:-0}"

mapfile -t NODES < <(scontrol show hostnames "${SLURM_NODELIST}")
HEAD_NODE="${NODES[0]}"
HEAD_IP="$(srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname --ip-address | awk '{print $1; exit}')"
if [[ -z "${HEAD_IP}" ]]; then
  HEAD_IP="$(getent ahostsv4 "${HEAD_NODE}" | awk '{print $1; exit}')"
fi
if [[ -z "${HEAD_IP}" ]]; then
  echo "Could not determine IPv4 address for ${HEAD_NODE}" >&2
  exit 2
fi

USED_PORTS="$(ss -Htan | awk -F':' '{print $NF}' | sort -u)"
RAY_PORT="$(comm -23 <(seq 15000 20000) <(echo "${USED_PORTS}") | shuf -n 1)"
export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
export VLLM_HOST_IP="${HEAD_IP}"
export MASTER_ADDR="${HEAD_IP}"

cleanup() {
  set +e
  echo "[$(date)] stopping Ray runtime"
  for NODE in "${NODES[@]}"; do
    srun --overlap --nodes=1 --ntasks=1 -w "${NODE}" bash -lc "
      source '${VLLM_ENV}/bin/activate'
      NODE_IP=\$(hostname --ip-address | awk '{print \$1; exit}')
      export PATH='${VLLM_ENV}/bin':\"\${PATH}\"
      export LD_LIBRARY_PATH='${VLLM_LIB_PATH}':\"\${LD_LIBRARY_PATH:-}\"
      export VLLM_HOST_IP=\"\${NODE_IP}\"
      export NCCL_SOCKET_IFNAME='${NCCL_SOCKET_IFNAME}'
      export GLOO_SOCKET_IFNAME='${GLOO_SOCKET_IFNAME}'
      export NCCL_IB_HCA='${NCCL_IB_HCA}'
      export NCCL_SOCKET_FAMILY='${NCCL_SOCKET_FAMILY}'
      export NCCL_IB_DISABLE='${NCCL_IB_DISABLE}'
      export VLLM_ALLREDUCE_USE_SYMM_MEM='${VLLM_ALLREDUCE_USE_SYMM_MEM}'
      export VLLM_USE_DEEP_GEMM='${VLLM_USE_DEEP_GEMM}'
      export VLLM_MOE_USE_DEEP_GEMM='${VLLM_MOE_USE_DEEP_GEMM}'
      export VLLM_USE_DEEP_GEMM_E8M0='${VLLM_USE_DEEP_GEMM_E8M0}'
      export VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES='${VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES}'
      ray stop --force || true
    " || true
  done
}
trap cleanup EXIT TERM INT

echo "[$(date)] nodes=${NODES[*]}"
echo "[$(date)] head=${HEAD_NODE} ip=${HEAD_IP} ray_port=${RAY_PORT}"

echo "[$(date)] compute-node preflight"
srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" bash -lc "
  source '${VLLM_ENV}/bin/activate'
  NODE_IP=\$(hostname --ip-address | awk '{print \$1; exit}')
  export PATH='${VLLM_ENV}/bin':\"\${PATH}\"
  export LD_LIBRARY_PATH='${VLLM_LIB_PATH}':\"\${LD_LIBRARY_PATH:-}\"
  export VLLM_HOST_IP=\"\${NODE_IP}\"
  export NCCL_SOCKET_IFNAME='${NCCL_SOCKET_IFNAME}'
  export GLOO_SOCKET_IFNAME='${GLOO_SOCKET_IFNAME}'
  export NCCL_IB_HCA='${NCCL_IB_HCA}'
  export NCCL_SOCKET_FAMILY='${NCCL_SOCKET_FAMILY}'
  export NCCL_IB_DISABLE='${NCCL_IB_DISABLE}'
  export NCCL_DEBUG='${NCCL_DEBUG}'
  export VLLM_ALLREDUCE_USE_SYMM_MEM='${VLLM_ALLREDUCE_USE_SYMM_MEM}'
  export VLLM_USE_DEEP_GEMM='${VLLM_USE_DEEP_GEMM}'
  export VLLM_MOE_USE_DEEP_GEMM='${VLLM_MOE_USE_DEEP_GEMM}'
  export VLLM_USE_DEEP_GEMM_E8M0='${VLLM_USE_DEEP_GEMM_E8M0}'
  export VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES='${VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES}'
  python - <<'PY'
import os
import shutil
import torch

print('ray_bin', shutil.which('ray'))
print('vllm_bin', shutil.which('vllm'))
print('model_path_exists', os.path.exists('${MODEL_PATH}'))
print('VLLM_HOST_IP', os.environ.get('VLLM_HOST_IP'))
print('NCCL_SOCKET_IFNAME', os.environ.get('NCCL_SOCKET_IFNAME'))
print('GLOO_SOCKET_IFNAME', os.environ.get('GLOO_SOCKET_IFNAME'))
print('VLLM_USE_DEEP_GEMM', os.environ.get('VLLM_USE_DEEP_GEMM'))
print('VLLM_MOE_USE_DEEP_GEMM', os.environ.get('VLLM_MOE_USE_DEEP_GEMM'))
print('torch_cuda', torch.cuda.is_available(), torch.cuda.device_count())
if torch.cuda.is_available():
    print('gpu0', torch.cuda.get_device_name(0))
import ray
import vllm
import vllm._C
print('ray_version', ray.__version__)
print('vllm_version', vllm.__version__)
print('vllm_extension_import_ok')
PY
"

echo "[$(date)] starting Ray head on ${HEAD_NODE}"
srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" bash -lc "
  source '${VLLM_ENV}/bin/activate'
  NODE_IP=\$(hostname --ip-address | awk '{print \$1; exit}')
  export PATH='${VLLM_ENV}/bin':\"\${PATH}\"
  export LD_LIBRARY_PATH='${VLLM_LIB_PATH}':\"\${LD_LIBRARY_PATH:-}\"
  export VLLM_HOST_IP=\"\${NODE_IP}\"
  export NCCL_SOCKET_IFNAME='${NCCL_SOCKET_IFNAME}'
  export GLOO_SOCKET_IFNAME='${GLOO_SOCKET_IFNAME}'
  export NCCL_IB_HCA='${NCCL_IB_HCA}'
  export NCCL_SOCKET_FAMILY='${NCCL_SOCKET_FAMILY}'
  export NCCL_IB_DISABLE='${NCCL_IB_DISABLE}'
  export NCCL_DEBUG='${NCCL_DEBUG}'
  export VLLM_ALLREDUCE_USE_SYMM_MEM='${VLLM_ALLREDUCE_USE_SYMM_MEM}'
  export VLLM_DISTRIBUTED_EXECUTOR_CONFIG='${VLLM_DISTRIBUTED_EXECUTOR_CONFIG}'
  export VLLM_USE_DEEP_GEMM='${VLLM_USE_DEEP_GEMM}'
  export VLLM_MOE_USE_DEEP_GEMM='${VLLM_MOE_USE_DEEP_GEMM}'
  export VLLM_USE_DEEP_GEMM_E8M0='${VLLM_USE_DEEP_GEMM_E8M0}'
  export VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES='${VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES}'
  export RAY_DEDUP_LOGS='${RAY_DEDUP_LOGS}'
  ray stop --force || true
  ray start --head --node-ip-address='${HEAD_IP}' \
    --temp-dir /tmp/${USER}/${SLURM_JOB_ID}/ray \
    --port='${RAY_PORT}' \
    --num-cpus='${SLURM_CPUS_PER_TASK}' \
    --num-gpus='${SLURM_GPUS_ON_NODE:-4}' \
    --min-worker-port 20001 \
    --max-worker-port 30000 \
    --block
" &
sleep 20

for NODE in "${NODES[@]:1}"; do
  echo "[$(date)] starting Ray worker on ${NODE}"
  srun --overlap --nodes=1 --ntasks=1 -w "${NODE}" bash -lc "
    source '${VLLM_ENV}/bin/activate'
    NODE_IP=\$(hostname --ip-address | awk '{print \$1; exit}')
    export PATH='${VLLM_ENV}/bin':\"\${PATH}\"
    export LD_LIBRARY_PATH='${VLLM_LIB_PATH}':\"\${LD_LIBRARY_PATH:-}\"
    export VLLM_HOST_IP=\"\${NODE_IP}\"
    export NCCL_SOCKET_IFNAME='${NCCL_SOCKET_IFNAME}'
    export GLOO_SOCKET_IFNAME='${GLOO_SOCKET_IFNAME}'
    export NCCL_IB_HCA='${NCCL_IB_HCA}'
    export NCCL_SOCKET_FAMILY='${NCCL_SOCKET_FAMILY}'
    export NCCL_IB_DISABLE='${NCCL_IB_DISABLE}'
    export NCCL_DEBUG='${NCCL_DEBUG}'
    export VLLM_ALLREDUCE_USE_SYMM_MEM='${VLLM_ALLREDUCE_USE_SYMM_MEM}'
    export VLLM_DISTRIBUTED_EXECUTOR_CONFIG='${VLLM_DISTRIBUTED_EXECUTOR_CONFIG}'
    export VLLM_USE_DEEP_GEMM='${VLLM_USE_DEEP_GEMM}'
    export VLLM_MOE_USE_DEEP_GEMM='${VLLM_MOE_USE_DEEP_GEMM}'
    export VLLM_USE_DEEP_GEMM_E8M0='${VLLM_USE_DEEP_GEMM_E8M0}'
    export VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES='${VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES}'
    export RAY_DEDUP_LOGS='${RAY_DEDUP_LOGS}'
    ray stop --force || true
    ray start --address='${RAY_ADDRESS}' \
      --num-cpus='${SLURM_CPUS_PER_TASK}' \
      --num-gpus='${SLURM_GPUS_ON_NODE:-4}' \
      --min-worker-port 20001 \
      --max-worker-port 30000 \
      --block
  " &
  sleep 5
done
sleep 20

TP_SIZE="${TP_SIZE:-$((SLURM_NNODES * SLURM_GPUS_ON_NODE))}"
echo "[$(date)] launching vLLM local DeepSeek-R1-0528 server tp=${TP_SIZE}"
srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" bash -lc "
  source '${VLLM_ENV}/bin/activate'
  NODE_IP=\$(hostname --ip-address | awk '{print \$1; exit}')
  export PATH='${VLLM_ENV}/bin':\"\${PATH}\"
  export LD_LIBRARY_PATH='${VLLM_LIB_PATH}':\"\${LD_LIBRARY_PATH:-}\"
  export RAY_ADDRESS='${RAY_ADDRESS}'
  export VLLM_HOST_IP=\"\${NODE_IP}\"
  export MASTER_ADDR='${HEAD_IP}'
  export NCCL_SOCKET_IFNAME='${NCCL_SOCKET_IFNAME}'
  export GLOO_SOCKET_IFNAME='${GLOO_SOCKET_IFNAME}'
  export NCCL_IB_HCA='${NCCL_IB_HCA}'
  export NCCL_SOCKET_FAMILY='${NCCL_SOCKET_FAMILY}'
  export NCCL_IB_DISABLE='${NCCL_IB_DISABLE}'
  export NCCL_DEBUG='${NCCL_DEBUG}'
  export VLLM_ALLREDUCE_USE_SYMM_MEM='${VLLM_ALLREDUCE_USE_SYMM_MEM}'
  export VLLM_DISTRIBUTED_EXECUTOR_CONFIG='${VLLM_DISTRIBUTED_EXECUTOR_CONFIG}'
  export VLLM_USE_DEEP_GEMM='${VLLM_USE_DEEP_GEMM}'
  export VLLM_MOE_USE_DEEP_GEMM='${VLLM_MOE_USE_DEEP_GEMM}'
  export VLLM_USE_DEEP_GEMM_E8M0='${VLLM_USE_DEEP_GEMM_E8M0}'
  export VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES='${VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES}'
  export RAY_DEDUP_LOGS='${RAY_DEDUP_LOGS}'
  export HF_HUB_OFFLINE='${HF_HUB_OFFLINE}'
  export TRANSFORMERS_OFFLINE='${TRANSFORMERS_OFFLINE}'
  vllm serve '${MODEL_PATH}' \
    --trust-remote-code \
    --distributed-executor-backend ray \
    --tensor-parallel-size '${TP_SIZE}' \
    --served-model-name '${SERVED_MODEL_NAME}' \
    --max-model-len '${MAX_MODEL_LEN}' \
    --gpu-memory-utilization '${GPU_MEMORY_UTILIZATION}' \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 \
    --disable-custom-all-reduce \
    --enforce-eager \
    --host 0.0.0.0 \
    --port '${PORT}'
"
