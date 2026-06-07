#!/usr/bin/env bash
#SBATCH --job-name=deepseek_v3_vllm_tp4pp2
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner_eng
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --chdir=/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling
#SBATCH --output=results/deepseek_v3_replacement/slurm/%j_vllm_tp8_ep.out
#SBATCH --error=results/deepseek_v3_replacement/slurm/%j_vllm_tp8_ep.err

set -euo pipefail

REPO_ROOT="/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling"
cd "${REPO_ROOT}"
mkdir -p results/deepseek_v3_replacement/slurm

export HF_HOME="${HF_HOME:-/n/holylabs/kempner_dev/Users/hsafaai/.cache/huggingface}"
unset TRANSFORMERS_CACHE
export RAY_TMPDIR="${SLURM_TMPDIR:-/tmp}/ray_${SLURM_JOB_ID}"
export VLLM_ENV="${VLLM_ENV:-/n/holylabs/kempner_dev/Users/hsafaai/.venvs/vllm-deepseek}"
export UV_LINK_MODE=copy

PYTHON_BIN="${PYTHON:-$(command -v python)}"
UV_BIN="${UV:-/n/home13/hsafaai/.local/bin/uv}"
MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-V3}"
PORT="${PORT:-8000}"
NETWORK_IFACE="${NETWORK_IFACE:-ib0}"
TP_SIZE="${TP_SIZE:-4}"
PP_SIZE="${PP_SIZE:-2}"

export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NETWORK_IFACE}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${NETWORK_IFACE}}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export MASTER_ADDR="${MASTER_ADDR:-}"

echo "[$(date)] installing/checking vLLM env at ${VLLM_ENV}"
if [[ ! -x "${VLLM_ENV}/bin/vllm" ]]; then
  "${UV_BIN}" venv "${VLLM_ENV}"
  "${UV_BIN}" pip install --python "${VLLM_ENV}/bin/python" -U vllm --torch-backend auto
fi
"${UV_BIN}" pip install --python "${VLLM_ENV}/bin/python" ray nvidia-cuda-runtime nvidia-cuda-nvrtc
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

mapfile -t NODES < <(scontrol show hostnames "${SLURM_NODELIST}")
HEAD_NODE="${NODES[0]}"
HEAD_IP="$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" bash -lc "
  ip -4 -o addr show dev '${NETWORK_IFACE}' | awk '{split(\$4, a, \"/\"); print a[1]; exit}'
")"
if [[ -z "${HEAD_IP}" ]]; then
  echo "Could not determine IPv4 address for ${NETWORK_IFACE} on ${HEAD_NODE}" >&2
  exit 2
fi
RAY_PORT=6379
export VLLM_HOST_IP="${HEAD_IP}"
export MASTER_ADDR="${HEAD_IP}"

cleanup() {
  set +e
  echo "[$(date)] stopping Ray runtime"
  for NODE in "${NODES[@]}"; do
    srun --overlap --nodes=1 --ntasks=1 -w "${NODE}" bash -lc "
      source '${VLLM_ENV}/bin/activate'
      export PATH='${VLLM_ENV}/bin':\"\${PATH}\"
      export LD_LIBRARY_PATH='${VLLM_LIB_PATH}':\"\${LD_LIBRARY_PATH:-}\"
      ray stop --force || true
    " || true
  done
}
trap cleanup EXIT TERM INT

echo "[$(date)] nodes=${NODES[*]}"
echo "[$(date)] head=${HEAD_NODE} ip=${HEAD_IP} iface=${NETWORK_IFACE}"
echo "[$(date)] GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "[$(date)] vLLM parallelism: tensor=${TP_SIZE} pipeline=${PP_SIZE}"

echo "[$(date)] compute-node import preflight"
srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" bash -lc "
  source '${VLLM_ENV}/bin/activate'
  export PATH='${VLLM_ENV}/bin':\"\${PATH}\"
  export LD_LIBRARY_PATH='${VLLM_LIB_PATH}':\"\${LD_LIBRARY_PATH:-}\"
  python - <<'PY'
import os
import shutil
import torch

print('ray_bin', shutil.which('ray'))
print('vllm_bin', shutil.which('vllm'))
print('network_iface', os.environ.get('GLOO_SOCKET_IFNAME'))
print('vllm_host_ip', os.environ.get('VLLM_HOST_IP'))
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

srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" bash -lc "
  source '${VLLM_ENV}/bin/activate'
  export PATH='${VLLM_ENV}/bin':\"\${PATH}\"
  export LD_LIBRARY_PATH='${VLLM_LIB_PATH}':\"\${LD_LIBRARY_PATH:-}\"
  export VLLM_HOST_IP='${HEAD_IP}'
  export MASTER_ADDR='${HEAD_IP}'
  export GLOO_SOCKET_IFNAME='${GLOO_SOCKET_IFNAME}'
  export NCCL_SOCKET_IFNAME='${NCCL_SOCKET_IFNAME}'
  export NCCL_IB_DISABLE='${NCCL_IB_DISABLE}'
  export NCCL_DEBUG='${NCCL_DEBUG}'
  export RAY_DEDUP_LOGS='${RAY_DEDUP_LOGS}'
  ray stop --force || true
  ray start --head --node-ip-address='${HEAD_IP}' --port=${RAY_PORT} --num-gpus=${SLURM_GPUS_ON_NODE:-4} --block
" &
HEAD_PID=$!
sleep 20

for NODE in "${NODES[@]:1}"; do
  srun --overlap --nodes=1 --ntasks=1 -w "${NODE}" bash -lc "
    source '${VLLM_ENV}/bin/activate'
    export PATH='${VLLM_ENV}/bin':\"\${PATH}\"
    export LD_LIBRARY_PATH='${VLLM_LIB_PATH}':\"\${LD_LIBRARY_PATH:-}\"
    export GLOO_SOCKET_IFNAME='${GLOO_SOCKET_IFNAME}'
    export NCCL_SOCKET_IFNAME='${NCCL_SOCKET_IFNAME}'
    export NCCL_IB_DISABLE='${NCCL_IB_DISABLE}'
    export NCCL_DEBUG='${NCCL_DEBUG}'
    export RAY_DEDUP_LOGS='${RAY_DEDUP_LOGS}'
    NODE_IP=\$(ip -4 -o addr show dev '${NETWORK_IFACE}' | awk '{split(\$4, a, \"/\"); print a[1]; exit}')
    if [[ -z \"\${NODE_IP}\" ]]; then
      echo 'Could not determine worker IP for ${NETWORK_IFACE} on ${NODE}' >&2
      exit 2
    fi
    ray stop --force || true
    ray start --address='${HEAD_IP}:${RAY_PORT}' --node-ip-address=\"\${NODE_IP}\" --num-gpus=${SLURM_GPUS_ON_NODE:-4} --block
  " &
done
sleep 30

echo "[$(date)] launching vLLM DeepSeek-V3 TP8+EP server"
srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" bash -lc "
  source '${VLLM_ENV}/bin/activate'
  export PATH='${VLLM_ENV}/bin':\"\${PATH}\"
  export LD_LIBRARY_PATH='${VLLM_LIB_PATH}':\"\${LD_LIBRARY_PATH:-}\"
  export VLLM_HOST_IP='${HEAD_IP}'
  export MASTER_ADDR='${HEAD_IP}'
  export GLOO_SOCKET_IFNAME='${GLOO_SOCKET_IFNAME}'
  export NCCL_SOCKET_IFNAME='${NCCL_SOCKET_IFNAME}'
  export NCCL_IB_DISABLE='${NCCL_IB_DISABLE}'
  export NCCL_DEBUG='${NCCL_DEBUG}'
  export RAY_DEDUP_LOGS='${RAY_DEDUP_LOGS}'
  vllm serve '${MODEL_NAME}' \
    --trust-remote-code \
    --distributed-executor-backend ray \
    --tensor-parallel-size '${TP_SIZE}' \
    --pipeline-parallel-size '${PP_SIZE}' \
    --disable-custom-all-reduce \
    --distributed-timeout-seconds 7200 \
    --enable-expert-parallel \
    --host 0.0.0.0 \
    --port '${PORT}'
"

wait "${HEAD_PID}"
