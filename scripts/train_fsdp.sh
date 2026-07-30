#!/bin/bash
# ============================================================================
# FSDP training: multi-GPU fully-sharded training via torchrun
# ============================================================================
#
# Launches train_experiments_fsdp.py under torchrun for
# FullyShardedDataParallel.  Model parameters are sharded across GPUs,
# enabling training of models that exceed single-GPU memory.
#
# Single-node (default 4 GPUs):
#   sbatch scripts/train_fsdp.sh configs/vision/alexnet_k1_einet.yaml
#
# Override GPU count:
#   sbatch --gpus-per-node=2 scripts/train_fsdp.sh configs/my_config.yaml 2
#
# Multi-node (2 nodes x 4 GPUs = 8 GPUs):
#   sbatch --nodes=2 scripts/train_fsdp.sh configs/my_config.yaml 4
# ============================================================================
#SBATCH --job-name=train_fsdp
#SBATCH --partition=kempner_h100_priority
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_fsdp_%j.out
#SBATCH --error=logs/train_fsdp_%j.err

set -eo pipefail

REPO_ROOT="/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Keep the launcher independent of a shell-initialized Conda session.  Override
# PYTHON_BIN at submission time when testing another environment:
#   sbatch --export=ALL,PYTHON_BIN=/path/to/python scripts/train_fsdp.sh ...
PYTHON_BIN="${PYTHON_BIN:-/n/holylabs/LABS/kempner_dev/Users/hsafaai/conda_envs/vdc_paper_v2/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python is not executable: $PYTHON_BIN" >&2
    exit 2
fi
unset PYTHONHOME

CONFIG="${1:-configs/vision/alexnet_k1_einet.yaml}"
NGPUS="${2:-4}"
NNODES="${SLURM_NNODES:-1}"

# torchrun otherwise defaults to one OpenMP thread per worker. Divide the
# allocated CPUs evenly across local GPU workers so large CPU-side topology
# initialization can use the resources requested from Slurm.
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-$NGPUS}"
DEFAULT_THREADS_PER_RANK=$(( CPUS_PER_TASK / NGPUS ))
if (( DEFAULT_THREADS_PER_RANK < 1 )); then
    DEFAULT_THREADS_PER_RANK=1
fi
THREADS_PER_RANK="${DENDRITIC_FSDP_CPU_THREADS_PER_RANK:-$DEFAULT_THREADS_PER_RANK}"
if ! [[ "$THREADS_PER_RANK" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: DENDRITIC_FSDP_CPU_THREADS_PER_RANK must be a positive integer" >&2
    exit 2
fi
export OMP_NUM_THREADS="$THREADS_PER_RANK"
export MKL_NUM_THREADS="$THREADS_PER_RANK"

# NCCL settings for FASRC cluster
export NCCL_SOCKET_IFNAME=ib0,eth0
export NCCL_IB_DISABLE=0

# Resolve the master node explicitly to IPv4. Some Kempner compute-node
# hostnames advertise an IPv6 address family that c10d cannot use even though
# their routable fabric address is IPv4.
MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_ADDR=$(getent ahostsv4 "$MASTER_HOST" | awk 'NR == 1 {print $1}')
if [[ -z "$MASTER_ADDR" ]]; then
    echo "ERROR: Could not resolve an IPv4 address for $MASTER_HOST" >&2
    exit 2
fi
export MASTER_ADDR
export MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 1000 ))

echo "=== FSDP Training ==="
echo "Config: $CONFIG"
echo "Nodes:  $NNODES"
echo "GPUs per node: $NGPUS"
echo "Total GPUs:    $(( NNODES * NGPUS ))"
echo "CPU threads per rank: $OMP_NUM_THREADS"
echo "Master: $MASTER_HOST ($MASTER_ADDR):$MASTER_PORT"
echo "Job ID: ${SLURM_JOB_ID}"
echo ""

# srun launches one Python distributed coordinator per node. A static
# rendezvous is appropriate because Slurm fixes the node count and assigns one
# coordinator rank (SLURM_PROCID) per node. Explicit numeric master/local
# addresses prevent torchrun from advertising compute-node hostnames.
export PYTHON_BIN CONFIG NGPUS NNODES
srun bash -c '
    LOCAL_HOST=$(hostname -s)
    LOCAL_ADDR=$(getent ahostsv4 "$LOCAL_HOST" | awk "NR == 1 {print \$1}")
    LOCAL_ADDR=${LOCAL_ADDR%% *}
    if [[ -z "$LOCAL_ADDR" ]]; then
        echo "ERROR: Could not resolve a local IPv4 address for $LOCAL_HOST" >&2
        exit 2
    fi
    exec "$PYTHON_BIN" -m torch.distributed.run \
        --nnodes="$NNODES" \
        --nproc_per_node="$NGPUS" \
        --node_rank="$SLURM_PROCID" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        --local_addr="$LOCAL_ADDR" \
        src/dendritic_modeling/scripts/training/train_experiments_fsdp.py "$CONFIG"
'

echo ""
echo "=== FSDP Training completed ==="
