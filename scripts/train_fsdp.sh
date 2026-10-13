#!/bin/bash
# ============================================================================
# FSDP training: multi-GPU fully-sharded training via torchrun
# ============================================================================
#
# Launches train_experiments_fsdp.py under torchrun for
# FullyShardedDataParallel.  Model parameters are sharded across GPUs,
# enabling training of models that exceed single-GPU memory.
#
# Submit from the checkout to run (its src/ is used), or set
# DENDRITIC_REPO_ROOT. The script names no partition or account: pass your
# site's values with sbatch flags (-p/--partition, -A/--account), or let Slurm
# use your defaults.
#
# Single-node (default 4 GPUs):
#   sbatch -p <partition> -A <account> scripts/train_fsdp.sh configs/config_exp_fsdp.yaml
#
# Override GPU count:
#   sbatch --gpus-per-node=2 scripts/train_fsdp.sh configs/my_config.yaml 2
#
# Multi-node (2 nodes x 4 GPUs = 8 GPUs):
#   sbatch --nodes=2 scripts/train_fsdp.sh configs/my_config.yaml 4
# ============================================================================
#SBATCH --job-name=train_fsdp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_fsdp_%j.out
#SBATCH --error=logs/train_fsdp_%j.err

set -eo pipefail

# The checkout to run: the submission directory unless DENDRITIC_REPO_ROOT is
# set (sbatch runs a spooled copy of this script, so its own path is not it).
REPO_ROOT="${DENDRITIC_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -d "$REPO_ROOT/src/dendritic_modeling" ]]; then
    echo "ERROR: $REPO_ROOT is not a dendritic-modeling checkout; submit from" \
        "one or set DENDRITIC_REPO_ROOT" >&2
    exit 2
fi
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# The interpreter of the environment to use; override at submission time:
#   sbatch --export=ALL,PYTHON_BIN=/path/to/python scripts/train_fsdp.sh ...
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: no executable Python; set PYTHON_BIN" >&2
    exit 2
fi
unset PYTHONHOME

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch scripts/train_fsdp.sh <config.yaml> [gpus_per_node]" >&2
    exit 2
fi
CONFIG="$1"
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG (relative to $REPO_ROOT)" >&2
    exit 2
fi
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

# NCCL network selection is site-specific. NCCL chooses an interface itself;
# if it picks the wrong one, export NCCL_SOCKET_IFNAME (for example your
# InfiniBand interface) before submitting. sbatch passes the environment on.

# Resolve the master node explicitly to IPv4. Some clusters advertise an IPv6
# address family for compute-node hostnames that c10d cannot use even though
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
echo "Checkout: $REPO_ROOT ($(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown))"
echo "Python: $PYTHON_BIN"
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
