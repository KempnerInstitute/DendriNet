#!/bin/bash
#SBATCH --job-name=cifar_2nodes_opt
#SBATCH --partition=kempner
#SBATCH --account=kempner_dev
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --mem=600G
#SBATCH --time=6:00:00
#SBATCH --output=logs/cifar_2nodes_%j.out
#SBATCH --error=logs/cifar_2nodes_%j.err

# CIFAR-10 2-Node Training - Optimized version with aggressive memory management

set -e  # Exit on any error

echo "=== SLURM Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Working Directory: $(pwd)"
echo "========================"

# NOTE: Don't load Python module - it conflicts with conda
# Only load CUDA if needed
# module load cuda/12.1.1-fasrc01

# Activate conda environment properly - first initialize conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /n/netscratch/kempner_dev/hsafaai/conda_envs/dendritic_env

# Show GPU info and verify conda
nvidia-smi
echo "Active conda env: $CONDA_DEFAULT_ENV"
echo "Python location: $(which python)"
echo "Conda prefix: $CONDA_PREFIX"

# Get free port function
get_free_port() {
    while true; do
        PORT=$(shuf -i 20000-30000 -n 1)
        if ! ss -lpn | grep -q ":$PORT "; then
            echo $PORT
            break
        fi
    done
}

# Set up distributed environment variables
export MASTER_ADDR=$(scontrol show hostnames | head -n 1)
export MASTER_PORT=$(get_free_port)
export PYTHONPATH=.:${PYTHONPATH}

# Additional environment variables for our training
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export OMP_NUM_THREADS=20

# Force IPv4 and network optimizations
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export NCCL_IGNORE_CPU_AFFINITY=1

# CUDA memory optimizations (less aggressive for smaller model)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.8,expandable_segments:True

# Python environment
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

echo "=== Network Configuration ==="
echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT" 
echo "World size: $WORLD_SIZE"
echo "PYTHONPATH: $PYTHONPATH"
echo "========================"

echo "=== Starting Training ==="
echo "Config: cifar_2nodes.yaml"
echo "Mixed precision: enabled"
echo "CPU offload: enabled"
echo "========================"

# Test Python environment first
echo "=== Environment Test ==="
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Check if config file exists
if [[ ! -f "configs/distributed/cifar_2nodes.yaml" ]]; then
    echo "ERROR: Config file configs/distributed/cifar_2nodes.yaml not found!"
    exit 1
fi

# Check if training script exists
if [[ ! -f "src/dendritic_modeling/scripts/train_large_model_fsdp.py" ]]; then
    echo "ERROR: Training script src/dendritic_modeling/scripts/train_large_model_fsdp.py not found!"
    exit 1
fi

# Create a wrapper script in the working directory (shared across nodes)
cat > run_training_wrapper.sh << 'EOF'
#!/bin/bash
set -e  # Exit on any error

# Initialize and activate conda environment on each node
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /n/netscratch/kempner_dev/hsafaai/conda_envs/dendritic_env

# Verify environment
echo "=== Node Environment Check ==="
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Conda prefix: $CONDA_PREFIX"

# Set environment variables
export PYTHONPATH=.:${PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.8,expandable_segments:True

# Test CUDA on this node
echo "=== CUDA Test on $(hostname) ==="
python -c "
import torch
import os
node_id = os.environ.get('SLURM_LOCALID', '?')
print(f'Node {node_id} ({os.uname().nodename}): PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA devices: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
" || {
    echo "CUDA test failed on $(hostname)"
    exit 1
}

# Ensure CIFAR-10 dataset is downloaded only once (global rank 0)
if [[ "${SLURM_PROCID:-0}" == "0" ]]; then
    echo "=== Pre-downloading CIFAR-10 dataset on rank 0 ==="
    python - << 'PY'
import os
from torchvision import datasets

root_dir = "/n/holylfs06/LABS/kempner_shared/Lab/data/cifar10/datasets/cifar"
os.makedirs(root_dir, exist_ok=True)

print("Downloading CIFAR-10 train split if needed ...")
datasets.CIFAR10(root=root_dir, train=True, download=True)
print("Downloading CIFAR-10 test split if needed ...")
datasets.CIFAR10(root=root_dir, train=False, download=True)
print("CIFAR-10 dataset is ready at", root_dir)
PY
    echo "=== Finished dataset download ==="
fi

sync  # simple barrier to make sure other ranks see the files

# Run the training with error handling
echo "=== Starting Training on $(hostname) at $(date) ==="
python src/dendritic_modeling/scripts/train_large_model_fsdp.py \
    configs/distributed/cifar_2nodes.yaml \
    --mixed_precision \
    --cpu_offload \
    --model_size small || {
    echo "Training failed on $(hostname) at $(date)"
    exit 1
}
echo "Training completed on $(hostname) at $(date)"
EOF

chmod +x run_training_wrapper.sh

# Run training with the wrapper script
echo "=== Launching distributed training ==="
srun --output=logs/cifar_2nodes_%j_%t.out --error=logs/cifar_2nodes_%j_%t.err ./run_training_wrapper.sh

# Clean up
rm -f run_training_wrapper.sh

echo "=== Job Completed ==="
echo "Job ID: $SLURM_JOB_ID completed at $(date)" 